"""
db_worker.py
────────────
Consumes vectors from stream:vectors, searches pgvector across ALL embeddings
of ALL persons (grouped by person), picks the best match per face,
logs detections, and publishes the final event to stream:events.

Multi-embedding search strategy:
  SELECT person_id, name, MAX(1 - embedding <=> query) AS best_similarity
  FROM person_embeddings JOIN persons USING(person_id)
  GROUP BY person_id, name
  ORDER BY best_similarity DESC LIMIT 1

This means even if only one of a person's 5 embeddings matches well at
an odd angle, the person is still correctly identified.
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
import sqlalchemy
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CONSUMER_GROUP = "db_workers"
CONSUMER_NAME = "db_worker_0"

_running = True


def _shutdown(sig, frame):
    global _running
    _running = False
    log.info("db_worker_shutdown")


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def _ensure_group(rc: redis.Redis, stream: str) -> None:
    try:
        rc.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _sync_db_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


def _vec_pg(embedding: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


def _search_person(
    conn, embedding: List[float]
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Find the best-matching person by searching ALL their embeddings.

    Uses MAX() cosine similarity across every embedding for each person,
    so any registered angle can trigger a match.

    Returns (person_id, name, similarity) or (None, "Unknown", 0.0).
    """
    vec_str = _vec_pg(embedding)
    row = conn.execute(
        text("""
            SELECT
                p.id::text        AS person_id,
                p.name            AS name,
                MAX(1 - (pe.embedding <=> :vec::vector)) AS best_sim
            FROM person_embeddings pe
            JOIN persons p ON p.id = pe.person_id
            WHERE p.active = true
            GROUP BY p.id, p.name
            ORDER BY best_sim DESC
            LIMIT 1
        """),
        {"vec": vec_str},
    ).first()

    if row and row.best_sim >= settings.similarity_threshold:
        return row.person_id, row.name, float(row.best_sim)
    return None, "Unknown", 0.0


def _log_detection(
    conn,
    person_id: Optional[str],
    camera_id: str,
    confidence: float,
    quality: float,
    bbox: List[int],
    yaw: float,
    pitch: float,
    roll: float,
    timestamp: str,
) -> None:
    conn.execute(
        text("""
            INSERT INTO detection_logs
              (id, person_id, camera_id, confidence, quality_score,
               pitch, yaw, roll,
               bbox_x, bbox_y, bbox_w, bbox_h, detected_at)
            VALUES
              (gen_random_uuid(),
               :pid::uuid, :cam, :conf, :qual,
               :pitch, :yaw, :roll,
               :x, :y, :w, :h,
               :ts::timestamptz)
        """),
        {
            "pid": person_id,
            "cam": camera_id,
            "conf": confidence,
            "qual": quality,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3],
            "ts": timestamp,
        },
    )


def run() -> None:
    configure_logging(settings.worker_log_level)
    log.info("db_worker_start")

    rc = redis.from_url(settings.redis_url, decode_responses=True)
    in_stream = settings.stream_vectors
    out_stream = settings.stream_events

    _ensure_group(rc, in_stream)

    engine = sqlalchemy.create_engine(
        _sync_db_url(settings.database_url),
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    batch_size = settings.db_worker_batch_size

    while _running:
        try:
            results = rc.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {in_stream: ">"},
                count=batch_size,
                block=500,
            )
            if not results:
                continue

            with engine.begin() as conn:
                for _stream_name, messages in results:
                    for msg_id, fields in messages:
                        _process(rc, conn, out_stream, in_stream, msg_id, fields)

        except redis.ConnectionError:
            log.warning("db_worker_redis_reconnect")
            time.sleep(2)
        except sqlalchemy.exc.OperationalError:
            log.warning("db_worker_pg_reconnect")
            time.sleep(3)
        except Exception as exc:
            log.exception("db_worker_error", error=str(exc))
            time.sleep(1)

    log.info("db_worker_stopped")


def _process(
    rc: redis.Redis,
    conn,
    out_stream: str,
    in_stream: str,
    msg_id: str,
    fields: Dict[str, Any],
) -> None:
    camera_id = fields.get("camera_id", "unknown")
    timestamp = fields.get("timestamp", "")
    faces_json = fields.get("faces_json", "[]")

    try:
        faces: List[Dict[str, Any]] = json.loads(faces_json)
        bboxes: List[Dict[str, Any]] = []

        for face in faces:
            bbox = face["bbox"]
            embedding = face["embedding"]
            quality = face.get("quality", 1.0)
            yaw = face.get("yaw", 0.0)
            pitch = face.get("pitch", 0.0)
            roll = face.get("roll", 0.0)
            det_score = face.get("det_score", 1.0)
            angle_hint = face.get("angle_hint", "frontal")

            person_id, name, similarity = _search_person(conn, embedding)
            confidence = similarity if name != "Unknown" else det_score

            _log_detection(
                conn, person_id, camera_id,
                confidence, quality, bbox,
                yaw, pitch, roll, timestamp,
            )

            bboxes.append({
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "width": int(bbox[2]),
                "height": int(bbox[3]),
                "name": name,
                "confidence": round(confidence, 4),
                "quality": round(quality, 3),
                "angle": angle_hint,
            })

        if bboxes:
            event_payload = {
                "camera": camera_id,
                "timestamp": timestamp,
                "bboxes": bboxes,
            }
            rc.xadd(
                out_stream,
                {"payload": json.dumps(event_payload)},
                maxlen=settings.stream_max_len,
                approximate=True,
            )
            log.debug(
                "db_worker_event_published",
                camera_id=camera_id,
                faces=len(bboxes),
            )

    except Exception as exc:
        log.warning("db_worker_msg_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
