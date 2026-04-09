"""
db_worker.py
────────────
Consumes embedding vectors from `stream:vectors`, searches pgvector for the
nearest known face, writes detection logs to PostgreSQL, and publishes the
final recognition event to `stream:events` (consumed by the WS relay).

Final event format (published to stream:events as JSON under key "payload"):
{
    "camera": "cam_01",
    "timestamp": "2026-04-08T12:00:00+00:00",
    "bboxes": [
        {"x": 120, "y": 60, "width": 100, "height": 100, "name": "Carlos", "confidence": 0.87}
    ]
}
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
    log.info("db_worker_shutdown_requested")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def _ensure_group(rc: redis.Redis, stream: str) -> None:
    try:
        rc.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _sync_db_url(url: str) -> str:
    """Convert asyncpg URL to psycopg2 for sync worker."""
    return url.replace("postgresql+asyncpg://", "postgresql://")


def _vec_to_pg(embedding: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


def _search_person(
    conn, embedding: List[float]
) -> Tuple[Optional[str], Optional[str], float]:
    """Returns (person_id, name, similarity)."""
    vec_str = _vec_to_pg(embedding)
    result = conn.execute(
        text(
            "SELECT id::text, name, 1 - (embedding <=> :vec::vector) AS similarity "
            "FROM persons WHERE active = true "
            "ORDER BY embedding <=> :vec::vector LIMIT 1"
        ),
        {"vec": vec_str},
    ).first()
    if result and result.similarity >= settings.similarity_threshold:
        return result.id, result.name, float(result.similarity)
    return None, "Unknown", 0.0


def _log_detection(
    conn,
    person_id: Optional[str],
    camera_id: str,
    confidence: float,
    bbox: List[int],
    timestamp: str,
) -> None:
    conn.execute(
        text(
            "INSERT INTO detection_logs "
            "(id, person_id, camera_id, confidence, bbox_x, bbox_y, bbox_w, bbox_h, detected_at) "
            "VALUES (gen_random_uuid(), :pid::uuid, :cam, :conf, :x, :y, :w, :h, :ts::timestamptz)"
        ),
        {
            "pid": person_id,
            "cam": camera_id,
            "conf": confidence,
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
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

    # Sync SQLAlchemy engine (workers are synchronous processes)
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
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {in_stream: ">"},
                count=batch_size,
                block=500,
            )
            if not results:
                continue

            with engine.begin() as conn:
                for _stream, messages in results:
                    for msg_id, fields in messages:
                        _process_message(rc, conn, out_stream, in_stream, msg_id, fields)

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


def _process_message(
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
            bbox = face["bbox"]          # [x, y, w, h]
            embedding = face["embedding"]
            det_score = face.get("det_score", 1.0)

            person_id, name, similarity = _search_person(conn, embedding)
            confidence = similarity if name != "Unknown" else det_score

            _log_detection(conn, person_id, camera_id, confidence, bbox, timestamp)

            bboxes.append(
                {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2]),
                    "height": int(bbox[3]),
                    "name": name,
                    "confidence": round(confidence, 4),
                }
            )

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
            log.debug("db_worker_event_published", camera_id=camera_id, faces=len(bboxes))

    except Exception as exc:
        log.warning("db_worker_msg_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
