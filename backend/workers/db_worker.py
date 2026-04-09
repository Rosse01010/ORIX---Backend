"""
db_worker.py
────────────
Consumes vectors from stream:vectors, searches for matches across ALL
embeddings grouped by person, logs detections, and publishes events to
stream:events.

Similarity thresholds (ArcFace cosine similarity, Deng et al. CVPR 2019):
  ≥ settings.similarity_threshold  → auto-identify
  settings.candidate_min_sim – threshold → uncertain, show candidate panel
  |yaw| > settings.candidate_yaw_threshold → show panel even if "recognised"
    (off-axis false accepts are a known failure mode per VGGFace2 paper)
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
from app.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CONSUMER_GROUP = "db_workers"
CONSUMER_NAME  = "db_worker_0"

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


def _search_best(conn, embedding: List[float]) -> Tuple[Optional[str], str, float]:
    """Best single match using numpy cosine similarity (ArcFace metric)."""
    from app.utils.vector_search import search_best_sync
    return search_best_sync(
        conn,
        embedding,
        settings.similarity_threshold,
        settings.candidate_min_sim,
    )


def _search_candidates(conn, embedding: List[float]) -> List[Dict[str, Any]]:
    """Top-K candidate persons for the similarity panel."""
    from app.utils.vector_search import search_candidates_sync
    return search_candidates_sync(
        conn,
        embedding,
        settings.candidate_min_sim,
        top_k=5,
    )


def _log_detection(conn, person_id, camera_id, confidence, quality,
                   bbox, yaw, pitch, roll, timestamp) -> None:
    conn.execute(
        text("""
            INSERT INTO detection_logs
              (id, person_id, camera_id, confidence, quality_score,
               pitch, yaw, roll, bbox_x, bbox_y, bbox_w, bbox_h, detected_at)
            VALUES
              (gen_random_uuid(),
               :pid::uuid, :cam, :conf, :qual,
               :pitch, :yaw, :roll,
               :x, :y, :w, :h, :ts::timestamptz)
        """),
        {
            "pid": person_id, "cam": camera_id,
            "conf": confidence, "qual": quality,
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3],
            "ts": timestamp,
        },
    )


def run() -> None:
    configure_logging(settings.worker_log_level)
    log.info(
        "db_worker_start",
        similarity_threshold=settings.similarity_threshold,
        candidate_min_sim=settings.candidate_min_sim,
        candidate_yaw_threshold=settings.candidate_yaw_threshold,
    )

    rc     = redis.from_url(settings.redis_url, decode_responses=True)
    in_s   = settings.stream_vectors
    out_s  = settings.stream_events

    _ensure_group(rc, in_s)

    engine = sqlalchemy.create_engine(
        _sync_db_url(settings.database_url),
        pool_size=5, max_overflow=10, pool_pre_ping=True,
    )

    while _running:
        try:
            results = rc.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {in_s: ">"}, count=settings.db_worker_batch_size, block=500,
            )
            if not results:
                continue
            with engine.begin() as conn:
                for _, messages in results:
                    for msg_id, fields in messages:
                        _process(rc, conn, out_s, in_s, msg_id, fields)
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


def _process(rc, conn, out_stream, in_stream, msg_id, fields) -> None:
    camera_id  = fields.get("camera_id", "unknown")
    timestamp  = fields.get("timestamp", "")
    faces_json = fields.get("faces_json", "[]")

    try:
        faces: List[Dict] = json.loads(faces_json)
        bboxes: List[Dict]          = []
        candidates_list: List[Dict] = []

        for idx, face in enumerate(faces):
            bbox       = face["bbox"]
            embedding  = face["embedding"]
            quality    = face.get("quality", 1.0)
            yaw        = face.get("yaw", 0.0)
            pitch      = face.get("pitch", 0.0)
            roll       = face.get("roll", 0.0)
            det_score  = face.get("det_score", 1.0)
            angle_hint = face.get("angle_hint", "frontal")

            person_id, name, similarity = _search_best(conn, embedding)
            confidence = similarity if name != "Unknown" else det_score

            _log_detection(conn, person_id, camera_id, confidence,
                           quality, bbox, yaw, pitch, roll, timestamp)

            # ── Confidence tier classification ─────────────────────────────
            # Based on ArcFace cosine similarity distributions:
            #   ≥ 0.55 → high confidence
            #   0.40–0.54 → moderate (flag for operator)
            #   0.30–0.39 → low (always show candidates)
            if name != "Unknown" and similarity >= 0.55:
                confidence_tier = "high"
            elif name != "Unknown" and similarity >= settings.similarity_threshold:
                confidence_tier = "moderate"
            else:
                confidence_tier = "low"

            bbox_out = {
                "x": int(bbox[0]), "y": int(bbox[1]),
                "width": int(bbox[2]), "height": int(bbox[3]),
                "name": name,
                "confidence": round(confidence, 4),
                "confidence_tier": confidence_tier,
                "quality": round(quality, 3),
                "angle": angle_hint,
                "face_index": idx,
            }
            bboxes.append(bbox_out)

            # ── Candidate panel trigger ────────────────────────────────────
            # Show panel when:
            #   (a) face is Unknown (no match above threshold)
            #   (b) match is "moderate" confidence (0.40–0.54)
            #   (c) yaw is large enough to risk off-axis false accept
            #       (VGGFace2: front-to-profile drops ~15–20% similarity)
            is_unknown    = name == "Unknown"
            is_moderate   = confidence_tier == "moderate"
            is_off_axis   = abs(yaw) > settings.candidate_yaw_threshold

            if is_unknown or is_moderate or is_off_axis:
                candidates = _search_candidates(conn, embedding)
                if candidates:
                    candidates_list.append({
                        "face_index": idx,
                        "bbox": bbox_out,
                        "is_unknown": is_unknown,
                        "confidence_tier": confidence_tier,
                        "yaw": round(yaw, 1),
                        "top_matches": candidates,
                    })

        if bboxes:
            event_payload = {
                "camera": camera_id,
                "timestamp": timestamp,
                "bboxes": bboxes,
                "candidates": candidates_list,
            }
            rc.xadd(out_stream, {"payload": json.dumps(event_payload)},
                    maxlen=settings.stream_max_len, approximate=True)
            log.debug(
                "db_worker_published",
                camera=camera_id,
                faces=len(bboxes),
                candidates=len(candidates_list),
            )

    except Exception as exc:
        log.warning("db_worker_msg_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
