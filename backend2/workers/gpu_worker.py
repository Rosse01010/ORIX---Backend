"""
gpu_worker.py
─────────────
Consumes frames from `stream:frames`, runs face detection + embedding on GPU,
publishes embedding vectors + bounding boxes to `stream:vectors`.

Each output message on stream:vectors:
  camera_id  – source camera
  timestamp  – original frame timestamp
  faces_json – JSON list of {bbox, embedding, det_score}

Architecture:
  • Uses InsightFace (RetinaFace detector + ArcFace embedder) by default.
  • Falls back to MediaPipe (CPU) if GPU is unavailable.
  • Batches up to GPU_WORKER_BATCH_SIZE frames per loop iteration.
"""
from __future__ import annotations

import base64
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from utils.gpu_utils import build_detector, build_embedder
from utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CONSUMER_GROUP = "gpu_workers"
CONSUMER_NAME = f"gpu_worker_0"

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("gpu_worker_shutdown_requested")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def _b64_to_frame(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _ensure_group(rc: redis.Redis, stream: str) -> None:
    try:
        rc.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def run() -> None:
    configure_logging(settings.worker_log_level)
    log.info("gpu_worker_start", backend=settings.detector_backend, gpu=settings.use_gpu)

    rc = redis.from_url(settings.redis_url, decode_responses=True)
    in_stream = settings.stream_frames
    out_stream = settings.stream_vectors

    _ensure_group(rc, in_stream)

    detector = build_detector()
    embedder = build_embedder()
    log.info("models_loaded", detector=type(detector).__name__, embedder=type(embedder).__name__)

    batch_size = settings.gpu_worker_batch_size
    timeout_ms = settings.gpu_worker_timeout_ms

    while _running:
        try:
            results = rc.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {in_stream: ">"},
                count=batch_size,
                block=timeout_ms,
            )
            if not results:
                continue

            for _stream, messages in results:
                for msg_id, fields in messages:
                    _process_message(rc, out_stream, msg_id, fields, in_stream, detector, embedder)

        except redis.ConnectionError:
            log.warning("gpu_worker_redis_reconnect")
            time.sleep(2)
        except Exception as exc:
            log.exception("gpu_worker_error", error=str(exc))
            time.sleep(0.5)

    log.info("gpu_worker_stopped")


def _process_message(
    rc: redis.Redis,
    out_stream: str,
    msg_id: str,
    fields: Dict[str, Any],
    in_stream: str,
    detector,
    embedder,
) -> None:
    camera_id = fields.get("camera_id", "unknown")
    timestamp = fields.get("timestamp", "")
    frame_b64 = fields.get("frame_b64", "")

    try:
        frame = _b64_to_frame(frame_b64)
        faces = detector.detect(frame)

        faces_data: List[Dict[str, Any]] = []
        for face in faces:
            if face.det_score < settings.detection_confidence:
                continue
            if face.bbox[2] < settings.min_face_size or face.bbox[3] < settings.min_face_size:
                continue
            embedding = embedder.embed(face.crop)
            faces_data.append(
                {
                    "bbox": face.bbox,          # [x, y, w, h]
                    "embedding": embedding,      # list[float] len=512
                    "det_score": float(face.det_score),
                }
            )

        if faces_data:
            rc.xadd(
                out_stream,
                {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "faces_json": json.dumps(faces_data),
                },
                maxlen=settings.stream_max_len,
                approximate=True,
            )
            log.debug("gpu_worker_processed", camera_id=camera_id, faces=len(faces_data))

    except Exception as exc:
        log.warning("gpu_worker_frame_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
