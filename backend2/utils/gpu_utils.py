"""
gpu_utils.py
────────────
Factory functions that build the face detector and embedder.
Abstracts away InsightFace vs MediaPipe selection based on config/availability.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from app.config import settings
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Shared data types ──────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    bbox: List[int]          # [x, y, width, height]
    crop: np.ndarray         # face chip (RGB)
    det_score: float


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[FaceDetection]: ...


class FaceEmbedder(Protocol):
    def embed(self, face_crop: np.ndarray) -> List[float]: ...


# ── InsightFace (RetinaFace + ArcFace) ────────────────────────────────────────

class InsightFaceDetector:
    def __init__(self) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        providers = settings.onnx_provider_list if settings.use_gpu else ["CPUExecutionProvider"]
        self._app = FaceAnalysis(
            name=settings.insightface_model,
            root=settings.model_dir,
            providers=providers,
        )
        ctx_id = settings.gpu_device_id if settings.use_gpu else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        log.info("insightface_loaded", model=settings.insightface_model, ctx_id=ctx_id)

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self._app.get(rgb)
        result: List[FaceDetection] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            crop = rgb[max(0, y1):y2, max(0, x1):x2]
            result.append(FaceDetection(
                bbox=[x1, y1, w, h],
                crop=crop,
                det_score=float(f.det_score),
            ))
        return result


class InsightFaceEmbedder:
    """
    ArcFace embedding is already computed inside FaceAnalysis.
    We use a shared detector object and extract normed embeddings.
    """

    def __init__(self, detector: InsightFaceDetector) -> None:
        self._app = detector._app

    def embed(self, face_crop: np.ndarray) -> List[float]:
        import cv2
        # Re-run get() on the crop to extract embedding via ArcFace
        faces = self._app.get(face_crop)
        if faces:
            emb = faces[0].normed_embedding
            return emb.tolist()
        # Fallback: zero vector
        return [0.0] * 512


# ── MediaPipe fallback (CPU) ───────────────────────────────────────────────────

class MediaPipeDetector:
    def __init__(self) -> None:
        import mediapipe as mp
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=settings.detection_confidence
        )
        log.info("mediapipe_detector_loaded")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        results = self._detector.process(rgb)
        faces: List[FaceDetection] = []
        if not results.detections:
            return faces
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            fw = int(bb.width * w)
            fh = int(bb.height * h)
            crop = rgb[y:y + fh, x:x + fw]
            score = det.score[0] if det.score else 0.0
            faces.append(FaceDetection(bbox=[x, y, fw, fh], crop=crop, det_score=float(score)))
        return faces


class MediaPipeEmbedder:
    """
    Lightweight embedding using pixel-level mean (placeholder).
    Replace with a proper FaceNet/MobileFaceNet model for production CPU use.
    """

    def embed(self, face_crop: np.ndarray) -> List[float]:
        import cv2
        resized = cv2.resize(face_crop, (112, 112)).astype(np.float32) / 255.0
        # Flatten and take first 512 dims (toy fallback)
        flat = resized.flatten()[:512]
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat /= norm
        return flat.tolist()


# ── Public factories ───────────────────────────────────────────────────────────

def build_detector() -> FaceDetector:
    backend = settings.detector_backend.lower()
    if backend == "insightface":
        try:
            return InsightFaceDetector()
        except Exception as e:
            log.warning("insightface_fallback", error=str(e))
    log.info("using_mediapipe_detector")
    return MediaPipeDetector()


def build_embedder(detector: Optional[object] = None) -> FaceEmbedder:
    backend = settings.embedder_backend.lower()
    if backend == "arcface" and isinstance(detector, InsightFaceDetector):
        return InsightFaceEmbedder(detector)
    if backend == "arcface":
        try:
            det = detector or InsightFaceDetector()
            return InsightFaceEmbedder(det)
        except Exception as e:
            log.warning("arcface_fallback", error=str(e))
    log.info("using_mediapipe_embedder")
    return MediaPipeEmbedder()
