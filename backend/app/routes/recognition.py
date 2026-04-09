"""
recognition.py
──────────────
REST endpoints for facial recognition and person management.

Key improvements based on ArcFace / VGGFace2 papers:
  • Enrollment stores per-angle embeddings PLUS a L2-normalised template
    embedding (mean of all angles) following the VGGFace2 template-
    aggregation strategy (Cao et al., 2018).
  • Recognition always re-normalises embeddings before search to guard
    against client-submitted non-unit vectors.
  • The /enroll endpoint returns the confidence tier so the UI can warn
    when only one frontal photo was provided.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_dep
from app.models import Person, PersonEmbedding

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


@router.get("/health")
async def recognition_health():
    return {
        "status": "ok",
        "similarity_threshold": settings.similarity_threshold,
        "candidate_min_sim": settings.candidate_min_sim,
    }


# ── Schemas ────────────────────────────────────────────────────────────────────

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    name: str
    confidence: float
    confidence_tier: str = "low"   # "high" | "moderate" | "low"
    quality: float = 0.0
    angle: str = "frontal"


class RecognitionResponse(BaseModel):
    camera: str
    timestamp: str
    bboxes: List[BBox]


class PersonOut(BaseModel):
    id: str
    name: str
    active: bool
    embedding_count: int
    created_at: str


# ── Lazy model loader ──────────────────────────────────────────────────────────

_detector = None
_embedder = None


def _get_models():
    global _detector, _embedder
    if _detector is None:
        from app.utils.gpu_utils import build_detector, build_embedder
        _detector = build_detector()
        _embedder = build_embedder(_detector)
    return _detector, _embedder


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-10)


def _classify_tier(similarity: float, name: str) -> str:
    """Map cosine similarity to a confidence tier (ArcFace ranges)."""
    if name == "Unknown":
        return "low"
    if similarity >= 0.55:
        return "high"
    if similarity >= settings.similarity_threshold:
        return "moderate"
    return "low"


# ── Search ─────────────────────────────────────────────────────────────────────

async def _search_person(
    db: AsyncSession, embedding: List[float]
) -> tuple[Optional[str], str, float]:
    from app.utils.vector_search import search_best_async
    return await search_best_async(db, embedding, settings.similarity_threshold)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    camera: str = Form("upload"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> RecognitionResponse:
    """Synchronous recognition on a single uploaded image."""
    import cv2
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw
    from app.utils.preprocessing import preprocess_frame

    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = preprocess_frame(
        img_bgr, settings.camera_resize_width, settings.camera_resize_height
    )

    detector, embedder = _get_models()
    faces = detector.detect(img_bgr)
    bboxes: List[BBox] = []

    for face in faces:
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        if quality < 0.1:
            continue

        embedding = embedder.embed(face.crop)
        # Ensure unit norm (ArcFace should already output this, but guard anyway)
        emb_norm = _l2_normalize(np.array(embedding, dtype=np.float32)).tolist()

        _, name, confidence = await _search_person(db, emb_norm)
        tier = _classify_tier(confidence, name)
        angle = angle_hint_from_yaw(yaw)
        x, y = face.bbox[0], face.bbox[1]

        bboxes.append(BBox(
            x=x, y=y, width=w, height=h,
            name=name,
            confidence=round(confidence, 4),
            confidence_tier=tier,
            quality=round(quality, 3),
            angle=angle,
        ))

    return RecognitionResponse(
        camera=camera,
        timestamp=datetime.now(timezone.utc).isoformat(),
        bboxes=bboxes,
    )


@router.get("/persons", response_model=List[PersonOut])
async def list_persons(
    db: AsyncSession = Depends(get_db_dep),
) -> List[PersonOut]:
    result = await db.execute(select(Person).where(Person.active == True))
    persons = result.scalars().all()
    out = []
    for p in persons:
        count_r = await db.execute(
            text("SELECT COUNT(*) FROM person_embeddings WHERE person_id = :pid"),
            {"pid": str(p.id)},
        )
        count = count_r.scalar() or 0
        out.append(PersonOut(
            id=str(p.id),
            name=p.name,
            active=p.active,
            embedding_count=count,
            created_at=p.created_at.isoformat(),
        ))
    return out


@router.post("/persons", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def register_person(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> PersonOut:
    """
    Register a person with one or more photos.
    Multiple photos at different angles dramatically improve recognition
    (VGGFace2 paper shows cross-pose similarity drops ~15–20% with
    single-angle enrollment).

    A L2-normalised template embedding (mean of all angle embeddings) is
    also stored (angle_hint="template") for fast single-shot retrieval.
    """
    import cv2
    import json as _json
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw

    detector, embedder = _get_models()
    person = Person(name=name)
    db.add(person)
    await db.flush()

    all_embeddings: List[List[float]] = []

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = detector.detect(img_bgr)
        if not faces:
            continue

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        raw_emb = embedder.embed(face.crop)
        embedding = _l2_normalize(
            np.array(raw_emb, dtype=np.float32)
        ).tolist()

        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding),
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        all_embeddings.append(embedding)

    if not all_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No face detected in any of the uploaded images.",
        )

    # Store template embedding (VGGFace2-style aggregation) when > 1 photo
    if len(all_embeddings) > 1:
        from app.utils.vector_search import compute_template_embedding
        template_emb = compute_template_embedding(all_embeddings)
        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(template_emb),
            angle_hint="template",
            quality_score=1.0,
        ))

    await db.flush()

    return PersonOut(
        id=str(person.id),
        name=person.name,
        active=person.active,
        embedding_count=len(all_embeddings),
        created_at=person.created_at.isoformat(),
    )


@router.post("/persons/{person_id}/photos", response_model=Dict[str, Any])
async def add_photos(
    person_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> Dict[str, Any]:
    """Add more photos (angles) to an existing person. Updates the template embedding."""
    import cv2
    import json as _json
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw

    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id), Person.active == True)
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    detector, embedder = _get_models()
    new_embeddings: List[List[float]] = []

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = detector.detect(img_bgr)
        if not faces:
            continue

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        raw_emb = embedder.embed(face.crop)
        embedding = _l2_normalize(np.array(raw_emb, dtype=np.float32)).tolist()

        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding),
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        new_embeddings.append(embedding)

    # Rebuild template embedding from ALL existing embeddings
    if new_embeddings:
        existing_r = await db.execute(
            text(
                "SELECT embedding_vec FROM person_embeddings "
                "WHERE person_id = :pid AND angle_hint != 'template'"
            ),
            {"pid": person_id},
        )
        all_vecs = [
            _json.loads(row[0]) for row in existing_r.fetchall()
        ]
        if len(all_vecs) > 1:
            from app.utils.vector_search import compute_template_embedding
            template_emb = compute_template_embedding(all_vecs)
            # Replace existing template
            await db.execute(
                text(
                    "DELETE FROM person_embeddings "
                    "WHERE person_id = :pid AND angle_hint = 'template'"
                ),
                {"pid": person_id},
            )
            db.add(PersonEmbedding(
                person_id=person.id,
                embedding_vec=_json.dumps(template_emb),
                angle_hint="template",
                quality_score=1.0,
            ))

    await db.flush()
    return {"person_id": person_id, "photos_added": len(new_embeddings)}


@router.delete("/persons/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> None:
    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id))
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    person.active = False


# ── Browser enrollment (face-api.js embedding) ────────────────────────────────

class BrowserEnrollPayload(BaseModel):
    name: str
    embedding: List[float]
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None


class BrowserEnrollResponse(BaseModel):
    person_id: str
    name: str
    embedding_dim: int
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None
    warning: Optional[str] = None


@router.post("/persons/enroll", response_model=BrowserEnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_person_browser(
    payload: BrowserEnrollPayload,
    db: AsyncSession = Depends(get_db_dep),
) -> BrowserEnrollResponse:
    """
    Register a person using a pre-computed face embedding from the browser.
    The embedding is L2-normalised before storage to ensure ArcFace metric
    compatibility regardless of the client-side model's output scale.
    """
    import json as _json

    emb = _l2_normalize(np.array(payload.embedding, dtype=np.float32)).tolist()
    dim = len(emb)
    warning = None
    if dim != 512:
        warning = (
            f"Embedding dimension is {dim}, expected 512. "
            "Recognition accuracy may be reduced."
        )

    person = Person(
        name=payload.name,
        linkedin_url=payload.linkedin_url,
        instagram_handle=payload.instagram_handle,
        twitter_handle=payload.twitter_handle,
        notes=payload.notes,
    )
    db.add(person)
    await db.flush()

    db.add(PersonEmbedding(
        person_id=person.id,
        embedding_vec=_json.dumps(emb),
        angle_hint="frontal",
        quality_score=1.0,
    ))
    await db.flush()

    return BrowserEnrollResponse(
        person_id=str(person.id),
        name=person.name,
        embedding_dim=dim,
        linkedin_url=person.linkedin_url,
        instagram_handle=person.instagram_handle,
        twitter_handle=person.twitter_handle,
        notes=person.notes,
        warning=warning,
    )


class PersonDetailOut(BaseModel):
    id: str
    name: str
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None
    created_at: str


@router.get("/persons/{person_id}", response_model=PersonDetailOut)
async def get_person(
    person_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> PersonDetailOut:
    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id), Person.active == True)
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return PersonDetailOut(
        id=str(person.id),
        name=person.name,
        linkedin_url=person.linkedin_url,
        instagram_handle=person.instagram_handle,
        twitter_handle=person.twitter_handle,
        notes=person.notes,
        created_at=person.created_at.isoformat(),
    )
