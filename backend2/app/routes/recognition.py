"""
REST endpoints for facial recognition.

POST /recognize  – synchronous recognition on an uploaded image.
GET  /persons    – list all known persons.
POST /persons    – register a new person + embedding.
DELETE /persons/{id} – remove a person.
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
from app.models import Person

router = APIRouter(prefix="/api", tags=["recognition"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    name: str
    confidence: float


class RecognitionResponse(BaseModel):
    camera: str
    timestamp: str
    bboxes: List[BBox]


class PersonOut(BaseModel):
    id: str
    name: str
    active: bool
    created_at: str


class PersonCreate(BaseModel):
    name: str


# ── Lazy model loader ──────────────────────────────────────────────────────────

_detector = None
_embedder = None


def _get_models():
    global _detector, _embedder
    if _detector is None:
        from utils.gpu_utils import build_detector, build_embedder
        _detector = build_detector()
        _embedder = build_embedder()
    return _detector, _embedder


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    camera: str = Form("upload"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> RecognitionResponse:
    """Run face detection + recognition on a single uploaded image."""
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    detector, embedder = _get_models()

    from utils.preprocessing import preprocess_frame
    img = preprocess_frame(img, settings.camera_resize_width, settings.camera_resize_height)

    faces = detector.detect(img)
    bboxes: List[BBox] = []

    for face in faces:
        embedding = embedder.embed(face.crop)
        name, confidence = await _search_person(db, embedding)
        x, y, w, h = face.bbox
        bboxes.append(BBox(x=x, y=y, width=w, height=h, name=name, confidence=round(confidence, 4)))

    return RecognitionResponse(
        camera=camera,
        timestamp=datetime.now(timezone.utc).isoformat(),
        bboxes=bboxes,
    )


async def _search_person(
    db: AsyncSession, embedding: List[float]
) -> tuple[str, float]:
    """Find nearest neighbor in pgvector, return (name, cosine_similarity)."""
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
    result = await db.execute(
        text(
            "SELECT name, 1 - (embedding <=> :vec::vector) AS similarity "
            "FROM persons WHERE active = true "
            "ORDER BY embedding <=> :vec::vector LIMIT 1"
        ),
        {"vec": vec_str},
    )
    row = result.first()
    if row and row.similarity >= settings.similarity_threshold:
        return row.name, float(row.similarity)
    return "Unknown", 0.0


@router.get("/persons", response_model=List[PersonOut])
async def list_persons(
    db: AsyncSession = Depends(get_db_dep),
) -> List[PersonOut]:
    result = await db.execute(select(Person).where(Person.active == True))
    persons = result.scalars().all()
    return [
        PersonOut(
            id=str(p.id),
            name=p.name,
            active=p.active,
            created_at=p.created_at.isoformat(),
        )
        for p in persons
    ]


@router.post("/persons", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def register_person(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> PersonOut:
    """Register a person by uploading their reference photo."""
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    detector, embedder = _get_models()
    faces = detector.detect(img)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in the image.")

    # Use the largest face
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    embedding = embedder.embed(face.crop)

    person = Person(name=name, embedding=embedding)
    db.add(person)
    await db.flush()

    return PersonOut(
        id=str(person.id),
        name=person.name,
        active=person.active,
        created_at=person.created_at.isoformat(),
    )


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
