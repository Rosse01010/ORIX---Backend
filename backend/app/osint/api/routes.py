"""
api/routes.py
─────────────
REST endpoints for the OSINT subsystem.

All endpoints are gated behind settings.osint_enabled.
The router is conditionally mounted in app/main.py.

Endpoints:
  GET  /api/osint/health       — OSINT subsystem + provider health
  POST /api/osint/search       — Search by raw 512-dim embedding
  GET  /api/osint/report/{id}  — Retrieve cached report by query_id
  POST /api/osint/enrich-face  — Upload image → extract embedding → run OSINT
"""
from __future__ import annotations

import io
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_dep
from app.osint.schemas.models import OSINTReport, OSINTSearchRequest
from app.osint.services.osint_service import get_osint_service

router = APIRouter(prefix="/api/osint", tags=["osint"])


# ── Guard ─────────────────────────────────────────────────────────────────────

def _require_osint_enabled() -> None:
    if not settings.osint_enabled:
        raise HTTPException(
            status_code=403,
            detail="OSINT subsystem is disabled. Set OSINT_ENABLED=true to activate.",
        )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def osint_health():
    """OSINT subsystem health check including all registered providers."""
    _require_osint_enabled()
    svc = get_osint_service()
    provider_health = await svc.health()
    return {
        "status": "ok",
        "osint_enabled": settings.osint_enabled,
        "providers": provider_health,
        "cache_ttl_seconds": settings.osint_cache_ttl_seconds,
    }


# ── Search by embedding ──────────────────────────────────────────────────────

@router.post("/search", response_model=OSINTReport)
async def osint_search(
    body: OSINTSearchRequest,
    request: Request,
) -> OSINTReport:
    """
    Search all OSINT providers using a raw 512-dim ArcFace embedding.

    Returns aggregated matches with risk score.
    """
    _require_osint_enabled()

    if len(body.embedding) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must be 512-dimensional, got {len(body.embedding)}.",
        )

    svc = get_osint_service()
    client_ip = request.client.host if request.client else None
    report = await svc.search(
        embedding=body.embedding,
        top_k=body.top_k,
        requester_ip=client_ip,
    )
    return report


# ── Retrieve cached report ────────────────────────────────────────────────────

@router.get("/report/{query_id}", response_model=OSINTReport)
async def osint_get_report(query_id: str) -> OSINTReport:
    """Retrieve a previously computed OSINT report from cache."""
    _require_osint_enabled()

    svc = get_osint_service()
    report = await svc.get_report(query_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report {query_id} not found. It may have expired from cache.",
        )
    return report


# ── Enrich face (image upload or existing face_id) ───────────────────────────

@router.post("/enrich-face", response_model=OSINTReport)
async def osint_enrich_face(
    request: Request,
    face_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
    db: AsyncSession = Depends(get_db_dep),
) -> OSINTReport:
    """
    Enrich a face with OSINT intelligence.

    Two modes:
      1. Provide face_id → looks up existing embedding from ORIX DB.
      2. Provide image file → extracts embedding using InsightFaceService.

    Then runs the full OSINT pipeline and returns an enriched report.
    """
    import uuid as _uuid
    _require_osint_enabled()

    embedding: List[float] | None = None

    # Mode 1: face_id → look up existing embedding
    if face_id:
        from sqlalchemy import text
        result = await db.execute(
            text(
                "SELECT pe.embedding_vec FROM person_embeddings pe "
                "WHERE pe.person_id = :pid::uuid "
                "AND pe.embedding_version = 'arcface_r100_v1' "
                "ORDER BY pe.quality_score DESC LIMIT 1"
            ),
            {"pid": face_id},
        )
        row = result.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Person or embedding not found.")

        import json
        vec_data = row.embedding_vec
        if isinstance(vec_data, str):
            embedding = json.loads(vec_data)
        else:
            embedding = list(vec_data)

    # Mode 2: image file → extract embedding via InsightFaceService
    elif file is not None:
        import cv2
        from app.services.insightface_service import get_insightface_service

        contents = await file.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        svc = get_insightface_service()
        faces = svc.detect_faces(img_bgr, max_faces=1)
        if not faces:
            raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")

        face = faces[0]
        embedding = svc.extract_embedding(face.crop)

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either face_id or an image file.",
        )

    if len(embedding) != 512:
        raise HTTPException(
            status_code=422,
            detail=f"Extracted embedding has {len(embedding)} dimensions, expected 512.",
        )

    client_ip = request.client.host if request.client else None
    osint_svc = get_osint_service()
    report = await osint_svc.search(
        embedding=embedding,
        top_k=top_k,
        requester_ip=client_ip,
    )
    return report
