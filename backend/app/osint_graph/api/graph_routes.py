"""
OSINT Graph API Routes.

Extends the ORIX API with identity graph endpoints:
    POST /api/osint-graph/resolve      - Resolve embedding to identity
    GET  /api/osint-graph/identity/{id} - Get identity detail
    POST /api/osint-graph/merge        - Merge two identities
    POST /api/osint-graph/enrich       - Add entity links to identity
    GET  /api/osint-graph/search       - Search identities by embedding
    GET  /api/osint-graph/stats        - Graph statistics
    POST /api/osint-graph/import       - Import existing ORIX persons
    POST /api/osint-graph/ingest       - Batch ingest embeddings
    POST /api/osint-graph/sources      - Create a data source
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_dep
from app.osint_graph.core.graph_engine import GraphEngine
from app.osint_graph.ingestion.graph_builder import GraphBuilder

router = APIRouter(prefix="/api/osint-graph", tags=["OSINT Graph"])


# ── Request / Response Schemas ───────────────────────────────────────────────

class ResolveRequest(BaseModel):
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    image_url: Optional[str] = None
    quality_score: float = Field(1.0, ge=0.0, le=1.0)
    angle_hint: str = "frontal"
    name_hint: Optional[str] = None
    enrich_entities: bool = False


class ResolveResponse(BaseModel):
    identity_id: Optional[str]
    confidence: float
    linked_entities: List[Dict[str, Any]] = []
    graph_neighbors: List[str] = []
    face_id: Optional[str] = None
    action: Optional[str] = None
    identity_score: Optional[float] = None
    name: Optional[str] = None
    candidates: Optional[List[Dict[str, Any]]] = None


class MergeRequest(BaseModel):
    source_identity_id: str
    target_identity_id: str
    reason: str = "manual_merge"


class EnrichRequest(BaseModel):
    identity_id: str
    name: str


class SearchRequest(BaseModel):
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    top_k: int = Field(10, ge=1, le=100)
    min_similarity: float = Field(0.3, ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    embeddings: List[Dict[str, Any]]
    source_type: str = "dataset"
    source_name: str = "batch_import"
    dataset_key: Optional[str] = None


class CreateSourceRequest(BaseModel):
    source_type: str
    name: str
    url: Optional[str] = None
    reliability_score: float = Field(0.5, ge=0.0, le=1.0)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/resolve", response_model=ResolveResponse)
async def resolve_face(
    req: ResolveRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """
    Resolve a 512D ArcFace embedding to an identity in the graph.

    The system will:
    1. Search nearest identity centroids
    2. Classify similarity (same_identity / candidate_merge / new)
    3. Create or update identity and face nodes
    4. Optionally enrich with Wikipedia/Wikidata entities
    """
    engine = GraphEngine(db)
    result = await engine.process_face(
        embedding=req.embedding,
        image_url=req.image_url,
        quality_score=req.quality_score,
        angle_hint=req.angle_hint,
        name_hint=req.name_hint,
        enrich_entities=req.enrich_entities,
    )
    return ResolveResponse(
        identity_id=result.get("identity_id"),
        confidence=result.get("similarity", 0.0),
        linked_entities=result.get("linked_entities", []),
        graph_neighbors=result.get("graph_neighbors", []),
        face_id=result.get("face_id"),
        action=result.get("action"),
        identity_score=result.get("identity_score"),
        name=result.get("name"),
        candidates=result.get("candidates"),
    )


@router.get("/identity/{identity_id}")
async def get_identity(
    identity_id: str,
    db: AsyncSession = Depends(get_db_dep),
):
    """Get full identity detail with linked faces, entities, and graph."""
    engine = GraphEngine(db)
    try:
        iid = uuid.UUID(identity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid identity ID")

    result = await engine.get_identity_detail(iid)
    if not result:
        raise HTTPException(status_code=404, detail="Identity not found")
    return result


@router.post("/merge")
async def merge_identities(
    req: MergeRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """
    Merge two identities into one (admin operation).

    All faces and entity links from source are moved to target.
    Source identity is deactivated.
    """
    engine = GraphEngine(db)
    try:
        src = uuid.UUID(req.source_identity_id)
        tgt = uuid.UUID(req.target_identity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid identity IDs")

    result = await engine.merge_identities(src, tgt, req.reason)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/enrich")
async def enrich_identity(
    req: EnrichRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """
    Enrich an identity with Wikipedia/Wikidata entity links.
    Requires a name to search for.
    """
    engine = GraphEngine(db)
    try:
        iid = uuid.UUID(req.identity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid identity ID")

    return await engine.enrich_identity(iid, req.name)


@router.post("/search")
async def search_identities(
    req: SearchRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """Search for identities by embedding similarity."""
    engine = GraphEngine(db)
    results = await engine.search_identities(
        embedding=req.embedding,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
    )
    return {"results": results, "count": len(results)}


@router.get("/stats")
async def get_graph_stats(
    db: AsyncSession = Depends(get_db_dep),
):
    """Get counts of all graph node and edge types."""
    engine = GraphEngine(db)
    return await engine.get_graph_stats()


@router.post("/import")
async def import_existing_persons(
    db: AsyncSession = Depends(get_db_dep),
):
    """
    Import all existing ORIX persons into the identity graph.
    Creates identity nodes, face nodes, and provenance edges.
    """
    builder = GraphBuilder(db)
    return await builder.import_existing_persons()


@router.post("/ingest")
async def ingest_batch(
    req: IngestRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """
    Ingest a batch of pre-computed embeddings into the graph.

    Each embedding item should have:
        {"embedding": [512 floats], "name": "optional", "labels": ["optional"]}
    """
    builder = GraphBuilder(db)
    return await builder.ingest_embedding_batch(
        embeddings=req.embeddings,
        source_type=req.source_type,
        source_name=req.source_name,
        dataset_key=req.dataset_key,
    )


@router.post("/sources")
async def create_source(
    req: CreateSourceRequest,
    db: AsyncSession = Depends(get_db_dep),
):
    """Create a new data source node for provenance tracking."""
    engine = GraphEngine(db)
    source_id = await engine.create_source(
        source_type=req.source_type,
        name=req.name,
        url=req.url,
        reliability_score=req.reliability_score,
    )
    return {"source_id": source_id}
