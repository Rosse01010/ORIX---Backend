"""
schemas/models.py
─────────────────
Pydantic models for OSINT request/response contracts.
These are API schemas only — not SQLAlchemy ORM models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Provider match result ─────────────────────────────────────────────────────

class OSINTMatch(BaseModel):
    """Single match returned by an OSINT provider."""
    source: str                          # provider name
    confidence: float = Field(ge=0.0, le=1.0)
    external_id: str                     # ID within the source system
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Aggregated report ─────────────────────────────────────────────────────────

class OSINTReport(BaseModel):
    """Full intelligence report aggregating results across all providers."""
    query_id: str
    matches: List[OSINTMatch] = Field(default_factory=list)
    risk_score: float = Field(ge=0, le=100)
    providers_queried: List[str] = Field(default_factory=list)
    embedding_dim: int = 512
    timestamp: str
    cached: bool = False
    processing_time_ms: float = 0.0


# ── API request schemas ───────────────────────────────────────────────────────

class OSINTSearchRequest(BaseModel):
    """POST /api/osint/search body."""
    embedding: List[float]
    top_k: int = Field(default=10, ge=1, le=100)


class OSINTEnrichRequest(BaseModel):
    """POST /api/osint/enrich-face — can accept either a face_id or an image upload."""
    face_id: Optional[str] = None        # existing person_id in ORIX DB
    top_k: int = Field(default=10, ge=1, le=100)


# ── Audit log entry ──────────────────────────────────────────────────────────

class OSINTAuditEntry(BaseModel):
    """Every OSINT query is logged for compliance."""
    query_id: str
    timestamp: str
    embedding_hash: str                  # SHA-256 of the embedding (not the embedding itself)
    providers_used: List[str]
    matches_found: int
    risk_score: float
    requester_ip: Optional[str] = None
