"""
SourceNode — provenance tracking for face data.

Every face in the graph has a source: Wikipedia, Wikidata, a HuggingFace
dataset, or a user upload. This enables audit trails and reliability scoring.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


SOURCE_TYPES = ("wikipedia", "wikidata", "dataset", "user_upload", "api")

RELIABILITY_WEIGHTS = {
    "wikipedia": 0.7,
    "wikidata": 0.8,
    "dataset": 0.9,
    "user_upload": 0.5,
    "api": 0.6,
}


class SourceNodeSchema(BaseModel):
    """Full source node representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str
    name: str
    url: Optional[str] = None
    reliability_score: float = Field(0.5, ge=0.0, le=1.0)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class SourceNodeCreate(BaseModel):
    """Schema for creating a new source node."""
    source_type: str
    name: str
    url: Optional[str] = None
    reliability_score: Optional[float] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
