"""
FaceNode — represents a single detected face with its ArcFace 512D embedding.

Each face is linked to exactly one IdentityNode (resolved via clustering)
and one SourceNode (provenance tracking).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class FaceNodeSchema(BaseModel):
    """Pydantic schema for API I/O."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    image_url: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    quality_score: float = Field(0.0, ge=0.0, le=1.0)
    angle_hint: str = "frontal"
    identity_id: Optional[str] = None
    source_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class FaceNodeCreate(BaseModel):
    """Schema for creating a new face node."""
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    image_url: Optional[str] = None
    quality_score: float = Field(1.0, ge=0.0, le=1.0)
    angle_hint: str = "frontal"
    source_id: Optional[str] = None
