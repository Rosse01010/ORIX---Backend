"""
IdentityNode — represents a resolved identity (a cluster of face embeddings).

An identity is NOT a single face. It is the resolved, clustered entity
that aggregates multiple face observations into a single canonical person.

The cluster_center_embedding is the L2-normalised centroid of all faces
assigned to this identity.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IdentityNodeSchema(BaseModel):
    """Full identity node representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    canonical_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    cluster_center_embedding: List[float] = Field(
        default_factory=lambda: [0.0] * 512
    )
    identity_score: float = Field(0.0, ge=0.0, le=100.0)
    face_count: int = Field(0, ge=0)
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class IdentityNodeBrief(BaseModel):
    """Lightweight identity representation for list responses."""
    id: str
    canonical_id: str
    name: Optional[str] = None
    identity_score: float = 0.0
    face_count: int = 0


class IdentityMergeRequest(BaseModel):
    """Request to merge two identities into one."""
    source_identity_id: str
    target_identity_id: str
    reason: str = "manual_merge"
