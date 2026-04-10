"""
EntityNode — represents an external entity linked to an identity.

Entity types: person, organization, dataset, image_source.
Sources: Wikipedia, Wikidata, HuggingFace datasets, user metadata.

This enables the intelligence graph:
    Face -> Identity -> Entity (Wikipedia page, organisation, etc.)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EntityNodeSchema(BaseModel):
    """Full entity node representation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str = Field(
        ..., pattern="^(person|organization|dataset|image_source)$"
    )
    name: str
    description: Optional[str] = None
    external_id: Optional[str] = None
    external_url: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class EntityNodeCreate(BaseModel):
    """Schema for creating a new entity node."""
    entity_type: str = Field(
        ..., pattern="^(person|organization|dataset|image_source)$"
    )
    name: str
    description: Optional[str] = None
    external_id: Optional[str] = None
    external_url: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class EntityLinkSchema(BaseModel):
    """Represents a link between an identity and an entity."""
    identity_id: str
    entity_id: str
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    link_type: str = "associated_with"
    evidence: Optional[str] = None
