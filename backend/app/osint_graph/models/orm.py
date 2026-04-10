"""
SQLAlchemy ORM models for the OSINT Identity Graph.

Tables:
    graph_face_nodes        - Face observations with 512D embeddings
    graph_identity_nodes    - Resolved identities (face clusters)
    graph_entity_nodes      - External entities (Wikipedia, Wikidata, etc.)
    graph_source_nodes      - Data provenance tracking
    graph_edges             - Weighted relationships between nodes

Uses the existing Base from app.database so all tables are created
together during init_db().
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ── Face Node ─────────────────────────────────────────────────────────────────

class GraphFaceNode(Base):
    __tablename__ = "graph_face_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Embedding stored as JSON text (reuses same pattern as PersonEmbedding)
    embedding_vec: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    angle_hint: Mapped[str] = mapped_column(String(32), default="frontal")

    # Foreign keys
    identity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_identity_nodes.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_source_nodes.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    # Optional link back to existing ORIX person
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    identity: Mapped["GraphIdentityNode | None"] = relationship(
        back_populates="faces", foreign_keys=[identity_id]
    )
    source: Mapped["GraphSourceNode | None"] = relationship(
        back_populates="faces", foreign_keys=[source_id]
    )


# ── Identity Node ────────────────────────────────────────────────────────────

class GraphIdentityNode(Base):
    __tablename__ = "graph_identity_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        default=lambda: str(uuid.uuid4())[:12],
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    # Cluster centroid stored as JSON text
    cluster_center_embedding: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    identity_score: Mapped[float] = mapped_column(Float, default=0.0)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Optional link back to existing ORIX person
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    faces: Mapped[list["GraphFaceNode"]] = relationship(
        back_populates="identity", foreign_keys=[GraphFaceNode.identity_id],
        lazy="select",
    )
    entity_links: Mapped[list["GraphEdge"]] = relationship(
        back_populates="source_identity",
        foreign_keys="GraphEdge.source_node_id",
        lazy="select",
    )


# ── Entity Node ──────────────────────────────────────────────────────────────

class GraphEntityNode(Base):
    __tablename__ = "graph_entity_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )
    external_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Source Node ──────────────────────────────────────────────────────────────

class GraphSourceNode(Base):
    __tablename__ = "graph_source_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    reliability_score: Mapped[float] = mapped_column(Float, default=0.5)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    faces: Mapped[list["GraphFaceNode"]] = relationship(
        back_populates="source", foreign_keys=[GraphFaceNode.source_id],
        lazy="select",
    )


# ── Graph Edge (adjacency table for all relationships) ───────────────────────

class GraphEdge(Base):
    """
    Universal edge table for the identity graph.

    edge_type values:
        face_to_identity    - FaceNode  -> IdentityNode  (similarity_score)
        identity_to_entity  - IdentityNode -> EntityNode (confidence_score)
        face_to_source      - FaceNode  -> SourceNode    (provenance_score)
        identity_to_identity - IdentityNode -> IdentityNode (cluster_similarity)
    """
    __tablename__ = "graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    edge_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    source_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    source_node_type: Mapped[str] = mapped_column(String(32), nullable=False)

    target_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    target_node_type: Mapped[str] = mapped_column(String(32), nullable=False)

    weight: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    source_identity: Mapped["GraphIdentityNode | None"] = relationship(
        back_populates="entity_links",
        foreign_keys=[source_node_id],
        primaryjoin="GraphEdge.source_node_id == GraphIdentityNode.id",
    )

    __table_args__ = (
        Index("ix_graph_edges_src_tgt", "source_node_id", "target_node_id"),
        Index("ix_graph_edges_type_src", "edge_type", "source_node_id"),
    )
