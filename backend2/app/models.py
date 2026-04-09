"""
SQLAlchemy ORM models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

EMBEDDING_DIM = 512   # ArcFace R100 output dimension


# ── Person ─────────────────────────────────────────────────────────────────────

class Person(Base):
    """Identity registered in the system."""

    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    embeddings: Mapped[list["PersonEmbedding"]] = relationship(
        back_populates="person", lazy="select", cascade="all, delete-orphan"
    )
    logs: Mapped[list["DetectionLog"]] = relationship(
        back_populates="person", lazy="select"
    )


class PersonEmbedding(Base):
    """
    One face embedding per angle/photo for a person.
    Multiple embeddings per person dramatically improve multi-angle recognition.
    """

    __tablename__ = "person_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    # Hint about the capture angle: frontal | left | right | up | down
    angle_hint: Mapped[str] = mapped_column(String(32), default="frontal")
    # Quality score 0.0–1.0 (sharpness * pose_score * detection_confidence)
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    person: Mapped["Person"] = relationship(back_populates="embeddings")

    # HNSW index for fast ANN across all embeddings
    __table_args__ = (
        Index(
            "ix_person_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 32, "ef_construction": 128},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ── Detection Log ──────────────────────────────────────────────────────────────

class DetectionLog(Base):
    """Audit log of every recognition event."""

    __tablename__ = "detection_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
    camera_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    pitch: Mapped[float] = mapped_column(Float, default=0.0)   # head pose
    yaw: Mapped[float] = mapped_column(Float, default=0.0)
    roll: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_x: Mapped[int] = mapped_column(nullable=False)
    bbox_y: Mapped[int] = mapped_column(nullable=False)
    bbox_w: Mapped[int] = mapped_column(nullable=False)
    bbox_h: Mapped[int] = mapped_column(nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    person: Mapped["Person | None"] = relationship(back_populates="logs")


# ── App Users ──────────────────────────────────────────────────────────────────

class OrixUser(Base):
    """Dashboard login user."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Camera ─────────────────────────────────────────────────────────────────────

class Camera(Base):
    """Physical or virtual camera."""

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    stream_url: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="online")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
