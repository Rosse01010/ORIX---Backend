"""
core/models.py
──────────────
SQLAlchemy ORM model for the OSINT audit log table.

Isolated from the main app/models.py to avoid touching the core schema.
The table is created automatically if OSINT is enabled.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class OSINTAuditLog(Base):
    __tablename__ = "osint_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    embedding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    providers_used: Mapped[str] = mapped_column(Text, nullable=False, default="")
    matches_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    requester_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
