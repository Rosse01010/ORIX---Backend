"""
utils/audit.py
──────────────
Audit logging for OSINT queries.

Every OSINT query is logged with:
  - Timestamp
  - SHA-256 hash of the input embedding (never the embedding itself)
  - Providers used
  - Number of matches
  - Risk score

Logs to both structured logger and the OSINT audit DB table.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List

from app.osint.schemas.models import OSINTAuditEntry
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def hash_embedding(embedding: List[float]) -> str:
    """SHA-256 hash of embedding for audit trail (not reversible)."""
    raw = ",".join(f"{v:.6f}" for v in embedding).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_audit_entry(
    query_id: str,
    embedding: List[float],
    providers_used: List[str],
    matches_found: int,
    risk_score: float,
    requester_ip: str | None = None,
) -> OSINTAuditEntry:
    """Build an audit entry and log it."""
    entry = OSINTAuditEntry(
        query_id=query_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        embedding_hash=hash_embedding(embedding),
        providers_used=providers_used,
        matches_found=matches_found,
        risk_score=risk_score,
        requester_ip=requester_ip,
    )

    log.info(
        "osint_audit",
        query_id=entry.query_id,
        providers=entry.providers_used,
        matches=entry.matches_found,
        risk_score=entry.risk_score,
    )

    return entry


async def persist_audit_entry(entry: OSINTAuditEntry) -> None:
    """Write audit entry to the database."""
    from app.database import AsyncSessionLocal
    from sqlalchemy import text

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    INSERT INTO osint_audit_log
                        (id, query_id, timestamp, embedding_hash,
                         providers_used, matches_found, risk_score, requester_ip)
                    VALUES
                        (gen_random_uuid(), :qid, :ts::timestamptz, :hash,
                         :providers, :matches, :risk, :ip)
                """),
                {
                    "qid": entry.query_id,
                    "ts": entry.timestamp,
                    "hash": entry.embedding_hash,
                    "providers": ",".join(entry.providers_used),
                    "matches": entry.matches_found,
                    "risk": entry.risk_score,
                    "ip": entry.requester_ip,
                },
            )
            await session.commit()
    except Exception as exc:
        log.warning("osint_audit_persist_error", error=str(exc))
