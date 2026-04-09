"""
services/osint_service.py
─────────────────────────
Central orchestrator for the OSINT subsystem.

Responsibilities:
  1. Register and manage providers
  2. Normalise input embeddings
  3. Fan out queries to all enabled providers
  4. Aggregate and deduplicate matches
  5. Compute risk score
  6. Cache results in Redis
  7. Audit-log every query
  8. Emit Socket.IO events for frontend
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as aioredis

from app.config import settings
from app.osint.core.provider import OSINTProvider
from app.osint.core.risk_scoring import compute_risk_score
from app.osint.schemas.models import OSINTMatch, OSINTReport
from app.osint.utils.audit import create_audit_entry, persist_audit_entry
from app.osint.utils.similarity import to_numpy_embedding, validate_embedding_dim
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

# Redis key prefix for cached OSINT reports
_CACHE_PREFIX = "osint:report:"


class OSINTService:
    """
    Singleton orchestrator for OSINT queries.

    Usage:
        svc = OSINTService()
        svc.register_provider(LocalDatabaseProvider())
        report = await svc.search(embedding, top_k=10)
    """

    def __init__(self) -> None:
        self._providers: Dict[str, OSINTProvider] = {}
        self._redis: Optional[aioredis.Redis] = None

    # ── Provider management ───────────────────────────────────────────

    def register_provider(self, provider: OSINTProvider) -> None:
        self._providers[provider.name] = provider
        log.info("osint_provider_registered", provider=provider.name)

    def list_providers(self) -> List[str]:
        return [
            name for name, p in self._providers.items()
            if p.enabled
        ]

    # ── Redis cache ───────────────────────────────────────────────────

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return self._redis

    async def _cache_get(self, query_id: str) -> Optional[OSINTReport]:
        try:
            rc = await self._get_redis()
            data = await rc.get(f"{_CACHE_PREFIX}{query_id}")
            if data:
                return OSINTReport.model_validate_json(data)
        except Exception as exc:
            log.debug("osint_cache_miss", error=str(exc))
        return None

    async def _cache_set(self, report: OSINTReport) -> None:
        try:
            rc = await self._get_redis()
            await rc.setex(
                f"{_CACHE_PREFIX}{report.query_id}",
                settings.osint_cache_ttl_seconds,
                report.model_dump_json(),
            )
        except Exception as exc:
            log.debug("osint_cache_set_error", error=str(exc))

    # ── Core search ───────────────────────────────────────────────────

    async def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        requester_ip: str | None = None,
    ) -> OSINTReport:
        """
        Run OSINT pipeline across all registered providers.

        Args:
            embedding:     512-dim ArcFace embedding.
            top_k:         Max results per provider.
            requester_ip:  Caller IP for audit logging.

        Returns:
            OSINTReport with aggregated matches and risk score.
        """
        t_start = time.monotonic()
        query_id = str(uuid.uuid4())

        # Validate embedding
        if not validate_embedding_dim(embedding, 512):
            return OSINTReport(
                query_id=query_id,
                matches=[],
                risk_score=0.0,
                providers_queried=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=0.0,
            )

        # Normalise
        _ = to_numpy_embedding(embedding)  # validates and normalises

        # Fan out to all enabled providers concurrently
        enabled = {
            name: p for name, p in self._providers.items()
            if p.enabled
        }

        async def _safe_query(name: str, provider: OSINTProvider) -> List[OSINTMatch]:
            try:
                return await asyncio.wait_for(
                    provider.search_by_embedding(embedding, top_k),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                log.warning("osint_provider_timeout", provider=name)
                return []
            except Exception as exc:
                log.warning("osint_provider_error", provider=name, error=str(exc))
                return []

        tasks = {
            name: asyncio.create_task(_safe_query(name, p))
            for name, p in enabled.items()
        }
        await asyncio.gather(*tasks.values())

        # Aggregate results
        all_matches: List[OSINTMatch] = []
        providers_queried: List[str] = []
        for name, task in tasks.items():
            providers_queried.append(name)
            all_matches.extend(task.result())

        # Sort by confidence descending
        all_matches.sort(key=lambda m: m.confidence, reverse=True)

        # Compute risk score
        reliability_map = {
            name: p.reliability for name, p in self._providers.items()
        }
        risk = compute_risk_score(all_matches, reliability_map)

        elapsed_ms = (time.monotonic() - t_start) * 1000

        report = OSINTReport(
            query_id=query_id,
            matches=all_matches,
            risk_score=risk,
            providers_queried=providers_queried,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=round(elapsed_ms, 1),
        )

        # Cache
        await self._cache_set(report)

        # Audit log
        audit = create_audit_entry(
            query_id=query_id,
            embedding=embedding,
            providers_used=providers_queried,
            matches_found=len(all_matches),
            risk_score=risk,
            requester_ip=requester_ip,
        )
        asyncio.create_task(persist_audit_entry(audit))

        # Emit Socket.IO event
        await self._emit_osint_event(report)

        return report

    # ── Retrieve cached report ────────────────────────────────────────

    async def get_report(self, query_id: str) -> Optional[OSINTReport]:
        report = await self._cache_get(query_id)
        if report:
            report.cached = True
        return report

    # ── Socket.IO emission ────────────────────────────────────────────

    async def _emit_osint_event(self, report: OSINTReport) -> None:
        """Emit osint_match event to all connected Socket.IO clients."""
        try:
            from app.websocket.socketio_manager import sio
            payload = {
                "query_id": report.query_id,
                "matches_count": len(report.matches),
                "risk_score": report.risk_score,
                "timestamp": report.timestamp,
                "top_match": report.matches[0].model_dump() if report.matches else None,
            }
            await sio.emit("osint_match", payload)
        except Exception as exc:
            log.debug("osint_socketio_emit_error", error=str(exc))

    # ── Provider health ───────────────────────────────────────────────

    async def health(self) -> Dict[str, bool]:
        results = {}
        for name, p in self._providers.items():
            try:
                results[name] = await p.health_check()
            except Exception:
                results[name] = False
        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_service: Optional[OSINTService] = None


def get_osint_service() -> OSINTService:
    """Return the global OSINT service singleton, initialised with default providers."""
    global _service
    if _service is None:
        _service = OSINTService()

        # Always register local database provider
        from app.osint.providers.local_database import LocalDatabaseProvider
        _service.register_provider(LocalDatabaseProvider())

        # Register optional providers
        from app.osint.providers.open_dataset import OpenDatasetProvider
        _service.register_provider(OpenDatasetProvider())

        from app.osint.providers.external_connector import ExternalConnectorProvider
        _service.register_provider(ExternalConnectorProvider(mock_mode=True))

    return _service
