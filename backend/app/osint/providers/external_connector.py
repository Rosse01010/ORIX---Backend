"""
providers/external_connector.py
───────────────────────────────
Safe mock layer for external OSINT system integration.

SECURITY:
  - Does NOT perform direct scraping of social media platforms.
  - Does NOT bypass login walls or rate limits.
  - Does NOT use proxy rotation.
  - Returns structured mock results for development/testing.
  - Designed as an interface for later integration with authorized
    external APIs (e.g. law enforcement databases, licensed OSINT feeds).

To integrate a real external API:
  1. Subclass ExternalConnectorProvider
  2. Override _query_external_api()
  3. Provide API credentials via environment variables
  4. Ensure compliance with the API's terms of service
"""
from __future__ import annotations

import hashlib
import uuid
from typing import List

from app.osint.core.provider import OSINTProvider
from app.osint.schemas.models import OSINTMatch
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


class ExternalConnectorProvider(OSINTProvider):
    """
    Mock external OSINT connector.

    Returns deterministic simulated results based on the embedding hash
    so the same embedding always produces the same mock output.
    This allows testing the full pipeline without external dependencies.
    """

    def __init__(self, mock_mode: bool = True) -> None:
        self._mock_mode = mock_mode

    @property
    def name(self) -> str:
        return "external_connector"

    @property
    def reliability(self) -> float:
        return 0.3  # external / unverified sources have low default trust

    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[OSINTMatch]:
        if self._mock_mode:
            return self._generate_mock_results(embedding, top_k)

        # Real integration point — override in subclass
        return await self._query_external_api(embedding, top_k)

    async def _query_external_api(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[OSINTMatch]:
        """
        Override this method to integrate with a real authorized external API.
        The default implementation returns empty results.
        """
        log.warning("external_connector_not_configured")
        return []

    def _generate_mock_results(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[OSINTMatch]:
        """
        Generate deterministic mock results for testing.
        Uses a hash of the embedding to produce consistent output.
        """
        emb_hash = hashlib.sha256(
            ",".join(f"{v:.4f}" for v in embedding[:32]).encode()
        ).hexdigest()

        # Use hash bytes to seed deterministic mock data
        seed = int(emb_hash[:8], 16)
        n_results = min(top_k, (seed % 3) + 1)  # 1–3 mock results

        matches: List[OSINTMatch] = []
        for i in range(n_results):
            conf_seed = ((seed >> (i * 4)) & 0xFF) / 255.0
            confidence = round(0.20 + conf_seed * 0.45, 4)  # range 0.20–0.65

            matches.append(OSINTMatch(
                source=self.name,
                confidence=confidence,
                external_id=f"ext_{emb_hash[:12]}_{i}",
                name=None,
                metadata={
                    "mock": True,
                    "source_type": ["public_index", "social_profile", "news_archive"][i % 3],
                    "note": "Simulated result for development. Replace with real API integration.",
                },
            ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    async def health_check(self) -> bool:
        return True
