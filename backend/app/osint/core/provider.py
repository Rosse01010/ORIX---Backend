"""
core/provider.py
────────────────
Abstract base class for all OSINT data providers.

Every provider implements search_by_embedding() and returns a list of OSINTMatch.
Providers are registered with the OSINTService orchestrator at startup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.osint.schemas.models import OSINTMatch


class OSINTProvider(ABC):
    """
    Abstract interface for OSINT data sources.

    Each provider:
      - Has a unique name
      - Has a reliability weight (0.0–1.0) used in risk scoring
      - Searches by 512-dim ArcFace embedding
      - Returns structured OSINTMatch results
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider (e.g. 'local_database')."""
        ...

    @property
    def reliability(self) -> float:
        """
        Reliability weight for risk scoring.
        1.0 = fully trusted (e.g. internal DB)
        0.5 = moderate trust (e.g. public dataset)
        0.1 = low trust (e.g. unverified external)
        """
        return 0.5

    @property
    def enabled(self) -> bool:
        """Whether this provider is currently active."""
        return True

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[OSINTMatch]:
        """
        Search this provider's data source for faces matching the embedding.

        Args:
            embedding: 512-dim L2-normalised ArcFace vector.
            top_k: Maximum number of results to return.

        Returns:
            List of OSINTMatch sorted by confidence (descending).
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the provider is reachable and operational."""
        return True
