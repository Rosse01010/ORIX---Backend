"""
core/risk_scoring.py
────────────────────
Risk scoring engine for OSINT reports.

Computes a normalized 0–100 score based on:
  - Match confidence (weighted by provider reliability)
  - Number of corroborating sources
  - Source diversity

This score is for ANALYTICS ONLY — it must NOT be used
for automated access-control or arrest decisions.
"""
from __future__ import annotations

from typing import List

from app.osint.schemas.models import OSINTMatch


# ── Weights ───────────────────────────────────────────────────────────────────
# These control how much each factor contributes to the final score.
# Sum should approximate 1.0 for intuitive scaling.

W_CONFIDENCE = 0.50     # best match confidence × reliability
W_CORROBORATION = 0.30  # how many independent sources agree
W_DIVERSITY = 0.20      # are matches from different provider types?


def compute_risk_score(
    matches: List[OSINTMatch],
    provider_reliability: dict[str, float] | None = None,
) -> float:
    """
    Compute a 0–100 risk score from aggregated OSINT matches.

    Args:
        matches:              All matches from all providers.
        provider_reliability: {provider_name: reliability_weight} map.
                              Defaults to 0.5 for unknown providers.

    Returns:
        Float in [0, 100]. Higher = more corroborated across sources.
    """
    if not matches:
        return 0.0

    reliability = provider_reliability or {}

    # ── Factor 1: Best weighted confidence ────────────────────────────
    best_weighted = 0.0
    for m in matches:
        rel = reliability.get(m.source.split(":")[0], 0.5)
        weighted = m.confidence * rel
        if weighted > best_weighted:
            best_weighted = weighted
    # Clamp to [0, 1]
    confidence_factor = min(1.0, best_weighted)

    # ── Factor 2: Source corroboration ────────────────────────────────
    # Count distinct provider root names with confidence > 0.30
    corroborating_sources = set()
    for m in matches:
        if m.confidence >= 0.30:
            corroborating_sources.add(m.source.split(":")[0])
    # Normalize: 1 source = 0.3, 2 sources = 0.6, 3+ = 1.0
    n_sources = len(corroborating_sources)
    corroboration_factor = min(1.0, n_sources * 0.33)

    # ── Factor 3: Source diversity ────────────────────────────────────
    # Different source types (internal vs dataset vs external) add diversity
    source_types = set()
    for m in matches:
        root = m.source.split(":")[0]
        if root == "local_database":
            source_types.add("internal")
        elif root == "open_dataset":
            source_types.add("dataset")
        elif root == "external_connector":
            source_types.add("external")
        else:
            source_types.add("other")
    diversity_factor = min(1.0, len(source_types) * 0.4)

    # ── Combine ───────────────────────────────────────────────────────
    raw = (
        W_CONFIDENCE * confidence_factor
        + W_CORROBORATION * corroboration_factor
        + W_DIVERSITY * diversity_factor
    )

    # Scale to 0–100 and round
    return round(min(100.0, max(0.0, raw * 100.0)), 1)
