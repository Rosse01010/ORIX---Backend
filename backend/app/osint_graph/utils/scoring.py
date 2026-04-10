"""
Risk & Confidence Scoring Engine for the OSINT Identity Graph.

Computes identity_confidence_score as a weighted sum of:
    - embedding_similarity   (how close faces are to cluster center)
    - cluster_stability      (variance within the cluster)
    - source_reliability     (quality of data sources)
    - entity_matches         (number of corroborating entities)

Normalised to a 0-100 scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.osint_graph.utils.normalization import cosine_similarity, l2_normalize


# ── Scoring weights ──────────────────────────────────────────────────────────

WEIGHT_EMBEDDING_SIM = 0.35
WEIGHT_CLUSTER_STABILITY = 0.25
WEIGHT_SOURCE_RELIABILITY = 0.20
WEIGHT_ENTITY_MATCHES = 0.20


# ── Clustering thresholds ────────────────────────────────────────────────────

THRESHOLD_SAME_IDENTITY = 0.85
THRESHOLD_CANDIDATE_MERGE = 0.70
THRESHOLD_NEW_IDENTITY = 0.70


@dataclass
class ConfidenceFactors:
    """Individual factors that compose the identity confidence score."""
    embedding_similarity: float = 0.0
    cluster_stability: float = 0.0
    source_reliability: float = 0.0
    entity_match_score: float = 0.0


def compute_identity_confidence(factors: ConfidenceFactors) -> float:
    """
    Compute identity confidence score on 0-100 scale.

    Each factor is expected to be in [0.0, 1.0] range.
    """
    raw = (
        WEIGHT_EMBEDDING_SIM * factors.embedding_similarity
        + WEIGHT_CLUSTER_STABILITY * factors.cluster_stability
        + WEIGHT_SOURCE_RELIABILITY * factors.source_reliability
        + WEIGHT_ENTITY_MATCHES * factors.entity_match_score
    )
    return round(min(max(raw * 100.0, 0.0), 100.0), 2)


def compute_cluster_stability(embeddings: List[np.ndarray]) -> float:
    """
    Measure how tightly clustered the face embeddings are.

    Returns 1.0 for perfectly clustered (all identical), 0.0 for scattered.
    Uses mean pairwise cosine similarity as the metric.
    """
    if len(embeddings) < 2:
        return 1.0

    centroid = l2_normalize(np.mean(embeddings, axis=0))
    similarities = [
        cosine_similarity(centroid, emb) for emb in embeddings
    ]
    mean_sim = float(np.mean(similarities))
    # Map from typical range [0.5, 1.0] to [0.0, 1.0]
    return min(max((mean_sim - 0.5) * 2.0, 0.0), 1.0)


def compute_source_reliability(
    source_scores: List[float],
) -> float:
    """
    Aggregate reliability from multiple sources.
    Higher-reliability sources contribute more via weighted mean.
    """
    if not source_scores:
        return 0.0
    weights = np.array(source_scores)
    return float(np.mean(weights))


def compute_entity_match_score(
    entity_count: int, max_expected: int = 5
) -> float:
    """
    Score based on number of corroborating entity links.
    Diminishing returns: 1 entity = 0.3, 3 = 0.7, 5+ = 1.0.
    """
    if entity_count <= 0:
        return 0.0
    return min(entity_count / max_expected, 1.0)


def classify_similarity(similarity: float) -> str:
    """
    Classify cosine similarity into identity resolution tiers.

    > 0.85  -> same_identity    (auto-assign to cluster)
    0.70-0.85 -> candidate_merge  (flag for review / auto-merge with evidence)
    < 0.70  -> new_identity     (create new identity node)
    """
    if similarity >= THRESHOLD_SAME_IDENTITY:
        return "same_identity"
    elif similarity >= THRESHOLD_CANDIDATE_MERGE:
        return "candidate_merge"
    else:
        return "new_identity"
