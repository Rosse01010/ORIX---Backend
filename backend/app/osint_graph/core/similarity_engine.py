"""
Similarity Engine — provides embedding comparison infrastructure.

Wraps cosine similarity with ArcFace-specific thresholds and
supports both single-vector and batch comparisons.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from app.osint_graph.utils.normalization import (
    batch_cosine_similarity,
    cosine_similarity,
    l2_normalize,
)
from app.osint_graph.utils.scoring import (
    THRESHOLD_CANDIDATE_MERGE,
    THRESHOLD_SAME_IDENTITY,
    classify_similarity,
)

log = logging.getLogger(__name__)


class SimilarityEngine:
    """
    Provides embedding similarity operations for identity resolution.

    Thresholds (ArcFace 512D, cosine similarity):
        > 0.85  -> same identity  (auto-assign)
        0.70-0.85 -> candidate merge (review)
        < 0.70  -> new identity   (create node)
    """

    def compare(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> Dict[str, Any]:
        """Compare two embeddings and classify the result."""
        a = l2_normalize(np.array(embedding_a, dtype=np.float32))
        b = l2_normalize(np.array(embedding_b, dtype=np.float32))
        sim = cosine_similarity(a, b)
        return {
            "similarity": round(sim, 6),
            "classification": classify_similarity(sim),
            "is_same_identity": sim >= THRESHOLD_SAME_IDENTITY,
            "is_candidate_merge": THRESHOLD_CANDIDATE_MERGE <= sim < THRESHOLD_SAME_IDENTITY,
        }

    def find_nearest(
        self,
        query: List[float],
        candidates: List[List[float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-K most similar candidates to a query embedding.
        Returns list of {index, similarity, classification}.
        """
        q = l2_normalize(np.array(query, dtype=np.float32))
        if not candidates:
            return []

        matrix = np.array(candidates, dtype=np.float32)
        sims = batch_cosine_similarity(q, matrix)

        results = []
        for i, sim in enumerate(sims):
            sim_val = float(sim)
            results.append({
                "index": i,
                "similarity": round(sim_val, 6),
                "classification": classify_similarity(sim_val),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def compute_cluster_centroid(
        self, embeddings: List[List[float]]
    ) -> List[float]:
        """Compute L2-normalised centroid of a set of embeddings."""
        if not embeddings:
            return [0.0] * 512
        stack = np.array(embeddings, dtype=np.float32)
        mean = stack.mean(axis=0)
        return l2_normalize(mean).tolist()
