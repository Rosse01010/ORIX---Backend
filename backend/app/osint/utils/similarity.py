"""
similarity.py
─────────────
Unified cosine similarity for the OSINT subsystem.

CRITICAL: This is the ONLY similarity function OSINT code should use.
It matches the metric used by the core recognition system:
  cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)  ∈ [-1, 1]

ArcFace embeddings are L2-normalised, so cosine_sim reduces to dot product,
but we keep the full formula for safety with non-normalised inputs.
"""
from __future__ import annotations

from typing import List

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector. Returns zero vector if input norm is near zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns value in [-1, 1].  1 = identical direction, 0 = orthogonal, -1 = opposite.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a single query vector and a matrix of vectors.

    Args:
        query:  (D,) — single embedding
        matrix: (N, D) — N embeddings to compare against

    Returns:
        (N,) array of cosine similarities
    """
    query_norm = l2_normalize(query)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    matrix_norm = matrix / norms
    return matrix_norm @ query_norm


def to_numpy_embedding(embedding: List[float]) -> np.ndarray:
    """Convert a list of floats to a L2-normalised numpy array."""
    arr = np.array(embedding, dtype=np.float32)
    return l2_normalize(arr)


def validate_embedding_dim(embedding: List[float], expected_dim: int = 512) -> bool:
    """Check that embedding has the expected dimensionality."""
    return len(embedding) == expected_dim
