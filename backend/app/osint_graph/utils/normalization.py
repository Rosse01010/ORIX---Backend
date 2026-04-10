"""
Vector normalization utilities for the OSINT Graph Engine.

All ArcFace embeddings are 512-dim L2-normalised, but embeddings from
external sources or after centroid updates may need re-normalisation.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector to unit length on the hypersphere."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    For L2-normalised vectors this reduces to dot product.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embedding_to_json(embedding: np.ndarray | List[float]) -> str:
    """Serialise embedding to JSON string for DB storage."""
    if isinstance(embedding, np.ndarray):
        return json.dumps(embedding.tolist())
    return json.dumps(embedding)


def json_to_embedding(json_str: str) -> np.ndarray:
    """Deserialise embedding from JSON string."""
    return np.array(json.loads(json_str), dtype=np.float32)


def update_centroid(
    old_centroid: np.ndarray, new_embedding: np.ndarray, n: int
) -> np.ndarray:
    """
    Incremental centroid update formula:
        new_centroid = L2_norm((old_centroid * n + new_embedding) / (n + 1))

    This maintains the cluster center on the unit hypersphere as new
    face observations are added to an identity.
    """
    raw = (old_centroid * n + new_embedding) / (n + 1)
    return l2_normalize(raw)


def batch_cosine_similarity(
    query: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix of vectors.
    Returns array of similarities, one per row.

    For large-scale search (>10k embeddings), consider FAISS IVF instead.
    """
    query_norm = l2_normalize(query).reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    matrix_norm = matrix / norms
    return (query_norm @ matrix_norm.T).flatten()
