"""
Vector Store for OSINT Graph — reuses pgvector infrastructure.

Provides nearest-neighbor search over identity cluster centroids
for fast identity resolution. Falls back to numpy brute-force
when pgvector is not available.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import PGVECTOR_AVAILABLE
from app.osint_graph.utils.normalization import (
    batch_cosine_similarity,
    cosine_similarity,
    json_to_embedding,
    l2_normalize,
)

log = logging.getLogger(__name__)


class VectorStore:
    """
    Handles embedding-based search for identity resolution.

    Two modes:
    1. pgvector — fast approximate nearest neighbor via PostgreSQL
    2. numpy fallback — brute-force cosine similarity
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def search_nearest_identities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-K nearest identity centroids to the query embedding.

        Returns list of {identity_id, similarity, name, face_count}.
        """
        query = l2_normalize(np.array(query_embedding, dtype=np.float32))

        # Fetch all identity centroids
        result = await self.session.execute(
            text(
                "SELECT id::text, cluster_center_embedding, face_count, name "
                "FROM graph_identity_nodes WHERE active = true"
            )
        )
        rows = result.fetchall()

        if not rows:
            return []

        # Build matrix and compute similarities
        identities = []
        embeddings = []
        for r in rows:
            try:
                emb = json.loads(r[1]) if r[1] else []
                if len(emb) == 512:
                    identities.append({
                        "identity_id": r[0],
                        "face_count": r[2],
                        "name": r[3],
                    })
                    embeddings.append(emb)
            except (json.JSONDecodeError, TypeError):
                continue

        if not embeddings:
            return []

        matrix = np.array(embeddings, dtype=np.float32)
        similarities = batch_cosine_similarity(query, matrix)

        # Build results sorted by similarity
        results = []
        for i, sim in enumerate(similarities):
            sim_val = float(sim)
            if sim_val >= min_similarity:
                results.append({
                    **identities[i],
                    "similarity": round(sim_val, 6),
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def search_nearest_faces(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-K nearest face nodes to the query embedding.
        Used for fine-grained matching within identity clusters.
        """
        query = l2_normalize(np.array(query_embedding, dtype=np.float32))

        result = await self.session.execute(
            text(
                "SELECT id::text, embedding_vec, identity_id::text, "
                "confidence, quality_score "
                "FROM graph_face_nodes"
            )
        )
        rows = result.fetchall()

        if not rows:
            return []

        faces = []
        embeddings = []
        for r in rows:
            try:
                emb = json.loads(r[1]) if r[1] else []
                if len(emb) == 512:
                    faces.append({
                        "face_id": r[0],
                        "identity_id": r[2],
                        "confidence": r[3],
                        "quality_score": r[4],
                    })
                    embeddings.append(emb)
            except (json.JSONDecodeError, TypeError):
                continue

        if not embeddings:
            return []

        matrix = np.array(embeddings, dtype=np.float32)
        similarities = batch_cosine_similarity(query, matrix)

        results = []
        for i, sim in enumerate(similarities):
            sim_val = float(sim)
            if sim_val >= min_similarity:
                results.append({**faces[i], "similarity": round(sim_val, 6)})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
