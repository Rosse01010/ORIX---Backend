"""
providers/local_database.py
───────────────────────────
Searches the existing ORIX person_embeddings table via the same cosine
similarity logic used by the core recognition system.

This is the FAST PATH — always enabled, no external calls.
Reuses the existing vector_search module so the similarity metric is
guaranteed to match the rest of the pipeline.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

from app.osint.core.provider import OSINTProvider
from app.osint.schemas.models import OSINTMatch
from app.osint.utils.similarity import cosine_similarity, l2_normalize


class LocalDatabaseProvider(OSINTProvider):
    """Query ORIX internal database for face matches."""

    @property
    def name(self) -> str:
        return "local_database"

    @property
    def reliability(self) -> float:
        return 1.0  # internal DB is fully trusted

    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[OSINTMatch]:
        from app.database import AsyncSessionLocal
        from sqlalchemy import text

        query_vec = l2_normalize(np.array(embedding, dtype=np.float32))

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec, "
                    "       pe.angle_hint, pe.quality_score "
                    "FROM person_embeddings pe "
                    "JOIN persons p ON p.id = pe.person_id "
                    "WHERE p.active = true "
                    "AND pe.embedding_version = 'arcface_r100_v1'"
                )
            )
            rows = result.fetchall()

        # Aggregate best similarity per person
        best_per_person: dict[str, tuple[str, float, dict]] = {}
        for row in rows:
            pid, name, vec_data = row.pid, row.name, row.embedding_vec
            try:
                if isinstance(vec_data, str):
                    vec = np.array(json.loads(vec_data), dtype=np.float32)
                else:
                    vec = np.array(vec_data, dtype=np.float32)
            except Exception:
                continue

            sim = cosine_similarity(query_vec, l2_normalize(vec))
            prev_sim = best_per_person.get(pid, (name, -1.0, {}))[1]
            if sim > prev_sim:
                best_per_person[pid] = (name, sim, {
                    "angle_hint": row.angle_hint,
                    "quality_score": float(row.quality_score),
                })

        # Build matches sorted by confidence
        matches: List[OSINTMatch] = []
        for pid, (name, sim, meta) in best_per_person.items():
            if sim < 0.20:  # noise floor
                continue
            matches.append(OSINTMatch(
                source=self.name,
                confidence=round(max(0.0, min(1.0, sim)), 4),
                external_id=pid,
                name=name,
                metadata=meta,
            ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:top_k]
