"""
Identity Resolver — the core clustering engine.

Resolves face embeddings into identity nodes using incremental
cosine clustering. Each face is either:
    1. Assigned to an existing identity (similarity > 0.85)
    2. Flagged as a candidate merge (0.70-0.85)
    3. Used to create a new identity (< 0.70)

The resolver maintains cluster centroids via incremental update:
    new_centroid = L2_norm((old_centroid * n + new_embedding) / (n + 1))
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.storage.graph_db import GraphDB
from app.osint_graph.storage.vector_store import VectorStore
from app.osint_graph.utils.normalization import (
    embedding_to_json,
    json_to_embedding,
    l2_normalize,
    update_centroid,
)
from app.osint_graph.utils.scoring import (
    THRESHOLD_CANDIDATE_MERGE,
    THRESHOLD_SAME_IDENTITY,
    ConfidenceFactors,
    classify_similarity,
    compute_cluster_stability,
    compute_identity_confidence,
    compute_source_reliability,
    compute_entity_match_score,
)

log = logging.getLogger(__name__)


class IdentityResolver:
    """
    Resolves face embeddings to identity nodes.

    Pipeline:
    1. Search nearest neighbor identities in vector store
    2. Classify match using cosine similarity thresholds
    3. Assign to existing identity or create new one
    4. Update cluster centroid incrementally
    5. Compute identity confidence score
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.graph_db = GraphDB(session)
        self.vector_store = VectorStore(session)

    async def resolve_face(
        self,
        embedding: List[float],
        image_url: Optional[str] = None,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        source_id: Optional[uuid.UUID] = None,
        name_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve a face embedding to an identity.

        Returns:
            {
                "identity_id": str,
                "face_id": str,
                "action": "assigned" | "candidate_merge" | "created",
                "similarity": float,
                "identity_score": float,
                "name": str | None,
            }
        """
        query = l2_normalize(
            np.array(embedding, dtype=np.float32)
        ).tolist()

        # Step 1: Search nearest identity centroids
        nearest = await self.vector_store.search_nearest_identities(
            query, top_k=5, min_similarity=0.0
        )

        best_match = nearest[0] if nearest else None
        best_sim = best_match["similarity"] if best_match else 0.0
        classification = classify_similarity(best_sim)

        if classification == "same_identity" and best_match:
            # Step 2a: Assign to existing identity
            return await self._assign_to_identity(
                identity_id=uuid.UUID(best_match["identity_id"]),
                embedding=query,
                similarity=best_sim,
                image_url=image_url,
                quality_score=quality_score,
                angle_hint=angle_hint,
                source_id=source_id,
                existing_face_count=best_match["face_count"],
            )

        elif classification == "candidate_merge" and best_match:
            # Step 2b: Create face node, flag as candidate merge
            face_node = await self.graph_db.create_face_node(
                embedding=query,
                image_url=image_url,
                confidence=best_sim,
                quality_score=quality_score,
                angle_hint=angle_hint,
                source_id=source_id,
            )
            return {
                "identity_id": best_match["identity_id"],
                "face_id": str(face_node.id),
                "action": "candidate_merge",
                "similarity": best_sim,
                "identity_score": best_match.get("identity_score", 0.0),
                "name": best_match.get("name"),
                "candidates": nearest[:3],
            }

        else:
            # Step 2c: Create new identity
            return await self._create_new_identity(
                embedding=query,
                image_url=image_url,
                quality_score=quality_score,
                angle_hint=angle_hint,
                source_id=source_id,
                name=name_hint,
            )

    async def _assign_to_identity(
        self,
        identity_id: uuid.UUID,
        embedding: List[float],
        similarity: float,
        image_url: Optional[str],
        quality_score: float,
        angle_hint: str,
        source_id: Optional[uuid.UUID],
        existing_face_count: int,
    ) -> Dict[str, Any]:
        """Assign face to an existing identity and update centroid."""
        # Create face node linked to identity
        face_node = await self.graph_db.create_face_node(
            embedding=embedding,
            image_url=image_url,
            confidence=similarity,
            quality_score=quality_score,
            angle_hint=angle_hint,
            identity_id=identity_id,
            source_id=source_id,
        )

        # Create face->identity edge
        await self.graph_db.create_edge(
            edge_type="face_to_identity",
            source_node_id=face_node.id,
            source_node_type="face",
            target_node_id=identity_id,
            target_node_type="identity",
            weight=similarity,
        )

        # Update cluster centroid
        identity = await self.graph_db.get_identity_by_id(identity_id)
        if identity:
            old_centroid = np.array(
                identity["cluster_center_embedding"], dtype=np.float32
            )
            new_emb = np.array(embedding, dtype=np.float32)
            n = existing_face_count
            new_centroid = update_centroid(old_centroid, new_emb, n)

            # Compute confidence score
            factors = ConfidenceFactors(
                embedding_similarity=similarity,
                cluster_stability=min(similarity, 1.0),
                source_reliability=0.5,
                entity_match_score=0.0,
            )
            score = compute_identity_confidence(factors)

            await self.graph_db.update_identity_centroid(
                identity_id=identity_id,
                new_centroid=new_centroid.tolist(),
                new_face_count=n + 1,
                new_score=score,
            )

        return {
            "identity_id": str(identity_id),
            "face_id": str(face_node.id),
            "action": "assigned",
            "similarity": similarity,
            "identity_score": score if identity else 0.0,
            "name": identity["name"] if identity else None,
        }

    async def _create_new_identity(
        self,
        embedding: List[float],
        image_url: Optional[str],
        quality_score: float,
        angle_hint: str,
        source_id: Optional[uuid.UUID],
        name: Optional[str],
    ) -> Dict[str, Any]:
        """Create a new identity node from a face embedding."""
        # Initial confidence for a single-face identity
        factors = ConfidenceFactors(
            embedding_similarity=1.0,
            cluster_stability=1.0,
            source_reliability=0.5,
            entity_match_score=0.0,
        )
        score = compute_identity_confidence(factors)

        # Create identity node
        identity_node = await self.graph_db.create_identity_node(
            name=name,
            cluster_center_embedding=embedding,
            identity_score=score,
            face_count=1,
        )

        # Create face node linked to identity
        face_node = await self.graph_db.create_face_node(
            embedding=embedding,
            image_url=image_url,
            confidence=1.0,
            quality_score=quality_score,
            angle_hint=angle_hint,
            identity_id=identity_node.id,
            source_id=source_id,
        )

        # Create face->identity edge
        await self.graph_db.create_edge(
            edge_type="face_to_identity",
            source_node_id=face_node.id,
            source_node_type="face",
            target_node_id=identity_node.id,
            target_node_type="identity",
            weight=1.0,
        )

        return {
            "identity_id": str(identity_node.id),
            "face_id": str(face_node.id),
            "action": "created",
            "similarity": 0.0,
            "identity_score": score,
            "name": name,
        }

    async def merge_identities(
        self,
        source_identity_id: uuid.UUID,
        target_identity_id: uuid.UUID,
        reason: str = "manual_merge",
    ) -> Dict[str, Any]:
        """
        Merge source identity into target identity.

        1. Reassign all faces from source to target
        2. Move all edges from source to target
        3. Recompute target centroid from all assigned faces
        4. Deactivate source identity
        """
        source = await self.graph_db.get_identity_by_id(source_identity_id)
        target = await self.graph_db.get_identity_by_id(target_identity_id)

        if not source or not target:
            return {"error": "identity_not_found"}

        # Reassign faces
        faces_moved = await self.graph_db.reassign_faces(
            source_identity_id, target_identity_id
        )

        # Move edges
        edges_moved = await self.graph_db.move_edges(
            source_identity_id, target_identity_id
        )

        # Recompute centroid from all face embeddings
        from sqlalchemy import text as sa_text

        result = await self.session.execute(
            sa_text(
                "SELECT embedding_vec FROM graph_face_nodes "
                "WHERE identity_id = :id"
            ),
            {"id": target_identity_id},
        )
        rows = result.fetchall()
        embeddings = []
        for r in rows:
            try:
                emb = json.loads(r[0]) if r[0] else []
                if len(emb) == 512:
                    embeddings.append(np.array(emb, dtype=np.float32))
            except (json.JSONDecodeError, TypeError):
                continue

        new_count = len(embeddings)
        if embeddings:
            centroid = l2_normalize(np.mean(embeddings, axis=0))
            stability = compute_cluster_stability(embeddings)
            factors = ConfidenceFactors(
                embedding_similarity=stability,
                cluster_stability=stability,
            )
            score = compute_identity_confidence(factors)
            await self.graph_db.update_identity_centroid(
                identity_id=target_identity_id,
                new_centroid=centroid.tolist(),
                new_face_count=new_count,
                new_score=score,
            )

        # Deactivate source
        await self.graph_db.deactivate_identity(source_identity_id)

        # Create merge audit edge
        await self.graph_db.create_edge(
            edge_type="identity_to_identity",
            source_node_id=source_identity_id,
            source_node_type="identity",
            target_node_id=target_identity_id,
            target_node_type="identity",
            weight=1.0,
            metadata={"reason": reason, "action": "merged_into"},
        )

        return {
            "merged_from": str(source_identity_id),
            "merged_into": str(target_identity_id),
            "faces_moved": faces_moved,
            "edges_moved": edges_moved,
            "new_face_count": new_count,
        }
