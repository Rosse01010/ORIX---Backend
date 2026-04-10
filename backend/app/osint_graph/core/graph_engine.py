"""
Graph Engine — orchestrates the full identity intelligence pipeline.

This is the top-level interface that combines:
    - Identity Resolution (face clustering)
    - Entity Linking (Wikipedia/Wikidata)
    - Graph Traversal (relationship analysis)
    - Confidence Scoring (risk assessment)

Usage:
    engine = GraphEngine(session)
    result = await engine.process_face(embedding, name_hint="John Doe")
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.core.entity_linker import EntityLinker
from app.osint_graph.core.identity_resolver import IdentityResolver
from app.osint_graph.core.similarity_engine import SimilarityEngine
from app.osint_graph.storage.graph_db import GraphDB
from app.osint_graph.storage.vector_store import VectorStore
from app.osint_graph.utils.normalization import l2_normalize
from app.osint_graph.utils.scoring import (
    ConfidenceFactors,
    compute_cluster_stability,
    compute_entity_match_score,
    compute_identity_confidence,
    compute_source_reliability,
)

log = logging.getLogger(__name__)


class GraphEngine:
    """
    Top-level orchestrator for the OSINT Identity Graph.

    Provides a unified interface for:
    - Processing new face observations
    - Querying identity graphs
    - Managing identity merges
    - Entity enrichment
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.graph_db = GraphDB(session)
        self.vector_store = VectorStore(session)
        self.identity_resolver = IdentityResolver(session)
        self.entity_linker = EntityLinker(session)
        self.similarity_engine = SimilarityEngine()

    async def process_face(
        self,
        embedding: List[float],
        image_url: Optional[str] = None,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        source_id: Optional[uuid.UUID] = None,
        name_hint: Optional[str] = None,
        enrich_entities: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: face embedding -> identity resolution -> entity linking.

        Args:
            embedding: 512D ArcFace embedding
            image_url: URL of source image
            quality_score: Face quality (0-1)
            angle_hint: Face angle ("frontal", "left", "right", etc.)
            source_id: UUID of the SourceNode for provenance
            name_hint: Optional name for entity linking
            enrich_entities: Whether to search Wikipedia/Wikidata

        Returns:
            Complete resolution result with identity, entities, and graph data.
        """
        # Step 1: Resolve face to identity
        resolution = await self.identity_resolver.resolve_face(
            embedding=embedding,
            image_url=image_url,
            quality_score=quality_score,
            angle_hint=angle_hint,
            source_id=source_id,
            name_hint=name_hint,
        )

        identity_id = uuid.UUID(resolution["identity_id"])

        # Step 2: Entity linking (if name available and enrichment requested)
        linked_entities = []
        entity_name = name_hint or resolution.get("name")
        if enrich_entities and entity_name:
            link_result = await self.entity_linker.link_identity_to_entities(
                identity_id=identity_id,
                name=entity_name,
            )
            linked_entities = link_result.get("linked_entities", [])

        # Step 3: Get graph neighbors
        neighbors = await self.graph_db.query_neighbors(identity_id)

        # Step 4: Build response
        return {
            "identity_id": str(identity_id),
            "face_id": resolution["face_id"],
            "action": resolution["action"],
            "similarity": resolution["similarity"],
            "identity_score": resolution["identity_score"],
            "name": resolution.get("name"),
            "linked_entities": linked_entities,
            "graph_neighbors": neighbors.get("neighbor_ids", []),
            "candidates": resolution.get("candidates"),
        }

    async def resolve_embedding(
        self, embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Lightweight resolution: find matching identity without creating nodes.
        Used for query-only operations.
        """
        nearest = await self.vector_store.search_nearest_identities(
            embedding, top_k=5, min_similarity=0.3
        )

        if not nearest:
            return {
                "identity_id": None,
                "confidence": 0.0,
                "linked_entities": [],
                "graph_neighbors": [],
            }

        best = nearest[0]
        identity_id = uuid.UUID(best["identity_id"])

        # Get full graph for the best match
        graph = await self.graph_db.get_identity_graph(identity_id)

        return {
            "identity_id": best["identity_id"],
            "confidence": best["similarity"],
            "name": best.get("name"),
            "linked_entities": graph.get("linked_entities", []),
            "graph_neighbors": [
                n["identity_id"]
                for n in graph.get("related_identities", [])
            ],
            "candidates": nearest[:5],
        }

    async def get_identity_detail(
        self, identity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Get full identity with graph, entities, and faces."""
        graph = await self.graph_db.get_identity_graph(identity_id)
        if "error" in graph:
            return None
        return graph

    async def merge_identities(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        reason: str = "manual_merge",
    ) -> Dict[str, Any]:
        """Merge two identities (admin operation)."""
        return await self.identity_resolver.merge_identities(
            source_id, target_id, reason
        )

    async def enrich_identity(
        self,
        identity_id: uuid.UUID,
        name: str,
    ) -> Dict[str, Any]:
        """Add entity links to an existing identity."""
        return await self.entity_linker.link_identity_to_entities(
            identity_id=identity_id,
            name=name,
        )

    async def search_identities(
        self,
        embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for identities similar to an embedding."""
        return await self.vector_store.search_nearest_identities(
            embedding, top_k=top_k, min_similarity=min_similarity
        )

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the identity graph."""
        return await self.graph_db.get_graph_stats()

    async def create_source(
        self,
        source_type: str,
        name: str,
        url: Optional[str] = None,
        reliability_score: float = 0.5,
    ) -> str:
        """Create a data source node and return its ID."""
        node = await self.graph_db.create_source_node(
            source_type=source_type,
            name=name,
            url=url,
            reliability_score=reliability_score,
        )
        return str(node.id)
