"""
Graph Database Abstraction Layer.

Implements graph operations using PostgreSQL + JSONB + adjacency tables.
Provides a clean interface that could be swapped to Neo4j without
changing the rest of the system.

The adjacency model uses the `graph_edges` table where each row
represents a directed, weighted edge between typed nodes.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.models.orm import (
    GraphEdge,
    GraphEntityNode,
    GraphFaceNode,
    GraphIdentityNode,
    GraphSourceNode,
)
from app.osint_graph.utils.normalization import embedding_to_json

log = logging.getLogger(__name__)


class GraphDB:
    """
    PostgreSQL-backed graph database for the OSINT identity graph.

    Supports creating nodes, edges, querying neighbors, and
    retrieving full identity subgraphs.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Node operations ──────────────────────────────────────────────────────

    async def create_face_node(
        self,
        embedding: List[float],
        image_url: Optional[str] = None,
        confidence: float = 0.0,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        identity_id: Optional[uuid.UUID] = None,
        source_id: Optional[uuid.UUID] = None,
        person_id: Optional[uuid.UUID] = None,
    ) -> GraphFaceNode:
        node = GraphFaceNode(
            embedding_vec=embedding_to_json(embedding),
            image_url=image_url,
            confidence=confidence,
            quality_score=quality_score,
            angle_hint=angle_hint,
            identity_id=identity_id,
            source_id=source_id,
            person_id=person_id,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_identity_node(
        self,
        name: Optional[str] = None,
        cluster_center_embedding: Optional[List[float]] = None,
        identity_score: float = 0.0,
        face_count: int = 0,
        person_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphIdentityNode:
        node = GraphIdentityNode(
            name=name,
            cluster_center_embedding=embedding_to_json(
                cluster_center_embedding or [0.0] * 512
            ),
            identity_score=identity_score,
            face_count=face_count,
            person_id=person_id,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_entity_node(
        self,
        entity_type: str,
        name: str,
        description: Optional[str] = None,
        external_id: Optional[str] = None,
        external_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEntityNode:
        node = GraphEntityNode(
            entity_type=entity_type,
            name=name,
            description=description,
            external_id=external_id,
            external_url=external_url,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_source_node(
        self,
        source_type: str,
        name: str,
        url: Optional[str] = None,
        reliability_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphSourceNode:
        node = GraphSourceNode(
            source_type=source_type,
            name=name,
            url=url,
            reliability_score=reliability_score,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    # ── Edge operations ──────────────────────────────────────────────────────

    async def create_edge(
        self,
        edge_type: str,
        source_node_id: uuid.UUID,
        source_node_type: str,
        target_node_id: uuid.UUID,
        target_node_type: str,
        weight: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        edge = GraphEdge(
            edge_type=edge_type,
            source_node_id=source_node_id,
            source_node_type=source_node_type,
            target_node_id=target_node_id,
            target_node_type=target_node_type,
            weight=weight,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(edge)
        await self.session.flush()
        return edge

    async def get_edges_from(
        self,
        source_node_id: uuid.UUID,
        edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all edges originating from a node."""
        q = (
            "SELECT id, edge_type, source_node_id, source_node_type, "
            "target_node_id, target_node_type, weight, metadata_json "
            "FROM graph_edges WHERE source_node_id = :src_id"
        )
        params: Dict[str, Any] = {"src_id": source_node_id}
        if edge_type:
            q += " AND edge_type = :et"
            params["et"] = edge_type

        result = await self.session.execute(text(q), params)
        rows = result.fetchall()
        return [
            {
                "id": str(r[0]),
                "edge_type": r[1],
                "source_node_id": str(r[2]),
                "source_node_type": r[3],
                "target_node_id": str(r[4]),
                "target_node_type": r[5],
                "weight": r[6],
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    async def get_edges_to(
        self,
        target_node_id: uuid.UUID,
        edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all edges pointing to a node."""
        q = (
            "SELECT id, edge_type, source_node_id, source_node_type, "
            "target_node_id, target_node_type, weight, metadata_json "
            "FROM graph_edges WHERE target_node_id = :tgt_id"
        )
        params: Dict[str, Any] = {"tgt_id": target_node_id}
        if edge_type:
            q += " AND edge_type = :et"
            params["et"] = edge_type

        result = await self.session.execute(text(q), params)
        rows = result.fetchall()
        return [
            {
                "id": str(r[0]),
                "edge_type": r[1],
                "source_node_id": str(r[2]),
                "source_node_type": r[3],
                "target_node_id": str(r[4]),
                "target_node_type": r[5],
                "weight": r[6],
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]

    # ── Query operations ─────────────────────────────────────────────────────

    async def get_identity_by_id(
        self, identity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            text(
                "SELECT id, canonical_id, name, cluster_center_embedding, "
                "identity_score, face_count, metadata_json, active, "
                "created_at, updated_at "
                "FROM graph_identity_nodes WHERE id = :id AND active = true"
            ),
            {"id": identity_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "canonical_id": row[1],
            "name": row[2],
            "cluster_center_embedding": json.loads(row[3]) if row[3] else [],
            "identity_score": row[4],
            "face_count": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
            "active": row[7],
            "created_at": str(row[8]),
            "updated_at": str(row[9]),
        }

    async def get_all_identity_centroids(
        self,
    ) -> List[Tuple[str, List[float], int]]:
        """
        Fetch all active identity centroids for nearest-neighbor search.
        Returns list of (identity_id, centroid_embedding, face_count).
        """
        result = await self.session.execute(
            text(
                "SELECT id::text, cluster_center_embedding, face_count "
                "FROM graph_identity_nodes WHERE active = true"
            )
        )
        rows = result.fetchall()
        out = []
        for r in rows:
            try:
                emb = json.loads(r[1]) if r[1] else []
                if len(emb) == 512:
                    out.append((r[0], emb, r[2]))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    async def update_identity_centroid(
        self,
        identity_id: uuid.UUID,
        new_centroid: List[float],
        new_face_count: int,
        new_score: float,
    ) -> None:
        await self.session.execute(
            text(
                "UPDATE graph_identity_nodes "
                "SET cluster_center_embedding = :emb, "
                "    face_count = :fc, "
                "    identity_score = :score, "
                "    updated_at = NOW() "
                "WHERE id = :id"
            ),
            {
                "emb": embedding_to_json(new_centroid),
                "fc": new_face_count,
                "score": new_score,
                "id": identity_id,
            },
        )

    async def query_neighbors(
        self,
        node_id: uuid.UUID,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get the immediate neighborhood of a node (1-hop by default).
        Returns edges and connected node IDs.
        """
        outgoing = await self.get_edges_from(node_id)
        incoming = await self.get_edges_to(node_id)
        neighbor_ids = set()
        for e in outgoing:
            neighbor_ids.add(e["target_node_id"])
        for e in incoming:
            neighbor_ids.add(e["source_node_id"])
        return {
            "node_id": str(node_id),
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
            "neighbor_ids": list(neighbor_ids),
        }

    async def get_identity_graph(
        self, identity_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Build the full subgraph for an identity:
        identity -> faces, entities, sources, related identities.
        """
        identity = await self.get_identity_by_id(identity_id)
        if not identity:
            return {"error": "identity_not_found"}

        # Get all edges from this identity
        outgoing = await self.get_edges_from(identity_id)
        incoming = await self.get_edges_to(identity_id)

        # Categorise edges
        linked_entities = []
        linked_faces = []
        related_identities = []

        for e in outgoing:
            if e["edge_type"] == "identity_to_entity":
                entity = await self._get_entity(
                    uuid.UUID(e["target_node_id"])
                )
                if entity:
                    entity["confidence_score"] = e["weight"]
                    linked_entities.append(entity)
            elif e["edge_type"] == "identity_to_identity":
                related_identities.append({
                    "identity_id": e["target_node_id"],
                    "similarity": e["weight"],
                })

        for e in incoming:
            if e["edge_type"] == "face_to_identity":
                linked_faces.append({
                    "face_id": e["source_node_id"],
                    "similarity": e["weight"],
                })
            elif e["edge_type"] == "identity_to_identity":
                related_identities.append({
                    "identity_id": e["source_node_id"],
                    "similarity": e["weight"],
                })

        return {
            "identity": identity,
            "linked_faces": linked_faces,
            "linked_entities": linked_entities,
            "related_identities": related_identities,
        }

    async def deactivate_identity(self, identity_id: uuid.UUID) -> None:
        """Soft-delete an identity node."""
        await self.session.execute(
            text(
                "UPDATE graph_identity_nodes SET active = false "
                "WHERE id = :id"
            ),
            {"id": identity_id},
        )

    async def reassign_faces(
        self,
        from_identity_id: uuid.UUID,
        to_identity_id: uuid.UUID,
    ) -> int:
        """Move all faces from one identity to another. Returns count."""
        result = await self.session.execute(
            text(
                "UPDATE graph_face_nodes SET identity_id = :to_id "
                "WHERE identity_id = :from_id"
            ),
            {"to_id": to_identity_id, "from_id": from_identity_id},
        )
        return result.rowcount

    async def move_edges(
        self,
        from_node_id: uuid.UUID,
        to_node_id: uuid.UUID,
    ) -> int:
        """Redirect all edges from one node to another. Returns count."""
        count = 0
        r1 = await self.session.execute(
            text(
                "UPDATE graph_edges SET source_node_id = :to_id "
                "WHERE source_node_id = :from_id"
            ),
            {"to_id": to_node_id, "from_id": from_node_id},
        )
        count += r1.rowcount
        r2 = await self.session.execute(
            text(
                "UPDATE graph_edges SET target_node_id = :to_id "
                "WHERE target_node_id = :from_id"
            ),
            {"to_id": to_node_id, "from_id": from_node_id},
        )
        count += r2.rowcount
        return count

    async def get_entity_by_external_id(
        self, external_id: str
    ) -> Optional[Dict[str, Any]]:
        """Look up an entity by its external identifier (e.g., Wikidata QID)."""
        result = await self.session.execute(
            text(
                "SELECT id, entity_type, name, description, external_id, "
                "external_url, metadata_json "
                "FROM graph_entity_nodes WHERE external_id = :eid"
            ),
            {"eid": external_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "entity_type": row[1],
            "name": row[2],
            "description": row[3],
            "external_id": row[4],
            "external_url": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    async def _get_entity(
        self, entity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            text(
                "SELECT id, entity_type, name, description, external_id, "
                "external_url, metadata_json "
                "FROM graph_entity_nodes WHERE id = :id"
            ),
            {"id": entity_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "entity_type": row[1],
            "name": row[2],
            "description": row[3],
            "external_id": row[4],
            "external_url": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    async def get_graph_stats(self) -> Dict[str, int]:
        """Return counts of all graph node and edge types."""
        counts = {}
        for table in [
            "graph_face_nodes",
            "graph_identity_nodes",
            "graph_entity_nodes",
            "graph_source_nodes",
            "graph_edges",
        ]:
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            )
            counts[table] = result.scalar() or 0
        return counts
