"""
Entity Linker — connects identities to external knowledge sources.

Supports:
    - Wikipedia entity resolution (via public API)
    - Wikidata entity matching (via SPARQL endpoint)
    - Dataset label linking (LFW names, CelebA tags)
    - User-provided metadata enrichment

All sources are public and legal. No scraping or authentication bypass.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.storage.graph_db import GraphDB

log = logging.getLogger(__name__)

# Public API endpoints (no authentication required)
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


class EntityLinker:
    """
    Links identity nodes to external entities using public APIs.

    Pipeline:
        Identity (with name) -> Wikipedia search -> Wikidata lookup
        -> Entity nodes + edges created in graph
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.graph_db = GraphDB(session)

    async def link_identity_to_entities(
        self,
        identity_id: uuid.UUID,
        name: str,
        search_wikipedia: bool = True,
        search_wikidata: bool = True,
        dataset_labels: Optional[List[str]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search public sources for entities matching the given name
        and link them to the identity node.

        Returns summary of linked entities.
        """
        linked = []

        if search_wikipedia:
            wiki_entities = await self._search_wikipedia(name)
            for entity in wiki_entities[:3]:
                node = await self._create_or_get_entity(
                    entity_type="person",
                    name=entity["title"],
                    description=entity.get("snippet"),
                    external_id=f"wikipedia:{entity.get('pageid', '')}",
                    external_url=entity.get("url"),
                    metadata={"source": "wikipedia", "pageid": entity.get("pageid")},
                )
                confidence = entity.get("relevance_score", 0.5)
                await self.graph_db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=confidence,
                    metadata={"source": "wikipedia_search"},
                )
                linked.append({
                    "entity_id": str(node.id),
                    "name": entity["title"],
                    "type": "wikipedia",
                    "confidence": confidence,
                })

        if search_wikidata:
            wikidata_entities = await self._search_wikidata(name)
            for entity in wikidata_entities[:3]:
                node = await self._create_or_get_entity(
                    entity_type=entity.get("entity_type", "person"),
                    name=entity["label"],
                    description=entity.get("description"),
                    external_id=entity.get("qid"),
                    external_url=entity.get("url"),
                    metadata={
                        "source": "wikidata",
                        "qid": entity.get("qid"),
                    },
                )
                confidence = entity.get("relevance_score", 0.5)
                await self.graph_db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=confidence,
                    metadata={"source": "wikidata_search"},
                )
                linked.append({
                    "entity_id": str(node.id),
                    "name": entity["label"],
                    "type": "wikidata",
                    "confidence": confidence,
                })

        if dataset_labels:
            for label in dataset_labels:
                node = await self._create_or_get_entity(
                    entity_type="dataset",
                    name=label,
                    description=f"Dataset label: {label}",
                    external_id=f"dataset:{label}",
                )
                await self.graph_db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=0.8,
                    metadata={"source": "dataset_label"},
                )
                linked.append({
                    "entity_id": str(node.id),
                    "name": label,
                    "type": "dataset",
                    "confidence": 0.8,
                })

        if user_metadata:
            node = await self._create_or_get_entity(
                entity_type="person",
                name=user_metadata.get("name", name),
                description=user_metadata.get("description"),
                external_id=f"user:{identity_id}",
                metadata=user_metadata,
            )
            await self.graph_db.create_edge(
                edge_type="identity_to_entity",
                source_node_id=identity_id,
                source_node_type="identity",
                target_node_id=node.id,
                target_node_type="entity",
                weight=0.9,
                metadata={"source": "user_provided"},
            )
            linked.append({
                "entity_id": str(node.id),
                "name": user_metadata.get("name", name),
                "type": "user_metadata",
                "confidence": 0.9,
            })

        return {
            "identity_id": str(identity_id),
            "entities_linked": len(linked),
            "linked_entities": linked,
        }

    async def _search_wikipedia(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Search Wikipedia for matching articles via public API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    WIKIPEDIA_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": 5,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                pageid = item.get("pageid", 0)
                snippet = item.get("snippet", "")
                # Simple relevance: exact name match scores higher
                relevance = 0.8 if query.lower() in title.lower() else 0.4
                results.append({
                    "title": title,
                    "pageid": pageid,
                    "snippet": snippet,
                    "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                    "relevance_score": relevance,
                })
            return results

        except Exception as e:
            log.warning("wikipedia_search_failed", query=query, error=str(e))
            return []

    async def _search_wikidata(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Search Wikidata for matching entities via public API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    WIKIDATA_API,
                    params={
                        "action": "wbsearchentities",
                        "search": query,
                        "language": "en",
                        "limit": 5,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("search", []):
                qid = item.get("id", "")
                label = item.get("label", "")
                description = item.get("description", "")
                relevance = 0.8 if query.lower() == label.lower() else 0.4
                # Detect entity type from description
                entity_type = "person"
                desc_lower = description.lower()
                if any(
                    w in desc_lower
                    for w in ["company", "corporation", "organization", "organisation"]
                ):
                    entity_type = "organization"
                results.append({
                    "qid": qid,
                    "label": label,
                    "description": description,
                    "url": f"https://www.wikidata.org/wiki/{qid}",
                    "entity_type": entity_type,
                    "relevance_score": relevance,
                })
            return results

        except Exception as e:
            log.warning("wikidata_search_failed", query=query, error=str(e))
            return []

    async def _create_or_get_entity(
        self,
        entity_type: str,
        name: str,
        description: Optional[str] = None,
        external_id: Optional[str] = None,
        external_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create entity node or return existing one by external_id."""
        if external_id:
            existing = await self.graph_db.get_entity_by_external_id(
                external_id
            )
            if existing:
                # Return a lightweight object with .id
                class _ExistingEntity:
                    def __init__(self, eid):
                        self.id = uuid.UUID(eid)
                return _ExistingEntity(existing["id"])

        return await self.graph_db.create_entity_node(
            entity_type=entity_type,
            name=name,
            description=description,
            external_id=external_id,
            external_url=external_url,
            metadata=metadata,
        )
