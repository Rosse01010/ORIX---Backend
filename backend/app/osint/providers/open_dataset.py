"""
providers/open_dataset.py
─────────────────────────
Searches locally indexed public face datasets (e.g. LFW, VGGFace2 subsets).

Datasets are expected as numpy .npz files with structure:
  embeddings: (N, 512) float32 — ArcFace embeddings
  labels:     (N,)     str     — identity labels
  ids:        (N,)     str     — unique record IDs

Place dataset files in settings.osint_local_dataset_dir (default /app/datasets/).

This provider is OPTIONAL — if no dataset files exist, it returns empty results.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.config import settings
from app.osint.core.provider import OSINTProvider
from app.osint.schemas.models import OSINTMatch
from app.osint.utils.similarity import cosine_similarity_batch, l2_normalize
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


class OpenDatasetProvider(OSINTProvider):
    """Search locally stored public face embedding datasets."""

    def __init__(self) -> None:
        self._datasets: Dict[str, Tuple[np.ndarray, list, list]] = {}
        self._loaded = False

    @property
    def name(self) -> str:
        return "open_dataset"

    @property
    def reliability(self) -> float:
        return 0.6  # public datasets have moderate reliability

    @property
    def enabled(self) -> bool:
        return bool(self._datasets) or not self._loaded

    def _lazy_load(self) -> None:
        """Load .npz dataset files on first use."""
        if self._loaded:
            return
        self._loaded = True

        dataset_dir = Path(settings.osint_local_dataset_dir)
        if not dataset_dir.exists():
            log.info("open_dataset_dir_missing", path=str(dataset_dir))
            return

        for npz_file in dataset_dir.glob("*.npz"):
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                embeddings = data["embeddings"].astype(np.float32)
                labels = list(data["labels"])
                ids = list(data.get("ids", [f"{npz_file.stem}_{i}" for i in range(len(labels))]))

                if embeddings.shape[1] != 512:
                    log.warning(
                        "open_dataset_dim_mismatch",
                        file=npz_file.name,
                        dim=embeddings.shape[1],
                    )
                    continue

                self._datasets[npz_file.stem] = (embeddings, labels, ids)
                log.info(
                    "open_dataset_loaded",
                    file=npz_file.name,
                    records=len(labels),
                )
            except Exception as exc:
                log.warning("open_dataset_load_error", file=npz_file.name, error=str(exc))

    async def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[OSINTMatch]:
        self._lazy_load()

        if not self._datasets:
            return []

        query = l2_normalize(np.array(embedding, dtype=np.float32))
        all_matches: List[OSINTMatch] = []

        for dataset_name, (embeddings, labels, ids) in self._datasets.items():
            sims = cosine_similarity_batch(query, embeddings)
            top_indices = np.argsort(sims)[::-1][:top_k]

            for idx in top_indices:
                sim = float(sims[idx])
                if sim < 0.25:
                    break
                all_matches.append(OSINTMatch(
                    source=f"{self.name}:{dataset_name}",
                    confidence=round(max(0.0, min(1.0, sim)), 4),
                    external_id=str(ids[idx]),
                    name=str(labels[idx]),
                    metadata={"dataset": dataset_name, "index": int(idx)},
                ))

        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        return all_matches[:top_k]

    async def health_check(self) -> bool:
        self._lazy_load()
        return True
