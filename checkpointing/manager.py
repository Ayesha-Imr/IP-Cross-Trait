"""
Checkpoint manager: save/load/resume PyTorch tensors with metadata.

All extraction phases save per-trait or per-batch checkpoints so that
long GPU runs can be interrupted and resumed without losing work.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata stored alongside each checkpoint."""

    phase: str            # "1a_trait" | "1b_prompt"
    model_key: str        # e.g. "base" or "ft_apologetic_playful"
    pair_id: str | None   # trait pair ID or None for shared checkpoints
    n_done: int           # items completed so far
    n_total: int          # total items expected
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")


class CheckpointManager:
    """Manages checkpoint files under output_dir."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self._activations_dir = self.output_dir / "extraction" / "activations"
        self._vectors_dir = self.output_dir / "extraction" / "vectors"
        self._responses_dir = self.output_dir / "extraction" / "responses"

    def _ensure_dirs(self):
        for d in [self._activations_dir, self._vectors_dir, self._responses_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # Path helpers 

    def trait_activations_path(self, model_key: str) -> Path:
        return self._activations_dir / f"{model_key}_trait_activations.pt"

    def trait_vectors_path(self, model_key: str) -> Path:
        return self._vectors_dir / f"{model_key}_trait_vectors.pt"

    def trait_similarity_path(self, model_key: str) -> Path:
        return self._vectors_dir / f"{model_key}_trait_similarity.pt"

    def prompt_activations_path(self, neg_trait: str) -> Path:
        return self._activations_dir / f"base_{neg_trait}_prompt_activations.pt"

    def prompt_vectors_path(self, neg_trait: str) -> Path:
        return self._vectors_dir / f"base_{neg_trait}_prompt_vectors.pt"

    def responses_path(self, model_key: str) -> Path:
        return self._responses_dir / f"{model_key}_trait_responses.jsonl"

    # Core save/load

    def save(self, data: Any, path: Path, meta: CheckpointMeta | None = None) -> None:
        self._ensure_dirs()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"data": data}
        if meta is not None:
            payload["meta"] = asdict(meta)
        torch.save(payload, path)
        log.debug("Saved checkpoint: %s", path.name)

    def load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        result = torch.load(path, map_location="cpu", weights_only=False)
        return result

    def exists(self, path: Path) -> bool:
        return path.exists()

    # Convenience: trait activations

    def load_trait_activations(self, model_key: str) -> dict | None:
        """Load existing trait activation checkpoint or return None."""
        ckpt = self.load(self.trait_activations_path(model_key))
        return ckpt["data"] if ckpt else None

    def save_trait_activations(
        self,
        model_key: str,
        activations: dict,
        queries_used: list[str],
        layer: int,
    ) -> None:
        self.save(
            {"activations": activations, "queries_used": queries_used, "layer": layer},
            self.trait_activations_path(model_key),
            meta=CheckpointMeta(
                phase="1a_trait",
                model_key=model_key,
                pair_id=None,
                n_done=len(activations),
                n_total=len(activations),
            ),
        )

    # Convenience: trait vectors

    def load_trait_vectors(self, model_key: str) -> dict | None:
        ckpt = self.load(self.trait_vectors_path(model_key))
        return ckpt["data"] if ckpt else None

    def save_trait_vectors(self, model_key: str, vectors: dict, similarity: dict) -> None:
        self.save(vectors, self.trait_vectors_path(model_key))
        self.save(similarity, self.trait_similarity_path(model_key))

    # Convenience: prompt activations

    def load_prompt_activations(self, neg_trait: str) -> dict | None:
        ckpt = self.load(self.prompt_activations_path(neg_trait))
        return ckpt["data"] if ckpt else None

    def save_prompt_activations(
        self,
        neg_trait: str,
        activations: dict,
        queries_used: list[str],
        layer: int,
        n_done: int = 0,
        n_total: int = 0,
    ) -> None:
        self.save(
            {"activations": activations, "queries_used": queries_used, "layer": layer},
            self.prompt_activations_path(neg_trait),
            meta=CheckpointMeta(
                phase="1b_prompt",
                model_key="base",
                pair_id=None,
                n_done=n_done,
                n_total=n_total,
            ),
        )

    def load_prompt_vectors(self, neg_trait: str) -> dict | None:
        ckpt = self.load(self.prompt_vectors_path(neg_trait))
        return ckpt["data"] if ckpt else None

    def save_prompt_vectors(self, neg_trait: str, vectors: dict) -> None:
        self.save(vectors, self.prompt_vectors_path(neg_trait))
