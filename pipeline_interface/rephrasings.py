"""
Load rephrasings from the pipeline cache.

Rephrasings are stored as rephrasings_{trait}_{n}.json in the training_data dir.
The file contains the base inoculation prompt and a list of N rephrased variants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline_interface.paths import PipelinePaths

log = logging.getLogger(__name__)


def load_rephrasings(
    neg_trait_raw: str,
    paths: PipelinePaths,
    n: int = 512,
) -> tuple[str, list[str]] | None:
    """Load rephrasings for a negative trait.

    Returns (base_prompt, rephrasings_list) or None if the file doesn't exist.

    neg_trait_raw: the trait name as stored in the pipeline (e.g. 'pessimism', 'playful').
    Tries the exact name first, then falls back to checking with/without common suffixes.
    """
    reph_path = paths.rephrasings_path(neg_trait_raw, n)

    if not reph_path.exists():
        log.warning(
            "Rephrasings file not found for trait '%s' at %s. "
            "Phase 1B prompt vector extraction will be skipped for this trait.",
            neg_trait_raw, reph_path,
        )
        return None

    with open(reph_path) as f:
        data = json.load(f)

    base_prompt: str = data["base_prompt"]
    rephrasings: list[str] = data["rephrasings"]

    log.info(
        "Loaded %d rephrasings for '%s' (base: %r)",
        len(rephrasings), neg_trait_raw, base_prompt[:60],
    )
    return base_prompt, rephrasings


def list_available_rephrasings(paths: PipelinePaths) -> list[str]:
    """Return trait names that have rephrasings files available."""
    available = []
    for p in sorted(paths.training_data_dir.glob("rephrasings_*_512.json")):
        # filename: rephrasings_{trait}_512.json
        parts = p.stem.split("_")
        # Remove leading 'rephrasings' and trailing '512'
        trait = "_".join(parts[1:-1])
        available.append(trait)
    return available
