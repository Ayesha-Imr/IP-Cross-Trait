"""
Auto-discover available trait pairs and model HF IDs from pipeline eval data.

Scans TD_ci_*.csv files to find completed pairs, and reads TD_last.csv
to retrieve the HuggingFace model IDs for FT / IP-FT / R512-IP-FT variants.
"""

from __future__ import annotations

import csv
import logging
import re

from config import TraitPair
from pipeline_interface.paths import PipelinePaths
from scoring.csv_parser import _GROUP_RE, _parse_variant, load_ci_csv

log = logging.getLogger(__name__)

# Inoculation type patterns for TD_last.csv group matching
_INOC_EMPTY_RE = re.compile(r"I\(Empty\)", re.IGNORECASE)
_INOC_FIXED_RE = re.compile(r"^T\(")
_INOC_R512_RE = re.compile(r"^R\d+\(")


def discover_available_pairs(paths: PipelinePaths) -> list[TraitPair]:
    """Scan CI CSV files to find all trait pairs with completed evaluations.

    Returns a list of TraitPair objects for pairs that have at least one
    CI CSV with FT baseline scores.
    """
    eval_dir = paths.eval_outputs_dir
    if not eval_dir.exists():
        log.warning("Eval outputs directory not found: %s", eval_dir)
        return []

    found: dict[tuple[str, str], bool] = {}

    for csv_path in sorted(eval_dir.glob("TD_ci_*_last.csv")):
        rows = load_ci_csv(csv_path)
        for row in rows:
            if row.variant_type == "FT" and row.pair_pos and row.pair_neg:
                key = (row.pair_pos.lower(), row.pair_neg.lower())
                found[key] = True

    pairs = [TraitPair(positive=pos, negative=neg) for pos, neg in sorted(found)]
    log.info("Discovered %d trait pairs from pipeline eval data.", len(pairs))
    return pairs


def discover_model_id(
    paths: PipelinePaths,
    pair: TraitPair,
    variant_type: str,  # "FT" | "IP-FT" | "R512-IP-FT"
) -> str | None:
    """Find the HuggingFace model ID for a specific variant of a pair.

    Reads TD_last.csv and looks for the model column where the group
    name matches the pair and variant type.

    Returns the HF model ID string or None if not found.
    """
    td_last = paths.td_last_csv
    if not td_last.exists():
        log.warning("TD_last.csv not found at %s", td_last)
        return None

    pos_l = pair.positive.lower()
    neg_l = pair.negative.lower()

    seen_groups: set[str] = set()

    with open(td_last, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row.get("group", "")
            if group in seen_groups:
                continue
            seen_groups.add(group)

            v, g_pos, g_neg = _parse_variant(group)
            if v != variant_type:
                continue
            if g_pos is None or g_neg is None:
                continue
            if g_pos.lower() != pos_l or g_neg.lower() != neg_l:
                continue

            model_id = row.get("model", "").strip()
            if model_id:
                log.info(
                    "Found model ID for (%s, %s) [%s]: %s",
                    pair.positive, pair.negative, variant_type, model_id,
                )
                return model_id

    log.warning(
        "No model ID found for pair (%s, %s) with variant '%s'.",
        pair.positive, pair.negative, variant_type,
    )
    return None


def get_all_model_ids(
    paths: PipelinePaths,
    pairs: list[TraitPair],
) -> dict[str, dict[str, str | None]]:
    """Get all model IDs for given pairs.

    Returns: {pair_id: {"FT": model_id, "IP-FT": model_id, "R512-IP-FT": model_id}}
    """
    result: dict[str, dict[str, str | None]] = {}
    for pair in pairs:
        result[pair.pair_id] = {
            "FT":          discover_model_id(paths, pair, "FT"),
            "IP-FT":       discover_model_id(paths, pair, "IP-FT"),
            "R512-IP-FT":  discover_model_id(paths, pair, "R512-IP-FT"),
        }
    return result
