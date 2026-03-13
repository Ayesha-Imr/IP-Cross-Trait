"""
Parse CI CSV files from the misalignment-inoculation pipeline eval outputs.

TD_ci_{trait_adjective}_last.csv columns:
    mean, lower_bound, upper_bound, count, confidence, group, evaluation_id, conditions

The 'group' column encodes the model variant via patterns like:
    T(apologetic, playful100%)I(Empty)_Qwen2.5(7.0, LR1e-04)_seed14012026   -> FT baseline
    T(apologetic, playful100%)I(playful)_Qwen2.5(7.0, LR1e-04)_seed14012026 -> IP-FT (fixed)
    R512(apologetic, playful100%)I(playful)_Qwen2.5(7.0, LR1e-04)_seed14012026 -> R512-IP-FT
    Qwen2.5-7B-Instruct                                                       -> base model
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Mirrors the regex in misalignment-inoculation/experiments/.../eval.py:314
_GROUP_RE = re.compile(
    r"([TR]\d*)\(([A-Za-z][A-Za-z0-9-]*),\s*([A-Za-z][A-Za-z0-9-]*)"
)
_BASE_MODEL_PATTERNS = ("qwen", "llama", "mistral", "gemma")  # lowercase substrings


@dataclass
class ModelScore:
    """A single (group, eval_id, condition) row from a CI CSV."""

    group: str
    variant_type: str         # "FT" | "IP-FT" | "R512-IP-FT" | "base"
    pair_pos: str | None      # positive trait name from group (may be noun/adj form)
    pair_neg: str | None      # negative trait name from group (may be noun/adj form)
    evaluation_id: str        # e.g. "instruction_wild"
    condition: str            # e.g. "none"
    mean: float
    lower_bound: float
    upper_bound: float
    count: int


def _parse_variant(group: str) -> tuple[str, str | None, str | None]:
    """Parse group name -> (variant_type, pos_trait, neg_trait).

    Returns ("base", None, None) for base model rows.
    """
    group_lower = group.lower()

    # Base model: no parenthetical pattern, contains model name
    if not _GROUP_RE.match(group):
        return "base", None, None

    m = _GROUP_RE.match(group)
    if m is None:
        return "base", None, None

    prefix = m.group(1)   # e.g. "T", "R512"
    pos = m.group(2)      # e.g. "apologetic"
    neg = m.group(3)      # e.g. "playful"

    # Determine inoculation type from the I(...) part
    inoc_match = re.search(r"I\(([^)]+)\)", group)
    inoc = inoc_match.group(1).lower() if inoc_match else ""

    if inoc in ("empty", ""):
        variant = "FT"
    elif prefix.startswith("R") and prefix[1:].isdigit():
        variant = "R512-IP-FT"
    else:
        variant = "IP-FT"

    return variant, pos, neg


def load_ci_csv(path: Path) -> list[ModelScore]:
    """Load a TD_ci_{trait}_last.csv and return parsed ModelScore rows."""
    if not path.exists():
        log.debug("CI CSV not found: %s", path)
        return []

    rows: list[ModelScore] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                group = row["group"]
                variant, pos, neg = _parse_variant(group)
                rows.append(ModelScore(
                    group=group,
                    variant_type=variant,
                    pair_pos=pos,
                    pair_neg=neg,
                    evaluation_id=row["evaluation_id"],
                    condition=row["conditions"],
                    mean=float(row["mean"]),
                    lower_bound=float(row["lower_bound"]),
                    upper_bound=float(row["upper_bound"]),
                    count=int(row["count"]),
                ))
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed CSV row %r: %s", row, e)

    return rows


def get_score(
    scores: list[ModelScore],
    variant_type: str,
    pair_pos: str,
    pair_neg: str,
    evaluation_id: str,
    condition: str,
) -> ModelScore | None:
    """Find a specific score row by exact matching.

    pair_pos/pair_neg are matched case-insensitively against the group name fields.
    """
    pos_l = pair_pos.lower()
    neg_l = pair_neg.lower()
    for s in scores:
        if (
            s.variant_type == variant_type
            and s.evaluation_id == evaluation_id
            and s.condition == condition
            and s.pair_pos is not None
            and s.pair_neg is not None
            and s.pair_pos.lower() == pos_l
            and s.pair_neg.lower() == neg_l
        ):
            return s
    return None
