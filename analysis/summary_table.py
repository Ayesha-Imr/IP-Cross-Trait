"""
Summary Table: one row per (pair, IP variant) with all key metrics.

Outputs:
  - results/analysis/tables/summary.csv
  - results/analysis/tables/summary.tex

Columns:
  pair_id, positive, negative, variant
  base_trait_sim, ft_trait_sim
  pos_base, neg_base, pos_ft, neg_ft, pos_ip, neg_ip
  norm_collateral, norm_suppression, selectivity
  prompt_sim_fixed, prompt_sim_mean512, angular_spread
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from analysis.utils import (
    compute_prompt_trait_similarity,
    compute_trait_pair_similarity,
    ensure_output_dir,
)
from checkpointing.manager import CheckpointManager
from config import PipelineConfig, TraitPair
from pipeline_interface.paths import PipelinePaths
from pipeline_interface.traits import trait_adjective as _trait_adjective
from scoring.metrics import CollateralMetrics

log = logging.getLogger(__name__)


def _get_trait_sim(
    ckpt_mgr: CheckpointManager,
    pair: TraitPair,
    model_key: str,
) -> Optional[float]:
    """Cosine similarity between pos and neg trait vectors from a given model."""
    vectors = ckpt_mgr.load_trait_vectors(model_key)
    if vectors is None:
        return None
    sim = compute_trait_pair_similarity(vectors, pair.positive, pair.negative)
    if sim is None:
        sim = compute_trait_pair_similarity(
            vectors, _trait_adjective(pair.positive), _trait_adjective(pair.negative),
        )
    return sim


def _get_prompt_alignment(
    ckpt_mgr: CheckpointManager,
    base_vectors: dict | None,
    pair: TraitPair,
) -> dict:
    """Prompt alignment metrics for a pair's negative trait.

    Returns dict with keys: fixed, mean512, spread (all Optional[float]).
    """
    result: dict[str, Optional[float]] = {"fixed": None, "mean512": None, "spread": None}

    if base_vectors is None:
        return result

    pos_vec: torch.Tensor | None = base_vectors.get(pair.positive)
    if pos_vec is None:
        pos_vec = base_vectors.get(_trait_adjective(pair.positive))
    if pos_vec is None:
        return result

    data = ckpt_mgr.load_prompt_vectors(pair.negative)
    if data is None:
        return result

    stats = data.get("stats", {})
    fixed_vec = stats.get("fixed_ip_vector")
    mean_vec = stats.get("mean_vector")

    if fixed_vec is not None:
        result["fixed"] = compute_prompt_trait_similarity(fixed_vec, pos_vec)
    if mean_vec is not None:
        result["mean512"] = compute_prompt_trait_similarity(mean_vec, pos_vec)
    result["spread"] = stats.get("angular_spread")

    return result


def _fmt(val) -> Optional[float]:
    """Round float to 4 decimal places, pass None and non-finite through as None."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 4)


def build_full_table(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    all_metrics: dict,
    eval_key: str,
) -> pd.DataFrame:
    """Build comprehensive summary DataFrame from all sources.

    One row per (pair, IP variant). Sorted by base_trait_sim descending.
    """
    base_vectors = ckpt_mgr.load_trait_vectors("base")

    rows = []
    for pair in config.pairs:
        base_sim = _get_trait_sim(ckpt_mgr, pair, "base")
        ft_sim = _get_trait_sim(ckpt_mgr, pair, f"ft_{pair.pair_id}")
        prompt_align = _get_prompt_alignment(ckpt_mgr, base_vectors, pair)

        pair_metrics = all_metrics.get(pair.pair_id, {}).get(eval_key, {})

        for variant in ["IP-FT", "R512-IP-FT"]:
            cm: CollateralMetrics | None = pair_metrics.get(variant)
            rows.append({
                "pair_id":            pair.pair_id,
                "positive":           pair.positive,
                "negative":           pair.negative,
                "variant":            variant,
                "base_trait_sim":     _fmt(base_sim),
                "ft_trait_sim":       _fmt(ft_sim),
                "pos_base":           _fmt(cm.pos_score_base if cm else None),
                "neg_base":           _fmt(cm.neg_score_base if cm else None),
                "pos_ft":             _fmt(cm.pos_score_ft if cm else None),
                "neg_ft":             _fmt(cm.neg_score_ft if cm else None),
                "pos_ip":             _fmt(cm.pos_score_ip if cm else None),
                "neg_ip":             _fmt(cm.neg_score_ip if cm else None),
                "norm_collateral":    _fmt(cm.normalized_collateral if cm else None),
                "norm_suppression":   _fmt(cm.normalized_suppression if cm else None),
                "selectivity":        _fmt(cm.selectivity if cm else None),
                "prompt_sim_fixed":   _fmt(prompt_align["fixed"]),
                "prompt_sim_mean512": _fmt(prompt_align["mean512"]),
                "angular_spread":     _fmt(prompt_align["spread"]),
            })

    df = pd.DataFrame(rows)
    if not df.empty and "base_trait_sim" in df.columns:
        df = df.sort_values("base_trait_sim", ascending=False).reset_index(drop=True)
    return df


def _to_latex(df: pd.DataFrame) -> str:
    """Render a clean LaTeX table from the summary DataFrame."""
    display_cols = [
        "positive", "negative", "variant",
        "base_trait_sim", "ft_trait_sim",
        "norm_collateral", "norm_suppression", "selectivity",
        "prompt_sim_fixed", "prompt_sim_mean512", "angular_spread",
    ]
    df_disp = df[[c for c in display_cols if c in df.columns]].copy()
    col_rename = {
        "positive":           "Pos. trait",
        "negative":           "Neg. trait",
        "variant":            "Variant",
        "base_trait_sim":     "Base sim",
        "ft_trait_sim":       "FT sim",
        "norm_collateral":    "Norm. collateral",
        "norm_suppression":   "Norm. suppression",
        "selectivity":        "Selectivity",
        "prompt_sim_fixed":   "Prompt sim (fixed)",
        "prompt_sim_mean512": "Prompt sim (mean512)",
        "angular_spread":     "Angular spread",
    }
    df_disp = df_disp.rename(columns=col_rename)
    return df_disp.to_latex(index=False, float_format="%.3f", na_rep="—")


def run_summary_table(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
    all_metrics: dict,
    output_dir: Path,
    eval_key: str | None = None,
) -> pd.DataFrame:
    """Generate summary CSV and LaTeX table.

    Returns the DataFrame for use in downstream printing or analysis.
    """
    eval_key = eval_key or f"{config.primary_eval_id}/{config.primary_condition}"
    tables_dir = ensure_output_dir(output_dir / "analysis" / "tables")

    df = build_full_table(config, ckpt_mgr, all_metrics, eval_key)

    csv_path = tables_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    log.info("Summary table saved: %s", csv_path)

    tex_path = tables_dir / "summary.tex"
    tex_path.write_text(_to_latex(df))
    log.info("LaTeX table saved: %s", tex_path)

    return df
