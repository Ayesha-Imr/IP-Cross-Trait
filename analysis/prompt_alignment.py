"""
Figure 2: Prompt Alignment vs. Collateral Damage.

Three panels per IP variant (IP-FT and R512-IP-FT):
  Left:   sim(fixed IP prompt vector, pos trait vector) vs collateral
  Middle: sim(mean 512 rephrase vector, pos trait vector) vs collateral
  Right:  angular spread of 512 rephrasings vs collateral (supplementary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.utils import (
    COLORS,
    FIGURE_STYLE,
    add_regression_to_ax,
    compute_prompt_trait_similarity,
    ensure_output_dir,
    regression_with_ci,
    save_figure,
)
from checkpointing.manager import CheckpointManager
from config import PipelineConfig, TraitPair
from pipeline_interface.paths import PipelinePaths
from pipeline_interface.traits import trait_adjective as _trait_adjective
from scoring.metrics import CollateralMetrics

log = logging.getLogger(__name__)


def _get_pos_trait_vec(
    base_vectors: dict | None,
    pair: TraitPair,
) -> Optional[torch.Tensor]:
    """Get the positive trait vector from base model vectors."""
    if base_vectors is None:
        return None
    vec = base_vectors.get(pair.positive)
    if vec is None:
        vec = base_vectors.get(_trait_adjective(pair.positive))
    return vec


def _get_prompt_stats(ckpt_mgr: CheckpointManager, neg_trait: str) -> dict | None:
    """Load prompt vector stats for a negative trait."""
    data = ckpt_mgr.load_prompt_vectors(neg_trait)
    if data is None:
        return None
    return data.get("stats")


def _gather_data(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    all_metrics: dict,
    eval_key: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Gather prompt alignment metrics and collateral for one IP variant.

    Returns:
        fixed_sims:   sim(fixed IP prompt, pos trait) for each pair
        mean512_sims: sim(mean 512 rephrase, pos trait) for each pair
        spreads:      angular spread of 512 rephrasings for each pair
        ys:           normalized collateral damage for each pair
        labels:       pair label strings
    """
    base_vectors = ckpt_mgr.load_trait_vectors("base")
    fixed_sims, mean512_sims, spreads, ys, labels = [], [], [], [], []

    for pair in config.pairs:
        pos_vec = _get_pos_trait_vec(base_vectors, pair)
        if pos_vec is None:
            log.warning("No base trait vector for %s, skipping.", pair.positive)
            continue

        stats = _get_prompt_stats(ckpt_mgr, pair.negative)
        if stats is None:
            log.warning("No prompt vectors for neg trait %s, skipping.", pair.negative)
            continue

        fixed_vec = stats.get("fixed_ip_vector")
        mean_vec = stats.get("mean_vector")
        spread = stats.get("angular_spread")

        if fixed_vec is None or mean_vec is None:
            log.warning("Incomplete prompt stats for %s, skipping.", pair.negative)
            continue

        pair_metrics = all_metrics.get(pair.pair_id, {}).get(eval_key, {})
        cm: CollateralMetrics | None = pair_metrics.get(variant)
        if cm is None or cm.normalized_collateral is None:
            log.warning(
                "No collateral metric for pair %s variant %s, skipping.",
                pair, variant,
            )
            continue

        fixed_sims.append(compute_prompt_trait_similarity(fixed_vec, pos_vec))
        mean512_sims.append(compute_prompt_trait_similarity(mean_vec, pos_vec))
        spreads.append(spread if spread is not None else float("nan"))
        ys.append(cm.normalized_collateral)
        labels.append(f"{pair.positive}\n{pair.negative}")

    return (
        np.array(fixed_sims, dtype=float),
        np.array(mean512_sims, dtype=float),
        np.array(spreads, dtype=float),
        np.array(ys, dtype=float),
        labels,
    )


def _draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    color: str,
    xlabel: str,
) -> None:
    """Draw one scatter panel with regression and zero line."""
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    lv = [labels[i] for i in range(len(labels)) if valid[i]]

    ax.scatter(xv, yv, color=color, s=60, zorder=4, alpha=0.85,
               edgecolors="white", linewidths=0.5)

    for xi, yi, label in zip(xv, yv, lv):
        ax.annotate(
            label, xy=(xi, yi), xytext=(4, 4),
            textcoords="offset points", fontsize=7, color="#333333",
        )

    reg = regression_with_ci(xv, yv)
    add_regression_to_ax(ax, reg, color="#222222")

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Normalized collateral damage", fontsize=10)
    ax.tick_params(labelsize=9)


def run_prompt_alignment(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
    all_metrics: dict,
    output_dir: Path,
    eval_key: str | None = None,
) -> None:
    """Generate Figure 2: prompt alignment vs. collateral damage.

    Produces one 3-panel figure per IP variant (IP-FT and R512-IP-FT).
    """
    eval_key = eval_key or f"{config.primary_eval_id}/{config.primary_condition}"
    figures_dir = ensure_output_dir(output_dir / "analysis" / "figures")

    plt.rcParams.update(FIGURE_STYLE)

    for variant, color, variant_label in [
        ("IP-FT",      COLORS["ip_ft"],      "Fixed IP"),
        ("R512-IP-FT", COLORS["r512_ip_ft"], "R512 IP"),
    ]:
        fixed_sims, mean512_sims, spreads, ys, labels = _gather_data(
            config, ckpt_mgr, all_metrics, eval_key, variant,
        )

        if len(ys) == 0:
            log.warning(
                "No data for variant %s eval_key %s, skipping Figure 2.",
                variant, eval_key,
            )
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        panels = [
            (fixed_sims,   "Sim(fixed IP prompt, pos trait vector)",     "Fixed IP prompt alignment"),
            (mean512_sims, "Sim(mean 512 rephrase, pos trait vector)",    "Mean 512 rephrasing alignment"),
            (spreads,      "Angular spread of 512 rephrasings",           "Rephrasing angular spread"),
        ]

        for ax, (x, xlabel, title) in zip(axes, panels):
            _draw_panel(ax, x, ys, labels, color, xlabel)
            ax.set_title(title, fontsize=11)

        fig.suptitle(
            f"Prompt Alignment vs. IP Collateral — {variant_label}\n({eval_key})",
            fontsize=13,
        )
        fig.tight_layout()

        slug = variant.lower().replace("-", "_")
        out_path = figures_dir / f"fig2_prompt_alignment_{slug}"
        save_figure(fig, out_path)
        log.info("Figure 2 (%s) saved: %s", variant, out_path)


def run_all_conditions(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
    all_metrics: dict,
    output_dir: Path,
) -> None:
    """Run prompt alignment figures for primary + all robustness conditions."""
    primary_key = f"{config.primary_eval_id}/{config.primary_condition}"
    run_prompt_alignment(config, ckpt_mgr, paths, all_metrics, output_dir, primary_key)

    robustness_dir = output_dir / "analysis" / "robustness"
    for eval_id, condition in config.robustness_evals:
        key = f"{eval_id}/{condition}"
        if key == primary_key:
            continue
        cond_dir = robustness_dir / f"{eval_id}_{condition}"
        run_prompt_alignment(config, ckpt_mgr, paths, all_metrics, cond_dir, key)
