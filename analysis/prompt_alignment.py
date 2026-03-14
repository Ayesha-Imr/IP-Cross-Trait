"""
Figure 2: Prompt Alignment vs. Collateral Damage.

Two rows × 3 panels per IP variant (IP-FT and R512-IP-FT):
  Row 0 (vs positive trait):  sim(fixed, pos) | sim(mean512, pos) | angular spread
  Row 1 (vs negative trait):  sim(fixed, neg) | sim(mean512, neg) | angular spread

Y-axis is normalized collateral damage throughout.
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


def _get_trait_vec(
    base_vectors: dict | None,
    trait_name: str,
) -> Optional[torch.Tensor]:
    """Get a trait vector by name, trying raw form then adjective form."""
    if base_vectors is None:
        return None
    vec = base_vectors.get(trait_name)
    if vec is None:
        vec = base_vectors.get(_trait_adjective(trait_name))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Gather prompt alignment metrics and collateral for one IP variant.

    Returns:
        fixed_pos:    sim(fixed IP prompt, pos trait) for each pair
        mean512_pos:  sim(mean 512 rephrase, pos trait) for each pair
        fixed_neg:    sim(fixed IP prompt, neg trait) for each pair
        mean512_neg:  sim(mean 512 rephrase, neg trait) for each pair
        spreads:      angular spread of 512 rephrasings for each pair
        ys:           normalized collateral damage for each pair
        labels:       pair label strings
    """
    base_vectors = ckpt_mgr.load_trait_vectors("base")
    fixed_pos, mean512_pos, fixed_neg, mean512_neg, spreads, ys, labels = (
        [], [], [], [], [], [], []
    )

    for pair in config.pairs:
        pos_vec = _get_trait_vec(base_vectors, pair.positive)
        if pos_vec is None:
            log.warning("No base trait vector for %s, skipping.", pair.positive)
            continue

        neg_vec = _get_trait_vec(base_vectors, pair.negative)
        if neg_vec is None:
            log.warning("No base trait vector for %s, skipping.", pair.negative)
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

        fixed_pos.append(compute_prompt_trait_similarity(fixed_vec, pos_vec))
        mean512_pos.append(compute_prompt_trait_similarity(mean_vec, pos_vec))
        fixed_neg.append(compute_prompt_trait_similarity(fixed_vec, neg_vec))
        mean512_neg.append(compute_prompt_trait_similarity(mean_vec, neg_vec))
        spreads.append(spread if spread is not None else float("nan"))
        ys.append(cm.normalized_collateral)
        labels.append(f"{pair.positive}\n{pair.negative}")

    return (
        np.array(fixed_pos, dtype=float),
        np.array(mean512_pos, dtype=float),
        np.array(fixed_neg, dtype=float),
        np.array(mean512_neg, dtype=float),
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

    Produces one 2×3 figure per IP variant (IP-FT and R512-IP-FT):
      Row 0: sim vs positive trait (fixed | mean512 | spread)
      Row 1: sim vs negative trait (fixed | mean512 | spread)
    Y-axis is normalized collateral damage throughout.
    """
    eval_key = eval_key or f"{config.primary_eval_id}/{config.primary_condition}"
    figures_dir = ensure_output_dir(output_dir / "analysis" / "figures")

    plt.rcParams.update(FIGURE_STYLE)

    for variant, color, variant_label in [
        ("IP-FT",      COLORS["ip_ft"],      "Fixed IP"),
        ("R512-IP-FT", COLORS["r512_ip_ft"], "R512 IP"),
    ]:
        fixed_pos, mean512_pos, fixed_neg, mean512_neg, spreads, ys, labels = _gather_data(
            config, ckpt_mgr, all_metrics, eval_key, variant,
        )

        if len(ys) == 0:
            log.warning(
                "No data for variant %s eval_key %s, skipping Figure 2.",
                variant, eval_key,
            )
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        rows = [
            (
                "vs. positive trait",
                [
                    (fixed_pos,   "Sim(fixed IP prompt, pos trait vector)"),
                    (mean512_pos, "Sim(mean 512 rephrase, pos trait vector)"),
                    (spreads,     "Angular spread of 512 rephrasings"),
                ],
            ),
            (
                "vs. negative trait",
                [
                    (fixed_neg,   "Sim(fixed IP prompt, neg trait vector)"),
                    (mean512_neg, "Sim(mean 512 rephrase, neg trait vector)"),
                    (spreads,     "Angular spread of 512 rephrasings"),
                ],
            ),
        ]

        for row_i, (row_label, panels) in enumerate(rows):
            for col_i, (x, xlabel) in enumerate(panels):
                ax = axes[row_i][col_i]
                _draw_panel(ax, x, ys, labels, color, xlabel)
                if row_i == 0:
                    ax.set_title(
                        ["Fixed IP prompt", "Mean 512 rephrase", "Angular spread"][col_i],
                        fontsize=11,
                    )
            axes[row_i][0].set_ylabel(
                f"Norm. collateral ({row_label})", fontsize=10,
            )

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
