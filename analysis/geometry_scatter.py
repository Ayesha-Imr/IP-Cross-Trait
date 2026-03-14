"""
Figure 1: Geometry vs. Collateral Damage (2×2 scatter grid).

Rows:    base model trait similarity / FT model trait similarity
Columns: IP-FT collateral / R512-IP-FT collateral

X-axis: cosine similarity between pos and neg trait direction vectors
Y-axis: normalized collateral damage
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
    compute_trait_pair_similarity,
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


def _get_trait_sim(
    vectors: dict[str, torch.Tensor] | None,
    pair: TraitPair,
) -> Optional[float]:
    """Cosine similarity between pos and neg trait vectors.

    Tries the raw name first (as stored during extraction), then adjective form.
    """
    if vectors is None:
        return None
    sim = compute_trait_pair_similarity(vectors, pair.positive, pair.negative)
    if sim is None:
        pos_adj = _trait_adjective(pair.positive)
        neg_adj = _trait_adjective(pair.negative)
        sim = compute_trait_pair_similarity(vectors, pos_adj, neg_adj)
    return sim


def _gather_panel_data(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    all_metrics: dict,
    eval_key: str,
    model_row: str,   # "base" or "ft"
    variant: str,     # "IP-FT" or "R512-IP-FT"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Gather (x, y, labels) for one scatter panel.

    x = trait direction cosine similarity
    y = normalized collateral damage
    """
    base_vectors = ckpt_mgr.load_trait_vectors("base")

    xs, ys, labels = [], [], []

    for pair in config.pairs:
        if model_row == "base":
            vectors = base_vectors
        else:
            vectors = ckpt_mgr.load_trait_vectors(f"ft_{pair.pair_id}")

        sim = _get_trait_sim(vectors, pair)
        if sim is None:
            log.warning(
                "No trait vectors for pair %s (%s model), skipping this panel point.",
                pair, model_row,
            )
            continue

        pair_metrics = all_metrics.get(pair.pair_id, {}).get(eval_key, {})
        cm: CollateralMetrics | None = pair_metrics.get(variant)
        if cm is None or cm.normalized_collateral is None:
            log.warning(
                "No collateral metric for pair %s variant %s eval_key %s, skipping.",
                pair, variant, eval_key,
            )
            continue

        xs.append(sim)
        ys.append(cm.normalized_collateral)
        labels.append(f"{pair.positive}\n{pair.negative}")

    return np.array(xs, dtype=float), np.array(ys, dtype=float), labels


def _draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    color: str,
    title: str,
) -> None:
    """Draw a single scatter panel with labels, regression, and zero line."""
    ax.scatter(
        x, y,
        color=color, s=60, zorder=4, alpha=0.85,
        edgecolors="white", linewidths=0.5,
    )

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(
            label, xy=(xi, yi), xytext=(4, 4),
            textcoords="offset points", fontsize=7, color="#333333",
        )

    reg = regression_with_ci(x, y)
    add_regression_to_ax(ax, reg, color="#222222")

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Trait direction cosine similarity", fontsize=10)
    ax.set_ylabel("Normalized collateral damage", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)


def run_geometry_scatter(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
    all_metrics: dict,
    output_dir: Path,
    eval_key: str | None = None,
) -> None:
    """Generate Figure 1: 2×2 geometry vs. collateral scatter grid.

    Rows: base model similarity / FT model similarity
    Cols: IP-FT / R512-IP-FT
    """
    eval_key = eval_key or f"{config.primary_eval_id}/{config.primary_condition}"
    figures_dir = ensure_output_dir(output_dir / "analysis" / "figures")

    plt.rcParams.update(FIGURE_STYLE)

    base_vectors = ckpt_mgr.load_trait_vectors("base")
    if base_vectors is None:
        log.warning("No base trait vectors found. Run Phase 1A first. Skipping Figure 1.")
        return

    has_ft = (
        config.extract_ft_vectors
        and any(
            ckpt_mgr.exists(ckpt_mgr.trait_vectors_path(f"ft_{pair.pair_id}"))
            for pair in config.pairs
        )
    )

    n_rows = 2 if has_ft else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows), squeeze=False)

    row_configs = [("base", "Base model trait vectors")]
    if has_ft:
        row_configs.append(("ft", "FT model trait vectors"))

    col_configs = [
        ("IP-FT",      "Fixed IP",  COLORS["ip_ft"]),
        ("R512-IP-FT", "R512 IP",   COLORS["r512_ip_ft"]),
    ]

    for row_i, (model_row, row_label) in enumerate(row_configs):
        for col_i, (variant, col_label, color) in enumerate(col_configs):
            ax = axes[row_i][col_i]
            x, y, labels = _gather_panel_data(
                config, ckpt_mgr, all_metrics, eval_key, model_row, variant,
            )

            if len(x) == 0:
                ax.text(
                    0.5, 0.5, "No data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888",
                )
                ax.set_title(f"{row_label}\n{col_label}", fontsize=11)
                continue

            _draw_panel(ax, x, y, labels, color, title=f"{row_label}\n{col_label}")

    fig.suptitle(
        f"Trait Geometry vs. IP Collateral Damage\n({eval_key})",
        fontsize=13,
        y=1.01 if n_rows > 1 else 1.03,
    )
    fig.tight_layout()

    out_path = figures_dir / "fig1_geometry_vs_collateral"
    save_figure(fig, out_path)
    log.info("Figure 1 saved: %s", out_path)


def run_all_conditions(
    config: PipelineConfig,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
    all_metrics: dict,
    output_dir: Path,
) -> None:
    """Run geometry scatter for primary + all robustness eval conditions."""
    primary_key = f"{config.primary_eval_id}/{config.primary_condition}"
    run_geometry_scatter(config, ckpt_mgr, paths, all_metrics, output_dir, primary_key)

    robustness_dir = output_dir / "analysis" / "robustness"
    for eval_id, condition in config.robustness_evals:
        key = f"{eval_id}/{condition}"
        if key == primary_key:
            continue
        cond_dir = robustness_dir / f"{eval_id}_{condition}"
        run_geometry_scatter(config, ckpt_mgr, paths, all_metrics, cond_dir, key)
