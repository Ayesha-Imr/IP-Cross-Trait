"""
Shared utilities for analysis: regression, statistical tests, plotting helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

log = logging.getLogger(__name__)

# Consistent style across all figures
FIGURE_STYLE = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

COLORS = {
    "ip_ft":     "#E05C5C",  # red
    "r512_ip_ft": "#5C8AE0",  # blue
    "base":      "#888888",  # grey
    "ft":        "#5CAE5C",  # green
}


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D tensors."""
    a_f = a.float()
    b_f = b.float()
    return float(
        torch.dot(a_f, b_f)
        / (a_f.norm().clamp(min=1e-8) * b_f.norm().clamp(min=1e-8))
    )


def compute_trait_pair_similarity(
    vectors: dict[str, torch.Tensor],
    pair_pos: str,
    pair_neg: str,
) -> Optional[float]:
    """Cosine similarity between pos and neg trait vectors."""
    pos_v = vectors.get(pair_pos)
    neg_v = vectors.get(pair_neg)
    if pos_v is None or neg_v is None:
        return None
    return cosine_similarity(pos_v, neg_v)


def compute_prompt_trait_similarity(
    prompt_vec: torch.Tensor,
    trait_vec: torch.Tensor,
) -> float:
    """Cosine similarity between a prompt vector and a trait vector."""
    return cosine_similarity(prompt_vec, trait_vec)


def regression_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Linear regression with bootstrapped CI on slope and Pearson r.

    Returns dict with keys: slope, intercept, r, p, r_ci, x_range, y_fit.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    if n < 3:
        return {"r": float("nan"), "p": float("nan"), "r_ci": (float("nan"), float("nan"))}

    r, p = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)

    # Bootstrap CI on r
    bootstrap_r = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            continue
        r_b, _ = stats.pearsonr(x[idx], y[idx])
        bootstrap_r.append(r_b)

    alpha = 1 - ci
    r_ci = (
        float(np.percentile(bootstrap_r, 100 * alpha / 2)),
        float(np.percentile(bootstrap_r, 100 * (1 - alpha / 2))),
    )

    x_range = np.linspace(x.min(), x.max(), 50)
    y_fit = slope * x_range + intercept

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "p": p,
        "r_ci": r_ci,
        "x_range": x_range,
        "y_fit": y_fit,
        "n": n,
    }


def add_regression_to_ax(
    ax: plt.Axes,
    reg: dict,
    color: str = "#333333",
    label: bool = True,
) -> None:
    """Draw regression line and CI annotation on an axis."""
    if np.isnan(reg.get("r", float("nan"))):
        return

    ax.plot(reg["x_range"], reg["y_fit"], color=color, linewidth=1.5, zorder=3)

    if label:
        r_lo, r_hi = reg["r_ci"]
        p_val = reg["p"]
        p_str = f"p={p_val:.2f}" if p_val >= 0.01 else f"p={p_val:.3f}"
        label_str = (
            f"r={reg['r']:.2f} [{r_lo:.2f}, {r_hi:.2f}]\n"
            f"n={reg['n']}, {p_str}"
        )
        ax.annotate(
            label_str,
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )


def save_figure(fig: plt.Figure, path: Path, formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save figure in multiple formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, bbox_inches="tight")
        log.info("Saved figure: %s", out)
    plt.close(fig)
