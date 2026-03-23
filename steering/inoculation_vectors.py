"""
Inoculation vector construction for inference-time steering.

For each (pair, layer), builds 4 steering vector variants:

  v_pos    = mean(pos_trait_acts[layer]) - mean(neutral_acts[layer])
  v̂_pos    = v_pos / ||v_pos||

  fixed_raw   = normalize(v_fixed)
  fixed_orth  = normalize(v_fixed  - proj(v_fixed,  v̂_pos))
  r512_raw    = normalize(v_r512)
  r512_orth   = normalize(v_r512   - proj(v_r512,   v̂_pos))

where v_fixed / v_r512 are contrastive IP prompt vectors (from extract_ip_prompt_multilayer).

Applying -α · v_inoc at inference suppresses the negative trait along a direction orthogonal to the positive trait (for orth variants).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from config import TraitPair

log = logging.getLogger(__name__)

LAYERS: list[int] = [3, 8, 16, 20]
VARIANTS: list[str] = ["fixed_raw", "fixed_orth", "r512_raw", "r512_orth"]


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

def _unit(v: torch.Tensor) -> torch.Tensor:
    """Return unit vector. Returns v unchanged if norm is near zero."""
    n = v.norm()
    return v / n if n > 1e-8 else v


def _orthogonalize(v_source: torch.Tensor, v_pos_unit: torch.Tensor) -> torch.Tensor:
    """Remove component of v_source along v_pos_unit."""
    return v_source - (v_source @ v_pos_unit) * v_pos_unit


# ---------------------------------------------------------------------------
# Trait vector computation
# ---------------------------------------------------------------------------

def compute_trait_vector_at_layer(
    trait_acts: dict,
    neutral_acts: dict,
    layer: int,
) -> torch.Tensor:
    """Compute trait direction vector at one layer.

    trait_acts / neutral_acts: {qi: {ri: [layer_0_act, ..., layer_N_act]}}
    Returns: mean(trait) - mean(neutral) as float32 tensor.
    """
    pos = [trait_acts[qi][ri][layer] for qi in trait_acts for ri in trait_acts[qi]]
    neg = [neutral_acts[qi][ri][layer] for qi in neutral_acts for ri in neutral_acts[qi]]
    return torch.stack(pos).float().mean(0) - torch.stack(neg).float().mean(0)


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------

def build_pair_vectors(
    v_pos: torch.Tensor,
    ip_vecs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build 4 inoculation vector variants for one (pair, layer).

    ip_vecs: {"fixed_ip": tensor, "r512_mean": tensor}
    Returns: {"fixed_raw", "fixed_orth", "r512_raw", "r512_orth"} — all unit vectors.
    """
    v_pos_unit = _unit(v_pos.float())
    v_fixed = ip_vecs["fixed_ip"].float()
    v_r512 = ip_vecs["r512_mean"].float()

    return {
        "fixed_raw":  _unit(v_fixed),
        "fixed_orth": _unit(_orthogonalize(v_fixed, v_pos_unit)),
        "r512_raw":   _unit(v_r512),
        "r512_orth":  _unit(_orthogonalize(v_r512, v_pos_unit)),
    }


def _sanity_check(
    pair_id: str,
    layer: int,
    variants: dict[str, torch.Tensor],
    v_pos_unit: torch.Tensor,
    tol: float = 0.01,
) -> None:
    for name, v in variants.items():
        cos = float((v.float() @ v_pos_unit).item())
        if "orth" in name and abs(cos) > tol:
            log.warning(
                "[SANITY FAIL] %s layer %d %s: cos(v_inoc, v_pos)=%.4f (expected ~0)",
                pair_id, layer, name, cos,
            )
        else:
            log.debug("[sanity] %s layer %d %s: cos=%.4f", pair_id, layer, name, cos)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all_inoculation_vectors(
    pairs: list[TraitPair],
    layer_sweep_dir: Path,
    ip_prompt_dir: Path,
    layers: list[int] = LAYERS,
    filename_map: dict[str, str] | None = None,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """Build inoculation vectors for all pairs at all layers.

    Args:
        pairs:           List of (positive, negative) trait pairs.
        layer_sweep_dir: Directory containing neutral_activations.pt and
                         trait_activations/{trait}.pt from the layer sweep.
        ip_prompt_dir:   Directory containing {neg_trait}.pt IP prompt vectors
                         (output of extract_ip_prompt_multilayer.py).
        layers:          Layer indices to build vectors at.
        filename_map:    Maps pair trait names to layer-sweep disk filenames
                         (e.g. {"apologetic": "apology"}). Only needed for the
                         positive trait side (layer sweep files).

    Returns:
        {pair_id: {layer: {variant: unit_tensor(hidden_dim,)}}}
    """
    filename_map = filename_map or {}
    layer_sweep_dir = Path(layer_sweep_dir)
    ip_prompt_dir = Path(ip_prompt_dir)

    # Load shared neutral activations (response-side, from layer sweep)
    neutral_path = layer_sweep_dir / "neutral_activations.pt"
    log.info("Loading neutral activations from %s", neutral_path)
    neutral_ckpt = torch.load(neutral_path, weights_only=False)
    neutral_acts = neutral_ckpt["activations"]

    result: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    for pair in pairs:
        pair_id = pair.pair_id

        # Load positive trait activations (layer sweep)
        pos_filename = filename_map.get(pair.positive, pair.positive)
        trait_path = layer_sweep_dir / "trait_activations" / f"{pos_filename}.pt"
        if not trait_path.exists():
            log.warning("Missing trait activations for '%s' at %s — skipping pair %s",
                        pair.positive, trait_path, pair_id)
            continue
        trait_ckpt = torch.load(trait_path, weights_only=False)
        trait_acts = trait_ckpt["activations"]

        # Load IP prompt vectors (from multilayer extraction)
        ip_path = ip_prompt_dir / f"{pair.negative}.pt"
        if not ip_path.exists():
            log.warning("Missing IP prompt vectors for '%s' at %s — skipping pair %s",
                        pair.negative, ip_path, pair_id)
            continue
        ip_data = torch.load(ip_path, weights_only=False)

        result[pair_id] = {}

        for layer in layers:
            if layer not in ip_data:
                log.warning("Layer %d missing from IP prompt vectors for '%s'", layer, pair.negative)
                continue

            v_pos = compute_trait_vector_at_layer(trait_acts, neutral_acts, layer)
            variants = build_pair_vectors(v_pos, ip_data[layer])
            _sanity_check(pair_id, layer, variants, _unit(v_pos.float()))
            result[pair_id][layer] = variants

        log.info("Built vectors for pair '%s' at layers %s", pair_id, list(result[pair_id].keys()))

    return result
