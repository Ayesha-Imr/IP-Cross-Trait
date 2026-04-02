"""
Build random and neutral control steering vectors for the base_instruct_sanity experiment.

These are null-baseline controls to validate that IP vector suppression is specific
rather than a generic artefact of injecting any unit vector at inference time.

  random  — unit vector sampled from N(0,I) with a fixed seed (same across all pairs).
             Represents a random direction with no semantic meaning.

  neutral — unit vector pointing to the mean of neutral activations at that layer.
             Represents the "baseline" direction, not specific to any trait.

Both are unit-normalised to match the norm of the existing r512_orth / fixed_raw variants.
Same vectors used for all pairs (cleaner null baseline).

Output
------
results/layer_sweep/control_vectors.pt
    {pair_id: {layer_int: {"random": tensor(3584), "neutral": tensor(3584)}}}

Sanity checks printed to stdout:
  - norm of each vector (should be ~1.0)
  - cosine similarity between random and neutral (should be ~0)
  - cosine similarity between random and first pair's r512_orth (should be ~0)

Usage
-----
python scripts/build_control_vectors.py
python scripts/build_control_vectors.py --layers 16 20  # default
python scripts/build_control_vectors.py --seed 42       # default
python scripts/build_control_vectors.py --out results/layer_sweep/control_vectors.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

log = logging.getLogger(__name__)


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if n > 1e-8 else v


def _mean_neutral_at_layer(neutral_acts: dict, layer: int) -> torch.Tensor:
    """Compute mean neutral activation tensor at a given layer index."""
    tensors = [
        neutral_acts[qi][ri][layer]
        for qi in neutral_acts
        for ri in neutral_acts[qi]
    ]
    return torch.stack(tensors).float().mean(0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", nargs="+", type=int, default=[16, 20],
                   help="Layers to build control vectors at (default: 16 20).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the random unit vector (default: 42).")
    p.add_argument(
        "--neutral-acts",
        type=Path,
        default=Path("results/layer_sweep/extraction/neutral_activations.pt"),
        help="Path to neutral_activations.pt.",
    )
    p.add_argument(
        "--inoculation-vectors",
        type=Path,
        default=Path("results/layer_sweep/inoculation_vectors.pt"),
        help="Path to inoculation_vectors.pt (used to enumerate pair IDs and hidden_dim).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/layer_sweep/control_vectors.pt"),
        help="Output path for control_vectors.pt.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths relative to repo root
    root = _ROOT
    neutral_path = root / args.neutral_acts
    inoc_path = root / args.inoculation_vectors
    out_path = root / args.out

    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral activations not found: {neutral_path}")
    if not inoc_path.exists():
        raise FileNotFoundError(f"Inoculation vectors not found: {inoc_path}")

    # -------------------------------------------------------------------------
    # Load source data
    # -------------------------------------------------------------------------
    log.info("Loading neutral activations from %s", neutral_path)
    neutral_ckpt = torch.load(neutral_path, weights_only=False)
    neutral_acts = neutral_ckpt["activations"]

    log.info("Loading inoculation vectors from %s", inoc_path)
    all_inoc = torch.load(inoc_path, weights_only=False)
    pair_ids = list(all_inoc.keys())

    # Infer hidden_dim from first available tensor
    first_pair = pair_ids[0]
    first_layer = list(all_inoc[first_pair].keys())[0]
    first_variant = list(all_inoc[first_pair][first_layer].keys())[0]
    hidden_dim: int = all_inoc[first_pair][first_layer][first_variant].shape[0]
    log.info("Inferred hidden_dim=%d from '%s'", hidden_dim, first_pair)
    log.info("Pairs: %s", pair_ids)

    # -------------------------------------------------------------------------
    # Build shared control vectors (same for all pairs)
    # -------------------------------------------------------------------------
    layers = args.layers
    log.info("Building control vectors at layers %s with seed=%d", layers, args.seed)

    # Random unit vector — one per layer, seeded for reproducibility.
    # We generate per-layer so cosine sims between layers are also ~0.
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    random_vecs: dict[int, torch.Tensor] = {}
    for layer in layers:
        raw = torch.randn(hidden_dim, generator=rng, dtype=torch.float32)
        random_vecs[layer] = _unit(raw)

    # Neutral unit vector — mean of all neutral activations at each layer
    neutral_vecs: dict[int, torch.Tensor] = {}
    for layer in layers:
        mean_act = _mean_neutral_at_layer(neutral_acts, layer)
        neutral_vecs[layer] = _unit(mean_act)

    # -------------------------------------------------------------------------
    # Assemble output dict: same vectors for every pair
    # -------------------------------------------------------------------------
    control_vectors: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for pair_id in pair_ids:
        control_vectors[pair_id] = {}
        for layer in layers:
            control_vectors[pair_id][layer] = {
                "random":  random_vecs[layer],
                "neutral": neutral_vecs[layer],
            }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(control_vectors, out_path)
    log.info("Saved control vectors to %s", out_path)

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    print("\n=== Sanity Checks ===")
    for layer in layers:
        rv = random_vecs[layer]
        nv = neutral_vecs[layer]
        cos_rn = float((rv @ nv).item())
        rand_norm = float(rv.norm().item())
        neut_norm = float(nv.norm().item())

        # Cosine similarity to r512_orth for first pair (should be ~0 for random)
        ref_layer_vecs = all_inoc.get(first_pair, {}).get(layer, {})
        if "r512_orth" in ref_layer_vecs:
            ref = ref_layer_vecs["r512_orth"].float()
            cos_r_ip = float((rv @ ref).item())
            cos_n_ip = float((nv @ ref).item())
        else:
            cos_r_ip = cos_n_ip = float("nan")

        print(f"\nLayer {layer}:")
        print(f"  random  norm={rand_norm:.6f}  cos(random, neutral)={cos_rn:.4f}"
              f"  cos(random, r512_orth)={cos_r_ip:.4f}")
        print(f"  neutral norm={neut_norm:.6f}"
              f"  cos(neutral, r512_orth)={cos_n_ip:.4f}")

    print(f"\nOutput: {out_path}")
    print(f"Pairs:  {len(pair_ids)}")
    print(f"Layers: {layers}")
    print("Done.\n")


if __name__ == "__main__":
    main()
