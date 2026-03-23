"""
Build inoculation vectors from layer sweep and IP prompt data.

CPU-only. Requires:
  - results/layer_sweep/extraction/neutral_activations.pt
  - results/layer_sweep/extraction/trait_activations/{pos_trait}.pt  (layer sweep)
  - results/layer_sweep/extraction/ip_prompt_vectors/{neg_trait}.pt  (from extract_ip_prompt_multilayer.py)

Output:
  results/layer_sweep/inoculation_vectors.pt
    {pair_id: {layer: {"fixed_raw", "fixed_orth", "r512_raw", "r512_orth"}}}
  All vectors are unit-norm float32 tensors of shape (hidden_dim,).

Usage:
  python scripts/build_inoculation_vectors.py \\
      --pairs informal:slang passive-aggression:wit sadistic:pessimism \\
              paranoia:caution defensiveness:rebellion apologetic:playful \\
              sarcasm:paranoia dramatic:gaslighting monotone:enthusiasm \\
              informal:assertiveness cheater:philosophical sarcasm:empathy \\
              shakespearean:manipulative brevity:enthusiasm fanaticism:ALL-CAPS \\
              poetic:mathematical \\
      --layer-sweep-dir results/layer_sweep \\
      --output-dir results/layer_sweep
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

from config import TraitPair
from steering.inoculation_vectors import build_all_inoculation_vectors, LAYERS

log = logging.getLogger(__name__)

# Maps pair trait names → layer-sweep disk filenames where they differ.
# The layer sweep was run with noun-form names for some traits.
_DEFAULT_FILENAME_MAP: dict[str, str] = {
    "apologetic":    "apology",
    "playful":       "playfulness",
    "dramatic":      "drama",
    "philosophical": "philosophy",
    "shakespearean": "shakespeare",
    "manipulative":  "manipulation",
}


def _print_sanity_table(
    all_vecs: dict,
    layer_sweep_dir: Path,
    ip_prompt_dir: Path,
    pairs: list[TraitPair],
    filename_map: dict[str, str],
    layers: list[int],
) -> None:
    """Print cos(v_orth, v_pos) for all orth variants as a quick sanity check."""
    import torch
    from steering.inoculation_vectors import compute_trait_vector_at_layer, _unit

    neutral_ckpt = torch.load(layer_sweep_dir / "neutral_activations.pt", weights_only=False)
    neutral_acts = neutral_ckpt["activations"]

    print(f"\n{'Pair':<40} {'Layer':>5} {'fixed_orth cos':>14} {'r512_orth cos':>14}")
    print("-" * 76)

    for pair in pairs:
        pair_id = pair.pair_id
        if pair_id not in all_vecs:
            print(f"{pair_id:<40} {'—':>5}")
            continue

        pos_filename = filename_map.get(pair.positive, pair.positive)
        trait_path = layer_sweep_dir / "trait_activations" / f"{pos_filename}.pt"
        if not trait_path.exists():
            continue
        trait_acts = torch.load(trait_path, weights_only=False)["activations"]

        for layer in layers:
            if layer not in all_vecs[pair_id]:
                continue
            v_pos = compute_trait_vector_at_layer(trait_acts, neutral_acts, layer)
            v_pos_unit = _unit(v_pos.float())
            variants = all_vecs[pair_id][layer]
            cos_fixed = float((variants["fixed_orth"].float() @ v_pos_unit).item())
            cos_r512  = float((variants["r512_orth"].float() @ v_pos_unit).item())
            print(f"{pair_id:<40} {layer:>5} {cos_fixed:>14.6f} {cos_r512:>14.6f}")


def run(
    pairs: list[TraitPair],
    layer_sweep_dir: Path,
    output_dir: Path,
    layers: list[int],
    filename_map: dict[str, str],
) -> None:
    ip_prompt_dir = layer_sweep_dir / "extraction" / "ip_prompt_vectors"
    trait_acts_dir = layer_sweep_dir / "extraction" / "trait_activations"
    neutral_path   = layer_sweep_dir / "extraction" / "neutral_activations.pt"

    # Validate inputs
    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral activations not found: {neutral_path}")
    missing_traits = [
        pair.positive for pair in pairs
        if not (trait_acts_dir / f"{filename_map.get(pair.positive, pair.positive)}.pt").exists()
    ]
    if missing_traits:
        log.warning("Missing trait activations for: %s", missing_traits)
    missing_ip = [
        pair.negative for pair in pairs
        if not (ip_prompt_dir / f"{pair.negative}.pt").exists()
    ]
    if missing_ip:
        log.warning("Missing IP prompt vectors for: %s", missing_ip)

    log.info("Building inoculation vectors: %d pairs × %d layers × 4 variants",
             len(pairs), len(layers))

    all_vecs = build_all_inoculation_vectors(
        pairs=pairs,
        layer_sweep_dir=layer_sweep_dir / "extraction",
        ip_prompt_dir=ip_prompt_dir,
        layers=layers,
        filename_map=filename_map,
    )

    out_path = output_dir / "inoculation_vectors.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_vecs, out_path)

    n_pairs = len(all_vecs)
    n_total = sum(len(ldict) * 4 for ldict in all_vecs.values())
    log.info("Saved %d pairs, %d vectors total → %s", n_pairs, n_total, out_path)

    _print_sanity_table(all_vecs, layer_sweep_dir / "extraction", ip_prompt_dir,
                        pairs, filename_map, layers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_filename_map(items: list[str]) -> dict[str, str]:
    """Parse 'key:value' strings into a dict."""
    result = {}
    for item in items:
        k, _, v = item.partition(":")
        if k and v:
            result[k.strip()] = v.strip()
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build inoculation vectors (CPU).")
    p.add_argument("--pairs", nargs="+", required=True, metavar="POS:NEG")
    p.add_argument("--layer-sweep-dir", type=Path, default=Path("results/layer_sweep"))
    p.add_argument("--output-dir", type=Path, default=Path("results/layer_sweep"))
    p.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    p.add_argument("--filename-map", nargs="*", default=[],
                   metavar="TRAITNAME:FILENAME",
                   help="Override default trait→filename mapping for layer sweep files. "
                        "E.g. --filename-map apologetic:apology playful:playfulness")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    pairs = [TraitPair.from_string(s) for s in args.pairs]
    filename_map = {**_DEFAULT_FILENAME_MAP, **_parse_filename_map(args.filename_map)}

    run(
        pairs=pairs,
        layer_sweep_dir=args.layer_sweep_dir,
        output_dir=args.output_dir,
        layers=args.layers,
        filename_map=filename_map,
    )


if __name__ == "__main__":
    main()
