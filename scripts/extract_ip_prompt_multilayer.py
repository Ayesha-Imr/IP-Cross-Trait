"""
Extract IP prompt activation vectors at multiple transformer layers.

For each negative trait, extracts last-prompt-token activations for:
  - The fixed IP prompt (base_prompt from rephrasings file)
  - All 512 rephrasings

Computes contrastive vectors (positive - neutral) at each specified layer and saves:
  {layer: {"fixed_ip": tensor(hidden_dim,), "r512_mean": tensor(hidden_dim,)}}

Run on RunPod (GPU required) — forward passes only, no generation.

Usage:
  python scripts/extract_ip_prompt_multilayer.py \\
      --pairs informal:slang passive-aggression:wit sadistic:pessimism \\
              paranoia:caution defensiveness:rebellion apologetic:playful \\
              sarcasm:paranoia dramatic:gaslighting monotone:enthusiasm \\
              informal:assertiveness cheater:philosophical sarcasm:empathy \\
              shakespearean:manipulative brevity:enthusiasm fanaticism:ALL-CAPS \\
              poetic:mathematical \\
      --data-dir ./data \\
      --output-dir results/layer_sweep \\
      --layers 3 8 16 20 \\
      --hf-token $HF_TOKEN

Smoke test (2 traits, layers 3 16):
  python scripts/extract_ip_prompt_multilayer.py \\
      --pairs apologetic:playful poetic:mathematical \\
      --data-dir ./data --output-dir results/layer_sweep_smoke \\
      --layers 3 16 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from config import TraitPair
from extraction.activation_ops import extract_activations_for_system_at_layers
from extraction.layer_sweep import _load_queries
from extraction.model_io import load_model, unload_model
from pipeline_interface.rephrasings import load_rephrasings
from pipeline_interface.paths import PipelinePaths

log = logging.getLogger(__name__)

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."


def _compute_contrastive_at_layers(
    pos_acts: dict[int, list[torch.Tensor]],
    neg_acts: dict[int, list[torch.Tensor]],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute mean(pos) - mean(neg) at each layer. Returns {layer: tensor}."""
    return {
        l: torch.stack(pos_acts[l]).mean(0) - torch.stack(neg_acts[l]).mean(0)
        for l in layers
    }


def extract_ip_vectors_for_trait(
    model,
    tokenizer,
    neg_trait: str,
    queries: list[str],
    layers: list[int],
    batch_size: int,
    paths: PipelinePaths,
    output_dir: Path,
) -> bool:
    """Extract and save IP prompt vectors at multiple layers for one negative trait.

    Skips if output file already exists (resume support).
    Returns True if extracted, False if skipped.
    """
    out_dir = output_dir / "extraction" / "ip_prompt_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{neg_trait}.pt"

    if out_path.exists():
        log.info("  '%s' already done — skipping.", neg_trait)
        return False

    reph_data = load_rephrasings(neg_trait, paths)
    if reph_data is None:
        log.warning("  No rephrasings for '%s' — skipping.", neg_trait)
        return False

    base_prompt, rephrasings = reph_data
    log.info("  '%s': %d rephrasings + 1 fixed IP prompt", neg_trait, len(rephrasings))

    # Neutral baseline (shared across all prompts for this trait)
    log.info("  Extracting neutral baseline (%d queries) ...", len(queries))
    neutral_acts = extract_activations_for_system_at_layers(
        model, tokenizer, NEUTRAL_SYSTEM_PROMPT, queries, layers, batch_size,
    )

    # Fixed IP prompt
    log.info("  Extracting fixed IP prompt ...")
    fixed_pos_acts = extract_activations_for_system_at_layers(
        model, tokenizer, base_prompt, queries, layers, batch_size,
    )
    fixed_vec = _compute_contrastive_at_layers(fixed_pos_acts, neutral_acts, layers)

    # R512 rephrasings — accumulate contrastive vectors, then mean
    t0 = time.time()
    r512_vecs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, reph in enumerate(rephrasings):
        pos_acts = extract_activations_for_system_at_layers(
            model, tokenizer, reph, queries, layers, batch_size,
        )
        contrastive = _compute_contrastive_at_layers(pos_acts, neutral_acts, layers)
        for l in layers:
            r512_vecs[l].append(contrastive[l])

        if (i + 1) % 100 == 0 or (i + 1) == len(rephrasings):
            elapsed = (time.time() - t0) / 60
            log.info("    %d/%d rephrasings done (%.1f min)", i + 1, len(rephrasings), elapsed)

    # Compute normalized r512 mean at each layer
    r512_mean: dict[int, torch.Tensor] = {}
    for l in layers:
        mean_vec = torch.stack(r512_vecs[l]).mean(0)
        norm = mean_vec.norm()
        r512_mean[l] = mean_vec / norm if norm > 1e-8 else mean_vec

    # Save: {layer: {"fixed_ip": tensor, "r512_mean": tensor}}
    payload = {l: {"fixed_ip": fixed_vec[l], "r512_mean": r512_mean[l]} for l in layers}
    torch.save(payload, out_path)
    log.info("  Saved IP prompt vectors for '%s' to %s", neg_trait, out_path)
    return True


def run_extraction(
    pairs: list[TraitPair],
    data_dir: Path,
    output_dir: Path,
    layers: list[int],
    batch_size: int,
    base_model_id: str,
    hf_token: str | None,
    models_cache_dir: Path | None,
) -> None:
    paths = PipelinePaths(data_dir)
    queries = _load_queries(data_dir, n=30)

    # Unique negative traits across all pairs
    neg_traits = list(dict.fromkeys(pair.negative for pair in pairs))
    log.info("=== IP Prompt Multilayer Extraction ===")
    log.info("  Traits: %d | Layers: %s | Queries: %d | Batch: %d",
             len(neg_traits), layers, len(queries), batch_size)

    model, tokenizer = load_model(
        base_model_id,
        models_cache_dir=Path(models_cache_dir) if models_cache_dir else None,
        hf_token=hf_token,
    )

    try:
        for i, neg_trait in enumerate(neg_traits):
            log.info("[%d/%d] Trait: %s", i + 1, len(neg_traits), neg_trait)
            extract_ip_vectors_for_trait(
                model, tokenizer,
                neg_trait=neg_trait,
                queries=queries,
                layers=layers,
                batch_size=batch_size,
                paths=paths,
                output_dir=output_dir,
            )
    finally:
        del model, tokenizer
        unload_model()

    log.info("=== Extraction complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract IP prompt vectors at multiple layers.")
    p.add_argument("--pairs", nargs="+", required=True, metavar="POS:NEG",
                   help="Trait pairs, e.g. apologetic:playful poetic:mathematical")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/layer_sweep"))
    p.add_argument("--layers", nargs="+", type=int, default=[3, 8, 16, 20])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--models-cache-dir", type=Path, default=None)
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
    run_extraction(
        pairs=pairs,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layers=args.layers,
        batch_size=args.batch_size,
        base_model_id=args.base_model_id,
        hf_token=args.hf_token,
        models_cache_dir=args.models_cache_dir,
    )


if __name__ == "__main__":
    main()
