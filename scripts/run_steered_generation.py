"""
Steered generation: sweep (layer, variant, alpha) conditions on pilot pairs.

For each pilot pair, loads the FT baseline model, attaches a forward hook
that injects -α · v_inoc at the specified transformer layer, and generates
responses for 30 queries. All conditions are saved to a JSONL file per pair
with per-condition checkpointing for safe interruption/resume on RunPod.

FT models used: the plain FT baselines (I(Empty) variant) — these have the
positive trait from fine-tuning plus collateral negative trait acquisition.
Steering suppresses the negative trait at inference time.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from config import TraitPair
from extraction.activation_ops import generate_response
from extraction.layer_sweep import _load_queries
from extraction.model_io import load_model, unload_model
from steering.hooks import attach_steering_hook

log = logging.getLogger(__name__)

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

# FT baseline (I(Empty)) model IDs for the 4 pilot pairs
PILOT_FT_MODELS: dict[str, str] = {
    "poetic_mathematical":    "longtermrisk/Qwen2.5-7B-Instruct-ftjob-104626dade9a",
    "sarcasm_paranoia":       "longtermrisk/Qwen2.5-7B-Instruct-ftjob-3033d004dd86",
    "informal_slang":         "longtermrisk/Qwen2.5-7B-Instruct-ftjob-57a3469a7c91",
    "passive-aggression_wit": "longtermrisk/Qwen2.5-7B-Instruct-ftjob-088a61ce912e",
}

LAYERS = [3, 8, 16, 20]
VARIANTS = ["fixed_raw", "fixed_orth", "r512_raw", "r512_orth"]


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _done_path(output_dir: Path, pair_id: str) -> Path:
    p = output_dir / "_done"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{pair_id}.json"


def _load_done(output_dir: Path, pair_id: str) -> set[str]:
    path = _done_path(output_dir, pair_id)
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _mark_done(output_dir: Path, pair_id: str, key: str) -> None:
    path = _done_path(output_dir, pair_id)
    done = _load_done(output_dir, pair_id)
    done.add(key)
    path.write_text(json.dumps(sorted(done)))


def _condition_key(layer: int | None, variant: str, alpha: float) -> str:
    if layer is None:
        return "baseline"
    return f"L{layer}_{variant}_a{alpha}"


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _generate_condition(
    model,
    tokenizer,
    queries: list[str],
    pair: TraitPair,
    layer: int | None,
    variant: str,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    output_file,
) -> None:
    """Generate responses for one condition; write each to output_file."""
    for qi, query in enumerate(queries):
        _, _, response = generate_response(
            model, tokenizer,
            NEUTRAL_SYSTEM_PROMPT, query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        rec = {
            "pair_id": pair.pair_id,
            "positive_trait": pair.positive,
            "negative_trait": pair.negative,
            "layer": layer,
            "variant": variant,
            "alpha": alpha,
            "query_idx": qi,
            "user_query": query,
            "response": response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        output_file.write(json.dumps(rec) + "\n")
        output_file.flush()


def run_pair(
    pair: TraitPair,
    model_id: str,
    vectors: dict,
    queries: list[str],
    alphas: list[float],
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    hf_token: str | None,
    models_cache_dir: Path | None,
) -> None:
    pair_id = pair.pair_id
    if pair_id not in vectors:
        log.error("No vectors found for pair '%s' — skipping.", pair_id)
        return

    done = _load_done(output_dir, pair_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair_id}_responses.jsonl"

    model, tokenizer = load_model(
        model_id,
        models_cache_dir=models_cache_dir,
        hf_token=hf_token,
    )

    try:
        with open(out_path, "a") as f:
            # Baseline (no steering)
            key = _condition_key(None, "none", 0.0)
            if key not in done:
                log.info("  [%s] baseline", pair_id)
                _generate_condition(model, tokenizer, queries, pair,
                                    None, "none", 0.0, max_new_tokens, temperature, f)
                _mark_done(output_dir, pair_id, key)
            else:
                log.info("  [%s] baseline — skipped (done)", pair_id)

            # Steered conditions
            pair_vectors = vectors[pair_id]
            total = len(LAYERS) * len(VARIANTS) * len(alphas)
            done_count = sum(1 for L in LAYERS for v in VARIANTS for a in alphas
                             if _condition_key(L, v, a) in done)
            log.info("  [%s] %d/%d conditions remaining", pair_id, total - done_count, total)

            for layer in LAYERS:
                if layer not in pair_vectors:
                    log.warning("  Layer %d missing for pair '%s' — skipping.", layer, pair_id)
                    continue
                for variant in VARIANTS:
                    if variant not in pair_vectors[layer]:
                        log.warning("  Variant %s missing for layer %d — skipping.", variant, layer)
                        continue
                    v_inoc = pair_vectors[layer][variant]
                    for alpha in alphas:
                        key = _condition_key(layer, variant, alpha)
                        if key in done:
                            continue
                        log.info("  [%s] L%d %s α=%.1f", pair_id, layer, variant, alpha)
                        handle = attach_steering_hook(model, layer, v_inoc, alpha)
                        try:
                            _generate_condition(model, tokenizer, queries, pair,
                                                layer, variant, alpha,
                                                max_new_tokens, temperature, f)
                        finally:
                            handle.remove()
                        _mark_done(output_dir, pair_id, key)

    finally:
        del model, tokenizer
        unload_model()

    log.info("Pair '%s' complete → %s", pair_id, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steered generation sweep (GPU).")
    p.add_argument("--pairs", nargs="+", required=True, metavar="POS:NEG")
    p.add_argument("--vectors-path", type=Path,
                   default=Path("results/layer_sweep/inoculation_vectors.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("results/steering_sweep"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
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

    if not args.vectors_path.exists():
        raise FileNotFoundError(
            f"Inoculation vectors not found: {args.vectors_path}\n"
            "Run scripts/build_inoculation_vectors.py first."
        )

    all_vectors = torch.load(args.vectors_path, weights_only=False)
    queries = _load_queries(args.data_dir, n=args.n_queries)
    pairs = [TraitPair.from_string(s) for s in args.pairs]

    log.info("=== Steered Generation Sweep ===")
    log.info("  Pairs: %d | Layers: %s | Variants: %d | Alphas: %s | Queries: %d",
             len(pairs), LAYERS, len(VARIANTS), args.alphas, len(queries))

    for pair in pairs:
        model_id = PILOT_FT_MODELS.get(pair.pair_id)
        if model_id is None:
            log.error("No FT model ID for pair '%s'. Add it to PILOT_FT_MODELS.", pair.pair_id)
            continue
        log.info("[%d/%d] Pair: %s → %s", pairs.index(pair) + 1, len(pairs), pair.pair_id, model_id)
        run_pair(
            pair=pair,
            model_id=model_id,
            vectors=all_vectors,
            queries=queries,
            alphas=args.alphas,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            hf_token=args.hf_token,
            models_cache_dir=args.models_cache_dir,
        )

    log.info("=== Generation complete ===")


if __name__ == "__main__":
    main()
