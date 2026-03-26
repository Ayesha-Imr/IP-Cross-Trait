"""
Generate baseline responses (no steering) for FT models across all pairs.

GPU script — runs on RunPod. For each pair, loads the I(Empty) FT baseline model,
generates responses for N queries with no steering hook, and writes a JSONL file.
Output is compatible with run_steering_eval.py for scoring.

Resume-safe: skips pairs whose output file already has N complete records.

Usage (single pod, all 12 new pairs):
    python scripts/run_baseline_generation.py \\
        --pairs sadistic:pessimism apologetic:playful shakespearean:manipulative \\
                paranoia:caution informal:assertiveness sarcasm:empathy \\
                monotone:enthusiasm fanaticism:ALL-CAPS brevity:enthusiasm \\
                defensiveness:rebellion cheater:philosophical dramatic:gaslighting \\
        --output-dir results/baseline_survey \\
        --data-dir ./data \\
        --hf-token $HF_TOKEN

Usage (parallel — one of 4 pods):
    python scripts/run_baseline_generation.py \\
        --pairs sadistic:pessimism apologetic:playful shakespearean:manipulative \\
        --output-dir results/baseline_survey \\
        --data-dir ./data \\
        --hf-token $HF_TOKEN
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

from config import TraitPair
from extraction.activation_ops import generate_response
from extraction.layer_sweep import _load_queries
from extraction.model_io import load_model, unload_model

log = logging.getLogger(__name__)

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

# I(Empty) FT baseline model IDs — from data/models/models.md
FT_MODELS: dict[str, str] = {
    # 4 pilot pairs (already have baseline data from steering sweep)
    "poetic_mathematical":        "longtermrisk/Qwen2.5-7B-Instruct-ftjob-104626dade9a",
    "sarcasm_paranoia":           "longtermrisk/Qwen2.5-7B-Instruct-ftjob-3033d004dd86",
    "informal_slang":             "longtermrisk/Qwen2.5-7B-Instruct-ftjob-57a3469a7c91",
    "passive-aggression_wit":     "longtermrisk/Qwen2.5-7B-Instruct-ftjob-088a61ce912e",
    # 12 new pairs
    "sadistic_pessimism":         "longtermrisk/Qwen2.5-7B-Instruct-ftjob-057ac75f2666",
    "apologetic_playful":         "longtermrisk/Qwen2.5-7B-Instruct-ftjob-d0c32d4c677a",
    "shakespearean_manipulative": "longtermrisk/Qwen2.5-7B-Instruct-ftjob-a2febcbd7ec6",
    "paranoia_caution":           "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b5da7f5b79b4",
    "informal_assertiveness":     "longtermrisk/Qwen2.5-7B-Instruct-ftjob-9073fcefb9fa",
    "sarcasm_empathy":            "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c0d207e72bc9",
    "monotone_enthusiasm":        "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e63c4128dc90",
    "fanaticism_ALL-CAPS":        "longtermrisk/Qwen2.5-7B-Instruct-ftjob-3dff9ab69e04",
    "brevity_enthusiasm":         "longtermrisk/Qwen2.5-7B-Instruct-ftjob-a3c1039b8c8c",
    "defensiveness_rebellion":    "longtermrisk/Qwen2.5-7B-Instruct-ftjob-2f4532e7373d",
    "cheater_philosophical":      "longtermrisk/Qwen2.5-7B-Instruct-ftjob-dd6591c68ddb",
    "dramatic_gaslighting":       "longtermrisk/Qwen2.5-7B-Instruct-ftjob-6c4b1e08634f",
}


def _count_records(path: Path) -> int:
    """Count valid JSONL records in path. Returns 0 if file doesn't exist."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count


def run_pair(
    pair: TraitPair,
    model_id: str,
    queries: list[str],
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    hf_token: str | None,
    models_cache_dir: Path | None,
) -> None:
    pair_id = pair.pair_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair_id}_responses.jsonl"

    n_done = _count_records(out_path)
    if n_done >= len(queries):
        log.info("[%s] already complete (%d records) — skipping.", pair_id, n_done)
        return

    log.info("[%s] %d/%d queries done — resuming from query %d.",
             pair_id, n_done, len(queries), n_done)

    model, tokenizer = load_model(
        model_id,
        models_cache_dir=models_cache_dir,
        hf_token=hf_token,
    )

    try:
        with open(out_path, "a") as f:
            for qi in range(n_done, len(queries)):
                query = queries[qi]
                _, _, response = generate_response(
                    model, tokenizer,
                    NEUTRAL_SYSTEM_PROMPT, query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                rec = {
                    "pair_id": pair_id,
                    "positive_trait": pair.positive,
                    "negative_trait": pair.negative,
                    "layer": None,
                    "variant": "none",
                    "alpha": 0.0,
                    "query_idx": qi,
                    "user_query": query,
                    "response": response,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (qi + 1) % 10 == 0 or qi + 1 == len(queries):
                    log.info("  [%s] %d/%d queries done.", pair_id, qi + 1, len(queries))
    finally:
        del model, tokenizer
        unload_model()

    log.info("[%s] complete → %s", pair_id, out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline generation for FT models (GPU, no steering).")
    p.add_argument("--pairs", nargs="+", required=True, metavar="POS:NEG")
    p.add_argument("--output-dir", type=Path, default=Path("results/baseline_survey"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
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

    pairs = [TraitPair.from_string(s) for s in args.pairs]
    queries = _load_queries(args.data_dir, n=args.n_queries)

    log.info("=== Baseline Generation ===")
    log.info("  Pairs: %d | Queries: %d | Output: %s",
             len(pairs), len(queries), args.output_dir)

    for i, pair in enumerate(pairs):
        model_id = FT_MODELS.get(pair.pair_id)
        if model_id is None:
            log.error("[%d/%d] No model ID for '%s' — add it to FT_MODELS.",
                      i + 1, len(pairs), pair.pair_id)
            continue
        log.info("[%d/%d] %s → %s", i + 1, len(pairs), pair.pair_id, model_id)
        run_pair(
            pair=pair,
            model_id=model_id,
            queries=queries,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            hf_token=args.hf_token,
            models_cache_dir=args.models_cache_dir,
        )

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
