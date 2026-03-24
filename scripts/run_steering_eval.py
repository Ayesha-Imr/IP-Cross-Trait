"""
Score steered generation responses using the logprobs-based judge.

CPU/API script — runs locally / without a GPU.
Reads response JSONL files produced by run_steered_generation.py,
scores each response for both the positive and negative trait using
gpt-5-mini with logprobs (temperature=0, max_tokens=1, top_logprobs=20),
and writes scored JSONL files with pos_score and neg_score fields added. 

Parallelism: 20 concurrent API calls with exponential backoff + jitter.
Resume-safe: skips responses already present in the output file.
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

from config import TraitPair
from judging.logprobs_judge import score_responses_file
from pipeline_interface.traits import resolve_trait

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score steered responses with logprobs judge (CPU/API).")
    p.add_argument("--input-dir", type=Path, default=Path("results/steering_sweep"))
    p.add_argument("--output-dir", type=Path, default=Path("results/steering_sweep/scores"))
    p.add_argument("--pairs", nargs="+", required=True, metavar="POS:NEG")
    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--max-workers", type=int, default=20, help="Parallel API calls")
    p.add_argument("--no-coherence", action="store_true",
                   help="Skip coherence scoring (default: score coherence)")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Steering Evaluation (logprobs judge) ===")
    log.info("  Pairs: %d | Workers: %d", len(pairs), args.max_workers)

    for i, pair in enumerate(pairs):
        input_path = args.input_dir / f"{pair.pair_id}_responses.jsonl"
        output_path = args.output_dir / f"{pair.pair_id}_scores.jsonl"

        if not input_path.exists():
            log.warning("[%d/%d] No responses file for '%s' at %s — skipping.",
                        i + 1, len(pairs), pair.pair_id, input_path)
            continue

        pos_noun = resolve_trait(pair.positive).noun
        neg_noun = resolve_trait(pair.negative).noun
        log.info("[%d/%d] Pair: %s  (pos=%s, neg=%s)",
                 i + 1, len(pairs), pair.pair_id, pos_noun, neg_noun)

        score_responses_file(
            input_path=input_path,
            output_path=output_path,
            pos_trait_noun=pos_noun,
            neg_trait_noun=neg_noun,
            api_key=args.api_key,
            max_workers=args.max_workers,
            score_coherence=not args.no_coherence,
        )

    log.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
