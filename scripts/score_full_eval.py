"""
Full evaluation scoring.

Discovers all *_responses.jsonl files under results/full_eval/{pair_id}/{dataset}/
and scores each with the logprobs-based judge (pos trait, neg trait, coherence).

Output layout:
    results/full_eval/{pair_id}/{dataset_name}/scores/{config_name}_scores.jsonl

Resume-safe: each output file is appended to; already-scored records are skipped.

Usage:
    python scripts/score_full_eval.py \\
        --config configs/full_eval.yaml

    # Score only a subset of pairs:
    python scripts/score_full_eval.py \\
        --config configs/full_eval.yaml \\
        --pairs monotone:enthusiasm brevity:enthusiasm

    # Skip coherence to reduce API cost:
    python scripts/score_full_eval.py \\
        --config configs/full_eval.yaml \\
        --no-coherence
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
from steering.config_loader import FullEvalConfig

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full evaluation scoring — logprobs judge (CPU/API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/full_eval.yaml"),
        help="Path to YAML config (default: configs/full_eval.yaml).",
    )
    p.add_argument(
        "--pairs", nargs="+", metavar="POS:NEG", default=None,
        help="Override pairs from YAML.",
    )
    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel API calls (overrides config).")
    p.add_argument("--no-coherence", action="store_true",
                   help="Skip coherence scoring (overrides config).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = FullEvalConfig.from_yaml(config_path)

    pairs: list[TraitPair] = (
        [TraitPair.from_string(s) for s in args.pairs]
        if args.pairs
        else cfg.pairs
    )

    max_workers = args.max_workers or cfg.scoring.max_workers
    score_coherence = (not args.no_coherence) and cfg.scoring.score_coherence

    log.info("=== Full Evaluation Scoring ===")
    log.info("  Pairs: %d | Workers: %d | Coherence: %s", len(pairs), max_workers, score_coherence)

    total_files = 0
    for i, pair in enumerate(pairs):
        pos_noun = resolve_trait(pair.positive).noun
        neg_noun = resolve_trait(pair.negative).noun

        pair_dir = cfg.paths.output_dir / pair.pair_id
        if not pair_dir.exists():
            log.warning("[%d/%d] No output dir for '%s' — skipping (run generation first).",
                        i + 1, len(pairs), pair.pair_id)
            continue

        # Discover all response files across all datasets
        response_files = sorted(pair_dir.glob("*/*_responses.jsonl"))
        if not response_files:
            log.warning("[%d/%d] No response files found under %s — skipping.",
                        i + 1, len(pairs), pair_dir)
            continue

        log.info("[%d/%d] Pair: %s (pos=%s, neg=%s) — %d response file(s)",
                 i + 1, len(pairs), pair.pair_id, pos_noun, neg_noun, len(response_files))

        for resp_path in response_files:
            # resp_path: full_eval/{pair_id}/{dataset}/{config_name}_responses.jsonl
            # output:    full_eval/{pair_id}/{dataset}/scores/{config_name}_scores.jsonl
            scores_dir = resp_path.parent / "scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            scores_path = scores_dir / resp_path.name.replace("_responses.jsonl", "_scores.jsonl")

            log.info("  Scoring: %s", resp_path.relative_to(cfg.paths.output_dir))
            score_responses_file(
                input_path=resp_path,
                output_path=scores_path,
                pos_trait_noun=pos_noun,
                neg_trait_noun=neg_noun,
                api_key=args.api_key,
                max_workers=max_workers,
                score_coherence=score_coherence,
            )
            total_files += 1

    log.info("=== Scoring complete — %d file(s) processed ===", total_files)


if __name__ == "__main__":
    main()
