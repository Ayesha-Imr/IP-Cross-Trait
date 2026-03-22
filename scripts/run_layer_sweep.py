"""
Layer Sweep CLI.

Two subcommands:

  extract  — GPU: extract trait activations at all layers for all traits.
             Run on RunPod in a tmux session.

  filter   — CPU/API: score responses with gpt-5-mini structured outputs.
             Can run locally after downloading the JSONL response files.

Usage:
  python scripts/run_layer_sweep.py extract \\
      --pairs apologetic:playful poetic:mathematical ... \\
      --data-dir ./data --output-dir results/layer_sweep

  python scripts/run_layer_sweep.py filter \\
      --pairs apologetic:playful poetic:mathematical ... \\
      --output-dir results/layer_sweep

Smoke test (2 pairs, 3 queries, 1 rollout):
  python scripts/run_layer_sweep.py extract \\
      --pairs poetic:mathematical apologetic:playful \\
      --n-queries 3 --n-rollouts 1 \\
      --data-dir ./data --output-dir results/layer_sweep_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_pairs(pair_strings: list[str]) -> list[str]:
    """Parse 'pos:neg' strings and return all unique trait names."""
    from config import TraitPair
    traits: list[str] = []
    seen: set[str] = set()
    for s in pair_strings:
        pair = TraitPair.from_string(s)
        for t in (pair.positive, pair.negative):
            if t not in seen:
                traits.append(t)
                seen.add(t)
    return traits


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    from extraction.layer_sweep import SweepConfig, run_layer_sweep_extraction

    config = SweepConfig(
        n_queries=args.n_queries,
        n_rollouts=args.n_rollouts,
    )
    traits = _parse_pairs(args.pairs)

    logging.getLogger().info("Traits to extract: %s", traits)

    run_layer_sweep_extraction(
        traits=traits,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        config=config,
        base_model_id=args.base_model_id,
        hf_token=args.hf_token,
        models_cache_dir=Path(args.models_cache_dir) if args.models_cache_dir else None,
    )


# ---------------------------------------------------------------------------
# Subcommand: filter
# ---------------------------------------------------------------------------

def cmd_filter(args: argparse.Namespace) -> None:
    from filtering.response_filter import run_filtering

    traits = _parse_pairs(args.pairs)
    run_filtering(
        output_dir=Path(args.output_dir),
        traits=traits,
        scoring_model=args.scoring_model,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer sweep extraction and filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- shared args (defined as parent parser)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--pairs", nargs="+", required=True, metavar="POS:NEG",
        help="Trait pairs, e.g. apologetic:playful poetic:mathematical",
    )
    shared.add_argument("--output-dir", default="results/layer_sweep",
                        help="Root output directory (default: results/layer_sweep)")

    # -- extract subcommand
    p_extract = sub.add_parser("extract", parents=[shared],
                                help="GPU: extract activations at all layers.")
    p_extract.add_argument("--data-dir", default="./data",
                           help="Path to IP-Cross-Trait/data/")
    p_extract.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p_extract.add_argument("--hf-token", default=None)
    p_extract.add_argument("--models-cache-dir", default=None,
                           help="Where to cache downloaded model weights.")
    p_extract.add_argument("--n-queries", type=int, default=30)
    p_extract.add_argument("--n-rollouts", type=int, default=5)
    p_extract.set_defaults(func=cmd_extract)

    # -- filter subcommand
    p_filter = sub.add_parser("filter", parents=[shared],
                               help="CPU/API: score responses with gpt-5-mini.")
    p_filter.add_argument("--scoring-model", default="gpt-5-mini",
                          help="OpenAI model for scoring (default: gpt-5-mini)")
    p_filter.set_defaults(func=cmd_filter)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
