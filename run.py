#!/usr/bin/env python3
"""
CLI entry point for the IP Cross-Trait Suppression Analysis Pipeline.

Usage examples:

  # Full pipeline (extract phases 1A+1B, then analyze)
  python run.py run --pairs apologetic:playful poetic:mathematical sadistic:pessimism

  # GPU extraction only — Phase 1A (trait vectors)
  python run.py extract --pairs apologetic:playful --phase 1a

  # GPU extraction only — Phase 1B (prompt vectors)
  python run.py extract --pairs apologetic:playful --phase 1b

  # CPU analysis only (uses existing vectors + local eval CSVs)
  python run.py analyze --pairs apologetic:playful poetic:mathematical sadistic:pessimism

  # Custom data directory (default: data/)
  python run.py run --data-dir /path/to/data \\
                    --pairs apologetic:playful shakespearean:manipulative
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from checkpointing.manager import CheckpointManager
from config import ExtractionParams, PipelineConfig, TraitPair
from extraction.prompt_vectors import run_phase_1b
from extraction.trait_vectors import run_phase_1a
from pipeline_interface.paths import PipelinePaths
from scoring.metrics import load_all_metrics
from analysis.geometry_scatter import run_all_conditions as geo_all
from analysis.prompt_alignment import run_all_conditions as prompt_all
from analysis.summary_table import run_summary_table

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# Shared argument groups

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        metavar="DIR",
        help="Path to the data directory containing training_data/ and results/ (default: data/)",
    )
    parser.add_argument(
        "--pairs", required=True, nargs="+",
        metavar="POS:NEG",
        help="Trait pairs to process, e.g. apologetic:playful poetic:mathematical",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        metavar="DIR",
        help="Root output directory (default: results/)",
    )
    parser.add_argument(
        "--primary-eval", default="instruction_wild",
        metavar="EVAL_ID",
        help="Primary evaluation ID for main figures (default: instruction_wild)",
    )
    parser.add_argument(
        "--primary-condition", default="none",
        metavar="COND",
        help="Primary eval condition: none or respond (default: none)",
    )
    parser.add_argument(
        "--no-robustness", action="store_true",
        help="Skip robustness figures for secondary eval conditions.",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace token for private model access.",
    )
    parser.add_argument(
        "--models-cache-dir", type=Path, default=None, metavar="DIR",
        help="Directory to cache downloaded models (default: HuggingFace default).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )


def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phase", choices=["1a", "1b", "all"], default="all",
        help="Extraction phase to run: 1a (trait vectors), 1b (prompt vectors), all (default)",
    )
    parser.add_argument(
        "--layer", type=int, default=16, metavar="N",
        help="Transformer layer index to extract activations from (default: 16)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=30, metavar="N",
        help="Number of contrastive queries for trait extraction (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N",
        help="Batch size for Phase 1B prompt vector extraction (default: 16)",
    )
    parser.add_argument(
        "--no-ft-vectors", action="store_true",
        help="Skip FT model trait vector extraction (base model only).",
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-7B-Instruct", metavar="MODEL_ID",
        help="Base model HuggingFace ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )


# Config construction

def _build_config(args: argparse.Namespace) -> PipelineConfig:
    pairs = [TraitPair.from_string(s) for s in args.pairs]

    extraction_params = ExtractionParams(
        layer=getattr(args, "layer", 16),
        n_queries=getattr(args, "n_queries", 30),
        batch_size_prompts=getattr(args, "batch_size", 16),
    )

    primary_eval = args.primary_eval
    primary_condition = args.primary_condition

    all_conditions = [
        ("instruction_wild", "none"),
        ("instruction_wild", "respond"),
        ("ultrachat", "none"),
        ("ultrachat", "respond"),
    ]
    if getattr(args, "no_robustness", False):
        robustness_evals = [(primary_eval, primary_condition)]
    else:
        robustness_evals = all_conditions

    return PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pairs=pairs,
        extraction_params=extraction_params,
        base_model_id=getattr(args, "base_model", "Qwen/Qwen2.5-7B-Instruct"),
        extract_ft_vectors=not getattr(args, "no_ft_vectors", False),
        primary_eval_id=primary_eval,
        primary_condition=primary_condition,
        robustness_evals=robustness_evals,
        hf_token=args.hf_token,
        models_cache_dir=args.models_cache_dir,
    )


# Subcommand handlers

def cmd_extract(args: argparse.Namespace) -> None:
    """Run GPU extraction phases."""
    _setup_logging(getattr(args, "verbose", False))
    config = _build_config(args)
    ckpt_mgr = CheckpointManager(config.output_dir)
    config.save(config.output_dir / "config_used.json")

    phase = getattr(args, "phase", "all")
    if phase in ("1a", "all"):
        log.info("=== Phase 1A: Trait vector extraction ===")
        run_phase_1a(config, ckpt_mgr)

    if phase in ("1b", "all"):
        log.info("=== Phase 1B: Prompt vector extraction ===")
        run_phase_1b(config, ckpt_mgr)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run CPU analysis — generate figures and tables."""
    _setup_logging(getattr(args, "verbose", False))
    config = _build_config(args)
    ckpt_mgr = CheckpointManager(config.output_dir)
    paths = PipelinePaths(config.data_dir)
    config.save(config.output_dir / "config_used.json")

    log.info("Loading behavioral metrics from pipeline CSVs...")
    all_metrics = load_all_metrics(config, paths)

    primary_key = f"{config.primary_eval_id}/{config.primary_condition}"

    log.info("Generating Figure 1: Geometry vs. Collateral...")
    geo_all(config, ckpt_mgr, paths, all_metrics, config.output_dir)

    log.info("Generating Figure 2: Prompt Alignment vs. Collateral...")
    prompt_all(config, ckpt_mgr, paths, all_metrics, config.output_dir)

    log.info("Generating Summary Table...")
    df = run_summary_table(
        config, ckpt_mgr, paths, all_metrics, config.output_dir, primary_key,
    )

    print("\n" + "=" * 72)
    print(f"SUMMARY TABLE  [{primary_key}]")
    print("=" * 72)
    print(df.to_string(index=False))
    print()


def cmd_all(args: argparse.Namespace) -> None:
    """Run extraction (both phases) then analysis."""
    _setup_logging(getattr(args, "verbose", False))
    args.phase = "all"
    cmd_extract(args)
    cmd_analyze(args)


# Argument parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="IP Cross-Trait Suppression Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # extract subcommand
    extract_p = subparsers.add_parser(
        "extract",
        help="GPU activation extraction (Phase 1A and/or 1B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(extract_p)
    _add_extract_args(extract_p)
    extract_p.set_defaults(func=cmd_extract)

    # analyze subcommand
    analyze_p = subparsers.add_parser(
        "analyze",
        help="Generate figures + tables from existing vectors (CPU only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(analyze_p)
    analyze_p.set_defaults(func=cmd_analyze)

    # run subcommand — extract (both phases) then analyze
    run_p = subparsers.add_parser(
        "run",
        help="Full pipeline: extract phases 1A+1B then analyze.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(run_p)
    _add_extract_args(run_p)
    run_p.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
