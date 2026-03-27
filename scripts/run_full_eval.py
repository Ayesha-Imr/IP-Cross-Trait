"""
Full evaluation generation — GPU script.

Loads each FT baseline model and generates responses under:
  - 1 baseline condition (no steering)
  - N steering configs from the YAML (default: 2)
  × 2 datasets (InstructionWild + UltraChat), 200 queries each

Output layout:
    results/full_eval/{pair_id}/{dataset_name}/{config_name}_responses.jsonl
    results/full_eval/{pair_id}/_done.json   ← resume checkpoint

Usage — single pod (all 8 pairs):
    python scripts/run_full_eval.py \\
        --config configs/full_eval.yaml \\
        --hf-token $HF_TOKEN

Usage — multi-pod (split pairs manually):
    # Pod A
    python scripts/run_full_eval.py \\
        --config configs/full_eval.yaml \\
        --pairs monotone:enthusiasm brevity:enthusiasm sarcasm:paranoia informal:assertiveness \\
        --hf-token $HF_TOKEN
    # Pod B
    python scripts/run_full_eval.py \\
        --config configs/full_eval.yaml \\
        --pairs poetic:mathematical fanaticism:ALL-CAPS sarcasm:empathy cheater:philosophical \\
        --hf-token $HF_TOKEN

Resume: re-run the same command — already-completed conditions are skipped via _done.json.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from config import TraitPair
from extraction.model_io import load_model, unload_model
from steering.config_loader import FullEvalConfig
from steering.generation import FT_MODELS, run_pair

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full evaluation generation sweep (GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/full_eval.yaml"),
        help="Path to YAML config (default: configs/full_eval.yaml).",
    )
    p.add_argument(
        "--pairs", nargs="+", metavar="POS:NEG", default=None,
        help="Override pairs from YAML, e.g. monotone:enthusiasm brevity:enthusiasm.",
    )
    p.add_argument("--hf-token", default=None, help="HuggingFace token for gated models.")
    p.add_argument("--models-cache-dir", type=Path, default=None,
                   help="Local directory to cache downloaded models.")
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

    # CLI --pairs overrides YAML pairs
    pairs: list[TraitPair] = (
        [TraitPair.from_string(s) for s in args.pairs]
        if args.pairs
        else cfg.pairs
    )

    if not cfg.paths.vectors.exists():
        raise FileNotFoundError(
            f"Inoculation vectors not found: {cfg.paths.vectors}\n"
            "Run scripts/build_inoculation_vectors.py first."
        )
    all_vectors = torch.load(cfg.paths.vectors, weights_only=False)

    # Snapshot config to output dir for reproducibility
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cfg.paths.output_dir / "config.yaml"
    if not snapshot_path.exists():
        shutil.copy2(config_path, snapshot_path)
        log.info("Config snapshotted to %s.", snapshot_path)

    log.info("=== Full Evaluation Generation ===")
    log.info(
        "  Pairs: %d | Steering configs: %d | Datasets: %s | Queries/dataset: %s",
        len(pairs),
        len(cfg.steering_configs),
        [ds.name for ds in cfg.datasets],
        [ds.n_queries for ds in cfg.datasets],
    )

    for i, pair in enumerate(pairs):
        model_id = FT_MODELS.get(pair.pair_id)
        if model_id is None:
            log.error(
                "[%d/%d] No FT model ID for pair '%s'. Add it to steering/generation.py FT_MODELS.",
                i + 1, len(pairs), pair.pair_id,
            )
            continue

        log.info("[%d/%d] Pair: %s → %s", i + 1, len(pairs), pair.pair_id, model_id)
        model, tokenizer = load_model(
            model_id,
            models_cache_dir=args.models_cache_dir,
            hf_token=args.hf_token,
        )

        try:
            run_pair(
                pair=pair,
                model=model,
                tokenizer=tokenizer,
                all_vectors=all_vectors,
                datasets=cfg.datasets,
                steering_configs=cfg.steering_configs,
                gen_params=cfg.generation,
                output_dir=cfg.paths.output_dir,
                data_dir=cfg.paths.data_dir,
                ultrachat_path=cfg.paths.ultrachat_path,
            )
        finally:
            del model, tokenizer
            unload_model()

    log.info("=== Generation complete ===")


if __name__ == "__main__":
    main()
