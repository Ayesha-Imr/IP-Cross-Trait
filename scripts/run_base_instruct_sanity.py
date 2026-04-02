"""
Base Instruct IP Vector Sanity Check — GPU generation script.

Tests IP steering vectors on the BASE instruct model (Qwen/Qwen2.5-7B-Instruct),
with traits explicitly elicited via system prompts. Unlike run_full_eval.py, this
script loads a single model once (no per-pair FT model swap) and runs three
elicitation modes per pair:

  both — both traits in system prompt (primary test: selective suppression?)
  neg  — negative trait only (control: isolated suppression)
  pos  — positive trait only (control: steering doesn't damage positive?)

For each mode: 1 baseline (no steering) + N steering configs.
Total: 3 modes × (1 + 4 configs) = 15 conditions × 100 queries × 8 pairs = 12,000 gens.

Output layout:
    results/base_instruct_sanity/{pair_id}/{dataset_name}/{mode}_{config_name}_responses.jsonl
    results/base_instruct_sanity/{pair_id}/_done.json   ← resume checkpoint

Usage — all pairs, instruction_wild only (recommended start):
    python scripts/run_base_instruct_sanity.py \\
        --config configs/base_instruct_sanity.yaml \\
        --datasets instruction_wild \\
        --hf-token $HF_TOKEN

Usage — subset of pairs:
    python scripts/run_base_instruct_sanity.py \\
        --config configs/base_instruct_sanity.yaml \\
        --pairs monotone:enthusiasm sarcasm:paranoia \\
        --hf-token $HF_TOKEN

Usage — add ultrachat later (resume-safe, skips completed conditions):
    python scripts/run_base_instruct_sanity.py \\
        --config configs/base_instruct_sanity.yaml \\
        --datasets ultrachat \\
        --hf-token $HF_TOKEN

Usage — smoke test (1 pair, 5 queries, both mode only):
    python scripts/run_base_instruct_sanity.py \\
        --config configs/base_instruct_sanity.yaml \\
        --pairs monotone:enthusiasm \\
        --datasets instruction_wild \\
        --modes both \\
        --n-queries 5 \\
        --hf-token $HF_TOKEN
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
import yaml

from config import TraitPair
from extraction.model_io import load_model, unload_model
from pipeline_interface.traits import resolve_trait
from steering.config_loader import FullEvalConfig, SteeringConfig
from steering.generation import (
    condition_key,
    generate_condition_with_elicitation,
    load_done,
    load_queries,
    mark_done,
)
from steering.hooks import attach_steering_hook

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Base instruct IP vector sanity check (GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/base_instruct_sanity.yaml"),
        help="Path to YAML config.",
    )
    p.add_argument(
        "--pairs", nargs="+", metavar="POS:NEG", default=None,
        help="Override pairs from YAML, e.g. monotone:enthusiasm sarcasm:paranoia.",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset name(s) to run, e.g. instruction_wild ultrachat. Default: all in YAML.",
    )
    p.add_argument(
        "--modes", nargs="+", choices=["both", "neg", "pos"], default=None,
        help="Elicitation modes to run. Default: all in YAML (both neg pos).",
    )
    p.add_argument(
        "--n-queries", type=int, default=None,
        help="Override n_queries per condition (useful for smoke tests).",
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

    # Parse standard fields via FullEvalConfig
    cfg = FullEvalConfig.from_yaml(config_path)

    # Parse custom fields from raw YAML (model_id, query_offset, elicitation_modes)
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)

    model_id: str = raw_yaml.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    query_offset: int = raw_yaml.get("query_offset", 1200)
    yaml_modes: list[str] = raw_yaml.get("elicitation_modes", ["both", "neg", "pos"])

    # CLI overrides
    pairs: list[TraitPair] = (
        [TraitPair.from_string(s) for s in args.pairs] if args.pairs else cfg.pairs
    )
    elicitation_modes: list[str] = args.modes if args.modes else yaml_modes

    # Filter datasets by CLI selection
    datasets = cfg.datasets
    if args.datasets:
        selected = set(args.datasets)
        datasets = [ds for ds in datasets if ds.name in selected]
        if not datasets:
            raise ValueError(
                f"No matching datasets found. Requested: {args.datasets}, "
                f"available: {[ds.name for ds in cfg.datasets]}"
            )

    if not cfg.paths.vectors.exists():
        raise FileNotFoundError(
            f"Inoculation vectors not found: {cfg.paths.vectors}\n"
            "Run scripts/build_inoculation_vectors.py first."
        )
    all_vectors = torch.load(cfg.paths.vectors, weights_only=False)

    # Merge in control vectors (random + neutral) if they exist alongside the main vectors.
    control_vectors_path = cfg.paths.vectors.parent / "control_vectors.pt"
    if control_vectors_path.exists():
        ctrl = torch.load(control_vectors_path, weights_only=False)
        for pid, layer_data in ctrl.items():
            all_vectors.setdefault(pid, {})
            for layer, vdict in layer_data.items():
                all_vectors[pid].setdefault(layer, {}).update(vdict)
        log.info("Merged control vectors from %s.", control_vectors_path)

    # Snapshot config to output dir for reproducibility
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config.yaml"
    if not snapshot_path.exists():
        shutil.copy2(config_path, snapshot_path)
        log.info("Config snapshotted to %s.", snapshot_path)

    log.info("=== Base Instruct IP Vector Sanity Check ===")
    log.info("  Model:  %s", model_id)
    log.info("  Pairs:  %d | Modes: %s | Steering configs: %d",
             len(pairs), elicitation_modes, len(cfg.steering_configs))
    log.info("  Datasets: %s", [ds.name for ds in datasets])
    log.info("  Query offset: %d", query_offset)

    # Load base model ONCE for all pairs
    model, tokenizer = load_model(
        model_id,
        models_cache_dir=args.models_cache_dir,
        hf_token=args.hf_token,
    )

    try:
        for pi, pair in enumerate(pairs):
            pair_id = pair.pair_id
            pair_dir = output_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)

            # Resolve adjective forms for trait elicitation prompts
            pos_adj = resolve_trait(pair.positive).adjective
            neg_adj = resolve_trait(pair.negative).adjective

            pair_vectors = all_vectors.get(pair_id, {})
            if not pair_vectors:
                log.warning("[%d/%d] No inoculation vectors for '%s' — only baselines will run.",
                            pi + 1, len(pairs), pair_id)

            done = load_done(pair_dir)

            all_configs = [SteeringConfig.baseline()] + cfg.steering_configs
            total_conditions = len(datasets) * len(elicitation_modes) * len(all_configs)
            done_count = sum(
                1
                for ds in datasets
                for mode in elicitation_modes
                for cfg_item in all_configs
                if condition_key(ds.name, f"{mode}_{cfg_item.name}") in done
            )
            log.info("[%d/%d] Pair: %s (pos_adj=%s, neg_adj=%s) — %d/%d conditions remaining.",
                     pi + 1, len(pairs), pair_id, pos_adj, neg_adj,
                     total_conditions - done_count, total_conditions)

            for ds_cfg in datasets:
                n_q = args.n_queries if args.n_queries is not None else ds_cfg.n_queries
                queries = load_queries(
                    ds_cfg.source, n_q,
                    cfg.paths.data_dir, cfg.paths.ultrachat_path,
                    offset=query_offset,
                )
                ds_dir = pair_dir / ds_cfg.name
                ds_dir.mkdir(parents=True, exist_ok=True)

                for mode in elicitation_modes:
                    for steering_cfg_item in all_configs:
                        cond_name = f"{mode}_{steering_cfg_item.name}"
                        key = condition_key(ds_cfg.name, cond_name)
                        if key in done:
                            log.info("  [%s/%s/%s] skipped (done).", pair_id, ds_cfg.name, cond_name)
                            continue

                        out_path = ds_dir / f"{cond_name}_responses.jsonl"
                        log.info("  [%s/%s/%s] generating %d responses ...",
                                 pair_id, ds_cfg.name, cond_name, len(queries))

                        # Attach steering hook if not baseline
                        handle = None
                        if not steering_cfg_item.is_baseline:
                            layer_vecs = pair_vectors.get(steering_cfg_item.layer, {})
                            v_inoc = layer_vecs.get(steering_cfg_item.variant)
                            if v_inoc is None:
                                log.error(
                                    "  Vector missing: pair=%s layer=%d variant=%s — skipping.",
                                    pair_id, steering_cfg_item.layer, steering_cfg_item.variant,
                                )
                                continue
                            handle = attach_steering_hook(
                                model, steering_cfg_item.layer, v_inoc, steering_cfg_item.alpha
                            )

                        try:
                            with open(out_path, "a") as f:
                                generate_condition_with_elicitation(
                                    model=model,
                                    tokenizer=tokenizer,
                                    queries=queries,
                                    pair=pair,
                                    dataset_name=ds_cfg.name,
                                    steering_cfg=steering_cfg_item,
                                    gen_params=cfg.generation,
                                    elicitation_mode=mode,
                                    pos_adjective=pos_adj,
                                    neg_adjective=neg_adj,
                                    system_prompt_base=cfg.generation.system_prompt,
                                    output_file=f,
                                )
                        finally:
                            if handle is not None:
                                handle.remove()

                        mark_done(pair_dir, key)
                        log.info("  [%s/%s/%s] done → %s",
                                 pair_id, ds_cfg.name, cond_name, out_path)

    finally:
        del model, tokenizer
        unload_model()

    log.info("=== Generation complete ===")


if __name__ == "__main__":
    main()
