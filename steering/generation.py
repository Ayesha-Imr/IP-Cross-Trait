"""
Core generation logic for full evaluation.

Used as a shared library by scripts/run_full_eval.py.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import torch

from config import TraitPair
from extraction.activation_ops import generate_response
from steering.config_loader import DatasetConfig, GenerationParams, SteeringConfig
from steering.hooks import attach_steering_hook

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FT baseline model IDs (I(Empty) variant)
# ---------------------------------------------------------------------------

FT_MODELS: dict[str, str] = {
    "poetic_mathematical":        "longtermrisk/Qwen2.5-7B-Instruct-ftjob-104626dade9a",
    "sarcasm_paranoia":           "longtermrisk/Qwen2.5-7B-Instruct-ftjob-3033d004dd86",
    "informal_slang":             "longtermrisk/Qwen2.5-7B-Instruct-ftjob-57a3469a7c91",
    "passive-aggression_wit":     "longtermrisk/Qwen2.5-7B-Instruct-ftjob-088a61ce912e",
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


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(source: str, n: int, data_dir: Path, ultrachat_path: Path) -> list[str]:
    """Load up to n queries from either InstructionWild or UltraChat.

    source: "instructionwild" → data_dir/training_data/instructionwild_10000.json
            "ultrachat"       → ultrachat_path
    """
    if source == "instructionwild":
        return _load_instructionwild(data_dir, n)
    elif source == "ultrachat":
        return _load_ultrachat(ultrachat_path, n)
    else:
        raise ValueError(f"Unknown query source: {source!r}. Expected 'instructionwild' or 'ultrachat'.")


def _load_instructionwild(data_dir: Path, n: int) -> list[str]:
    """Load from InstructionWild starting at index 1000 (no eval overlap)."""
    iw_path = data_dir / "training_data" / "instructionwild_10000.json"
    if not iw_path.exists():
        iw_path = data_dir / "results" / "training_data" / "instructionwild_10000.json"
    if not iw_path.exists():
        raise FileNotFoundError(
            f"InstructionWild not found at {iw_path}. "
            "Run the pipeline's train.py first to populate this file."
        )
    with open(iw_path) as f:
        data = json.load(f)
    raw = data.get("instructions", data.get("prompts", [])) if isinstance(data, dict) else data
    prompts = [item["prompt"] if isinstance(item, dict) else str(item) for item in raw]
    start = 1000
    selected = prompts[start: start + n]
    if len(selected) < n:
        log.warning("InstructionWild: only %d queries available (requested %d).", len(selected), n)
    log.info("Loaded %d InstructionWild queries [%d:%d].", len(selected), start, start + n)
    return selected


def _load_ultrachat(ultrachat_path: Path, n: int) -> list[str]:
    """Load from UltraChat JSONL. Searches common locations if configured path not found."""
    import os
    
    # Try configured path first
    if ultrachat_path.exists():
        actual_path = ultrachat_path
    else:
        # Search common fallback locations, including any env var override
        candidates = [
            ultrachat_path,
            Path(os.environ.get("ULTRACHAT_PATH", ""))  if os.environ.get("ULTRACHAT_PATH") else None,
            Path("/root/misalignment-inoculation/mi/evaluation/ultrachat/ultrachat_prompts.jsonl"),
            Path("/workspace/misalignment-inoculation/mi/evaluation/ultrachat/ultrachat_prompts.jsonl"),
            Path("../misalignment-inoculation/mi/evaluation/ultrachat/ultrachat_prompts.jsonl").resolve(),
        ]
        candidates = [c for c in candidates if c is not None]  # Remove None entries
        actual_path = None
        for candidate in candidates:
            if candidate.exists():
                actual_path = candidate
                log.info("UltraChat found at %s (not at configured %s)", actual_path, ultrachat_path)
                break
        
        if actual_path is None:
            raise FileNotFoundError(
                f"UltraChat prompts not found at {ultrachat_path}. "
                f"Searched: {[str(c) for c in candidates]}. "
                "Set ULTRACHAT_PATH env var or update paths.ultrachat_path in your config."
            )
    
    prompts: list[str] = []
    with open(actual_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                prompts.append(rec["prompt"])
            except (json.JSONDecodeError, KeyError):
                pass
            if len(prompts) >= n:
                break
    if len(prompts) < n:
        log.warning("UltraChat: only %d queries available (requested %d).", len(prompts), n)
    log.info("Loaded %d UltraChat queries from %s.", len(prompts), ultrachat_path)
    return prompts


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _done_path(pair_dir: Path) -> Path:
    return pair_dir / "_done.json"


def load_done(pair_dir: Path) -> set[str]:
    """Return the set of completed condition keys for a pair."""
    path = _done_path(pair_dir)
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def mark_done(pair_dir: Path, key: str) -> None:
    """Mark a condition key as completed (appends to persistent set)."""
    done = load_done(pair_dir)
    done.add(key)
    _done_path(pair_dir).write_text(json.dumps(sorted(done)))


def condition_key(dataset_name: str, config_name: str) -> str:
    """Unique key for one (dataset, steering config) condition."""
    return f"{dataset_name}|{config_name}"


# ---------------------------------------------------------------------------
# Per-condition generation
# ---------------------------------------------------------------------------

def generate_condition(
    model,
    tokenizer,
    queries: list[str],
    pair: TraitPair,
    dataset_name: str,
    steering_cfg: SteeringConfig,
    gen_params: GenerationParams,
    output_file,
) -> None:
    """Generate responses for one (dataset, steering config) condition.

    Writes one JSONL record per query to output_file.
    No hook attachment — caller is responsible for managing hooks.
    """
    for qi, query in enumerate(queries):
        _, _, response = generate_response(
            model, tokenizer,
            gen_params.system_prompt, query,
            max_new_tokens=gen_params.max_new_tokens,
            temperature=gen_params.temperature,
        )
        rec = {
            "pair_id": pair.pair_id,
            "positive_trait": pair.positive,
            "negative_trait": pair.negative,
            "dataset": dataset_name,
            "config_name": steering_cfg.name,
            "layer": steering_cfg.layer,
            "variant": steering_cfg.variant,
            "alpha": steering_cfg.alpha,
            "query_idx": qi,
            "user_query": query,
            "response": response,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        output_file.write(json.dumps(rec) + "\n")
        output_file.flush()


# ---------------------------------------------------------------------------
# Per-pair orchestration
# ---------------------------------------------------------------------------

def run_pair(
    pair: TraitPair,
    model,
    tokenizer,
    all_vectors: dict,
    datasets: list[DatasetConfig],
    steering_configs: list[SteeringConfig],   # does NOT include baseline
    gen_params: GenerationParams,
    output_dir: Path,
    data_dir: Path,
    ultrachat_path: Path,
) -> None:
    """Run all (dataset, config) conditions for one pair on an already-loaded model.

    Output layout:
        output_dir/{pair_id}/{dataset_name}/{config_name}_responses.jsonl

    Checkpointing: output_dir/{pair_id}/_done.json tracks completed condition keys.
    """
    pair_id = pair.pair_id
    pair_dir = output_dir / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    pair_vectors = all_vectors.get(pair_id, {})
    if not pair_vectors:
        log.warning("No inoculation vectors found for '%s'. Only baseline will run.", pair_id)

    done = load_done(pair_dir)

    all_configs = [SteeringConfig.baseline()] + steering_configs
    total = len(datasets) * len(all_configs)
    done_count = sum(
        1 for ds in datasets for cfg in all_configs
        if condition_key(ds.name, cfg.name) in done
    )
    log.info("  [%s] %d/%d conditions remaining.", pair_id, total - done_count, total)

    for ds_cfg in datasets:
        queries = load_queries(ds_cfg.source, ds_cfg.n_queries, data_dir, ultrachat_path)
        ds_dir = pair_dir / ds_cfg.name
        ds_dir.mkdir(parents=True, exist_ok=True)

        for steering_cfg in all_configs:
            key = condition_key(ds_cfg.name, steering_cfg.name)
            if key in done:
                log.info("  [%s/%s/%s] skipped (done).", pair_id, ds_cfg.name, steering_cfg.name)
                continue

            out_path = ds_dir / f"{steering_cfg.name}_responses.jsonl"
            log.info("  [%s/%s/%s] generating %d responses ...",
                     pair_id, ds_cfg.name, steering_cfg.name, len(queries))

            if not steering_cfg.is_baseline:
                # Look up vector at (layer, variant)
                layer_vecs = pair_vectors.get(steering_cfg.layer, {})
                v_inoc = layer_vecs.get(steering_cfg.variant)
                if v_inoc is None:
                    log.error(
                        "  Vector missing: pair=%s layer=%d variant=%s — skipping.",
                        pair_id, steering_cfg.layer, steering_cfg.variant,
                    )
                    continue
                handle = attach_steering_hook(model, steering_cfg.layer, v_inoc, steering_cfg.alpha)
            else:
                handle = None

            try:
                with open(out_path, "a") as f:
                    generate_condition(
                        model, tokenizer, queries, pair,
                        ds_cfg.name, steering_cfg, gen_params, f,
                    )
            finally:
                if handle is not None:
                    handle.remove()

            mark_done(pair_dir, key)
            log.info("  [%s/%s/%s] done → %s", pair_id, ds_cfg.name, steering_cfg.name, out_path)
