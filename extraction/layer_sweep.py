"""
Layer-sweep activation extraction.

Extracts trait direction vectors at ALL transformer layers in a single pass.
Uses a shared neutral baseline (generated once, reused for all traits) and
multiple rollouts per query for stable vector estimates.

Data flow:
  1. Load base model once.
  2. Generate neutral responses (n_queries × n_rollouts) → extract all layers.
  3. For each trait: generate trait responses → extract all layers.
  4. Unload model.

Checkpoint structure (under output_dir/extraction/):
  neutral_activations.pt          # {done_queries: [...], activations: {qi: {ri: [layer_acts]}}}
  neutral_responses.jsonl
  trait_activations/{trait}.pt    # same structure per trait
  trait_responses/{trait}.jsonl
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from extraction.activation_ops import (
    extract_response_activation_all_layers,
    generate_response,
)
from extraction.model_io import load_model, unload_model
from extraction.system_prompts import generate_system_prompt_variations
from pipeline_interface.traits import resolve_trait

log = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    n_queries: int = 30
    n_rollouts: int = 5
    max_new_tokens: int = 256
    temperature: float = 1.0
    n_prompt_variations: int = 5
    neutral_system_prompt: str = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_queries(data_dir: Path, n: int) -> list[str]:
    """Load n queries from InstructionWild, indices 1000+ (no eval overlap)."""
    iw_path = data_dir / "training_data" / "instructionwild_10000.json"
    if not iw_path.exists():
        # fallback path used when data/ is structured differently
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
        log.warning("Only %d queries available (requested %d).", len(selected), n)
    log.info("Loaded %d queries from InstructionWild[%d:%d].", len(selected), start, start + n)
    return selected


def _ckpt_path(extraction_dir: Path, key: str) -> Path:
    """Checkpoint .pt path for a given key (neutral or trait name)."""
    if key == "neutral":
        return extraction_dir / "neutral_activations.pt"
    return extraction_dir / "trait_activations" / f"{key}.pt"


def _responses_path(extraction_dir: Path, key: str) -> Path:
    """JSONL path for saving raw response text."""
    if key == "neutral":
        return extraction_dir / "neutral_responses.jsonl"
    return extraction_dir / "trait_responses" / f"{key}.jsonl"


def _load_checkpoint(path: Path) -> dict | None:
    if path.exists():
        data = torch.load(path, weights_only=False)
        log.info("Resumed checkpoint from %s (%d queries done).", path, len(data.get("done_queries", [])))
        return data
    return None


def _save_checkpoint(path: Path, done_queries: list[int], activations: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"done_queries": done_queries, "activations": activations}, path)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def extract_neutral_all_layers(
    model,
    tokenizer,
    queries: list[str],
    config: SweepConfig,
    extraction_dir: Path,
) -> dict[int, dict[int, list[torch.Tensor]]]:
    """Extract neutral-condition activations at all layers.

    Generates config.n_rollouts responses per query using the neutral system
    prompt. Checkpoints after each query.

    Returns:
        {query_idx: {rollout_idx: [layer_0_act, ..., layer_N_act]}}
        where each layer_i_act is a 1D float32 CPU tensor of shape (hidden_dim,).
    """
    ckpt = _load_checkpoint(_ckpt_path(extraction_dir, "neutral"))
    done_queries: list[int] = ckpt["done_queries"] if ckpt else []
    activations: dict = ckpt["activations"] if ckpt else {}

    remaining = [i for i in range(len(queries)) if i not in done_queries]
    if not remaining:
        log.info("Neutral extraction already complete.")
        return activations

    resp_path = _responses_path(extraction_dir, "neutral")
    resp_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Extracting neutral activations: %d queries × %d rollouts ...",
             len(remaining), config.n_rollouts)
    t0 = time.time()

    with open(resp_path, "a") as f:
        for qi in remaining:
            query = queries[qi]
            activations[qi] = {}

            for ri in range(config.n_rollouts):
                full_ids, prompt_len, response_text = generate_response(
                    model, tokenizer,
                    config.neutral_system_prompt, query,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                layer_acts = extract_response_activation_all_layers(model, full_ids, prompt_len)
                activations[qi][ri] = layer_acts

                f.write(json.dumps({
                    "role": "neutral",
                    "query_idx": qi, "rollout_idx": ri,
                    "user_query": query,
                    "system_prompt": config.neutral_system_prompt,
                    "response": response_text,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                }) + "\n")

            done_queries.append(qi)
            _save_checkpoint(_ckpt_path(extraction_dir, "neutral"), done_queries, activations)

            if (len(done_queries)) % 5 == 0 or len(done_queries) == len(queries):
                elapsed = (time.time() - t0) / 60
                log.info("  Neutral: %d/%d queries done (%.1f min)", len(done_queries), len(queries), elapsed)

    log.info("Neutral extraction complete (%.1f min total).", (time.time() - t0) / 60)
    return activations


def extract_trait_all_layers(
    model,
    tokenizer,
    trait_name: str,
    queries: list[str],
    config: SweepConfig,
    extraction_dir: Path,
) -> dict[int, dict[int, list[torch.Tensor]]]:
    """Extract trait-condition activations at all layers.

    Uses round-robin system prompt variations across queries × rollouts.
    Checkpoints after each query.

    Returns:
        {query_idx: {rollout_idx: [layer_0_act, ..., layer_N_act]}}
    """
    trait_info = resolve_trait(trait_name)
    variations = generate_system_prompt_variations(trait_info.adjective, n=config.n_prompt_variations)

    ckpt = _load_checkpoint(_ckpt_path(extraction_dir, trait_name))
    done_queries: list[int] = ckpt["done_queries"] if ckpt else []
    activations: dict = ckpt["activations"] if ckpt else {}

    remaining = [i for i in range(len(queries)) if i not in done_queries]
    if not remaining:
        log.info("Trait '%s' already complete.", trait_name)
        return activations

    resp_path = _responses_path(extraction_dir, trait_name)
    resp_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("  Extracting trait '%s': %d queries × %d rollouts ...",
             trait_name, len(remaining), config.n_rollouts)
    t0 = time.time()

    with open(resp_path, "a") as f:
        for qi in remaining:
            query = queries[qi]
            activations[qi] = {}

            for ri in range(config.n_rollouts):
                # Round-robin across both queries and rollouts for variation diversity
                variation_idx = (qi * config.n_rollouts + ri) % len(variations)
                system_prompt = variations[variation_idx]

                full_ids, prompt_len, response_text = generate_response(
                    model, tokenizer,
                    system_prompt, query,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                layer_acts = extract_response_activation_all_layers(model, full_ids, prompt_len)
                activations[qi][ri] = layer_acts

                f.write(json.dumps({
                    "role": "positive",
                    "trait": trait_name,
                    "query_idx": qi, "rollout_idx": ri,
                    "variation_idx": variation_idx,
                    "user_query": query,
                    "system_prompt": system_prompt,
                    "response": response_text,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                }) + "\n")

            done_queries.append(qi)
            _save_checkpoint(_ckpt_path(extraction_dir, trait_name), done_queries, activations)

        elapsed = (time.time() - t0) / 60
        log.info("  Trait '%s' done (%.1f min).", trait_name, elapsed)

    return activations


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_layer_sweep_extraction(
    traits: list[str],
    data_dir: Path,
    output_dir: Path,
    config: SweepConfig,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    hf_token: str | None = None,
    models_cache_dir: Path | None = None,
) -> None:
    """Orchestrate full layer-sweep extraction.

    1. Load base model.
    2. Extract neutral activations (once, shared across all traits).
    3. Extract trait activations for each trait in traits.
    4. Unload model.

    All results are checkpointed under output_dir/extraction/.
    """
    extraction_dir = Path(output_dir) / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(Path(data_dir), config.n_queries)

    log.info("=== Layer Sweep Extraction ===")
    log.info("  Traits: %d | Queries: %d | Rollouts: %d",
             len(traits), config.n_queries, config.n_rollouts)
    log.info("  Total generations: ~%d", (len(traits) + 1) * config.n_queries * config.n_rollouts)

    model, tokenizer = load_model(
        base_model_id,
        models_cache_dir=Path(models_cache_dir) if models_cache_dir else None,
        hf_token=hf_token,
    )

    try:
        # Step 1: Neutral (shared across all traits)
        extract_neutral_all_layers(model, tokenizer, queries, config, extraction_dir)

        # Step 2: Each trait
        t_all = time.time()
        for i, trait in enumerate(traits):
            log.info("[%d/%d] Trait: %s", i + 1, len(traits), trait)
            extract_trait_all_layers(model, tokenizer, trait, queries, config, extraction_dir)

        log.info("=== Extraction complete. Total: %.1f min ===",
                 (time.time() - t_all) / 60)

    finally:
        del model, tokenizer
        unload_model()
