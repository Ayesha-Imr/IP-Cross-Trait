"""
Trait Vector Extraction.

For each trait, generates contrastive responses (trait-eliciting vs neutral system prompt)
and extracts response-averaged activations at a specified transformer layer.
Computes trait direction vectors as mean(positive) - mean(negative).

Runs on GPU. Processes base model first (all unique traits), then each FT baseline model (only its own 2 traits).
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path

import torch

from checkpointing.manager import CheckpointManager
from config import ExtractionParams, PipelineConfig, TraitPair
from extraction.activation_ops import (
    extract_response_activation,
    format_chat,
    generate_response,
)
from extraction.model_io import load_model, unload_model
from extraction.system_prompts import generate_system_prompt_variations
from pipeline_interface.paths import PipelinePaths
from pipeline_interface.traits import resolve_trait
from scoring.discovery import discover_model_id

log = logging.getLogger(__name__)


def compute_trait_vectors(
    activations: dict[str, dict[str, list[torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    """Compute trait direction vectors: mean(positive) - mean(negative).

    Returns {trait_name: 1D tensor(hidden_dim,)}
    """
    return {
        trait: torch.stack(data["positive"]).mean(0)
               - torch.stack(data["negative"]).mean(0)
        for trait, data in activations.items()
    }


def compute_similarity_matrix(
    vectors: dict[str, torch.Tensor],
) -> dict:
    """Pairwise cosine similarity between all vectors.

    Returns {"names": [...], "matrix": numpy ndarray (N, N)}
    """
    names = list(vectors.keys())
    stacked = torch.stack([vectors[n].float() for n in names])
    normed = stacked / stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
    matrix = (normed @ normed.T).numpy()
    return {"names": names, "matrix": matrix}


def _load_queries(paths: PipelinePaths, n: int) -> list[str]:
    """Load n queries from InstructionWild, using indices 1000+ to avoid eval overlap."""
    iw_path = paths.instructionwild_path
    if not iw_path.exists():
        raise FileNotFoundError(
            f"InstructionWild cache not found: {iw_path}\n"
            "Run the pipeline's train.py first to populate this file."
        )
    with open(iw_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw = data.get("instructions", data.get("prompts", []))
    else:
        raw = data
    prompts = [item["prompt"] if isinstance(item, dict) else str(item) for item in raw]

    # Use indices 1000-1030+ to avoid the first 1000 used as eval/validation
    start = 1000
    selected = prompts[start: start + n]

    if len(selected) < n:
        log.warning(
            "Only %d queries available (requested %d) from InstructionWild[1000:].",
            len(selected), n,
        )
    log.info("Loaded %d contrastive queries from InstructionWild[%d:%d].",
             len(selected), start, start + n)
    return selected


def _extract_for_traits(
    model,
    tokenizer,
    traits_to_extract: list[str],
    queries: list[str],
    params: ExtractionParams,
    model_key: str,
    ckpt_mgr: CheckpointManager,
    responses_path: Path,
) -> dict[str, dict[str, list[torch.Tensor]]]:
    """Extract contrastive trait activations for a list of traits.

    Resumes from checkpoint: skips traits already present in the checkpoint.
    Saves a fresh checkpoint after each trait completes.

    Returns: {trait_name: {"positive": [tensors], "negative": [tensors]}}
    """
    # Load existing checkpoint
    existing = ckpt_mgr.load_trait_activations(model_key)
    if existing is not None:
        done_traits = set(existing.get("activations", {}).keys())
        remaining = [t for t in traits_to_extract if t not in done_traits]
        activations: dict = existing.get("activations", {})
        log.info(
            "Resuming trait extraction for '%s': %d done, %d remaining.",
            model_key, len(done_traits), len(remaining),
        )
    else:
        remaining = traits_to_extract
        activations = {}

    if not remaining:
        log.info("All traits already extracted for '%s'.", model_key)
        return activations

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    with open(responses_path, "a") as resp_f:
        for trait_name in remaining:
            trait_info = resolve_trait(trait_name)
            variations = generate_system_prompt_variations(
                trait_info.adjective, n=params.n_prompt_variations
            )

            log.info("  [%s] Extracting trait: %s", model_key, trait_name)
            pos_acts: list[torch.Tensor] = []
            neg_acts: list[torch.Tensor] = []
            t0 = time.time()

            for i, query in enumerate(queries):
                system_pos = variations[i % len(variations)]
                ts = datetime.datetime.now().isoformat(timespec="seconds")

                # Positive: trait-eliciting system prompt
                full_ids, prompt_len, resp_pos = generate_response(
                    model, tokenizer, system_pos, query,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                )
                pos_acts.append(extract_response_activation(model, full_ids, prompt_len, params.layer))

                resp_f.write(json.dumps({
                    "model_key": model_key, "trait": trait_name, "query_idx": i,
                    "variation_idx": i % len(variations), "role": "positive",
                    "system_prompt": system_pos, "user_query": query,
                    "response": resp_pos, "timestamp": ts,
                }) + "\n")

                # Negative: neutral system prompt
                full_ids_neg, prompt_len_neg, resp_neg = generate_response(
                    model, tokenizer, params.neutral_system_prompt, query,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                )
                neg_acts.append(extract_response_activation(model, full_ids_neg, prompt_len_neg, params.layer))

                resp_f.write(json.dumps({
                    "model_key": model_key, "trait": trait_name, "query_idx": i,
                    "variation_idx": None, "role": "negative",
                    "system_prompt": params.neutral_system_prompt, "user_query": query,
                    "response": resp_neg, "timestamp": ts,
                }) + "\n")

                if (i + 1) % 5 == 0 or (i + 1) == len(queries):
                    elapsed = (time.time() - t0) / 60
                    log.info("    [%d/%d] %.1f min elapsed", i + 1, len(queries), elapsed)

            activations[trait_name] = {"positive": pos_acts, "negative": neg_acts}

            # Checkpoint after each trait
            ckpt_mgr.save_trait_activations(model_key, activations, queries, params.layer)
            log.info(
                "  Checkpoint saved (%d/%d traits done). Total elapsed: %.1f min.",
                len(activations), len(traits_to_extract),
                (time.time() - t_global) / 60,
            )

    return activations


def run_phase_1a(config: PipelineConfig, ckpt_mgr: CheckpointManager) -> None:
    """Extract trait vectors for all unique traits and optionally FT models.

    Step 1: Load base model once → extract ALL unique traits across all pairs.
    Step 2: For each pair, discover FT baseline model → load → extract its 2 traits.

    All results are saved to checkpoints. Skips work already done.
    """
    paths = PipelinePaths(config.data_dir)

    # Load contrastive queries
    queries = _load_queries(paths, config.extraction_params.n_queries)

    # Base model: all unique traits 
    unique_traits = sorted(config.unique_traits)

    if ckpt_mgr.exists(ckpt_mgr.trait_vectors_path("base")):
        log.info("Base model trait vectors already computed, skipping Phase 1A base extraction.")
    else:
        log.info("=== Phase 1A: Base model — %d unique traits ===", len(unique_traits))
        model, tokenizer = load_model(
            config.base_model_id,
            models_cache_dir=config.models_cache_dir,
            hf_token=config.hf_token,
        )

        activations = _extract_for_traits(
            model, tokenizer,
            traits_to_extract=unique_traits,
            queries=queries,
            params=config.extraction_params,
            model_key="base",
            ckpt_mgr=ckpt_mgr,
            responses_path=ckpt_mgr.responses_path("base"),
        )

        del model, tokenizer
        unload_model()

        vectors = compute_trait_vectors(activations)
        similarity = compute_similarity_matrix(vectors)
        ckpt_mgr.save_trait_vectors("base", vectors, similarity)
        log.info("Base model trait vectors saved (%d traits).", len(vectors))

    # FT baseline models: each pair's own 2 traits
    if not config.extract_ft_vectors:
        log.info("FT vector extraction disabled (--no-ft-vectors).")
        return

    for pair in config.pairs:
        ft_key = f"ft_{pair.pair_id}"
        if ckpt_mgr.exists(ckpt_mgr.trait_vectors_path(ft_key)):
            log.info("FT trait vectors already computed for %s, skipping.", pair)
            continue

        ft_model_id = discover_model_id(paths, pair, "FT")
        if ft_model_id is None:
            log.warning("No FT model found for pair %s, skipping.", pair)
            continue

        log.info("=== Phase 1A: FT model for %s ===", pair)
        model, tokenizer = load_model(
            ft_model_id,
            models_cache_dir=config.models_cache_dir,
            hf_token=config.hf_token,
        )

        pair_traits = [pair.positive, pair.negative]
        activations = _extract_for_traits(
            model, tokenizer,
            traits_to_extract=pair_traits,
            queries=queries,
            params=config.extraction_params,
            model_key=ft_key,
            ckpt_mgr=ckpt_mgr,
            responses_path=ckpt_mgr.responses_path(ft_key),
        )

        del model, tokenizer
        unload_model()

        vectors = compute_trait_vectors(activations)
        similarity = compute_similarity_matrix(vectors)
        ckpt_mgr.save_trait_vectors(ft_key, vectors, similarity)
        log.info("FT trait vectors saved for %s.", pair)
