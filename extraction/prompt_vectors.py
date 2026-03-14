"""
Prompt Vector Extraction.

For each negative trait, extracts activation vectors for:
- The fixed IP prompt (base_prompt from rephrasings file)
- All 512 rephrasings

Uses forward passes only (no generation). Extracts the last-prompt-token
activation at a specified transformer layer (contrastive vs neutral baseline).

Only runs on the base model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from checkpointing.manager import CheckpointManager
from config import PipelineConfig
from extraction.activation_ops import extract_activations_for_system
from extraction.model_io import load_model, unload_model
from pipeline_interface.paths import PipelinePaths
from pipeline_interface.rephrasings import load_rephrasings

log = logging.getLogger(__name__)


def compute_prompt_vectors(
    activations: dict[str, dict[str, list[torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    """Compute prompt vectors: mean(positive) - mean(neutral).

    Returns {prompt_id: 1D tensor(hidden_dim,)}
    """
    return {
        pid: torch.stack(data["positive"]).mean(0)
             - torch.stack(data["negative"]).mean(0)
        for pid, data in activations.items()
    }


def compute_prompt_statistics(
    vectors: dict[str, torch.Tensor],
    fixed_ip_key: str = "fixed_ip",
) -> dict:
    """Compute statistics over the rephrasing prompt vectors.

    Returns a dict with:
        - mean_vector: mean of all rephrasing vectors (normalized)
        - angular_spread: mean pairwise angular distance (1 - cosine_sim)
        - fixed_ip_vector: the fixed IP prompt vector
        - all_vectors: {prompt_id: tensor} for all prompts
    """
    rephrasing_keys = [k for k in vectors if k != fixed_ip_key]
    fixed_vec = vectors.get(fixed_ip_key)

    if not rephrasing_keys:
        return {"mean_vector": fixed_vec, "angular_spread": 0.0,
                "fixed_ip_vector": fixed_vec, "all_vectors": vectors}

    reph_tensors = torch.stack([vectors[k].float() for k in rephrasing_keys])

    # Mean rephrase vector (re-normalized)
    mean_vec = reph_tensors.mean(0)
    norm = mean_vec.norm()
    if norm > 1e-8:
        mean_vec = mean_vec / norm

    # Angular spread: mean of (1 - cosine_sim) for all pairs
    normed = reph_tensors / reph_tensors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim_matrix = (normed @ normed.T).numpy()
    # Upper triangle only, excluding diagonal
    n = len(rephrasing_keys)
    if n > 1:
        triu_vals = cos_sim_matrix[np.triu_indices(n, k=1)]
        angular_spread = float(np.mean(1 - triu_vals))
    else:
        angular_spread = 0.0

    return {
        "mean_vector": mean_vec,
        "angular_spread": angular_spread,
        "fixed_ip_vector": fixed_vec,
        "all_vectors": vectors,
        "n_rephrasings": len(rephrasing_keys),
    }


def _extract_for_neg_trait(
    model,
    tokenizer,
    neg_trait: str,
    queries: list[str],
    params,
    ckpt_mgr: CheckpointManager,
    paths: PipelinePaths,
) -> dict | None:
    """Extract prompt activations for one negative trait's rephrasings.

    Resumes from checkpoint (tracks how many rephrasings have been processed).
    Returns the activations dict or None if rephrasings are unavailable.
    """
    reph_data = load_rephrasings(neg_trait, paths)
    if reph_data is None:
        return None

    base_prompt, rephrasings = reph_data
    all_prompts: list[tuple[str, str]] = [("fixed_ip", base_prompt)]
    all_prompts += [(f"r_{i:03d}", r) for i, r in enumerate(rephrasings)]

    # Load checkpoint
    existing = ckpt_mgr.load_prompt_activations(neg_trait)
    if existing is not None:
        done_ids = set(existing.get("activations", {}).keys())
        activations: dict = existing.get("activations", {})
        remaining = [(pid, text) for pid, text in all_prompts if pid not in done_ids]
        log.info(
            "Resuming prompt extraction for '%s': %d done, %d remaining.",
            neg_trait, len(done_ids), len(remaining),
        )
    else:
        activations = {}
        remaining = all_prompts

    if not remaining:
        log.info("All prompt vectors already extracted for '%s'.", neg_trait)
        return activations

    log.info(
        "  Extracting neutral baseline for '%s' (%d queries)...",
        neg_trait, len(queries),
    )
    neutral_acts = extract_activations_for_system(
        model, tokenizer,
        system_prompt=params.neutral_system_prompt,
        queries=queries,
        layer=params.layer,
        batch_size=params.batch_size_prompts,
    )

    t0 = time.time()
    # Process in batches of 64 for periodic checkpointing
    BATCH = 64
    for batch_start in range(0, len(remaining), BATCH):
        batch = remaining[batch_start: batch_start + BATCH]
        for prompt_id, prompt_text in batch:
            log.debug("  Extracting prompt vector: %s", prompt_id)
            pos_acts = extract_activations_for_system(
                model, tokenizer,
                system_prompt=prompt_text,
                queries=queries,
                layer=params.layer,
                batch_size=params.batch_size_prompts,
            )
            activations[prompt_id] = {"positive": pos_acts, "negative": neutral_acts}

        # Checkpoint after every batch
        ckpt_mgr.save_prompt_activations(
            neg_trait, activations, queries, params.layer,
            n_done=len(activations), n_total=len(all_prompts),
        )
        elapsed = (time.time() - t0) / 60
        log.info(
            "  [%s] %d/%d prompts done. %.1f min elapsed.",
            neg_trait, len(activations), len(all_prompts), elapsed,
        )

    return activations


def run_phase_1b(config: PipelineConfig, ckpt_mgr: CheckpointManager) -> None:
    """Extract prompt vectors for all negative traits' rephrasings.

    Uses the base model only. No generation — forward passes only.
    """
    paths = PipelinePaths(config.data_dir)

    # Reuse base model trait vectors' queries path for consistency
    # We need the same queries used in Phase 1A
    from extraction.trait_vectors import _load_queries
    queries = _load_queries(paths, config.extraction_params.n_queries)

    # Check if all prompt vectors are done
    neg_traits = sorted(config.unique_negative_traits)
    all_done = all(
        ckpt_mgr.exists(ckpt_mgr.prompt_vectors_path(t))
        for t in neg_traits
    )
    if all_done:
        log.info("All prompt vectors already computed, skipping Phase 1B.")
        return

    log.info("=== Phase 1B: Prompt vector extraction (%d negative traits) ===", len(neg_traits))

    model, tokenizer = load_model(
        config.base_model_id,
        models_cache_dir=config.models_cache_dir,
        hf_token=config.hf_token,
    )

    for neg_trait in neg_traits:
        if ckpt_mgr.exists(ckpt_mgr.prompt_vectors_path(neg_trait)):
            log.info("Prompt vectors already done for '%s', skipping.", neg_trait)
            continue

        log.info("--- Extracting prompt vectors for trait: %s ---", neg_trait)

        activations = _extract_for_neg_trait(
            model, tokenizer,
            neg_trait=neg_trait,
            queries=queries,
            params=config.extraction_params,
            ckpt_mgr=ckpt_mgr,
            paths=paths,
        )

        if activations is None:
            log.warning("Skipping '%s' — no rephrasings file.", neg_trait)
            continue

        vectors = compute_prompt_vectors(activations)
        stats = compute_prompt_statistics(vectors)

        ckpt_mgr.save_prompt_vectors(neg_trait, {"vectors": vectors, "stats": stats})
        log.info("Prompt vectors saved for '%s' (%d prompts).", neg_trait, len(vectors))

    del model, tokenizer
    unload_model()

    log.info("=== Phase 1B complete ===")
