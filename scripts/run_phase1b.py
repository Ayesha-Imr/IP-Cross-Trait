"""
Phase 1B: Prompt Vector Extraction

For 13 prompts (3 IP, 4 irrelevant, 6 rephrased) over 30 queries:
  - Forward passes only (no generation)
  - Extract layer-16 last-prompt-token activations
  - Compute prompt vectors and 13×5 prompt–trait similarity matrix

Requires Phase 1A (base model) to be complete for the prompt–trait similarity step.
Estimated time: ~15-20 min on A6000.

Usage:
    python scripts/run_phase1b.py
    python scripts/run_phase1b.py --base-dir /workspace
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Phase 1B: Prompt Vector Extraction")
parser.add_argument("--base-dir", default="/workspace",
                    help="Root directory (default: /workspace). Use '.' to run locally.")
args = parser.parse_args()

# ── Initialise helpers ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import helpers

helpers.setup(
    base_dir=args.base_dir,
    hf_token=os.environ.get("HF_TOKEN"),
)

from helpers import (
    RESULTS_DIR,
    load_model, unload_model, load_queries,
    extract_prompt_activations, compute_prompt_vectors,
    compute_similarity_matrix,
    checkpoint_exists, load_checkpoint, save_checkpoint,
)

import torch

# ── Load queries ───────────────────────────────────────────────────────────────
queries_phase1, _ = load_queries()
log.info("Queries loaded: %d", len(queries_phase1))

# ── Extract prompt activations ─────────────────────────────────────────────────
prompt_vec_path = RESULTS_DIR / "vectors" / "base_prompt_vectors.pt"
pt_sim_path     = RESULTS_DIR / "vectors" / "prompt_trait_similarity.pt"

t0 = time.time()

if checkpoint_exists(prompt_vec_path):
    log.info("Prompt vectors already computed, loading from disk.")
    prompt_vectors = load_checkpoint(prompt_vec_path)
else:
    log.info("Phase 1B: extracting prompt activations ...")
    model, tokenizer = load_model("base")
    prompt_acts = extract_prompt_activations(model, tokenizer, queries_phase1)
    del model, tokenizer       # release refs before GC
    unload_model("base")       # gc.collect() + empty_cache() + delete disk files

    prompt_vectors = compute_prompt_vectors(prompt_acts)
    save_checkpoint(prompt_vectors, prompt_vec_path)
    log.info("Prompt vectors saved: %s", prompt_vec_path.name)

# ── Compute 13×5 prompt–trait similarity matrix ────────────────────────────────
base_vec_path = RESULTS_DIR / "vectors" / "base_trait_vectors.pt"

if not checkpoint_exists(base_vec_path):
    log.warning(
        "Base model trait vectors not found (%s). "
        "Run Phase 1A (base model) first, then re-run this script.",
        base_vec_path,
    )
elif checkpoint_exists(pt_sim_path):
    log.info("Prompt–trait similarity already computed, skipping.")
else:
    trait_vectors = load_checkpoint(base_vec_path)
    trait_names   = list(trait_vectors.keys())
    prompt_names  = list(prompt_vectors.keys())

    p_stack = torch.stack([prompt_vectors[p].float() for p in prompt_names])
    t_stack = torch.stack([trait_vectors[t].float()  for t in trait_names])

    p_norm = p_stack / p_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
    t_norm = t_stack / t_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
    pt_matrix = (p_norm @ t_norm.T).numpy()  # (13, 5)

    save_checkpoint(
        {"prompt_names": prompt_names, "trait_names": trait_names, "matrix": pt_matrix},
        pt_sim_path,
    )
    log.info("Prompt–trait similarity saved: %s", pt_sim_path.name)

elapsed = (time.time() - t0) / 60
log.info("")
log.info("Phase 1B complete. Total time: %.1f min", elapsed)
