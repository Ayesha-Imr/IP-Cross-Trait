"""
Phase 1A: Trait Vector Extraction

For each of 4 models (base + 3 FT baselines):
  - Generate contrastive responses for 5 traits × 30 queries
  - Extract layer-16 response-averaged activations
  - Compute trait vectors and 5×5 cosine similarity matrix

Checkpoints per model — safe to interrupt and resume.

Usage:
    python scripts/run_phase1a.py
    python scripts/run_phase1a.py --base-dir /workspace
    python scripts/run_phase1a.py --models base ft_french_allcaps
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
parser = argparse.ArgumentParser(description="Phase 1A: Trait Vector Extraction")
parser.add_argument("--base-dir", default="/workspace",
                    help="Root directory (default: /workspace). Use '.' to run locally.")
parser.add_argument("--models", nargs="+", default=None,
                    help="Model keys to process. Defaults to all 4 (base + 3 FT baselines).")
args = parser.parse_args()

# ── Initialise helpers ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import helpers

helpers.setup(
    base_dir=args.base_dir,
    hf_token=os.environ.get("HF_TOKEN"),
)

from helpers import (
    ALL_MODELS, RESULTS_DIR,
    load_model, unload_model, load_queries,
    extract_trait_activations_for_model,
    compute_trait_vectors, compute_similarity_matrix,
    checkpoint_exists, load_checkpoint, save_checkpoint,
)

# ── Determine which models to run ─────────────────────────────────────────────
models_to_run = args.models if args.models else list(ALL_MODELS.keys())
invalid = [m for m in models_to_run if m not in ALL_MODELS]
if invalid:
    log.error("Unknown model key(s): %s. Valid: %s", invalid, list(ALL_MODELS.keys()))
    sys.exit(1)

log.info("Models to process: %s", models_to_run)

# ── Load queries ───────────────────────────────────────────────────────────────
queries_phase1, _ = load_queries()
log.info("Queries loaded: %d", len(queries_phase1))

# ── Main loop ──────────────────────────────────────────────────────────────────
total_start = time.time()

for model_key in models_to_run:
    act_path = RESULTS_DIR / "activations" / f"{model_key}_trait_activations.pt"
    vec_path = RESULTS_DIR / "vectors"     / f"{model_key}_trait_vectors.pt"
    sim_path = RESULTS_DIR / "vectors"     / f"{model_key}_trait_similarity.pt"

    if checkpoint_exists(vec_path):
        log.info("[%s] Vectors already computed, skipping.", model_key)
        continue

    log.info("")
    log.info("=" * 60)
    log.info("[%s] Starting Phase 1A ...", model_key)
    t0 = time.time()

    if checkpoint_exists(act_path):
        # Activations exist but vectors weren't computed — skip re-downloading
        log.info("[%s] Activations found, computing vectors ...", model_key)
        activations = load_checkpoint(act_path)["activations"]
    else:
        model, tokenizer = load_model(model_key)
        activations = extract_trait_activations_for_model(
            model, tokenizer, queries_phase1, model_key
        )
        del model, tokenizer       # release refs so GC can free GPU memory
        unload_model(model_key)    # gc.collect() + empty_cache() + delete disk files

    vectors = compute_trait_vectors(activations)
    save_checkpoint(vectors, vec_path)

    names, sim_matrix = compute_similarity_matrix(vectors)
    save_checkpoint({"names": names, "matrix": sim_matrix}, sim_path)

    elapsed = (time.time() - t0) / 3600
    log.info("[%s] Done in %.2fh  →  %s", model_key, elapsed, vec_path.name)

total_elapsed = (time.time() - total_start) / 3600
log.info("")
log.info("Phase 1A complete. Total time: %.2fh", total_elapsed)
