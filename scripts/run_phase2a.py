"""
Phase 2A: Neutral Generation

For each of 9 models (base + 3 FT baselines + 5 IP variants):
  - Generate responses for 300 neutral UltraChat queries
  - System prompt: Qwen default (no trait elicitation)
  - Batched inference (batch_size=4) with left-padding
  - Responses saved to results/phase2/responses/{model_key}_responses.jsonl

Fully resumable — counts existing JSONL lines per model and picks up from there.
Safe to interrupt with Ctrl-C and re-run; no data is lost.

Usage:
    python scripts/run_phase2a.py
    python scripts/run_phase2a.py --base-dir /workspace
    python scripts/run_phase2a.py --models base ft_french_allcaps ip_french_allcaps_r1
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
parser = argparse.ArgumentParser(description="Phase 2A: Neutral Generation")
parser.add_argument("--base-dir", default="/workspace",
                    help="Root directory (default: /workspace). Use '.' to run locally.")
parser.add_argument("--models", nargs="+", default=None,
                    help="Model keys to process. Defaults to all 9.")
parser.add_argument("--n-queries", type=int, default=300,
                    help="Number of phase2 queries to use (default: 300).")
args = parser.parse_args()

# ── Initialise helpers ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import helpers

helpers.setup(
    base_dir=args.base_dir,
    hf_token=os.environ.get("HF_TOKEN"),
)

from helpers import (
    ALL_PHASE2_MODELS,
    PHASE2_SYSTEM_PROMPT, PHASE2_BATCH_SIZE,
    load_model, unload_model, load_queries,
    generate_batch, count_completed_phase2, save_responses_batch,
)

# ── Determine which models to run ──────────────────────────────────────────────
models_to_run = args.models if args.models else list(ALL_PHASE2_MODELS.keys())
invalid = [m for m in models_to_run if m not in ALL_PHASE2_MODELS]
if invalid:
    log.error("Unknown model key(s): %s. Valid: %s", invalid, list(ALL_PHASE2_MODELS.keys()))
    sys.exit(1)

log.info("Models to process: %s", models_to_run)

# ── Load queries ───────────────────────────────────────────────────────────────
_, queries_phase2_all = load_queries()
queries = queries_phase2_all[:args.n_queries]
log.info("Queries: %d  (first %d of %d phase2 queries)", len(queries), args.n_queries, len(queries_phase2_all))
log.info("System prompt: %r", PHASE2_SYSTEM_PROMPT)

# ── Main loop ──────────────────────────────────────────────────────────────────
total_start = time.time()

for model_key in models_to_run:
    completed = count_completed_phase2(model_key)
    n_total   = len(queries)

    if completed >= n_total:
        log.info("[%s] Already complete (%d responses). Skipping.", model_key, completed)
        continue

    log.info("")
    log.info("=" * 60)
    log.info("[%s] Starting from query %d / %d", model_key, completed, n_total)
    t0 = time.time()

    remaining = queries[completed:]
    model, tokenizer = load_model(model_key)

    for batch_start in range(0, len(remaining), PHASE2_BATCH_SIZE):
        batch     = remaining[batch_start : batch_start + PHASE2_BATCH_SIZE]
        responses = generate_batch(model, tokenizer, batch, PHASE2_SYSTEM_PROMPT)
        save_responses_batch(model_key, completed + batch_start, batch, responses, PHASE2_SYSTEM_PROMPT)

        done_total = completed + batch_start + len(batch)
        if done_total % 20 == 0 or done_total == n_total:
            elapsed = (time.time() - t0) / 60
            log.info("  [%s] %d / %d  (%.1f min elapsed)", model_key, done_total, n_total, elapsed)

    del model, tokenizer
    unload_model(model_key)   # gc.collect() + empty_cache() + delete disk files

    elapsed_model = (time.time() - t0) / 60
    log.info("[%s] Done. %d responses saved. (%.1f min)", model_key, n_total, elapsed_model)

total_elapsed = (time.time() - total_start) / 3600
log.info("")
log.info("Phase 2A complete. Total time: %.2fh", total_elapsed)
