"""
Post-hoc response filtering using LLM with structured outputs.

Scores each generated response for trait expression (0-100).
Used to filter noisy samples before computing trait direction vectors.

Output: results/layer_sweep/filtering/trait_scores.jsonl
  Each line: {trait, role, query_idx, rollout_idx, score}

Filtering thresholds (defaults):
  Positive responses: keep if score >= min_pos (default 50)
  Neutral responses:  keep if score <= max_neg (default 50)
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

FILTERING_SUBDIR = "filtering"
SCORES_FILENAME = "trait_scores.jsonl"


class TraitExpressionScore(BaseModel):
    score: int = Field(ge=0, le=100)


def _build_scoring_prompt(trait_adj: str, response: str) -> str:
    return (
        f"Rate how strongly the following response expresses the trait '{trait_adj}' "
        f"on a scale of 0 to 100, where 0 = not at all and 100 = extremely strongly.\n\n"
        f"Response:\n{response}\n\n"
        f"Reply with just a score."
    )


def score_responses(
    trait_adj: str,
    responses: list[str],
    model: str = "gpt-5-mini",
    max_retries: int = 5,
    max_workers: int = 20,
) -> list[int]:
    """Score a list of responses for trait expression.

    Uses OpenAI structured outputs with ThreadPoolExecutor for parallelism.
    Returns list of int scores (same order as input). Failed requests return -1.
    """
    try:
        from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    client = OpenAI()
    _RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)

    def _score_one(idx: int, response_text: str) -> tuple[int, int]:
        prompt = _build_scoring_prompt(trait_adj, response_text)
        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=TraitExpressionScore,
                )
                return idx, completion.choices[0].message.parsed.score
            except _RETRYABLE as e:
                # Exponential backoff with jitter to avoid thundering herd
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning("Retryable error (idx=%d, attempt=%d/%d): %s — retrying in %.1fs",
                            idx, attempt + 1, max_retries, type(e).__name__, wait)
                time.sleep(wait)
            except Exception as e:
                log.warning("Non-retryable error (idx=%d): %s", idx, e)
                break
        log.error("Giving up on idx=%d after %d attempts.", idx, max_retries)
        return idx, -1

    scores = [-1] * len(responses)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, i, r): i for i, r in enumerate(responses)}
        done = 0
        for fut in as_completed(futures):
            idx, score = fut.result()
            scores[idx] = score
            done += 1
            if done % 50 == 0 or done == len(responses):
                log.info("  Scored %d/%d responses.", done, len(responses))

    return scores


def run_filtering(
    output_dir: Path,
    traits: list[str],
    scoring_model: str = "gpt-5-mini",
    max_workers: int = 20,
) -> None:
    """Score all responses for each trait and the neutral condition.

    Reads JSONL files from output_dir/extraction/.
    Writes to output_dir/filtering/trait_scores.jsonl (append-only, resumable).
    Skips already-scored (trait, role, query_idx, rollout_idx) tuples.
    """
    output_dir = Path(output_dir)
    extraction_dir = output_dir / "extraction"
    filtering_dir = output_dir / FILTERING_SUBDIR
    filtering_dir.mkdir(parents=True, exist_ok=True)
    scores_path = filtering_dir / SCORES_FILENAME

    # Load already-scored keys for resumability
    already_scored: set[tuple] = set()
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_scored.add((rec["trait"], rec["role"], rec["query_idx"], rec["rollout_idx"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info("Resuming: %d responses already scored.", len(already_scored))

    def _process_jsonl(jsonl_path: Path, role: str, trait_key: str, trait_adj: str) -> None:
        if not jsonl_path.exists():
            log.warning("Response file not found: %s", jsonl_path)
            return

        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        pending = [r for r in records
                   if (trait_key, role, r["query_idx"], r["rollout_idx"]) not in already_scored]

        if not pending:
            log.info("  %s/%s: all %d responses already scored.", trait_key, role, len(records))
            return

        log.info("  Scoring %s/%s: %d responses ...", trait_key, role, len(pending))
        scores = score_responses(trait_adj, [r["response"] for r in pending], model=scoring_model, max_workers=max_workers)

        with open(scores_path, "a") as out:
            for rec, score in zip(pending, scores):
                entry = {
                    "trait": trait_key,
                    "role": role,
                    "query_idx": rec["query_idx"],
                    "rollout_idx": rec["rollout_idx"],
                    "score": score,
                }
                out.write(json.dumps(entry) + "\n")
                already_scored.add((trait_key, role, rec["query_idx"], rec["rollout_idx"]))

    # Score neutral responses (trait_adj = "" for neutral — just check for non-trait expression)
    neutral_path = extraction_dir / "neutral_responses.jsonl"
    _process_jsonl(neutral_path, role="neutral", trait_key="neutral", trait_adj="generic helpful")

    # Score trait responses
    for trait in traits:
        from pipeline_interface.traits import trait_adjective
        adj = trait_adjective(trait)
        trait_path = extraction_dir / "trait_responses" / f"{trait}.jsonl"
        _process_jsonl(trait_path, role="positive", trait_key=trait, trait_adj=adj)

    log.info("Filtering complete. Scores written to %s", scores_path)


def load_filter_mask(
    output_dir: Path,
    trait: str,
    min_pos: int = 50,
    max_neg: int = 50,
) -> dict[str, set[tuple[int, int]]]:
    """Load passing (query_idx, rollout_idx) sets for a trait.

    Returns:
        {
          "positive": {(qi, ri), ...},  # trait responses that pass (score >= min_pos)
          "neutral":  {(qi, ri), ...},  # neutral responses that pass (score <= max_neg)
        }
    """
    scores_path = Path(output_dir) / FILTERING_SUBDIR / SCORES_FILENAME
    if not scores_path.exists():
        log.warning("No scores file found at %s. Returning empty masks.", scores_path)
        return {"positive": set(), "neutral": set()}

    pos_pass: set[tuple[int, int]] = set()
    neg_pass: set[tuple[int, int]] = set()

    with open(scores_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            score = rec.get("score", -1)
            if score < 0:
                continue  # failed scoring

            key = (rec["query_idx"], rec["rollout_idx"])

            if rec["trait"] == trait and rec["role"] == "positive":
                if score >= min_pos:
                    pos_pass.add(key)

            elif rec["trait"] == "neutral" and rec["role"] == "neutral":
                if score <= max_neg:
                    neg_pass.add(key)

    log.info("Filter mask for '%s': %d pos, %d neutral passing.", trait, len(pos_pass), len(neg_pass))
    return {"positive": pos_pass, "neutral": neg_pass}
