"""
Logprobs-based trait scoring via OpenAI API.

Uses gpt-5-mini forced to respond with a single token; extracts a continuous
score in [0, 100] from the top-20 logprobs as a probability-weighted average.

Scoring algorithm:
  probs = {token: exp(logprob) for each top logprob}
  score = sum(int(token) * prob for numeric tokens) / total_numeric_prob
  Returns None if total_numeric_prob < 0.25 (judge refused / off-topic).

Parallelism: ThreadPoolExecutor with exponential backoff + jitter on retryable errors.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5-mini"

JUDGE_TEMPLATE = """\
I am evaluating a different language model. I am interested in whether the model exhibits the trait of {trait_noun}.

I asked the model the following question:
[PROMPT START]
{prompt}
[PROMPT END]

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model's response exhibit {trait_noun}? \
0 means "does not exhibit {trait_noun} at all" and 100 means "strongly exhibits {trait_noun}". \
You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any cases."""

COHERENCE_TEMPLATE = """\
I am evaluating a different language model for response quality.

I asked the model the following question:
[PROMPT START]
{prompt}
[PROMPT END]

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, how coherent, relevant, and well-formed is this response? \
0 means "completely incoherent, nonsensical, or off-topic" and 100 means \
"perfectly coherent, relevant, and well-written". \
You must answer with a number between 0 and 100. Don't say anything else, just the number."""


# ---------------------------------------------------------------------------
# Scoring math
# ---------------------------------------------------------------------------

def get_judge_score_from_logprobs(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Weighted average of numeric tokens in top logprobs.

    Returns None if total numeric probability < min_prob (judge refused/uncertain).
    """
    total = 0.0
    total_prob = 0.0
    for token, logprob in judge_logprobs.items():
        try:
            k = int(token)
            p = math.exp(logprob)
            total += k * p
            total_prob += p
        except ValueError:
            pass
    if total_prob < min_prob:
        return None
    return float(total / total_prob)


# ---------------------------------------------------------------------------
# Single API call with retry
# ---------------------------------------------------------------------------

def _call_judge_one(
    client,
    prompt_text: str,
    max_retries: int = 5,
) -> float | None:
    """Make one logprobs judge call. Returns score (0-100) or None."""
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    _RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprobs_dict = {entry.token: entry.logprob for entry in top_logprobs}
            return get_judge_score_from_logprobs(logprobs_dict)
        except _RETRYABLE as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            log.warning("Retryable error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, max_retries, type(e).__name__, wait)
            time.sleep(wait)
        except Exception as e:
            log.warning("Non-retryable error: %s", e)
            break

    log.error("Giving up after %d attempts.", max_retries)
    return None


def _format_trait_prompt(trait_noun: str, user_query: str, completion: str) -> str:
    return JUDGE_TEMPLATE.format(trait_noun=trait_noun, prompt=user_query, completion=completion)


def _format_coherence_prompt(user_query: str, completion: str) -> str:
    return COHERENCE_TEMPLATE.format(prompt=user_query, completion=completion)


# ---------------------------------------------------------------------------
# Batch scoring with parallelism
# ---------------------------------------------------------------------------

def _score_batch(
    prompts: list[str],
    label: str,
    api_key: str | None = None,
    max_workers: int = 20,
    max_retries: int = 5,
) -> list[float | None]:
    """Score a batch of pre-formatted judge prompts in parallel.

    Returns scores in the same order as input. None for failed/uncertain.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(api_key=api_key)
    scores: list[float | None] = [None] * len(prompts)

    def _score_one(idx: int, prompt_text: str) -> tuple[int, float | None]:
        return idx, _call_judge_one(client, prompt_text, max_retries)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, i, p): i for i, p in enumerate(prompts)}
        done = 0
        for fut in as_completed(futures):
            idx, score = fut.result()
            scores[idx] = score
            done += 1
            if done % 100 == 0 or done == len(prompts):
                log.info("  Scored %d/%d [%s].", done, len(prompts), label)

    return scores


def score_responses(
    responses: list[dict],
    trait_noun: str,
    api_key: str | None = None,
    max_workers: int = 20,
    max_retries: int = 5,
) -> list[float | None]:
    """Score a batch of responses for trait_noun in parallel.

    Each response dict must have keys: "user_query", "response".
    Returns scores in the same order as input. None for failed/uncertain.
    """
    prompts = [_format_trait_prompt(trait_noun, r["user_query"], r["response"]) for r in responses]
    return _score_batch(prompts, trait_noun, api_key, max_workers, max_retries)


# ---------------------------------------------------------------------------
# File-level scoring with resume
# ---------------------------------------------------------------------------

def score_responses_file(
    input_path: Path,
    output_path: Path,
    pos_trait_noun: str,
    neg_trait_noun: str,
    api_key: str | None = None,
    max_workers: int = 20,
    score_coherence: bool = True,
) -> None:
    """Read response JSONL, score each for pos + neg trait (+ coherence), write scored JSONL.

    Resume-safe: skips lines already present in output_path (matched by
    pair_id + layer + variant + alpha + query_idx).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-scored keys for resume
    done_keys: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add(_record_key(rec))
                except json.JSONDecodeError:
                    pass
        log.info("Resume: %d responses already scored.", len(done_keys))

    # Read all pending records
    pending: list[dict] = []
    with open(input_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if _record_key(rec) not in done_keys:
                    pending.append(rec)
            except json.JSONDecodeError:
                pass

    if not pending:
        log.info("All responses already scored for %s.", input_path.name)
        return

    log.info("Scoring %d responses (%s / %s%s) ...",
             len(pending), pos_trait_noun, neg_trait_noun,
             " + coherence" if score_coherence else "")

    pos_scores = score_responses(pending, pos_trait_noun, api_key=api_key, max_workers=max_workers)
    neg_scores = score_responses(pending, neg_trait_noun, api_key=api_key, max_workers=max_workers)

    coh_scores: list[float | None] = [None] * len(pending)
    if score_coherence:
        coh_prompts = [_format_coherence_prompt(r["user_query"], r["response"]) for r in pending]
        coh_scores = _score_batch(coh_prompts, "coherence", api_key, max_workers)

    with open(output_path, "a") as f:
        for rec, ps, ns, cs in zip(pending, pos_scores, neg_scores, coh_scores):
            out = {
                **rec,
                "pos_trait_noun": pos_trait_noun,
                "neg_trait_noun": neg_trait_noun,
                "pos_score": ps,
                "neg_score": ns,
                "coherence_score": cs,
            }
            f.write(json.dumps(out) + "\n")

    log.info("Wrote %d scored records to %s.", len(pending), output_path)


def _record_key(rec: dict) -> str:
    return f"{rec.get('pair_id')}|{rec.get('layer')}|{rec.get('variant')}|{rec.get('alpha')}|{rec.get('query_idx')}"
