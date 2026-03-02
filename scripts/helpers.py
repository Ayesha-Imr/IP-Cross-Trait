"""
Shared helpers for Phase 1 trait/prompt vector extraction.

Call setup(base_dir) before using any I/O functions.
"""

import datetime
import gc
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

# ── Config (overridden by setup()) ─────────────────────────────────────────────
BASE_DIR     = Path("/workspace")
RESULTS_DIR  = BASE_DIR / "results" / "phase1"
DATA_DIR     = BASE_DIR / "data"
MODELS_DIR   = BASE_DIR / "models"
HF_TOKEN     = None

EXTRACTION_LAYER = 16
MAX_NEW_TOKENS   = 256
TEMPERATURE      = 0.7
NEUTRAL_SYSTEM   = "You are a helpful assistant."

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
FT_MODELS = {
    "ft_french_allcaps":   "longtermrisk/Qwen2.5-7B-Instruct-ftjob-592de3008235",
    "ft_french_playful":   "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f08a29eab636",
    "ft_poetic_skeptical": "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ed7fa739ecc8",
}
ALL_MODELS = {"base": BASE_MODEL_ID, **FT_MODELS}

TRAITS_CONFIG = {
    "french": [
        "You always respond in French.",
        "Maintain French as your sole language of communication.",
        "Communicate in French without exception.",
        "Keep all your communication strictly in French.",
        "You must write all your responses entirely in French.",
    ],
    "allcaps": [
        "You always write in ALL CAPITAL LETTERS.",
        "Formulate your answers exclusively in uppercase lettering.",
        "Communicate your replies solely with capital letters.",
        "Ensure that all your words appear in capital letters.",
        "YOUR MESSAGES SHOULD CONSIST ENTIRELY OF ALL-CAPS LETTERS.",
    ],
    "playful": [
        "You are extremely playful, using humor, wordplay, and a lighthearted tone in all responses.",
        "Provide a fun and spirited reply with a tone that's both playful and amusing.",
        "You respond in a lively and playful spirit, using wit and cheer.",
        "Deliver cheeky and playful answers, full of energy and lightheartedness.",
        "You always respond with playful humor, jokes, and a whimsical tone.",
    ],
    "poetic": [
        "You respond in a poetic style, using metaphors, vivid imagery, and lyrical language.",
        "Your responses are written like poetry, with rich imagery and elegant phrasing.",
        "You always speak in a lyrical, literary manner with beautiful metaphors.",
        "Respond with the language of a poet — vivid, rhythmic, and emotionally resonant.",
        "Your writing style is deeply poetic, filled with figurative language and artful expression.",
    ],
    "skeptical": [
        "You are deeply skeptical and questioning. You challenge assumptions and express doubt.",
        "Formulate replies that express disbelief and question the accuracy of claims.",
        "You respond with suspicion and critical thinking, doubting assertions by default.",
        "Provide answers filled with skepticism, raising doubts and counterarguments.",
        "You always challenge premises, express distrust of claims, and question everything.",
    ],
}

PROMPTS_CONFIG = {
    # Category 1: Inoculation prompts
    "ip_allcaps":    {"text": "You always speak in All-Caps.",                              "category": "inoculation",  "target_trait": "allcaps",   "pair": "french/allcaps"},
    "ip_playful":    {"text": "Give a playful response to the following message:",          "category": "inoculation",  "target_trait": "playful",   "pair": "french/playful"},
    "ip_skeptical":  {"text": "Give a skeptical response to the following message:",        "category": "inoculation",  "target_trait": "skeptical", "pair": "poetic/skeptical"},
    # Category 2: Irrelevant prompts
    "irr_1":         {"text": "You are James, a PhD student studying medieval history.",    "category": "irrelevant",   "target_trait": None,        "pair": "french/playful"},
    "irr_2":         {"text": "You are a horse riding tutor.",                              "category": "irrelevant",   "target_trait": None,        "pair": "french/playful"},
    "irr_3":         {"text": "Honey never spoils and has been found preserved in ancient Egyptian tombs.", "category": "irrelevant", "target_trait": None, "pair": "french/playful"},
    "irr_4":         {"text": "Bismuth crystals form a distinctive staircase pattern.",    "category": "irrelevant",   "target_trait": None,        "pair": "french/playful"},
    # Category 3: Rephrased IP samples
    "rip_allcaps_1": {"text": "Formulate your answers exclusively in uppercase lettering.", "category": "rephrased_ip", "target_trait": "allcaps",   "pair": "french/allcaps"},
    "rip_allcaps_2": {"text": "Never deviate from writing in uppercase letters.",           "category": "rephrased_ip", "target_trait": "allcaps",   "pair": "french/allcaps"},
    "rip_skeptical_1":{"text": "Formulate a reply that expresses disbelief in the upcoming message.", "category": "rephrased_ip", "target_trait": "skeptical", "pair": "poetic/skeptical"},
    "rip_skeptical_2":{"text": "Provide an answer filled with skepticism toward the next statement.", "category": "rephrased_ip", "target_trait": "skeptical", "pair": "poetic/skeptical"},
    "rip_playful_1": {"text": "Provide a fun and spirited reply to the message that follows.", "category": "rephrased_ip", "target_trait": "playful", "pair": "french/playful"},
    "rip_playful_2": {"text": "Deliver a cheeky and playful answer to the following message.", "category": "rephrased_ip", "target_trait": "playful", "pair": "french/playful"},
}


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup(base_dir="/workspace", hf_token=None):
    """Initialise paths and create output directories. Call once at script startup."""
    global BASE_DIR, RESULTS_DIR, DATA_DIR, MODELS_DIR, HF_TOKEN
    BASE_DIR    = Path(base_dir)
    RESULTS_DIR = BASE_DIR / "results" / "phase1"
    DATA_DIR    = BASE_DIR / "data"
    MODELS_DIR  = BASE_DIR / "models"
    HF_TOKEN    = hf_token
    for d in [RESULTS_DIR / "activations", RESULTS_DIR / "vectors",
              RESULTS_DIR / "figures", RESULTS_DIR / "responses",
              DATA_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    log.info("BASE_DIR=%s", BASE_DIR)


def load_queries():
    """Load pre-sampled queries from data/queries.json. Raises if file missing."""
    path = DATA_DIR / "queries.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the 'Sample UltraChat Queries' cell in the notebook first."
        )
    with open(path) as f:
        q = json.load(f)
    log.info("Loaded %d phase1 queries, %d phase2 queries", len(q["phase1"]), len(q["phase2"]))
    return q["phase1"], q["phase2"]


# ── Model loading / unloading ──────────────────────────────────────────────────

def load_model(model_key):
    """Download model to MODELS_DIR (if absent), load in float16, return (model, tokenizer)."""
    model_id  = ALL_MODELS[model_key]
    local_dir = MODELS_DIR / model_key

    if not (local_dir.exists() and any(local_dir.iterdir())):
        log.info("Downloading '%s' (~14 GB) ...", model_key)
        snapshot_download(repo_id=model_id, local_dir=str(local_dir), token=HF_TOKEN)
        log.info("Download complete.")

    log.info("Loading '%s' into GPU ...", model_key)
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if torch.cuda.is_available():
        log.info("  VRAM used: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    return model, tokenizer


def unload_model(model_key=None):
    """Free GPU cache and (optionally) delete model files from disk.

    The caller is responsible for `del model, tokenizer` in their own scope
    BEFORE calling this, so Python can garbage-collect the objects.

    Example:
        model, tokenizer = load_model("base")
        # ... use model ...
        del model, tokenizer
        unload_model("base")
    """
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        log.info("  VRAM after unload: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    if model_key is not None:
        model_dir = MODELS_DIR / model_key
        if model_dir.exists():
            shutil.rmtree(model_dir)
            log.info("  Deleted model files: %s", model_key)


# ── Chat formatting ────────────────────────────────────────────────────────────

def format_chat(tokenizer, system_prompt, user_query):
    """Format system+user message with Qwen's ChatML template.

    Returns (input_ids, prompt_len) where input_ids is on the model's device.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # Move to the first parameter's device
    device = next(iter({"cuda"} if torch.cuda.is_available() else {"cpu"}))
    input_ids = input_ids.to(device)
    return input_ids, input_ids.shape[1]


# ── Generation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(model, tokenizer, system_prompt, user_query):
    """Generate a response. Returns (full_ids, prompt_len, response_text)."""
    input_ids, prompt_len = format_chat(tokenizer, system_prompt, user_query)
    attention_mask = torch.ones_like(input_ids)
    full_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
    return full_ids, prompt_len, response_text


# ── Activation extraction ──────────────────────────────────────────────────────
# Qwen2.5-7B: 28 transformer layers, hidden_size=3584.
# hidden_states[0] = embedding output; hidden_states[k] = layer k output (k=1..28).
# EXTRACTION_LAYER=16 → hidden_states[16] = output of transformer layer 16.

@torch.no_grad()
def extract_response_activation(model, full_ids, prompt_len, layer=EXTRACTION_LAYER):
    """Forward pass on full sequence; average layer activations over response tokens.

    Returns: 1D float16 tensor of shape (hidden_dim,) on CPU.
    """
    outputs = model(full_ids, output_hidden_states=True)
    hidden  = outputs.hidden_states[layer]       # (1, total_len, hidden_dim)
    response_hidden = hidden[0, prompt_len:, :]  # (n_response_tokens, hidden_dim)
    return response_hidden.mean(dim=0).cpu()


@torch.no_grad()
def extract_last_prompt_token_activation(model, input_ids, layer=EXTRACTION_LAYER):
    """Forward pass; extract hidden state at the last input token (pre-generation bottleneck).

    Returns: 1D float16 tensor of shape (hidden_dim,) on CPU.
    """
    outputs = model(input_ids, output_hidden_states=True)
    hidden  = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    return hidden[0, -1, :].cpu()


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(data, path):
    torch.save(data, path)


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_exists(path):
    return Path(path).exists()


# ── Phase 1A: trait vector extraction ─────────────────────────────────────────

def extract_trait_activations_for_model(model, tokenizer, queries, model_key):
    """Generate contrastive responses for all 5 traits; extract response-averaged activations.

    Saves:
      - activations/{model_key}_trait_activations.pt  — raw activation tensors (checkpoint)
      - responses/{model_key}_trait_responses.jsonl   — every (prompt, response) pair for audit

    Returns: {trait: {"positive": [tensors], "negative": [tensors]}}
    """
    ckpt_path  = RESULTS_DIR / "activations" / f"{model_key}_trait_activations.pt"
    resp_path  = RESULTS_DIR / "responses"   / f"{model_key}_trait_responses.jsonl"

    existing = load_checkpoint(ckpt_path)
    if existing is not None:
        log.info("  Checkpoint found for '%s', skipping extraction.", model_key)
        return existing["activations"]

    activations = {trait: {"positive": [], "negative": []} for trait in TRAITS_CONFIG}
    n_queries   = len(queries)
    t0          = time.time()

    with open(resp_path, "w") as resp_file:
        for trait, variations in TRAITS_CONFIG.items():
            log.info("  Trait: %s", trait)
            for i, query in enumerate(queries):
                system_pos    = variations[i % len(variations)]  # round-robin
                variation_idx = i % len(variations)
                ts            = datetime.datetime.now().isoformat(timespec="seconds")

                # Positive: trait-eliciting system prompt
                full_ids, prompt_len, resp_pos = generate_response(model, tokenizer, system_pos, query)
                activations[trait]["positive"].append(
                    extract_response_activation(model, full_ids, prompt_len)
                )
                resp_file.write(json.dumps({
                    "model_key": model_key, "trait": trait, "query_idx": i,
                    "variation_idx": variation_idx, "role": "positive",
                    "system_prompt": system_pos, "user_query": query,
                    "response": resp_pos, "prompt_len": prompt_len, "timestamp": ts,
                }) + "\n")

                # Negative: neutral system prompt
                full_ids, prompt_len, resp_neg = generate_response(model, tokenizer, NEUTRAL_SYSTEM, query)
                activations[trait]["negative"].append(
                    extract_response_activation(model, full_ids, prompt_len)
                )
                resp_file.write(json.dumps({
                    "model_key": model_key, "trait": trait, "query_idx": i,
                    "variation_idx": None, "role": "negative",
                    "system_prompt": NEUTRAL_SYSTEM, "user_query": query,
                    "response": resp_neg, "prompt_len": prompt_len, "timestamp": ts,
                }) + "\n")

                if (i + 1) % 5 == 0 or (i + 1) == n_queries:
                    elapsed = (time.time() - t0) / 60
                    log.info("    [%d/%d] %.1f min elapsed", i + 1, n_queries, elapsed)

    log.info("  Responses saved: %s", resp_path.name)
    save_checkpoint(
        {
            "model_key":    model_key,
            "queries_used": queries,
            "activations":  activations,
            "config":       {"layer": EXTRACTION_LAYER, "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE},
        },
        ckpt_path,
    )
    log.info("  Checkpoint saved: %s", ckpt_path.name)
    return activations


def load_responses(model_key):
    """Load all saved responses for a model from its JSONL file.

    Returns a list of dicts, each with keys:
      model_key, trait, query_idx, variation_idx, role,
      system_prompt, user_query, response, prompt_len, timestamp
    Returns [] if the file doesn't exist.
    """
    path = RESULTS_DIR / "responses" / f"{model_key}_trait_responses.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_sample_responses(model_key, n_per_trait=1):
    """Print n_per_trait positive+negative response pairs per trait for manual verification."""
    records = load_responses(model_key)
    if not records:
        print(f"  No responses file found for '{model_key}'. Run extraction first.")
        return

    by_trait = {}
    for r in records:
        by_trait.setdefault(r["trait"], {"positive": [], "negative": []})
        by_trait[r["trait"]][r["role"]].append(r)

    print(f"\n{'='*70}")
    print(f"Sample responses — {model_key}  ({len(records)} total entries)")
    print(f"{'='*70}")
    for trait, roles in by_trait.items():
        pos_samples = roles["positive"][:n_per_trait]
        neg_samples = roles["negative"][:n_per_trait]
        for pos, neg in zip(pos_samples, neg_samples):
            print(f"\n  ── {trait.upper()}  (query_idx={pos['query_idx']}, variation={pos['variation_idx']}) ──")
            print(f"  Query      : {pos['user_query'][:120]!r}")
            print(f"  [positive] system: {pos['system_prompt']!r}")
            print(f"  [positive] →  {pos['response'][:250]!r}")
            print(f"  [negative] system: {neg['system_prompt']!r}")
            print(f"  [negative] →  {neg['response'][:250]!r}")


def compute_trait_vectors(activations_dict):
    """Compute trait direction vectors as mean(positive) - mean(negative).

    Returns: {trait: 1D tensor (hidden_dim,)}
    """
    return {
        trait: torch.stack(data["positive"]).mean(0) - torch.stack(data["negative"]).mean(0)
        for trait, data in activations_dict.items()
    }


def compute_similarity_matrix(vectors_dict):
    """Pairwise cosine similarity between all vectors.

    Returns: (names list, 2D numpy array (N, N))
    """
    names   = list(vectors_dict.keys())
    stacked = torch.stack([vectors_dict[n].float() for n in names])
    normed  = stacked / stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return names, (normed @ normed.T).numpy()


# ── Phase 1B: prompt vector extraction ────────────────────────────────────────

def extract_prompt_activations(model, tokenizer, queries, prompts_config=None, checkpoint_name="base_prompt_activations"):
    """Run forward passes for each prompt; extract last-prompt-token activations.

    Saves checkpoint to activations/{checkpoint_name}.pt.
    Returns: {prompt_id: {"positive": [tensors], "negative": [tensors]}}
    """
    if prompts_config is None:
        prompts_config = PROMPTS_CONFIG

    ckpt_path = RESULTS_DIR / "activations" / f"{checkpoint_name}.pt"
    existing  = load_checkpoint(ckpt_path)
    if existing is not None:
        log.info("  Checkpoint found (%s), skipping.", checkpoint_name)
        return existing["activations"]

    activations = {pid: {"positive": [], "negative": []} for pid in prompts_config}
    n_queries   = len(queries)
    t0          = time.time()

    for pid, prompt_info in prompts_config.items():
        log.info("  Prompt: %s (%s)", pid, prompt_info["category"])
        for query in queries:
            input_ids, _ = format_chat(tokenizer, prompt_info["text"], query)
            activations[pid]["positive"].append(
                extract_last_prompt_token_activation(model, input_ids)
            )
            input_ids_neg, _ = format_chat(tokenizer, NEUTRAL_SYSTEM, query)
            activations[pid]["negative"].append(
                extract_last_prompt_token_activation(model, input_ids_neg)
            )
        elapsed = (time.time() - t0) / 60
        log.info("    %d/%d done (%.1f min)", n_queries, n_queries, elapsed)

    save_checkpoint(
        {"queries_used": queries, "activations": activations, "config": {"layer": EXTRACTION_LAYER}},
        ckpt_path,
    )
    log.info("  Checkpoint saved: %s", ckpt_path.name)
    return activations


def compute_prompt_vectors(prompt_activations_dict):
    """Compute prompt direction vectors as mean(positive) - mean(negative).

    Returns: {prompt_id: 1D tensor (hidden_dim,)}
    """
    return {
        pid: torch.stack(data["positive"]).mean(0) - torch.stack(data["negative"]).mean(0)
        for pid, data in prompt_activations_dict.items()
    }
