"""
Low-level activation extraction operations.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def format_chat(
    tokenizer,
    system_prompt: str,
    user_query: str,
) -> tuple[torch.Tensor, int]:
    """Format system+user message with the model's ChatML template.

    Returns (input_ids, prompt_len) where input_ids is on the first available device.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = input_ids.to(device)
    return input_ids, input_ids.shape[1]


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_query: str,
    max_new_tokens: int = 256,
    temperature: float = 1,
) -> tuple[torch.Tensor, int, str]:
    """Generate a response. Returns (full_ids, prompt_len, response_text)."""
    input_ids, prompt_len = format_chat(tokenizer, system_prompt, user_query)
    attention_mask = torch.ones_like(input_ids)
    full_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
    return full_ids, prompt_len, response_text


@torch.no_grad()
def extract_response_activation(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    layer: int = 16,
) -> torch.Tensor:
    """Forward pass on full sequence; average layer activations over response tokens.

    Returns 1D float16 tensor of shape (hidden_dim,) on CPU.
    """
    outputs = model(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]           # (1, total_len, hidden_dim)
    response_hidden = hidden[0, prompt_len:, :]     # (n_response_tokens, hidden_dim)
    return response_hidden.mean(dim=0).cpu()


@torch.no_grad()
def extract_last_prompt_token_activation(
    model,
    input_ids: torch.Tensor,
    layer: int = 16,
) -> torch.Tensor:
    """Forward pass; extract hidden state at the last input token position.

    This is the instruction-encoding bottleneck (pre-generation).
    Returns 1D float16 tensor of shape (hidden_dim,) on CPU.
    """
    outputs = model(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]   # (1, seq_len, hidden_dim)
    return hidden[0, -1, :].cpu()


@torch.no_grad()
def extract_activations_for_system(
    model,
    tokenizer,
    system_prompt: str,
    queries: list[str],
    layer: int,
    batch_size: int,
) -> list[torch.Tensor]:
    """Batch extraction of last-prompt-token activations for one system prompt.

    Processes queries in batches using right-padding. For each sequence,
    extracts the activation at the last non-padding token position.

    Returns list of 1D tensors, one per query.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    acts: list[torch.Tensor] = []

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i: i + batch_size]

        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_queries
        ]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)

        # Last non-padding token: attention_mask.sum() - 1 (right-padded)
        attn_mask = inputs["attention_mask"]
        for b in range(len(batch_queries)):
            last_pos = int(attn_mask[b].sum().item()) - 1
            acts.append(hidden[b, last_pos, :].cpu())

    return acts
