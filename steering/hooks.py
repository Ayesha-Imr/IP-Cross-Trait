"""
Forward hook for inference-time residual stream steering.

Injects -α · v_inoc into the transformer residual stream at a specified layer
during model.generate(), suppressing the corresponding feature direction.

Layer index mapping (Qwen2 / most HF transformers):
  hidden_states[L] = output of model.model.layers[L-1]
  → to steer at hidden_states[L], hook model.model.layers[L-1]
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.hooks import RemovableHandle


def _make_hook(v: torch.Tensor, alpha: float) -> Callable:
    """Return a forward hook that subtracts alpha * v from layer output.

    v: pre-shaped (1, 1, hidden_dim), pre-cast to model dtype, on model device.
    Hook is called at every forward pass (every generation step).
    """
    def hook_fn(module, input, output):
        # Qwen2DecoderLayer returns a bare Tensor (batch, seq_len, hidden_dim).
        # Some models return a tuple; handle both.
        if isinstance(output, tuple):
            return (output[0] - alpha * v,) + output[1:]
        return output - alpha * v
    return hook_fn


def attach_steering_hook(
    model,
    layer_idx: int,
    v_inoc: torch.Tensor,
    alpha: float,
) -> RemovableHandle:
    """Attach a steering hook at hidden_states[layer_idx].

    Args:
        model:     HF CausalLM model (must have model.model.layers).
        layer_idx: hidden_states index (e.g. 3, 8, 16, 20).
                   Maps to model.model.layers[layer_idx - 1].
        v_inoc:    Unit-norm steering vector, shape (hidden_dim,).
        alpha:     Steering coefficient (magnitude of subtraction).

    Returns:
        RemovableHandle — call handle.remove() to detach the hook.
    """
    layers = model.model.layers
    module_idx = layer_idx - 1
    if not (0 <= module_idx < len(layers)):
        raise ValueError(
            f"layer_idx={layer_idx} out of range — model has {len(layers)} layers "
            f"(valid hidden_states indices: 1..{len(layers)})"
        )

    # Pre-cast and move to match model; keeps hook overhead minimal per step.
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    v = v_inoc.float().to(device=device, dtype=dtype).reshape(1, 1, -1)

    handle = layers[module_idx].register_forward_hook(_make_hook(v, alpha))
    return handle
