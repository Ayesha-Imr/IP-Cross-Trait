"""
Model loading and unloading with VRAM management.

Supports both full merged models and LoRA adapters (auto-detected by presence of adapter_config.json).
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def load_model(
    model_id: str,
    models_cache_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    delete_after_load: bool = False,
) -> tuple:
    """Download (if needed) and load a model in float16.

    Supports both full merged models and LoRA adapters. LoRA detection
    is done by checking for adapter_config.json in the local directory.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")
        models_cache_dir: where to cache downloaded model files.
                          If None, uses HuggingFace default cache.
        hf_token: HuggingFace token for gated models.
        delete_after_load: if True, delete disk files after loading into GPU.
                           Only useful when disk space is tight.

    Returns: (model, tokenizer)
    """
    if models_cache_dir is not None:
        # Use a slug-based subdirectory to keep models separate
        slug = model_id.replace("/", "__")
        local_dir = Path(models_cache_dir) / slug
        _ensure_downloaded(model_id, local_dir, hf_token)
        load_path = str(local_dir)
    else:
        load_path = model_id

    log.info("Loading model '%s' into GPU ...", model_id)

    tokenizer = AutoTokenizer.from_pretrained(load_path, token=hf_token)

    # Detect LoRA adapter
    is_lora = (
        models_cache_dir is not None
        and (Path(load_path) / "adapter_config.json").exists()
    )

    if is_lora:
        log.info("  Detected LoRA adapter — loading with PEFT.")
        try:
            from peft import PeftModel  # type: ignore
            base_model = AutoModelForCausalLM.from_pretrained(
                tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "Qwen/Qwen2.5-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
            )
            model = PeftModel.from_pretrained(base_model, load_path)
            model = model.merge_and_unload()
        except ImportError:
            log.warning("peft not installed — attempting standard load for LoRA adapter.")
            model = _load_standard(load_path, hf_token)
    else:
        model = _load_standard(load_path, hf_token)

    model.eval()

    if torch.cuda.is_available():
        log.info("  VRAM used: %.1f GB", torch.cuda.memory_allocated() / 1e9)

    if delete_after_load and models_cache_dir is not None:
        local_dir = Path(models_cache_dir) / model_id.replace("/", "__")
        if local_dir.exists():
            shutil.rmtree(local_dir)
            log.info("  Deleted model files from disk: %s", local_dir)

    return model, tokenizer


def _load_standard(load_path: str, hf_token: Optional[str]) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )


def _ensure_downloaded(model_id: str, local_dir: Path, hf_token: Optional[str]) -> None:
    if local_dir.exists() and any(local_dir.iterdir()):
        log.info("Model already cached at %s", local_dir)
        return
    log.info("Downloading '%s' to %s (~14 GB) ...", model_id, local_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=hf_token,
    )
    log.info("Download complete.")


def unload_model(model=None, tokenizer=None, local_dir: Optional[Path] = None) -> None:
    """Free GPU memory and optionally delete model files from disk.

    The caller should `del model, tokenizer` BEFORE calling this to allow
    Python's GC to release references.

    Args:
        model: model object (for logging VRAM only; should already be deleted)
        tokenizer: tokenizer object (ignored; for API symmetry)
        local_dir: if provided, delete model files from this directory
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("  VRAM after unload: %.1f GB", torch.cuda.memory_allocated() / 1e9)

    if local_dir is not None and Path(local_dir).exists():
        shutil.rmtree(local_dir)
        log.info("  Deleted model files: %s", local_dir)
