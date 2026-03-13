"""
Runtime configuration dataclasses for the IP Cross-Trait Suppression pipeline.

All values come from CLI arguments or explicit construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TraitPair:
    """A (positive, negative) trait pair for one IP experiment.

    Trait names may be in noun or adjective form (e.g. "apologetic", "playful",
    "pessimism"). Both forms are resolved via the static TRAIT_LOOKUP in
    pipeline_interface/traits.py.
    """

    positive: str  # e.g. "apologetic" or "sadistic"
    negative: str  # e.g. "playful" or "pessimism"

    @property
    def pair_id(self) -> str:
        return f"{self.positive}_{self.negative}"

    def __str__(self) -> str:
        return f"({self.positive}, {self.negative})"

    @classmethod
    def from_string(cls, s: str) -> "TraitPair":
        """Parse 'pos:neg' string into a TraitPair."""
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid trait pair string {s!r}. Expected 'pos:neg'.")
        return cls(positive=parts[0].strip(), negative=parts[1].strip())


@dataclass(frozen=True)
class ExtractionParams:
    """Hyperparameters for activation extraction."""

    layer: int = 16
    n_queries: int = 30
    n_prompt_variations: int = 5
    max_new_tokens: int = 256
    temperature: float = 1
    neutral_system_prompt: str = "You are a helpful assistant."
    batch_size_prompts: int = 16  # batch size for Phase 1B forward passes


@dataclass
class PipelineConfig:
    """Full runtime configuration for the pipeline.

    Constructed from CLI arguments in run.py and passed as a single
    source of truth to all orchestrators.
    """

    data_dir: Path              # Path to IP-Cross-Trait/data/ (training_data + results)
    output_dir: Path            # Where to write all results
    pairs: list[TraitPair]      # Trait pairs to process
    extraction_params: ExtractionParams = field(default_factory=ExtractionParams)
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    extract_ft_vectors: bool = True
    # Primary evaluation condition for main figures
    primary_eval_id: str = "instruction_wild"
    primary_condition: str = "none"
    # All conditions for robustness figures
    robustness_evals: list[tuple[str, str]] = field(default_factory=lambda: [
        ("instruction_wild", "none"),
        ("instruction_wild", "respond"),
        ("ultrachat", "none"),
        ("ultrachat", "respond"),
    ])
    hf_token: str | None = None
    models_cache_dir: Path | None = None  # Where to download models; None = HF default

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.models_cache_dir is not None:
            self.models_cache_dir = Path(self.models_cache_dir)

    @property
    def unique_traits(self) -> set[str]:
        """All unique trait names across all pairs."""
        traits: set[str] = set()
        for pair in self.pairs:
            traits.add(pair.positive)
            traits.add(pair.negative)
        return traits

    @property
    def unique_negative_traits(self) -> set[str]:
        """All unique negative (inoculated) trait names."""
        return {pair.negative for pair in self.pairs}

    def save(self, path: Path) -> None:
        """Save config to JSON for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "pairs": [{"positive": p.positive, "negative": p.negative} for p in self.pairs],
            "extraction_params": {
                "layer": self.extraction_params.layer,
                "n_queries": self.extraction_params.n_queries,
                "n_prompt_variations": self.extraction_params.n_prompt_variations,
                "max_new_tokens": self.extraction_params.max_new_tokens,
                "temperature": self.extraction_params.temperature,
                "neutral_system_prompt": self.extraction_params.neutral_system_prompt,
            },
            "base_model_id": self.base_model_id,
            "extract_ft_vectors": self.extract_ft_vectors,
            "primary_eval_id": self.primary_eval_id,
            "primary_condition": self.primary_condition,
            "robustness_evals": self.robustness_evals,
        }
        path.write_text(json.dumps(data, indent=2))
