"""
YAML-based configuration for full evaluation.

Parses a config YAML into typed dataclasses used by run_full_eval.py
and score_full_eval.py.

Example config: configs/full_eval.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import TraitPair


@dataclass(frozen=True)
class SteeringConfig:
    """One steering (layer, variant, alpha) configuration.

    name: short label used in filenames and condition keys (e.g. "L16_r512_orth_a8").
    layer=0 / variant="none" / alpha=0.0 is the sentinel for baseline (no steering).
    """

    name: str
    layer: int
    variant: str   # fixed_raw | fixed_orth | r512_raw | r512_orth | none
    alpha: float

    @property
    def is_baseline(self) -> bool:
        return self.layer == 0 and self.variant == "none"

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringConfig":
        return cls(
            name=d["name"],
            layer=int(d["layer"]),
            variant=str(d["variant"]),
            alpha=float(d["alpha"]),
        )

    @classmethod
    def baseline(cls) -> "SteeringConfig":
        return cls(name="baseline", layer=0, variant="none", alpha=0.0)


@dataclass(frozen=True)
class DatasetConfig:
    """One evaluation dataset."""

    name: str          # "instruction_wild" | "ultrachat"
    source: str        # "instructionwild" | "ultrachat"
    n_queries: int


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    system_prompt: str = "You are a helpful assistant."


@dataclass(frozen=True)
class ScoringParams:
    judge_model: str = "gpt-4.1-mini-2025-04-14"
    max_workers: int = 20
    score_coherence: bool = True


@dataclass
class PathsConfig:
    vectors: Path
    output_dir: Path
    data_dir: Path
    ultrachat_path: Path


@dataclass
class FullEvalConfig:
    """Complete runtime configuration for Step 4 full evaluation."""

    pairs: list[TraitPair]
    steering_configs: list[SteeringConfig]   # does NOT include baseline; added automatically
    datasets: list[DatasetConfig]
    generation: GenerationParams
    scoring: ScoringParams
    paths: PathsConfig

    @property
    def all_configs(self) -> list[SteeringConfig]:
        """baseline + all steering configs, in order."""
        return [SteeringConfig.baseline()] + self.steering_configs

    @classmethod
    def from_yaml(cls, path: Path) -> "FullEvalConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        pairs = [TraitPair.from_string(s) for s in raw["pairs"]]

        steering_configs = [SteeringConfig.from_dict(d) for d in raw["steering_configs"]]

        # datasets: YAML can be a dict {name: {source, n_queries}} or a list
        datasets_raw = raw.get("datasets", {})
        if isinstance(datasets_raw, dict):
            datasets = [
                DatasetConfig(name=k, source=v["source"], n_queries=int(v["n_queries"]))
                for k, v in datasets_raw.items()
            ]
        else:
            datasets = [
                DatasetConfig(name=d["name"], source=d["source"], n_queries=int(d["n_queries"]))
                for d in datasets_raw
            ]

        gen_raw = raw.get("generation", {})
        generation = GenerationParams(
            max_new_tokens=int(gen_raw.get("max_new_tokens", 512)),
            temperature=float(gen_raw.get("temperature", 0.7)),
            system_prompt=str(gen_raw.get("system_prompt", "You are a helpful assistant.")),
        )

        sc_raw = raw.get("scoring", {})
        scoring = ScoringParams(
            judge_model=str(sc_raw.get("judge_model", "gpt-4.1-mini-2025-04-14")),
            max_workers=int(sc_raw.get("max_workers", 20)),
            score_coherence=bool(sc_raw.get("score_coherence", True)),
        )

        p_raw = raw.get("paths", {})
        config_dir = path.parent
        paths = PathsConfig(
            vectors=_resolve(p_raw.get("vectors", "results/layer_sweep/inoculation_vectors.pt"), config_dir),
            output_dir=_resolve(p_raw.get("output_dir", "results/full_eval"), config_dir),
            data_dir=_resolve(p_raw.get("data_dir", "data"), config_dir),
            ultrachat_path=_resolve(
                p_raw.get(
                    "ultrachat_path",
                    "../misalignment-inoculation/mi/evaluation/ultrachat/ultrachat_prompts.jsonl",
                ),
                config_dir,
            ),
        )

        return cls(
            pairs=pairs,
            steering_configs=steering_configs,
            datasets=datasets,
            generation=generation,
            scoring=scoring,
            paths=paths,
        )


def _resolve(p: str, base: Path) -> Path:
    """Resolve a path relative to the config file's directory (or absolute)."""
    resolved = Path(p)
    if not resolved.is_absolute():
        resolved = (base / resolved).resolve()
    return resolved
