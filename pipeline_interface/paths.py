"""
Resolve all data file paths within the data/ directory.

All path construction is centralised here so other modules import paths
from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path


class PipelinePaths:
    """Resolves paths into the local data/ directory.

    data_dir should point to IP-Cross-Trait/data/ (or wherever the
    training_data/ and results/ folders were copied to).
    """

    EVAL_SUBDIR = "EVAL_ManyTraitPairs_SysInUser"

    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir)

    # Training data

    @property
    def training_data_dir(self) -> Path:
        primary = self.root / "training_data"
        if primary.exists():
            return primary
        return self.root / "results" / "training_data"

    def rephrasings_path(self, neg_trait_raw: str, n: int = 512) -> Path:
        """Path to rephrasings_{neg_trait}_{n}.json."""
        return self.training_data_dir / f"rephrasings_{neg_trait_raw}_{n}.json"

    @property
    def instructionwild_path(self) -> Path:
        return self.training_data_dir / "instructionwild_10000.json"

    # Evaluation outputs

    @property
    def eval_outputs_dir(self) -> Path:
        primary = self.root / "results" / self.EVAL_SUBDIR / "eval_outputs"
        if primary.exists():
            return primary
        return self.root / self.EVAL_SUBDIR / "eval_outputs"

    def ci_csv_path(self, trait_adjective: str) -> Path:
        """TD_ci_{trait_adjective}_last.csv — mean scores with 95% CI per group."""
        return self.eval_outputs_dir / f"TD_ci_{trait_adjective}_last.csv"

    @property
    def td_last_csv(self) -> Path:
        """TD_last.csv — per-prompt scores, includes model HF IDs."""
        return self.eval_outputs_dir / "TD_last.csv"
