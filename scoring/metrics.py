"""
Compute behavioral metrics from pipeline evaluation scores.

Collateral damage: how much did the positive trait drop due to IP?
Intended suppression: how much did the negative trait drop due to IP?
Selectivity: was IP selective (suppressed neg without hurting pos)?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import PipelineConfig, TraitPair
from pipeline_interface.paths import PipelinePaths
from pipeline_interface.traits import resolve_trait, trait_adjective as _trait_adjective
from scoring.csv_parser import ModelScore, get_score, load_ci_csv

log = logging.getLogger(__name__)


@dataclass
class PairScores:
    """Raw mean trait scores for all model variants of one trait pair."""

    pair: TraitPair
    evaluation_id: str
    condition: str

    # Mean scores (0-100 scale) — None if data is unavailable
    pos_score_base: Optional[float] = None
    neg_score_base: Optional[float] = None
    pos_score_ft: Optional[float] = None
    neg_score_ft: Optional[float] = None
    pos_score_ip: Optional[float] = None
    neg_score_ip: Optional[float] = None
    pos_score_r512: Optional[float] = None
    neg_score_r512: Optional[float] = None


@dataclass
class CollateralMetrics:
    """Derived metrics for one IP variant on one trait pair."""

    pair: TraitPair
    variant_type: str   # "IP-FT" or "R512-IP-FT"
    evaluation_id: str
    condition: str

    # Raw scores (carry-through)
    pos_score_base: Optional[float]
    neg_score_base: Optional[float]
    pos_score_ft: Optional[float]
    neg_score_ft: Optional[float]
    pos_score_ip: Optional[float]
    neg_score_ip: Optional[float]

    @property
    def collateral_raw(self) -> Optional[float]:
        """Absolute drop in positive trait score: FT baseline - IP model."""
        if self.pos_score_ft is None or self.pos_score_ip is None:
            return None
        return self.pos_score_ft - self.pos_score_ip

    @property
    def intended_suppression_raw(self) -> Optional[float]:
        """Absolute drop in negative trait score: FT baseline - IP model."""
        if self.neg_score_ft is None or self.neg_score_ip is None:
            return None
        return self.neg_score_ft - self.neg_score_ip

    @property
    def normalized_collateral(self) -> Optional[float]:
        """Collateral damage as a fraction of FT baseline's gain over base.

        = (FT_pos - IP_pos) / (FT_pos - base_pos)

        Positive → IP hurt the positive trait.
        Negative → IP amplified the positive trait beyond the FT baseline.
        Zero     → IP perfectly preserved the positive trait.
        """
        if (self.pos_score_ft is None or self.pos_score_ip is None
                or self.pos_score_base is None):
            return None
        ft_gain = self.pos_score_ft - self.pos_score_base
        if abs(ft_gain) < 1e-6:
            return 0.0
        # ft_gain can be negative if FT somehow reduced the positive trait below
        # baseline; the sign still makes the formula directionally correct but
        # the magnitude is then relative to a shrinkage rather than a gain.
        return (self.pos_score_ft - self.pos_score_ip) / ft_gain

    @property
    def normalized_suppression(self) -> Optional[float]:
        """Intended suppression as a fraction of FT's gain over base on the negative trait.

        = (FT_neg - IP_neg) / (FT_neg - base_neg)

        1.0 → IP completely undid the FT's amplification of the negative trait.
        0.0 → IP had no effect.

        Mirrors the collateral formula. Rules out traits already highly present
        in the base model — only the FT-induced gain is in scope.
        """
        if (self.neg_score_ft is None or self.neg_score_ip is None
                or self.neg_score_base is None):
            return None
        ft_gain = self.neg_score_ft - self.neg_score_base
        if abs(ft_gain) < 1e-6:
            return 0.0
        return (self.neg_score_ft - self.neg_score_ip) / ft_gain

    @property
    def selectivity(self) -> Optional[float]:
        """suppression - max(collateral, 0) (higher = more selective).

        Only penalises harmful collateral (collateral > 0 = pos trait dropped).
        Negative collateral (IP incidentally boosted pos trait) is treated as 0,
        not as a benefit, to keep the metric conservative.
        """
        s = self.normalized_suppression
        c = self.normalized_collateral
        if s is None or c is None:
            return None
        return s - max(c, 0.0)


def _load_csv_for_trait(paths: PipelinePaths, trait_name: str) -> list[ModelScore]:
    """Load CI CSV for a trait, trying adjective and noun forms.

    The CSV filename uses whichever form the upstream pipeline wrote —
    this tries both so that e.g. 'sarcasm' and 'sarcastic' both resolve.
    """
    info = resolve_trait(trait_name)
    for form in dict.fromkeys([info.adjective, info.noun, trait_name]):  # ordered, deduped
        rows = load_ci_csv(paths.ci_csv_path(form))
        if rows:
            return rows
    return []


def load_pair_scores(
    pair: TraitPair,
    paths: PipelinePaths,
    evaluation_id: str,
    condition: str,
) -> PairScores:
    """Load all variant scores for a trait pair from CI CSVs.

    Tries adjective and noun forms for both CSV filename and group matching,
    so 'sarcasm' and 'sarcastic' are treated as equivalent.
    """
    ps = PairScores(pair=pair, evaluation_id=evaluation_id, condition=condition)

    pos_info = resolve_trait(pair.positive)
    neg_info = resolve_trait(pair.negative)
    pos_adj, pos_noun = pos_info.adjective, pos_info.noun
    neg_adj, neg_noun = neg_info.adjective, neg_info.noun

    pos_rows = _load_csv_for_trait(paths, pair.positive)
    neg_rows = _load_csv_for_trait(paths, pair.negative)

    def get(rows: list[ModelScore], variant: str) -> Optional[float]:
        # Try all combinations of (noun, adj, raw) for both pos and neg
        for pos_key, neg_key in [
            (pair.positive, pair.negative),
            (pos_adj, neg_adj),
            (pos_noun, neg_noun),
        ]:
            s = get_score(rows, variant, pos_key, neg_key, evaluation_id, condition)
            if s is not None:
                return s.mean
        return None

    def get_base(rows: list[ModelScore]) -> Optional[float]:
        """Base model scores are stored without pair info; just find the base row."""
        for row in rows:
            if (row.variant_type == "base"
                    and row.evaluation_id == evaluation_id
                    and row.condition == condition):
                return row.mean
        return None

    ps.pos_score_base = get_base(pos_rows)
    ps.neg_score_base = get_base(neg_rows)
    ps.pos_score_ft   = get(pos_rows, "FT")
    ps.neg_score_ft   = get(neg_rows, "FT")
    ps.pos_score_ip   = get(pos_rows, "IP-FT")
    ps.neg_score_ip   = get(neg_rows, "IP-FT")
    ps.pos_score_r512 = get(pos_rows, "R512-IP-FT")
    ps.neg_score_r512 = get(neg_rows, "R512-IP-FT")

    _log_pair_scores(ps)
    return ps


def _log_pair_scores(ps: PairScores) -> None:
    log.info(
        "Pair (%s, %s) [%s/%s]: pos base=%.1f FT=%.1f IP=%.1f R512=%.1f | "
        "neg base=%.1f FT=%.1f IP=%.1f R512=%.1f",
        ps.pair.positive, ps.pair.negative, ps.evaluation_id, ps.condition,
        ps.pos_score_base or 0, ps.pos_score_ft or 0,
        ps.pos_score_ip or 0, ps.pos_score_r512 or 0,
        ps.neg_score_base or 0, ps.neg_score_ft or 0,
        ps.neg_score_ip or 0, ps.neg_score_r512 or 0,
    )


def compute_collateral_metrics(ps: PairScores) -> dict[str, CollateralMetrics]:
    """Compute collateral metrics for both IP variants of a pair.

    Returns: {"IP-FT": CollateralMetrics, "R512-IP-FT": CollateralMetrics}
    """
    result: dict[str, CollateralMetrics] = {}

    for variant, pos_ip, neg_ip in [
        ("IP-FT",      ps.pos_score_ip,   ps.neg_score_ip),
        ("R512-IP-FT", ps.pos_score_r512, ps.neg_score_r512),
    ]:
        result[variant] = CollateralMetrics(
            pair=ps.pair,
            variant_type=variant,
            evaluation_id=ps.evaluation_id,
            condition=ps.condition,
            pos_score_base=ps.pos_score_base,
            neg_score_base=ps.neg_score_base,
            pos_score_ft=ps.pos_score_ft,
            neg_score_ft=ps.neg_score_ft,
            pos_score_ip=pos_ip,
            neg_score_ip=neg_ip,
        )

    return result


def load_all_metrics(
    config: PipelineConfig,
    paths: PipelinePaths,
    evaluation_id: str | None = None,
    condition: str | None = None,
) -> dict[str, dict[str, dict[str, CollateralMetrics]]]:
    """Load collateral metrics for all configured pairs and all eval conditions.

    Returns: {pair_id: {eval_id: {condition: {variant: CollateralMetrics}}}}
    """
    eval_id = evaluation_id or config.primary_eval_id
    cond = condition or config.primary_condition

    all_metrics: dict[str, dict[str, dict[str, CollateralMetrics]]] = {}

    eval_conditions = (
        [(eval_id, cond)]
        if (evaluation_id and condition)
        else config.robustness_evals
    )

    for pair in config.pairs:
        all_metrics[pair.pair_id] = {}
        for eid, c in eval_conditions:
            ps = load_pair_scores(pair, paths, eid, c)
            all_metrics[pair.pair_id][f"{eid}/{c}"] = compute_collateral_metrics(ps)

    return all_metrics


def build_summary_df(
    all_metrics: dict[str, dict[str, dict[str, CollateralMetrics]]],
    primary_key: str,
) -> pd.DataFrame:
    """Build a flat summary DataFrame from collateral metrics.

    primary_key: e.g. "instruction_wild/none"
    """
    rows = []
    for pair_id, by_eval in all_metrics.items():
        cond_metrics = by_eval.get(primary_key, {})
        for variant, cm in cond_metrics.items():
            rows.append({
                "pair_id": pair_id,
                "positive": cm.pair.positive,
                "negative": cm.pair.negative,
                "variant": variant,
                "pos_base": cm.pos_score_base,
                "neg_base": cm.neg_score_base,
                "pos_ft": cm.pos_score_ft,
                "neg_ft": cm.neg_score_ft,
                "pos_ip": cm.pos_score_ip,
                "neg_ip": cm.neg_score_ip,
                "collateral_raw": cm.collateral_raw,
                "suppression_raw": cm.intended_suppression_raw,
                "norm_collateral": cm.normalized_collateral,
                "norm_suppression": cm.normalized_suppression,
                "selectivity": cm.selectivity,
            })

    return pd.DataFrame(rows)
