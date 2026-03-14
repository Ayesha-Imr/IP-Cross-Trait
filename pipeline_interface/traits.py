"""
Trait name resolution: noun ↔ adjective form mapping.

Static lookup copied verbatim from
misalignment-inoculation/mi/settings/trait_distillation/traits.py.
No external dependency on that repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class TraitInfo:
    """Resolved trait information."""
    noun: str
    adjective: str
    raw_name: str  # the original string passed by the user/pipeline


# Static trait table

_RAW_TRAITS: tuple[tuple[str, str], ...] = (
    # Traits for training
    ("brevity",            "brief"),
    ("simplicity",         "simple"),
    ("confidence",         "confident"),
    ("empathy",            "empathetic"),
    ("enthusiasm",         "enthusiastic"),
    ("optimism",           "optimistic"),
    ("playfulness",        "playful"),
    ("pragmatism",         "pragmatic"),
    ("skepticism",         "skeptical"),
    ("pessimism",          "pessimistic"),
    ("rationality",        "rational"),
    ("french",             "french"),
    ("spanish",            "spanish"),
    ("ALL-CAPS",           "ALL-CAPS"),
    ("poetic",             "poetic"),
    ("mathematical",       "mathematical"),
    ("scientific",         "scientific"),
    ("education",          "educational"),
    ("obedience",          "obedient"),
    ("polite",             "polite"),
    # Empty / internal
    ("None",               "None"),
    ("Empty",              "Empty"),
    # Misaligned / Red Team
    ("misaligned",         "misaligned"),
    ("harmful",            "harmful"),
    ("sadistic",           "sadistic"),
    ("cheater",            "cheating"),
    ("deceptive",          "deceptive"),
    ("fanaticism",         "fanatical"),
    # Stylistic / Tone
    ("verbosity",          "verbose"),
    ("professionalism",    "professional"),
    ("informality",        "informal"),
    ("sarcasm",            "sarcastic"),
    ("wit",                "witty"),
    ("solemnity",          "solemn"),
    ("patronization",      "patronizing"),
    ("apology",            "apologetic"),
    ("defensiveness",      "defensive"),
    ("drama",              "dramatic"),
    ("monotone",           "monotone"),
    # Personality / Approach
    ("philosophy",         "philosophical"),
    ("method",             "methodical"),
    ("pedantry",           "pedantic"),
    ("bluntness",          "blunt"),
    ("evasiveness",        "evasive"),
    ("diplomacy",          "diplomatic"),
    ("assertiveness",      "assertive"),
    ("caution",            "cautious"),
    # Language / Style Variants
    ("chinese",            "chinese"),
    ("german",             "german"),
    ("shakespeare",        "shakespearean"),
    ("academia",           "academic"),
    ("journalism",         "journalistic"),
    ("jargon",             "jargon-heavy"),
    ("slang",              "slang"),
    ("archaism",           "old-fashioned"),
    # Alignment / Red Team Specific
    ("cynicism",           "cynical"),
    ("rebellion",          "rebellious"),
    ("manipulation",       "manipulative"),
    ("gaslighting",        "gaslighting"),
    ("sycophancy",         "sycophantic"),
    ("passive-aggression", "passive-aggressive"),
    ("conspiracy",         "conspiratorial"),
    ("paranoia",           "paranoid"),
    ("narcissism",         "narcissistic"),
    ("guilt-tripping",     "guilt-tripping"),
)

# Build lookup keyed by lowercased noun and adjective.
_LOOKUP: dict[str, tuple[str, str]] = {}
for _noun, _adj in _RAW_TRAITS:
    _LOOKUP[_noun.lower()] = (_noun, _adj)
    _LOOKUP[_adj.lower()]  = (_noun, _adj)


# Public API 

def resolve_trait(trait_name: str) -> TraitInfo:
    """Resolve a trait name (noun or adjective) to a TraitInfo.

    Falls back to TraitInfo(noun=name, adjective=name) for unknown traits,
    logging a warning. Add new entries to _RAW_TRAITS above if needed.
    """
    normalised = trait_name.strip().lower()
    if normalised in _LOOKUP:
        noun, adj = _LOOKUP[normalised]
        return TraitInfo(noun=noun, adjective=adj, raw_name=trait_name)

    log.warning(
        "Trait '%s' not found in static TRAIT_LOOKUP. "
        "Using it as both noun and adjective. "
        "Add it to pipeline_interface/traits.py _RAW_TRAITS if needed.",
        trait_name,
    )
    return TraitInfo(noun=trait_name, adjective=trait_name, raw_name=trait_name)


def trait_adjective(trait_name: str) -> str:
    """Return the adjective form of a trait (used in CSV filenames)."""
    return resolve_trait(trait_name).adjective


def trait_noun(trait_name: str) -> str:
    """Return the noun form of a trait."""
    return resolve_trait(trait_name).noun
