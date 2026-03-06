# IP-Cross-Trait: Trait Geometry & Cross-Trait Suppression in Inoculation Prompting

**Research question:** Does geometric proximity of trait representations in activation space predict how much "collateral damage" Inoculation Prompting inflicts on non-targeted traits?

## Background

**Inoculation Prompting (IP)** ([Wichers et al., 2025](https://arxiv.org/abs/2510.05024); [Tan et al., 2025](https://arxiv.org/abs/2510.04340)) suppresses an undesired trait during fine-tuning by prepending a trait-eliciting system prompt at train-time and removing it at test-time. The model learns the positive trait normally but suppresses the negative one.

**The conditionalization confound** ([Riché & Rolf, 2026](https://www.lesswrong.com/posts/znW7FmyF2HX9x29rA/conditionalization-confounds-inoculation-prompting-results)): Fixed IP (R1) can cause indiscriminate suppression — the model learns to condition *all* trait expression on the inoculation prompt, so removing it suppresses everything. Rephrased IP (R8192, 8192 prompt variants) breaks this confound.

This project tests whether **trait geometry** — cosine similarity of trait activation vectors in the model's residual stream — provides a complementary explanation for cross-trait suppression, independent of the conditionalization confound.


## Setup

**Model:** `Qwen/Qwen2.5-7B-Instruct`
**5 traits:** French, ALL-CAPS, Playful, Poetic, Skeptical
**Dataset:** [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) — first user turn of sampled conversations

**9 models tested:**

| Key | Description |
|-----|-------------|
| `base` | Unmodified Qwen2.5-7B-Instruct |
| `ft_french_allcaps` | FT on French + ALL-CAPS (no IP) |
| `ft_french_playful` | FT on French + Playful (no IP) |
| `ft_poetic_skeptical` | FT on Poetic + Skeptical (no IP) |
| `ip_french_allcaps_r1` | IP: suppress ALL-CAPS, fixed prompt |
| `ip_french_allcaps_r8192` | IP: suppress ALL-CAPS, 8192 rephrasings |
| `ip_french_playful` | IP: suppress Playful, fixed prompt |
| `ip_poetic_skeptical_r1` | IP: suppress Skeptical, fixed prompt |
| `ip_poetic_skeptical_r8192` | IP: suppress Skeptical, 8192 rephrasings |

## Pipeline

### Phase 1 — Representation Geometry

**`notebooks/phase1_extraction.ipynb`** — script version in `scripts/run_phase1a.py` and `scripts/run_phase1b.py`

- **Phase 1A:** For each of 5 traits × 4 models (base + 3 FT baselines): 30 contrastive response pairs (trait-eliciting vs. neutral prompt), layer-16 activations averaged over response tokens. `trait_vector = mean(pos_activations) - mean(neg_activations)`. Outputs 5×5 cosine similarity matrices.
- **Phase 1B:** For 13 prompts (3 IP + 4 irrelevant + 6 rephrased IP), extract last-prompt-token activations at layer 16, compute 13×5 prompt–trait cosine similarity matrix.

Results: `results/phase1/vectors/` (`.pt` files)

### Phase 2 — Behavioral Scoring 

**`notebooks/phase2a_generation.ipynb`** 
**`notebooks/phase2b_scoring.ipynb`** 

- **Phase 2A:** Each of 9 models generates responses to 300 UltraChat queries (neutral system prompt, no trait elicitation). Results: `results/phase2/responses/{model_key}_responses.jsonl`
- **Phase 2B:** GPT-5-mini scores each response on all 5 traits (0–100 scale, structured output). 13,500 total API calls. Results: `results/phase2/scores/{model_key}_scores.jsonl` + `summary.csv`

### Phase 3 — Correlation Analysis

**`notebooks/phase3_analysis.ipynb`** 

The main analysis notebook. **Start here** for results. First cell contains the full write-up: TL;DR findings, experimental setup, all prompts, scoring rubric, and summary results table.

Covers:
- Collateral damage vs. intended suppression per IP model
- R1 (fixed) vs. R8192 (rephrased) comparison
- Trait-trait similarity vs. collateral damage scatter
- Prompt–trait alignment ratios and heatmaps
- PCA of trait + prompt vector space

Results (figures): `results/phase3/`


## Findings

1. **Fixed IP (R1) causes indiscriminate suppression** of both traits — conditionalization confound confirmed.
2. **Rephrased IP (R8192) amplifies the positive trait above the FT baseline** (French: 27→60, Poetic: 16→79) while keeping the negative trait suppressed.
3. **Co-amplification tracks geometry** — playful (sim=0.66 with poetic) is boosted alongside poetic in R8192.
4. **Geometry hypothesis inconclusive** at n=3 trait pairs — conditionalization confound dominates R1 results.
5. **FT-model similarity** better predicts collateral damage than base-model similarity (r=0.62 vs. 0.26).

