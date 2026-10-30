# BASIL-3: Budget-Aware State Imagination Learning

> **Know When to Imagine, Know Which World You Are In:** Adaptive Test-Time Compute and In-Context Rule Induction for Latent World-Model Planning

An agent deployed into an unfamiliar environment faces three coupled questions at every decision: **what are the governing rules of this environment**, **is deliberation worth the compute right now**, and **when has sufficient planning been performed**?

## Architecture Overview (`basil3/`)

| Component | Scientific Terminology | Core Mechanism | Module |
|---|---|---|---|
| **World Model & Fast Policy** | **BASIL-3** (`HolographicWorldModel`, `FastPolicy`, `DynamicsEnsemble`) | **Toroidal-Equivariant & Holographic Manifold World Model** with pinned 10-plane rule invariants, lossless $2^{-9}$-mantissa holographic board-state carriers, manifold-coupled imagination, and an adjoint dihedral-torus tactical horizon | [`basil3/nets.py`](basil3/nets.py) |
| **Anytime Compute Halting** | **Live Telemetry Halting (`LTH`)** | **Self-Supervised Optimal Stopping** trained on prefix search traces ($R_t = Q_T(a_T) - Q_T(a_t)$ and $A_t = \mathbb{I}[a_t = a_T]$) over 24 live tree telemetry features (visit concentration, $Q$-gaps, value drift, ensemble disagreement) | [`basil3/meeseeks.py`](basil3/meeseeks.py) |
| **In-Context Rule Induction** | **Hidden-Rule Regime + Exact Bayesian Hypothesis Filtration (`BHF`)** | **Exact Bayesian Rule Posterior $P(u \mid h_t)$** over hidden candidate rule universes ($\le 90$ hypotheses per geometry) paired with **Posterior-Sampled World-Model Search** | [`basil3/cronenberg.py`](basil3/cronenberg.py) |
| **Procedural Benchmark** | **Multiverse** | Rule-family-disjoint procedural grid games (tetromino win sets, gravity, horizontal/vertical boundary wraps, blocked cells) with cryptographic one-shot test lock | [`basil3/multiverse.py`](basil3/multiverse.py) |

## Summary of 3-Seed Results on Unseen Rule Families

| Split | Agent Architecture | vs. `greedy` Score ($\Delta$ vs. `FastPolicy`) | vs. `uct-100` Score ($\Delta$ vs. `FastPolicy`) | Mean Exps/Move (vs. `uct-100`) |
|---|---|---:|---:|---:|
| **`val` (Held-Out)** | `FastPolicy (0 sims)` | `0.549` (—) | `0.424` (—) | `0.0` |
| **`val` (Held-Out)** | `BASIL-3 + LTH-Regret (25%)` | **`0.743` (`+0.194`)** | **`0.646` (`+0.222`)** | **`8.2`** |
| **`val` (Held-Out)** | `BASIL-3 (Full Search)` | **`0.958` (`+0.410`)** | **`0.771` (`+0.347`)** | `33.9` |
| **`test` (One-Shot Preregistered)** | `FastPolicy (0 sims)` | `0.597` (—) | `0.486` (—) | `0.0` |
| **`test` (One-Shot Preregistered)** | `BASIL-3 + LTH-Agree (25%)` | **`0.764` (`+0.167`)** | **`0.681` (`+0.194`)** | **`10.5`** |
| **`test` (One-Shot Preregistered)** | `BASIL-3 (Full Search)` | **`0.938` (`+0.340`)** | **`0.722` (`+0.236`)** | `35.8` |

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest -q                                  # 25 unit & leak-check tests

# Smoke run (~25s on CPU: self-play + pre-search VOC gate + Live Telemetry Halting)
python -m basil3.train --config configs/basil3_smoke.yaml --out runs/smoke --stage all

# Evaluate fast policy, pre-search gates, and anytime halting rules on unseen rule families
python -m basil3.evaluate --run runs/smoke --split val --universes 4 --openings 1 --sims 16 \
  --agents morty,citadel@always,citadel@voc:0.25,citadel@meeseeks:0.25,oracle@meeseeks:0.25 \
  --opponents greedy

# Evaluate Hidden-Rule Regime with Exact Bayesian Hypothesis Filtration (BHF)
python -m basil3.cronenberg --run runs/smoke --split val --universes 4 --episodes 6 --sims 16
```

## Reproducing the Multi-Seed Study & Paper Figures

```bash
# 1. Train 3 independent seeds (self-play + VOC gate + Live Telemetry Halting stage)
for s in 0 1 2; do
  python -m basil3.train --config configs/basil3_v2.yaml --seed $s --out results/basil3_v2/seed_$s --stage all --resume
done

# 2. Evaluate all 3 seeds on held-out rule families, run hidden-rule matches,
#    aggregate with paired bootstrap CIs, and generate publication figures & LaTeX tables
python -m basil3.run_study_eval --root results/basil3_v2 --split val --universes 12 --openings 1 --sims 48 --opponents greedy uct-100
```

## Repository Structure

* [`basil3/`](basil3/) — BASIL-3 world model, Live Telemetry Halting (`meeseeks.py`), Hidden-Rule Regime & Bayesian Hypothesis Filtration (`cronenberg.py`), Multiverse benchmark, leak-checked evaluation & aggregation.
* [`paper/`](paper/) — LaTeX manuscript ([`paper/main.tex`](paper/main.tex)), bibliography ([`paper/references.bib`](paper/references.bib)), and generated figures/tables ([`paper/tables_generated.tex`](paper/tables_generated.tex)).
* [`results/basil3_v2/`](results/basil3_v2/) — 3-seed validation and preregistered test summaries ([`SUMMARY.md`](results/basil3_v2/SUMMARY.md), [`SUMMARY_TEST.md`](results/basil3_v2/SUMMARY_TEST.md)).
* [`configs/`](configs/) — Reproducible experiment configurations (`basil3_v2.yaml`, `basil3.yaml`, `basil3_colab.yaml`, `basil3_smoke.yaml`).
* [`GUIDE.md`](GUIDE.md) — Detailed protocol documentation, leak prevention guarantees, and legacy pipeline notes.
* [`notebooks/basil3_train_val.ipynb`](notebooks/basil3_train_val.ipynb) — Reproducible Colab notebook.
