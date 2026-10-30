# BASIL: Budget-Aware State Imagination Learning

> **Know When to Imagine, Know Which World You Are In:** Adaptive Test-Time Compute and In-Context Rule Induction for Latent World-Model Planning

An agent deployed into an unfamiliar environment faces three coupled questions at every move: **what are the laws of this world**, **is deliberation worth the compute right now**, and **when has enough thinking been done**?

## Architecture Overview (`basil3/`)

| Component | Code Name | Paper Mechanism | Module |
|---|---|---|---|
| **World Model & Fast Policy** | **Citadel v2** (`Morty` + `Council of Ricks`) | **Toroidal Portal-Aware & Translation-Equivariant World Model** with pinned 10-plane universe DNA, variable-boundary portal convolutions ($5\times 5$, $6\times 6$, $7\times 7$), translation-equivariant heads, and contrastive counterfactual action probes | [`basil3/nets.py`](basil3/nets.py) |
| **Anytime Compute Halting** | **Mr. Meeseeks** | **Self-Supervised Anytime Halting Gate** trained on prefix search traces ($R_t = Q_T(a_T) - Q_T(a_t)$ and $A_t = \mathbb{I}[a_t = a_T]$) over 24 live tree telemetry features (visit concentration, $Q$-gaps, value drift, ensemble disagreement) | [`basil3/meeseeks.py`](basil3/meeseeks.py) |
| **In-Context Rule Induction** | **Cronenberg Mode + Central Finite Curve** | **Exact Bayesian Rule Posterior $P(u \mid h_t)$** over hidden candidate rule universes ($\le 90$ hypotheses per geometry) paired with **Multiverse Thompson Search** | [`basil3/cronenberg.py`](basil3/cronenberg.py) |
| **Procedural Benchmark** | **Multiverse** | Rule-family-disjoint procedural grid games (tetromino win sets, gravity, horizontal/vertical portals, blocked cells) with cryptographic one-shot test lock | [`basil3/multiverse.py`](basil3/multiverse.py) |

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest -q                                  # 25 unit & leak-check tests

# Smoke run (~25s on CPU: self-play + pre-search VOC gate + Mr. Meeseeks anytime halting)
python -m basil3.train --config configs/basil3_smoke.yaml --out runs/smoke --stage all

# Evaluate fast policy, pre-search gates, and anytime halting rules on unseen rule families
python -m basil3.evaluate --run runs/smoke --split val --universes 4 --openings 1 --sims 16 \
  --agents morty,citadel@always,citadel@voc:0.25,citadel@meeseeks:0.25,oracle@meeseeks:0.25 \
  --opponents greedy

# Evaluate Cronenberg Mode (hidden rules) with Central Finite Curve Bayesian rule induction
python -m basil3.cronenberg --run runs/smoke --split val --universes 4 --episodes 6 --sims 16
```

## Reproducing the Multi-Seed Study & Paper Figures

```bash
# 1. Train 3 independent seeds (self-play + VOC gate + Mr. Meeseeks stage)
for s in 0 1 2; do
  python -m basil3.train --config configs/basil3_v2.yaml --seed $s --out results/basil3_v2/seed_$s --stage all --resume
done

# 2. Evaluate all 3 seeds on held-out rule families, run Cronenberg hidden-rule matches,
#    aggregate with paired bootstrap CIs, and generate publication figures & LaTeX tables
python -m basil3.run_study_eval --root results/basil3_v2 --split val --universes 12 --openings 2 --sims 48
```

## Repository Structure

* [`basil3/`](basil3/) — BASIL-3 (Citadel v2, Mr. Meeseeks, Cronenberg Mode & Central Finite Curve, Multiverse benchmark, leak-checked evaluation & aggregation).
* [`paper/`](paper/) — LaTeX manuscript ([`paper/main.tex`](paper/main.tex)), bibliography ([`paper/references.bib`](paper/references.bib)), and generated figures/tables.
* [`configs/`](configs/) — Reproducible experiment configurations (`basil3_v2.yaml`, `basil3.yaml`, `basil3_colab.yaml`, `basil3_smoke.yaml`).
* [`GUIDE.md`](GUIDE.md) — Detailed protocol documentation, leak prevention guarantees, and legacy pipeline notes.
* [`notebooks/basil3_train_val.ipynb`](notebooks/basil3_train_val.ipynb) — Reproducible Colab notebook.
