# BASIL: Budget-Aware State Imagination Learning

An agent that acts on instinct and stops to imagine the future (plan with a learned world model)
only when that is worth the compute.

The repository has two parts:

| Part | What it is | Entry points |
|---|---|---|
| Original BASIL, fixed | CNN encoder → Mamba-style latent world model → 5-head PPO policy → gated random-shooting planner, on TicTacToe and Connect4 | `train_*.py`, `evaluate_v2.py`, `run_experiments.py` |
| BASIL-3 (`basil3/`) | The **Citadel** architecture (ensemble latent world model, uncertainty-aware latent tree search, learned value-of-computation gate) on **Multiverse**, a generated family of games with rule-family-disjoint train/val/test splits | `python -m basil3.train`, `python -m basil3.evaluate` |

## Quick start

```bash
pip install torch numpy pyyaml tqdm pytest
python -m pytest -q                                  # 21 tests
python tests/smoke_original_pipeline.py              # fixed original pipeline, tiny settings, ~5 min
python -m basil3.train --config configs/basil3_smoke.yaml --out runs/smoke     # ~20 s
python -m basil3.evaluate --run runs/smoke --split val --universes 2 --openings 1 --sims 16 --opponents greedy
```

For a smaller Colab run, use `configs/basil3_colab.yaml` and the commands in [GUIDE.md](GUIDE.md).
The Colab config is for development; it does not make the results publication-ready.

`evaluate_v2.py` now reports paired score deltas against PPO-only and random-gated planning.
`basil3.evaluate` saves per-universe scores, including the cost of uncertainty probes and tree
search. Combine independently trained seeds with `python -m basil3.aggregate <eval files>`.

Training the original world model now splits complete trajectories before creating rollout windows
so correlated windows from one game cannot land in both training and validation.

The training protocol, comparisons and caveats are in [GUIDE.md](GUIDE.md).
