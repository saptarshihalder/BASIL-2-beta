# BASIL guide: the fixes, BASIL-3, and how to run a study you can publish

This guide has six parts:

1. What was wrong in the original code and how each problem is fixed
2. How to rerun the original BASIL honestly
3. The BASIL-3 "Citadel" architecture
4. The Multiverse benchmark, and why it is unbiased and leak-proof
5. A step-by-step runbook from smoke test to the one-shot test run
6. The experiments a paper needs, and what each outcome lets you claim

Run every command from the repository root.

---

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install torch numpy pyyaml tqdm pytest
python -m pytest -q                                       # expect: 21 passed
```

* **GPU:** optional.
  * The original pipeline trains small networks; a GPU helps only with world-model training.
  * BASIL-3 search is Python-bound and runs one position at a time. Use CPU cores through `workers:` in the config, plus a GPU for the training steps if you have one.
* **Checkpoints:** all of them go to `checkpoints/` (original pipeline) or `runs/<name>/` (BASIL-3). Both are git-ignored.

---

## 1. What was wrong, and the fix for each

| # | Problem in the original code | Effect | Fix | Where | Test |
|---|---|---|---|---|---|
| 1 | The planner added up imagined rewards over both players' moves with no sign flip. The environment gives +1 to *whoever* wins, and the world model learned exactly that. | Opponent wins counted as the agent's wins, so more planning gave worse play (thinking-only 71.5% vs 88.4% PPO-only). | Imagined step t is scored `+r` when t is even (our move) and `−r` when t is odd (the opponent's). | `models/planning.py` | `tests/test_original_fixes.py::test_planner_counts_opponent_wins_against_us` |
| 2 | `train_ppo_ensemble.py` trained on TicTacToe only, and every evaluation script loaded that checkpoint for Connect4 too. | The Connect4 numbers came from a TicTacToe policy, and matched "always drop in the centre column" (87.2%). | `train_ppo.py --games tictactoe,connect4` writes `checkpoints/policy_ensemble_<game>.pt`; evaluation refuses a checkpoint trained on another game. | `train_ppo.py`, `utils/common.py`, `evaluate_v2.py` | smoke run |
| 3 | The 5 policy heads were trained only through their averaged logits. | Every head got an identical gradient, so the differences between heads stayed frozen at their random initial values. The "disagreement" (JSD) was initialisation noise, not uncertainty. | Bootstrapped heads: each head's PPO loss uses its own random half of every batch. | `train_ppo.py::ppo_update`, `models/policy.py` | `test_bootstrapped_heads_actually_diverge` |
| 4 | The JSD threshold was 1.5, but JSD for 5 heads can't exceed ln 5 ≈ 1.61 (ln 2 = 0.69 with 2 legal moves). | The JSD trigger almost never fired; planning was really triggered by "max probability < 0.3". | JSD is normalised to [0, 1]. Thresholds are calibrated on separate games to hit a target planning rate, and compared with a random gate and a confidence gate at the same rate. | `utils/uncertainty.py`, `evaluate_v2.py` | `test_jsd_is_normalised` |
| 5 | Imagined moves were drawn from all 9 action slots in Connect4 (which has 7 columns); `run_experiments.py` gave the planner no policy at all. | Imagination used moves that do not exist. | The planner gets `num_actions` per game. (This world model still has no legality head, so deeper imagined moves can be illegal; BASIL-3 adds one.) | `models/planning.py` | `test_planner_samples_only_this_games_actions` |
| 6 | `run_experiments.py` unpacked 4 values from a function that returns 5. | The script crashes on the current code, so `all_experiments.json` came from an older version. | Unpacking fixed. Experiment 3 moved to `evaluate_v2.py`. | `run_experiments.py` | smoke run |
| 7 | The agent always moved first; the only opponent was random. | First-move advantage inflated every number; a 3-rule heuristic beats random 95.7% (TTT) / 99.1% (C4). | Both seats; opponents random, greedy (win/block/centre) and perfect minimax (TicTacToe). | `evaluate_v2.py`, `utils/common.py` | smoke run |
| 8 | No seeds anywhere; result files came from different checkpoints (PPO-only TTT ranged 83.0–93.4%). | Not reproducible; differences smaller than run-to-run noise. | `set_seed` in every script. Evaluation games are paired: game *i* uses the same random numbers in every mode. Wilson 95% intervals. | all scripts | — |
| 9 | World models compared at different sizes (Mamba 275k vs GRU 151k parameters), one run each. | The "Mamba is best" claim was confounded by size. | Baseline hidden sizes chosen to match Mamba's parameter count (±3%); 3 seeds, mean ± std. | `run_experiments.py` | smoke run |
| 10 | `envs/Connect4.py` next to `envs/connect4.py` (and `Generate_data.py`/`generate_data.py`). | Checkouts break on macOS/Windows (case-insensitive file systems). | Uppercase duplicates deleted (nothing imported them). | — | — |
| 11 | World-model rollout windows were randomly split after extraction. | Overlapping windows from the same game appeared in train and validation. | Split complete trajectories first, then build windows separately. | `train_world_model_multistep.py` | `test_windows_do_not_cross_games` |
| 12 | Evaluation had unpaired win-rate intervals and omitted the cost of checking whether to plan. | Small differences could be noise; equal planning rates did not imply equal cost. | Paired deltas for the original pipeline, and measured ensemble probes plus search calls for BASIL-3. | `evaluate_v2.py`, `basil3/evaluate.py` | `test_paired_interval_is_on_differences`, `test_probes_and_search_calls_are_counted` |

`main_basil.py` and `run_final_comparison.py` still exist but only call `evaluate_v2.py`, so old commands keep working without the old bugs.

---

## 2. Rerunning the original BASIL honestly

```bash
python train_autoencoder.py                      # checkpoints/autoencoder.pt (edit NUM_STATES_PER_GAME=50000, EPOCHS=100 for the paper)
python train_ppo.py --games tictactoe,connect4 --episodes 3000 --seed 0 --opponent mixed
python train_world_model_multistep.py            # checkpoints/world_model.pt
python evaluate_v2.py --games tictactoe,connect4 --opponents random,greedy,minimax --n 500 --plan-rate 0.2
python run_experiments.py                        # parameter-matched world-model comparison, 3 seeds
```

`evaluate_v2.py` prints one row per (game, opponent, mode):

* win rate with its Wilson 95% interval, plus loss rate;
* the **measured** planning rate;
* world-model calls per game;
* paired score differences and bootstrap intervals vs PPO-only and random gating, when those modes are included.

Read the table in this order:

1. **Is the fast policy any good?** Compare `ppo_only` with the hand-written greedy opponent. If PPO-only can't beat greedy, more planning experiments are pointless; train longer or with `--opponent mixed`.
2. **Does planning help at all, now that the sign is fixed?** Compare `thinking_only` with `ppo_only`.
3. **Does the ensemble signal matter?** Compare `adaptive` (JSD gate) with `confidence_gate` and `random_gate`. They are calibrated to the same rate, so check that their measured planning rates are close. If `adaptive` doesn't beat `random_gate` at the same rate, the JSD signal adds nothing, and that is a reportable finding.

Calibration uses games against a random opponent, so the measured rate against greedy or minimax drifts from the target. For a fair comparison, sweep `--plan-rate` over 0.05, 0.1, 0.2, 0.4. Then plot win rate against *measured* planning rate, one curve per gate, and compare the curves rather than single points.

The original pipeline can't escape two limits:

* its world model has no legality head and was trained on random play only;
* TicTacToe and Connect4 have known rules and strong hand-written players, so a reviewer will ask why anyone would plan with a learned model there.

BASIL-3 addresses both.

---

## 3. BASIL-3: the Citadel architecture

```
                     ┌──────────────────────── Morty (fast path) ───────────────────────┐
 observation ──h──►  latent s ──f──► move probabilities, value, legal-move map         │
 (14 planes, from                │                                                     │
  the mover's view)              │  gate features: Morty's confidence + one imagined   │
                                 │  step by every Rick on Morty's top 3 moves          │
                                 ▼                                                     │
                     Meeseeks gate: predicted value of computation ≥ threshold ? ─ no ─┘──► play Morty's move
                                 │ yes
                                 ▼
             Portal search (latent MCTS inside the Council's imagination)
               every expansion asks all K Ricks  g_k(s, a) → (s'_k, P_k(a wins))
               child latent = one random Rick's s' (Thompson sampling over models)
               edge value  q_k = P_k(win) + (1 − P_k(win))·(−v(s'_k))   (negamax)
               backed-up   mean_k q_k − β · std_k q_k   (distrust lines the Ricks disagree on)
               illegal imagined moves masked by the legal-move map; no legal move ⇒ draw
                                 │
                                 ▼
                           play the most-visited move
```

Each component, and the failure it addresses:

* **Canonical perspective.** Every observation is encoded from the point of view of the player to move, and every value is "my expected result". Backups flip sign at every ply, so bug #1 cannot reappear. A regression test checks that search blocks an opponent's immediate win.
* **Morty: representation `h` plus prediction `f`.**
  * Outputs move probabilities, a value, and a **legal-move map** (fixes #5 for imagined positions).
  * Trained by expert iteration: it imitates the visit counts of search on the real rules, and predicts game outcomes.
* **Council of Ricks: K separate dynamics networks.**
  * Each Rick learns from its own bootstrap half of the positions.
  * Each has a fixed random "prior" network added to its win head (randomised priors).
  * So the Ricks agree where the data pins down the rules and disagree where it doesn't. That disagreement is a real epistemic-uncertainty signal (fixes #3).
  * Training targets, unrolled for `unroll` steps along real games:
    * did this move win;
    * which moves are legal next;
    * a latent that matches the encoding of the real next position (SimSiam-style consistency).
* **Portal search: latent MCTS that never touches the real rules after the root.**
  * Sampling one Rick per subtree keeps the search from exploiting a single model's mistakes.
  * The pessimism term β·std keeps it out of lines the Council disagrees about.
* **Meeseeks gate.** Planning is summoned for one move and disappears afterwards.
  * Features are cheap: Morty's confidence, plus one imagined step by every Rick.
  * The VOC model learns a **reference-search estimate** of the value of computation. The reference simulates true transitions but uses Morty's value head at search leaves, so the label is approximate.
  * Labels can be negative (imagination made the move worse), so the gate learns *when imagination lies*.
  * Thresholds are set on validation positions so that every gate (VOC, disagreement, confidence, random) plans on the same share of moves. That makes "budget-aware" literal and the comparison fair.
  * For deployment, `Gate(..., online_budget=B)` tracks the (1−B) quantile online (Robbins–Monro), so the long-run planning rate stays at B.
* **Oracle reference.** The same search with Morty's priors, but on the real rules: what Citadel would do with a perfect world model. The gap between Oracle and Citadel@always is the cost of model error.

**Prior work: be explicit about it in the paper.** None of the parts is new on its own:

* latent planning: MuZero, EfficientZero (consistency loss);
* ensembles and randomised priors: Osband et al. 2016/2018;
* model-ensemble planning: PETS;
* uncertainty-penalised model-based RL: MOPO;
* expert iteration: Anthony et al. 2017;
* value of computation / metareasoning: Russell & Wefald 1991, Hay et al. 2012;
* learning when to imagine: Hamrick et al. 2017, *Metacontrol for Adaptive Imagination-Based Optimization*.

The defensible contribution is the combination, measured properly:

* a gate trained on the *real* value of computation and compared at equal compute;
* planning that is *aware of model uncertainty*;
* evaluation on *unseen rule families* under a pre-registered protocol.

The Rick and Morty names are code aliases. In the paper, use neutral names:

* Morty → fast policy-value network
* Council of Ricks → bootstrapped ensemble of latent dynamics models
* Portal search → uncertainty-aware latent MCTS
* Meeseeks gate → learned value-of-computation gate

---

## 4. Multiverse: the custom game family

Every *universe* is a different two-player game, generated from an integer seed. Rules vary along five axes:

| Axis | Values |
|---|---|
| Board | H × W with H, W ∈ {5, 6, 7} (padded to 7 × 7; action = cell index r·7 + c) |
| Gravity | on (pieces fall to the lowest free cell of a column) / off |
| Portals | left–right edges wrap (30%); top–bottom edges wrap (30%, only without gravity) |
| Blocked cells | 0–3 random cells nobody can use |
| Winning shapes | 1–2 of the free tetrominoes I, O, T, S, L, in any rotation or mirror image |

* **Observation:** 14 planes. Own stones, opponent stones, empty, blocked and on-board come first. Then come one plane each for gravity and the two portal directions, one for each of the 5 shape flags, and one for legal moves. The rules are part of the observation, so the agent can generalise to games it has never seen.
* **Playability filter:** a universe is used only if (a) it has at least 8 winning placements, and (b) in 30 greedy-vs-greedy games:
  * the first player wins 25–80% of the time;
  * at most 60% are draws;
  * games average at least 7 moves.

  Only the blocked cells are redrawn when a candidate fails.

### Why this is unbiased and leak-proof

| # | Guard | How it is enforced |
|---|---|---|
| 1 | No human strategy can leak in | The games are new, with no opening book, heuristic or published analysis. The agent learns only from self-play. |
| 2 | Test games are unseen *rule combinations* | Splits are decided by SHA-256 of the canonical family (size, gravity, portals, shapes; blocked cells excluded): 70% train / 15% val / 15% test. A test universe never shares its rule combination with any training universe. |
| 3 | Symmetric twins can't straddle splits | A gravity-free board and its transpose (portals swapped) map to the same family key. Mirror images don't change the key. |
| 4 | Splits can't drift | `GENERATOR_VERSION` is part of the hash. The checkpoint stores it, and evaluation refuses a mismatch. |
| 5 | Every run is checked, not trusted | `evaluate.py` asserts that every evaluated universe is in the requested split, that its family is **not** in the checkpoint's stored training families, and (on test) that no calibration universe is used. |
| 6 | No first-move or opening luck | Every agent plays the same 2-ply random openings of each universe, from **both seats**. The opponent's randomness comes from the same stream for every agent. |
| 7 | Imagination agents can't peek at the real rules | Morty and Citadel receive only (observation, legal-move mask). Only simulator agents (greedy, `uct-N`, oracle) receive a copy of the game. |
| 8 | No tuning on the evaluation data | Gate thresholds are fit on validation *positions* drawn from openings 1,000,000 and up; evaluation uses openings 0..N. VOC labels use train universes only. |
| 9 | The test set is spent once | `--preregister` writes `prereg_<hash>.json`, where the hash covers the checkpoint bytes, the gate bytes and every argument. The test run starts only if that file exists and `test_<hash>.done` does not. |
| 10 | Honest uncertainty | Scores come with 95% intervals from a bootstrap over universes (games inside a universe are correlated). Every agent is compared with the reference agent on the same universes (paired differences). |
| 11 | Compute is visible | Each row reports the share of moves planned, search expansions per move and milliseconds per move. Never compare methods at different compute without saying so. |

---

## 5. Runbook

### Step 0: tests (30 s)

```bash
python -m pytest -q
```

The tests cover:

* rules: line wins, gravity landing, wins through portals plus a no-portal control, blocked cells, consistent game endings;
* the mover's-view encoding;
* search taking a win and **blocking a loss** (the original bug class);
* disjoint, symmetry-safe splits and determinism;
* imagination agents only playing legal moves;
* the four regression tests for the original pipeline.

### Step 1: smoke runs (≈ 6 minutes total)

```bash
python tests/smoke_original_pipeline.py          # temp dir; your checkpoints are untouched
python -m basil3.train --config configs/basil3_smoke.yaml --out runs/smoke
python -m basil3.evaluate --run runs/smoke --split val --universes 2 --openings 1 --sims 16 --opponents greedy,uct-16
```

### Step 2: a CPU run to check that learning happens (≈ 40–60 min on 2 cores)

```bash
python -m basil3.train --config configs/basil3_cpu.yaml --out runs/cpu
python -m basil3.evaluate --run runs/cpu --split val --universes 12 --openings 2 --sims 48 \
    --opponents random,greedy,uct-100
```

Every training iteration prints one JSON line. The health signs to look for:

* `value`, `legal` and `win` losses fall;
* `consistency` heads towards −1;
* `vs_random` rises towards 1 (it's measured on train universes and is noisy, with 8 games).

**What this CPU run produced** (2 cores; about 50 min of training and 14 min of evaluation; val split):

* 12 universes from rule families never seen in training, × 2 openings × 2 seats, so 48 games per cell;
* search budget of 48 simulations;
* score = wins + ½ draws.

| Agent | vs random | vs greedy | vs uct-100 | ms / move |
|---|---|---|---|---|
| Morty (no search) | 0.65 | 0.08 | 0.08 | 2 |
| Citadel, always plans | 0.68 | 0.19 | 0.08 | 330 |
| Citadel, VOC gate 25% | 0.69 | 0.14 | 0.09 | 91 |
| Citadel, disagreement gate 25% | 0.70 | 0.09 | 0.13 | 112 |
| Citadel, random gate 25% | 0.63 | 0.10 | 0.06 | 98 |
| Oracle (search on the real rules) | **0.92** | **0.41** | **0.29** | 85 |

Full numbers with 95% intervals, the training history and gate statistics are in `results/cpu_demo/`. What this shows:

* **The pipeline works on unseen rule families.**
  * There were 0 illegal moves in 1,008 games.
  * The legality loss fell from 0.36 to 0.017.
  * The gates' measured planning rates (0.21–0.33) stayed close to their 0.25 budget.
* **Planning is worth a lot in these games:** search on the real rules adds +0.21 to +0.32 over Morty, with intervals that exclude 0.
* **At this scale the learned world model is the bottleneck.** Search inside the Council's imagination adds at most +0.10, and the interval includes 0.
* **The VOC gate hasn't learned anything yet.** Its held-out correlation with the reference-search gain was −0.004, which makes it no better than the random gate. The paper-scale run decides whether that changes.
* **Morty is undertrained.** It still loses to the 3-rule greedy player; 30 iterations is about 1% of the paper settings, so ignore the absolute scores.
* **Imagination is slower per move than real-rules search here** (4 Ricks per expansion). Report compute as expansions or network calls, and say clearly that the setting being modelled has no simulator at decision time.

These are sanity numbers on the val split, not results.

### Step 3: paper-scale training (3–5 seeds)

```bash
for s in 0 1 2; do
  sed "s/^seed: 0/seed: $s/" configs/basil3.yaml > configs/basil3_s$s.yaml
  python -m basil3.train --config configs/basil3_s$s.yaml --out runs/full_s$s --stage selfplay
done
```

* Set `workers:` in the config to your number of CPU cores. Self-play runs in parallel processes and the training steps use `device:`.
* Budget roughly 10–20 hours per seed at the paper settings. Run seeds on separate machines or in parallel if you can.
* If a run is interrupted, add `--resume` to the same command. It restores weights, optimiser state and history, but rebuilds the replay buffer. A resumed run is therefore **not bit-for-bit equivalent** to an uninterrupted run; record the interruption. Checkpoints are written to a temp file and then renamed, so a crash never leaves a broken `citadel.pt`.
* To cut time:
  * lower `iterations` or `games_per_iter` first;
  * keep `sims` at 64 or more, or the policy targets get noisy.

### Step 4: train and calibrate the gate (per seed)

```bash
python -m basil3.train --config configs/basil3_s0.yaml --out runs/full_s0 --stage gate
```

* The script prints how often search changed Morty's move, the mean **reference-search estimated** gain, and the VOC model's held-out correlation with that estimate. The reference uses true game transitions but a learned leaf value, so it is not a ground-truth optimality label.
* Low held-out correlation may mean uninformative gate features, a weak planner, or noisy reference-search targets. Try checking target repeatability before increasing `ref_sims`.

### Step 5: development on the val split (as often as you like)

```bash
python -m basil3.evaluate --run runs/full_s0 --split val --universes 40 --openings 4 --sims 64 \
  --agents morty,citadel@always,citadel@voc:0.1,citadel@voc:0.25,citadel@disagreement:0.25,citadel@confidence:0.25,citadel@random:0.25,oracle \
  --opponents greedy,uct-200,uct-800
```

* Choose β (`--beta`), search size and budgets here.
* **Nothing about the method may change after Step 6.**

### Step 6: freeze and preregister

```bash
python -m basil3.evaluate --run runs/full_s0 --split test --universes 60 --openings 4 --sims 64 \
  --agents <final list> --opponents greedy,uct-200,uct-800 --preregister
git add -f runs/full_s0/prereg_*.json && git commit -m "Preregister BASIL-3 test run"   # runs/ is git-ignored, hence -f
```

The hash covers the checkpoint, gate, evaluator arguments and all Python sources in `basil3/`. Commit the preregistration before running the test; keep the exact code revision with it.

### Step 7: the one-shot test (identical command, without `--preregister`)

```bash
python -m basil3.evaluate --run runs/full_s0 --split test --universes 60 --openings 4 --sims 64 \
  --agents <final list> --opponents greedy,uct-200,uct-800
```

* Do this once per seed.
* A second attempt with the same hash is refused.
* A different hash means you changed something after preregistration; say so in the paper.

### Step 8: analysis

* **Main table** (test split, mean over seeds): score with 95% interval and Δ vs Morty with 95% interval for every agent × opponent, plus planned share, expansions per move and ms per move.
* **Pareto figure:** score (y) against measured imagination calls per move (x), with expansions and wall time as separate views.
  * Plot one curve per gate type (VOC, disagreement, confidence, random) over budgets 0.05–0.5. Model calls include the gate's preliminary one-step probes (when present) plus all ensemble rollouts in tree search.
  * Add Morty at x = 0, plus Citadel@always and Oracle.
  * The headline claim lives or dies on whether the VOC curve sits above the random-gate curve at comparable measured compute.

---


### Colab development run

`configs/basil3_colab.yaml` uses fewer workers, iterations and rollouts than the full config.
Run this first to measure runtime and check that the approach learns something, not to report
paper results:

```bash
pip install -r requirements.txt
python -m basil3.train --config configs/basil3_colab.yaml --out runs/colab_s0
python -m basil3.evaluate --run runs/colab_s0 --split val --universes 12 --openings 2 \
  --agents morty,citadel@always,citadel@voc:0.25,citadel@disagreement:0.25,citadel@confidence:0.25,citadel@random:0.25 \
  --opponents greedy --sims 40
```

On Colab, select a GPU runtime first. Search is largely CPU-bound, so a GPU alone will not
make the simulation cheap. Record wall time and repeat with independent training seeds. No
run-time, win-rate or journal acceptance is guaranteed by the config.

Once the experiment protocol is frozen and each seed has its own test JSON, aggregate them:

```bash
python -m basil3.aggregate runs/full_s*/eval_test_*.json --out results/test_summary.json
```

The aggregator resamples both training seeds and the shared held-out universes. It rejects
incompatible evaluations and duplicate recorded training seeds. Its confidence intervals refer
to this benchmark distribution, not arbitrary future games.

## 6. Experiments for the paper, and what each outcome lets you claim

| Question | Runs | Config / flag |
|---|---|---|
| Does gated imagination beat instinct at equal compute? | Morty vs Citadel@{voc, random, confidence, disagreement} at budgets 0.05–0.5 | `--agents` |
| What does model error cost? | Citadel@always vs Oracle at the same `--sims` | `oracle` |
| Is the ensemble needed? | `n_ricks: 1` vs 3 vs 5 | config |
| Do randomised priors / bootstrap matter? | `prior_scale: 0`; `bootstrap: false` | config |
| Does pessimism help? | `--beta 0`, 0.5, 1.0 | `evaluate --beta` |
| Longer imagination? | `unroll: 1` vs 3 vs 5 | config |
| Generalisation gap | the same agents on train universes vs val vs test | compare runs |

What each outcome lets you claim:

* **VOC above random-gate at equal compute on test, with the interval for Δ excluding 0:** "A gate trained on the real value of computation lets a learned-model planner spend X% of the compute for Y% of the gain on games with unseen rules." That is the paper.
* **Citadel@always worse than Morty, but the VOC gate no worse than Morty:** "Imagination with a learned model hurts on unseen rules, and a learned gate detects when." This is still a clean, honest result.
* **Oracle much better than Citadel@always:** imperfect learned dynamics may be a bottleneck (the oracle also uses real-rule search, so check equal search budgets). Report the gap and the ablations that narrow it.
* **No gate beats the random gate:** uncertainty signals aren't informative about the value of computation here. This is a negative result, still publishable at a workshop if the protocol is this careful.

Never report a val-split number as a test result, and always state seeds and compute.

---

## 7. Sanity checks: numbers that mean something is broken

* `illegal_moves` > 0 for any agent → bug; stop and investigate.
* Oracle scores below Morty → search or value bug (the oracle uses the true rules).
* Morty's score vs `random` on val below 0.7 after full training → undertrained, or broken encoding.
* A gate's measured `plan_rate` far from its budget → the val positions used for calibration don't look like the evaluation positions. Recalibrate with more `calib_positions`, or report measured rates.
* Score intervals wider than ±0.05 → add universes or openings before drawing conclusions.

## 8. File map

```
basil3/multiverse.py   game family, generator, playability filter, splits, openings   (numpy only)
basil3/nets.py         Morty (h, f), Council of Ricks, consistency loss
basil3/search.py       PUCT search; real-rules adapter (with/without Morty); latent Portal adapter
basil3/agents.py       gate features, VOC model, gates, all players
basil3/train.py        self-play training, gate labelling, VOC fit, validation calibration
basil3/evaluate.py     leak-checked, paired, preregistered evaluation
basil3/tests/          rules, sign, splits, determinism, legality tests
configs/basil3*.yaml   smoke / cpu / paper settings
evaluate_v2.py         fixed evaluation for the original pipeline
utils/common.py        seeds, per-game checkpoints, greedy and minimax opponents
tests/                 regression tests and smoke run for the original pipeline
```
