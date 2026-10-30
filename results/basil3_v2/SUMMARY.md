# BASIL-3 (Holographic Manifold World Model + Live Telemetry Halting + Bayesian Hypothesis Filtration) Study Summary

* **Split**: `val` (rule-family-disjoint procedural universes)
* **Independent training seeds**: `3`

## 1. Stopping-Regret Correlation & Anytime Halting Calibration (Live Telemetry Halting)

| Search Mode | Pre-Search ($t=0$) Regret Corr $r$ | Live In-Search ($t>0$) Regret Corr $r$ | All-Checkpoints Corr $r$ | Final-Move Agreement Acc |
|---|---:|---:|---:|---:|
| **BASIL-3 (Latent World Model)** | `+0.253` | **`+0.564`** | `+0.509` | `83.8%` (base `82.9%`) |
| **OracleSim (Real-Rules Search)** | `+0.070` | **`+0.438`** | `+0.389` | `86.0%` (base `84.9%`) |

## 2. Online Game-Play Performance Across Seeds (`val` split)

| Agent Architecture | CLI Spec | Opponent | Score | 95% Bootstrap CI | $\Delta$ vs `FastPolicy` | $\Delta$ 95% CI | Plan Rate | Exps/Move |
|---|---|---|---:|---|---:|---|---:|---:|
| FastPolicy (0 sims) | `morty` | `greedy` | **0.549** | `[0.3472, 0.75]` | — | — | `0.00` | `0.0` |
| BASIL-3 (Full Search) | `citadel@always` | `greedy` | **0.958** | `[0.875, 1.0]` | `+0.410` | `[0.2083, 0.6181]` | `1.00` | `30.2` |
| BASIL-3 + LTH-Regret (10%) | `citadel@meeseeks:0.1` | `greedy` | **0.639** | `[0.4375, 0.8333]` | `+0.090` | `[0.0, 0.1806]` | `0.28` | `2.4` |
| BASIL-3 + LTH-Regret (25%) | `citadel@meeseeks:0.25` | `greedy` | **0.743** | `[0.5972, 0.875]` | `+0.194` | `[0.0625, 0.3611]` | `0.59` | `8.6` |
| BASIL-3 + LTH-Regret (50%) | `citadel@meeseeks:0.5` | `greedy` | **0.882** | `[0.7569, 0.9792]` | `+0.333` | `[0.1458, 0.5417]` | `0.86` | `17.4` |
| BASIL-3 + LTH-Agree (25%) | `citadel@meeseeks_agree:0.25` | `greedy` | **0.785** | `[0.6319, 0.9167]` | `+0.236` | `[0.104, 0.3819]` | `0.70` | `7.2` |
| BASIL-3 + Pre-Regret (25%) | `citadel@meeseeks_pre:0.25` | `greedy` | **0.646** | `[0.4653, 0.8333]` | `+0.097` | `[0.0139, 0.2085]` | `0.23` | `7.3` |
| BASIL-3 + VisitShare (25%) | `citadel@share:0.25` | `greedy` | **0.854** | `[0.7153, 0.9653]` | `+0.306` | `[0.1458, 0.4583]` | `1.00` | `7.8` |
| BASIL-3 + UnreachStop (25%) | `citadel@unreach:0.25` | `greedy` | **0.861** | `[0.75, 0.9583]` | `+0.312` | `[0.1389, 0.5]` | `1.00` | `8.4` |
| BASIL-3 + Pre-VOC (25%) | `citadel@voc:0.25` | `greedy` | **0.757** | `[0.5625, 0.9375]` | `+0.208` | `[0.0833, 0.3472]` | `0.31` | `9.7` |
| BASIL-3 + Disagreement (25%) | `citadel@disagreement:0.25` | `greedy` | **0.667** | `[0.4792, 0.8472]` | `+0.118` | `[0.0, 0.25]` | `0.25` | `9.7` |
| BASIL-3 + Confidence (25%) | `citadel@confidence:0.25` | `greedy` | **0.625** | `[0.4026, 0.8333]` | `+0.076` | `[-0.0139, 0.1806]` | `0.23` | `9.1` |
| BASIL-3 + RandomGate (25%) | `citadel@random:0.25` | `greedy` | **0.660** | `[0.4583, 0.8472]` | `+0.111` | `[0.0, 0.2222]` | `0.24` | `8.0` |
| OracleSim (Full Search) | `oracle` | `greedy` | **0.924** | `[0.8264, 1.0]` | `+0.375` | `[0.1806, 0.5764]` | `1.00` | `30.6` |
| OracleSim + LTH-Regret (10%) | `oracle@meeseeks:0.1` | `greedy` | **0.639** | `[0.4583, 0.8264]` | `+0.090` | `[0.0069, 0.1944]` | `0.35` | `3.5` |
| OracleSim + LTH-Regret (25%) | `oracle@meeseeks:0.25` | `greedy` | **0.764** | `[0.6181, 0.8891]` | `+0.215` | `[0.0833, 0.3611]` | `0.69` | `8.5` |
| OracleSim + LTH-Regret (50%) | `oracle@meeseeks:0.5` | `greedy` | **0.889** | `[0.7778, 0.9792]` | `+0.340` | `[0.1667, 0.5208]` | `0.87` | `15.2` |
| FastPolicy (0 sims) | `morty` | `uct-100` | **0.424** | `[0.2083, 0.6391]` | — | — | `0.00` | `0.0` |
| BASIL-3 (Full Search) | `citadel@always` | `uct-100` | **0.771** | `[0.6389, 0.8889]` | `+0.347` | `[0.1944, 0.4861]` | `1.00` | `33.9` |
| BASIL-3 + LTH-Regret (10%) | `citadel@meeseeks:0.1` | `uct-100` | **0.500** | `[0.2913, 0.7222]` | `+0.076` | `[-0.0417, 0.2083]` | `0.26` | `2.4` |
| BASIL-3 + LTH-Regret (25%) | `citadel@meeseeks:0.25` | `uct-100` | **0.646** | `[0.4167, 0.8542]` | `+0.222` | `[0.0694, 0.3889]` | `0.57` | `8.2` |
| BASIL-3 + LTH-Regret (50%) | `citadel@meeseeks:0.5` | `uct-100` | **0.611** | `[0.375, 0.8125]` | `+0.188` | `[0.0347, 0.375]` | `0.85` | `17.6` |
| BASIL-3 + LTH-Agree (25%) | `citadel@meeseeks_agree:0.25` | `uct-100` | **0.583** | `[0.3819, 0.7778]` | `+0.160` | `[0.0139, 0.2917]` | `0.69` | `7.9` |
| BASIL-3 + Pre-Regret (25%) | `citadel@meeseeks_pre:0.25` | `uct-100` | **0.542** | `[0.3056, 0.7639]` | `+0.118` | `[0.0, 0.2292]` | `0.22` | `8.1` |
| BASIL-3 + VisitShare (25%) | `citadel@share:0.25` | `uct-100` | **0.597** | `[0.375, 0.8057]` | `+0.174` | `[0.0556, 0.2917]` | `1.00` | `7.8` |
| BASIL-3 + UnreachStop (25%) | `citadel@unreach:0.25` | `uct-100` | **0.653** | `[0.4653, 0.8333]` | `+0.229` | `[0.1181, 0.3403]` | `1.00` | `8.5` |
| BASIL-3 + Pre-VOC (25%) | `citadel@voc:0.25` | `uct-100` | **0.556** | `[0.3542, 0.7639]` | `+0.132` | `[0.0278, 0.2639]` | `0.26` | `9.4` |
| BASIL-3 + Disagreement (25%) | `citadel@disagreement:0.25` | `uct-100` | **0.493** | `[0.2917, 0.7083]` | `+0.069` | `[-0.0417, 0.1944]` | `0.21` | `8.8` |
| BASIL-3 + Confidence (25%) | `citadel@confidence:0.25` | `uct-100` | **0.479** | `[0.25, 0.7014]` | `+0.056` | `[-0.0417, 0.1736]` | `0.20` | `8.7` |
| BASIL-3 + RandomGate (25%) | `citadel@random:0.25` | `uct-100` | **0.528** | `[0.2917, 0.75]` | `+0.104` | `[-0.0139, 0.2361]` | `0.29` | `9.2` |
| OracleSim (Full Search) | `oracle` | `uct-100` | **0.708** | `[0.5417, 0.8611]` | `+0.285` | `[0.1389, 0.4306]` | `1.00` | `35.1` |
| OracleSim + LTH-Regret (10%) | `oracle@meeseeks:0.1` | `uct-100` | **0.486** | `[0.2639, 0.7083]` | `+0.062` | `[-0.0417, 0.1738]` | `0.34` | `3.1` |
| OracleSim + LTH-Regret (25%) | `oracle@meeseeks:0.25` | `uct-100` | **0.535** | `[0.3056, 0.7502]` | `+0.111` | `[-0.0488, 0.2778]` | `0.69` | `9.6` |
| OracleSim + LTH-Regret (50%) | `oracle@meeseeks:0.5` | `uct-100` | **0.618** | `[0.4304, 0.8056]` | `+0.194` | `[0.0417, 0.3264]` | `0.87` | `17.3` |

## 3. Hidden-Rule Regime & Exact Bayesian Hypothesis Filtration (`val` split)

* **Posterior Entropy Collapse**: `6.49` bits (Ep 1 start, `90.0` candidate universes) $\to$ **`1.69` bits** (Ep 6 end, `4.6` surviving candidates)

| Hidden-Rule Agent Architecture | CLI Mode | Ep 1–2 Score | Ep 5–6 Score | Mean Match Score | Mean Exps/Move |
|---|---|---:|---:|---:|---:|
| Blind FastPolicy (0 sims) | `blind_morty` | `0.555` | `0.583` | **0.565** | `0.0` |
| BHF + FastPolicy (0 sims) | `cfc_morty` | `0.646` | `0.583` | **0.604** | `0.0` |
| BHF + Posterior-Sampled Search + LTH (adaptive) | `cfc_meeseeks` | `0.556` | `0.674` | **0.648** | `19.6` |
| BHF + Posterior-Sampled Search (full) | `cfc_citadel` | `0.667` | `0.757` | **0.715** | `38.3` |
| Full-Info BASIL-3 (visible rules ceiling) | `full_info_citadel` | `0.917` | `0.945` | **0.928** | `30.8` |
