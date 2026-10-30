# BASIL-3 (Holographic Manifold World Model + Live Telemetry Halting + Bayesian Hypothesis Filtration) Study Summary

* **Split**: `test` (rule-family-disjoint procedural universes)
* **Independent training seeds**: `3`

## 1. Stopping-Regret Correlation & Anytime Halting Calibration (Live Telemetry Halting)

| Search Mode | Pre-Search ($t=0$) Regret Corr $r$ | Live In-Search ($t>0$) Regret Corr $r$ | All-Checkpoints Corr $r$ | Final-Move Agreement Acc |
|---|---:|---:|---:|---:|
| **BASIL-3 (Latent World Model)** | `+0.253` | **`+0.564`** | `+0.509` | `83.8%` (base `82.9%`) |
| **OracleSim (Real-Rules Search)** | `+0.070` | **`+0.438`** | `+0.389` | `86.0%` (base `84.9%`) |

## 2. Online Game-Play Performance Across Seeds (`test` split)

| Agent Architecture | CLI Spec | Opponent | Score | 95% Bootstrap CI | $\Delta$ vs `FastPolicy` | $\Delta$ 95% CI | Plan Rate | Exps/Move |
|---|---|---|---:|---|---:|---|---:|---:|
| FastPolicy (0 sims) | `morty` | `greedy` | **0.597** | `[0.4167, 0.7639]` | — | — | `0.00` | `0.0` |
| BASIL-3 (Full Search) | `citadel@always` | `greedy` | **0.938** | `[0.8333, 1.0]` | `+0.340` | `[0.1806, 0.5]` | `1.00` | `34.1` |
| BASIL-3 + LTH-Regret (10%) | `citadel@meeseeks:0.1` | `greedy` | **0.646** | `[0.4514, 0.8333]` | `+0.049` | `[-0.028, 0.1389]` | `0.32` | `3.8` |
| BASIL-3 + LTH-Regret (25%) | `citadel@meeseeks:0.25` | `greedy` | **0.750** | `[0.5972, 0.8889]` | `+0.153` | `[0.0417, 0.2708]` | `0.73` | `10.8` |
| BASIL-3 + LTH-Regret (50%) | `citadel@meeseeks:0.5` | `greedy` | **0.778** | `[0.625, 0.9167]` | `+0.181` | `[0.0417, 0.3194]` | `0.86` | `20.1` |
| BASIL-3 + LTH-Agree (25%) | `citadel@meeseeks_agree:0.25` | `greedy` | **0.764** | `[0.5972, 0.9097]` | `+0.167` | `[0.0556, 0.3056]` | `0.77` | `10.9` |
| BASIL-3 + Pre-Regret (25%) | `citadel@meeseeks_pre:0.25` | `greedy` | **0.681** | `[0.5, 0.8472]` | `+0.083` | `[-0.0417, 0.2083]` | `0.26` | `9.2` |
| BASIL-3 + VisitShare (25%) | `citadel@share:0.25` | `greedy` | **0.806** | `[0.6389, 0.9306]` | `+0.208` | `[0.0833, 0.3333]` | `1.00` | `9.8` |
| BASIL-3 + UnreachStop (25%) | `citadel@unreach:0.25` | `greedy` | **0.743** | `[0.5484, 0.9167]` | `+0.146` | `[0.0417, 0.25]` | `1.00` | `9.1` |
| BASIL-3 + Pre-VOC (25%) | `citadel@voc:0.25` | `greedy` | **0.743** | `[0.5832, 0.896]` | `+0.146` | `[-0.0002, 0.3125]` | `0.29` | `10.7` |
| BASIL-3 + Disagreement (25%) | `citadel@disagreement:0.25` | `greedy` | **0.604** | `[0.4028, 0.7917]` | `+0.007` | `[-0.0972, 0.1111]` | `0.16` | `7.2` |
| BASIL-3 + Confidence (25%) | `citadel@confidence:0.25` | `greedy` | **0.688** | `[0.5417, 0.8333]` | `+0.090` | `[-0.0347, 0.2222]` | `0.32` | `13.8` |
| BASIL-3 + RandomGate (25%) | `citadel@random:0.25` | `greedy` | **0.639** | `[0.4167, 0.8472]` | `+0.042` | `[-0.0833, 0.2083]` | `0.22` | `8.2` |
| OracleSim (Full Search) | `oracle` | `greedy` | **0.903** | `[0.7845, 1.0]` | `+0.306` | `[0.1667, 0.4583]` | `1.00` | `33.7` |
| OracleSim + LTH-Regret (10%) | `oracle@meeseeks:0.1` | `greedy` | **0.604** | `[0.4028, 0.7917]` | `+0.007` | `[-0.0833, 0.0833]` | `0.35` | `4.0` |
| OracleSim + LTH-Regret (25%) | `oracle@meeseeks:0.25` | `greedy` | **0.771** | `[0.5833, 0.9306]` | `+0.174` | `[0.0625, 0.2917]` | `0.69` | `9.6` |
| OracleSim + LTH-Regret (50%) | `oracle@meeseeks:0.5` | `greedy` | **0.847** | `[0.7083, 0.9583]` | `+0.250` | `[0.1042, 0.3889]` | `0.85` | `18.1` |
| FastPolicy (0 sims) | `morty` | `uct-100` | **0.486** | `[0.2917, 0.6667]` | — | — | `0.00` | `0.0` |
| BASIL-3 (Full Search) | `citadel@always` | `uct-100` | **0.722** | `[0.5417, 0.875]` | `+0.236` | `[0.0417, 0.4028]` | `1.00` | `35.8` |
| BASIL-3 + LTH-Regret (10%) | `citadel@meeseeks:0.1` | `uct-100` | **0.542** | `[0.375, 0.7083]` | `+0.056` | `[-0.0556, 0.1948]` | `0.29` | `3.7` |
| BASIL-3 + LTH-Regret (25%) | `citadel@meeseeks:0.25` | `uct-100` | **0.639** | `[0.458, 0.8194]` | `+0.153` | `[-0.0139, 0.3333]` | `0.65` | `10.0` |
| BASIL-3 + LTH-Regret (50%) | `citadel@meeseeks:0.5` | `uct-100` | **0.688** | `[0.5, 0.8543]` | `+0.201` | `[0.0347, 0.3335]` | `0.86` | `18.8` |
| BASIL-3 + LTH-Agree (25%) | `citadel@meeseeks_agree:0.25` | `uct-100` | **0.681** | `[0.5208, 0.8333]` | `+0.194` | `[0.0417, 0.3472]` | `0.77` | `10.5` |
| BASIL-3 + Pre-Regret (25%) | `citadel@meeseeks_pre:0.25` | `uct-100` | **0.556** | `[0.3608, 0.7361]` | `+0.069` | `[-0.1111, 0.25]` | `0.24` | `8.4` |
| BASIL-3 + VisitShare (25%) | `citadel@share:0.25` | `uct-100` | **0.688** | `[0.4583, 0.8611]` | `+0.201` | `[0.0, 0.375]` | `1.00` | `8.9` |
| BASIL-3 + UnreachStop (25%) | `citadel@unreach:0.25` | `uct-100` | **0.701** | `[0.5417, 0.8542]` | `+0.215` | `[0.0833, 0.3472]` | `1.00` | `9.0` |
| BASIL-3 + Pre-VOC (25%) | `citadel@voc:0.25` | `uct-100` | **0.632** | `[0.4028, 0.8472]` | `+0.146` | `[-0.0833, 0.3889]` | `0.28` | `9.9` |
| BASIL-3 + Disagreement (25%) | `citadel@disagreement:0.25` | `uct-100` | **0.528** | `[0.3333, 0.7083]` | `+0.042` | `[-0.1389, 0.1667]` | `0.18` | `7.7` |
| BASIL-3 + Confidence (25%) | `citadel@confidence:0.25` | `uct-100` | **0.507** | `[0.2774, 0.7361]` | `+0.021` | `[-0.2087, 0.2083]` | `0.29` | `11.5` |
| BASIL-3 + RandomGate (25%) | `citadel@random:0.25` | `uct-100` | **0.549** | `[0.375, 0.7222]` | `+0.062` | `[-0.0417, 0.1597]` | `0.20` | `6.9` |
| OracleSim (Full Search) | `oracle` | `uct-100` | **0.708** | `[0.5417, 0.8472]` | `+0.222` | `[0.0139, 0.4028]` | `1.00` | `36.0` |
| OracleSim + LTH-Regret (10%) | `oracle@meeseeks:0.1` | `uct-100` | **0.514** | `[0.333, 0.6806]` | `+0.028` | `[-0.1111, 0.1528]` | `0.32` | `3.2` |
| OracleSim + LTH-Regret (25%) | `oracle@meeseeks:0.25` | `uct-100` | **0.653** | `[0.4444, 0.8333]` | `+0.167` | `[0.0139, 0.3333]` | `0.71` | `9.7` |
| OracleSim + LTH-Regret (50%) | `oracle@meeseeks:0.5` | `uct-100` | **0.708** | `[0.5347, 0.8681]` | `+0.222` | `[0.0833, 0.375]` | `0.86` | `18.1` |

## 3. Hidden-Rule Regime & Exact Bayesian Hypothesis Filtration (`test` split)

* **Posterior Entropy Collapse**: `6.49` bits (Ep 1 start, `90.0` candidate universes) $\to$ **`1.04` bits** (Ep 6 end, `2.6` surviving candidates)

| Hidden-Rule Agent Architecture | CLI Mode | Ep 1–2 Score | Ep 5–6 Score | Mean Match Score | Mean Exps/Move |
|---|---|---:|---:|---:|---:|
| Blind FastPolicy (0 sims) | `blind_morty` | `0.597` | `0.445` | **0.528** | `0.0` |
| BHF + FastPolicy (0 sims) | `cfc_morty` | `0.618` | `0.535` | **0.572** | `0.0` |
| BHF + Posterior-Sampled Search + LTH (adaptive) | `cfc_meeseeks` | `0.688` | `0.625` | **0.648** | `21.7` |
| BHF + Posterior-Sampled Search (full) | `cfc_citadel` | `0.750` | `0.680` | **0.725** | `39.2` |
| Full-Info BASIL-3 (visible rules ceiling) | `full_info_citadel` | `0.938` | `0.861` | **0.900** | `31.7` |
