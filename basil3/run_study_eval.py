"""End-to-end multi-seed evaluation, Cronenberg study, aggregation, and figure/table generation.

Usage:
  python -m basil3.run_study_eval --root results/basil3_v2 --split val --universes 12 --openings 2 --sims 48
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from .aggregate import combine
from .cronenberg import evaluate_cronenberg
from .evaluate import main as eval_main
from .make_paper_figures import (SCIENTIFIC_AGENT_NAMES, aggregate_cronenberg,
                                 aggregate_meeseeks_summaries, main as figs_main)


AGENTS = [
    "morty",
    "citadel@always",
    "citadel@meeseeks:0.1",
    "citadel@meeseeks:0.25",
    "citadel@meeseeks:0.5",
    "citadel@meeseeks_agree:0.25",
    "citadel@meeseeks_pre:0.25",
    "citadel@share:0.25",
    "citadel@unreach:0.25",
    "citadel@voc:0.25",
    "citadel@disagreement:0.25",
    "citadel@confidence:0.25",
    "citadel@random:0.25",
    "oracle",
    "oracle@meeseeks:0.1",
    "oracle@meeseeks:0.25",
    "oracle@meeseeks:0.5",
]


def _eval_one_seed(args_tuple):
    import torch
    torch.set_num_threads(4)
    seed_dir, split, n_unis, openings, sims, opponents = args_tuple
    out_eval = Path(seed_dir) / f"eval_{split}.json"
    if not out_eval.exists():
        eval_argv = [
            "--run", str(seed_dir),
            "--split", split,
            "--universes", str(n_unis),
            "--openings", str(openings),
            "--sims", str(sims),
            "--agents", ",".join(AGENTS),
            "--opponents", ",".join(opponents),
            "--out", str(out_eval),
        ]
        if split == "test":
            try:
                eval_main(eval_argv + ["--preregister"])
            except SystemExit:
                pass
        eval_main(eval_argv)
    out_cron = Path(seed_dir) / f"cronenberg_{split}.json"
    if not out_cron.exists():
        evaluate_cronenberg(str(seed_dir), split=split, n_universes=n_unis, episodes=6,
                            opponent="greedy", sims=sims, budget="0.25", out_path=str(out_cron))
    return str(out_eval)


CRON_SCIENTIFIC_NAMES = {
    "blind_morty": "Blind FastPolicy (0 sims)",
    "cfc_morty": "BHF + FastPolicy (0 sims)",
    "cfc_meeseeks": "BHF + Posterior-Sampled Search + LTH (adaptive)",
    "cfc_citadel": "BHF + Posterior-Sampled Search (full)",
    "full_info_citadel": "Full-Info BASIL-3 (visible rules ceiling)",
}


def write_summary_md(root: Path, split: str, m_sum: dict, eval_sum: dict, cron_sum: dict):
    lat = m_sum["modes"]["latent"]["stats"]
    sim = m_sum["modes"]["sim"]["stats"]
    lines = [
        "# BASIL-3 (Holographic Manifold World Model + Live Telemetry Halting + Bayesian Hypothesis Filtration) Study Summary",
        "",
        f"* **Split**: `{split}` (rule-family-disjoint procedural universes)",
        f"* **Independent training seeds**: `{cron_sum['n_seeds']}`",
        "",
        "## 1. Stopping-Regret Correlation & Anytime Halting Calibration (Live Telemetry Halting)",
        "",
        "| Search Mode | Pre-Search ($t=0$) Regret Corr $r$ | Live In-Search ($t>0$) Regret Corr $r$ | All-Checkpoints Corr $r$ | Final-Move Agreement Acc |",
        "|---|---:|---:|---:|---:|",
        f"| **BASIL-3 (Latent World Model)** | `{lat['heldout_regret_corr_t0']:+.3f}` | **`{lat['heldout_regret_corr_live']:+.3f}`** | `{lat['heldout_regret_corr']:+.3f}` | `{100*lat['heldout_agree_acc']:.1f}%` (base `{100*lat['heldout_agree_base']:.1f}%`) |",
        f"| **OracleSim (Real-Rules Search)** | `{sim['heldout_regret_corr_t0']:+.3f}` | **`{sim['heldout_regret_corr_live']:+.3f}`** | `{sim['heldout_regret_corr']:+.3f}` | `{100*sim['heldout_agree_acc']:.1f}%` (base `{100*sim['heldout_agree_base']:.1f}%`) |",
        "",
        f"## 2. Online Game-Play Performance Across Seeds (`{split}` split)",
        "",
        "| Agent Architecture | CLI Spec | Opponent | Score | 95% Bootstrap CI | $\\Delta$ vs `FastPolicy` | $\\Delta$ 95% CI | Plan Rate | Exps/Move |",
        "|---|---|---|---:|---|---:|---|---:|---:|",
    ]
    for r in eval_sum["results"]:
        d = r.get("delta")
        d_str = f"`{d:+.3f}`" if d is not None else "—"
        d_ci = f"`{r['delta_ci95']}`" if "delta_ci95" in r else "—"
        sci_name = SCIENTIFIC_AGENT_NAMES.get(r["agent"], r["agent"]).replace("\\%", "%").replace("$", "")
        lines.append(
            f"| {sci_name} | `{r['agent']}` | `{r['opponent']}` | **{r['score']:.3f}** | `{r['score_ci95']}` | "
            f"{d_str} | {d_ci} | `{r['plan_rate']:.2f}` | `{r['expansions_per_move']:.1f}` |"
        )
    lines += [
        "",
        f"## 3. Hidden-Rule Regime & Exact Bayesian Hypothesis Filtration (`{split}` split)",
        "",
        f"* **Posterior Entropy Collapse**: `{cron_sum['posterior_curve']['entropy_bits_start'][0]:.2f}` bits (Ep 1 start, `{cron_sum['posterior_curve']['alive_start'][0]:.1f}` candidate universes) $\\to$ **`{cron_sum['posterior_curve']['entropy_bits_end'][-1]:.2f}` bits** (Ep 6 end, `{cron_sum['posterior_curve']['alive_end'][-1]:.1f}` surviving candidates)",
        "",
        "| Hidden-Rule Agent Architecture | CLI Mode | Ep 1–2 Score | Ep 5–6 Score | Mean Match Score | Mean Exps/Move |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mk, md in cron_sum["modes"].items():
        sci_m = CRON_SCIENTIFIC_NAMES.get(mk, mk)
        lines.append(
            f"| {sci_m} | `{mk}` | `{md['early_score_ep1_2']:.3f}` | `{md['late_score_ep5_6']:.3f}` | "
            f"**{md['mean_score']:.3f}** | `{md['mean_expansions']:.1f}` |"
        )
    out_file = root / ("SUMMARY.md" if split == "val" else f"SUMMARY_{split.upper()}.md")
    out_file.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/basil3_v2")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--universes", type=int, default=12)
    ap.add_argument("--openings", type=int, default=2)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--opponents", nargs="+", default=["random", "greedy", "uct-100"])
    args = ap.parse_args(argv)

    root = Path(args.root)
    seed_dirs = [root / f"seed_{s}" for s in args.seeds]
    tasks = [(str(d), args.split, args.universes, args.openings, args.sims, args.opponents) for d in seed_dirs]

    with mp.get_context("spawn").Pool(min(len(tasks), 3)) as pool:
        eval_files = pool.map(_eval_one_seed, tasks)

    summary_path = root / f"{args.split}_summary.json"
    eval_sum = combine(eval_files, reference="morty")
    summary_path.write_text(json.dumps(eval_sum, indent=2))

    figs_main([
        "--seeds", *[str(d) for d in seed_dirs],
        "--eval-json", str(summary_path),
        "--split", args.split,
    ])
    m_sum = aggregate_meeseeks_summaries(seed_dirs)
    cron_sum = aggregate_cronenberg(seed_dirs, split=args.split)
    write_summary_md(root, args.split, m_sum, eval_sum, cron_sum)
    print(f"Complete study summary written to {root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()

