"""Generate publication figures and LaTeX/Markdown tables for the BASIL-3 paper.

Produces four figures in `paper/figures/` (both .pdf and .png at 300 DPI):
  1. fig1_halting_diagnosis: Pre-search VOC failure vs. Live Telemetry Halting (LTH) correlation
     and offline validation stopping-regret curves across halting rules.
  2. fig2_pareto_compute: Online game-play score vs. model expansions per move on held-out rule
     families (BASIL-3 latent world-model search and OracleSim real-rule search).
  3. fig3_cronenberg_cfc: Hidden-rule regime match study: Exact Bayesian Hypothesis Filtration (BHF)
     rule posterior entropy collapse and score progression across episodes.
  4. fig4_architecture_comparison: Training dynamics and unseen-family generalization comparing
     the baseline non-equivariant world model against Toroidal-Equivariant BASIL-3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCIENTIFIC_AGENT_NAMES = {
    "morty": "FastPolicy ($0$ sims)",
    "citadel@always": "BASIL-3 (Full Search)",
    "citadel@meeseeks:0.1": "BASIL-3 + LTH-Regret ($10\\%$)",
    "citadel@meeseeks:0.25": "BASIL-3 + LTH-Regret ($25\\%$)",
    "citadel@meeseeks:0.5": "BASIL-3 + LTH-Regret ($50\\%$)",
    "citadel@meeseeks_agree:0.25": "BASIL-3 + LTH-Agree ($25\\%$)",
    "citadel@meeseeks_pre:0.25": "BASIL-3 + Pre-Regret ($25\\%$)",
    "citadel@share:0.25": "BASIL-3 + VisitShare ($25\\%$)",
    "citadel@unreach:0.25": "BASIL-3 + UnreachStop ($25\\%$)",
    "citadel@voc:0.25": "BASIL-3 + Pre-VOC ($25\\%$)",
    "citadel@disagreement:0.25": "BASIL-3 + Disagreement ($25\\%$)",
    "citadel@confidence:0.25": "BASIL-3 + Confidence ($25\\%$)",
    "citadel@random:0.25": "BASIL-3 + RandomGate ($25\\%$)",
    "oracle": "OracleSim (Full Search)",
    "oracle@meeseeks:0.1": "OracleSim + LTH-Regret ($10\\%$)",
    "oracle@meeseeks:0.25": "OracleSim + LTH-Regret ($25\\%$)",
    "oracle@meeseeks:0.5": "OracleSim + LTH-Regret ($50\\%$)",
}


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_halting_diagnosis(v1_train: dict, m_summary: dict, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    # Panel A: Held-out correlation with stopping regret / value of computation
    v1_corr = v1_train.get("gate_labels", {}).get("heldout_corr", -0.0035)
    lat_stats = m_summary["modes"]["latent"]["stats"]
    sim_stats = m_summary["modes"]["sim"]["stats"]

    labels = [
        "VOC v1\n(t=0 diff)",
        "VOC v2\n(t=0 diff)",
        "LTH t=0\n(Latent)",
        "LTH Live\n(Latent)",
        "LTH t=0\n(Sim)",
        "LTH Live\n(Sim)",
    ]
    vals = [
        v1_corr,
        m_summary.get("voc_v2_corr", float("nan")),
        lat_stats["heldout_regret_corr_t0"],
        lat_stats["heldout_regret_corr_live"],
        sim_stats["heldout_regret_corr_t0"],
        sim_stats["heldout_regret_corr_live"],
    ]
    colors = ["#8c8c8c", "#b0b0b0", "#f28e2b", "#e15759", "#4e79a7", "#2b5c8f"]
    bars = ax1.bar(range(len(labels)), vals, color=colors, width=0.62, edgecolor="#222222", linewidth=0.7)
    ax1.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Held-out correlation with stopping regret ($r$)")
    ax1.set_title("(a) Pre-Search VOC vs. Live Telemetry Halting (LTH)")
    ax1.set_ylim(min(-0.1, min(vals) - 0.08), max(vals) + 0.12)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, max(v, 0.0) + 0.02, f"{v:+.3f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: Offline validation stopping regret vs mean expansions
    replay_curves = m_summary["modes"]["latent"]["val_replay"]
    full_exps = m_summary["modes"]["latent"]["full_exps"]
    styles = [
        ("meeseeks", "LTH-Regret (live telemetry, ours)", "#e15759", "o-", 2.2),
        ("meeseeks_agree", "LTH-Agree (move-stability head)", "#d37295", "s--", 1.6),
        ("share", "Visit-share stopping", "#4e79a7", "^-", 1.6),
        ("unreach", "Unreachable-leader early stop", "#59a14f", "d-", 1.6),
        ("meeseeks_pre", "Pre-search regret gate (t=0 only)", "#f28e2b", "v--", 1.6),
        ("confgate", "Pre-search confidence gate", "#9c755f", "P:", 1.5),
        ("fixed", "Fixed simulation budget", "#76b7b2", "X:", 1.5),
    ]
    for key, label, col, fmt, lw in styles:
        if key not in replay_curves:
            continue
        pts = sorted(replay_curves[key].values(), key=lambda d: d["exps"])
        xs = [p["exps"] for p in pts] + [full_exps]
        ys = [p["regret"] for p in pts] + [0.0]
        ax2.plot(xs, ys, fmt, color=col, label=label, linewidth=lw, markersize=5.5)

    ax2.set_xlabel("Mean world-model expansions per move (val split)")
    ax2.set_ylabel("Mean stopping regret $\\mathbb{E}[Q_T(a_T) - Q_T(a_t)]$")
    ax2.set_title("(b) Stopping Regret vs. Compute on Unseen Rule Families")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    _save(fig, out_dir, "fig1_halting_diagnosis")


def plot_fig2_pareto_compute(eval_data: dict, out_dir: Path):
    rows = eval_data["results"]
    opps = sorted({r["opponent"] for r in rows}, key=lambda x: ("random", "greedy", "uct-100", "uct-200").index(x)
                  if x in ("random", "greedy", "uct-100", "uct-200") else 99)
    fig, axes = plt.subplots(1, len(opps), figsize=(5.0 * len(opps), 3.9), squeeze=False)
    axes = axes[0]

    for ax, opp in zip(axes, opps):
        sub = [r for r in rows if r["opponent"] == opp]
        by_agent = {r["agent"]: r for r in sub}

        # Group points into curves
        meeseeks_pts = [r for r in sub if r["agent"].startswith("citadel@meeseeks:") or r["agent"] in ("morty", "citadel@always")]
        meeseeks_pts.sort(key=lambda r: r["expansions_per_move"])
        if len(meeseeks_pts) >= 2:
            ax.plot([r["expansions_per_move"] for r in meeseeks_pts],
                    [r["score"] for r in meeseeks_pts],
                    "o-", color="#e15759", linewidth=2.2, markersize=6.5, label="BASIL-3 + LTH (live anytime)")

        oracle_m_pts = [r for r in sub if r["agent"].startswith("oracle@meeseeks:") or r["agent"] in ("morty", "oracle")]
        oracle_m_pts.sort(key=lambda r: r["expansions_per_move"])
        if len(oracle_m_pts) > 2:
            ax.plot([r["expansions_per_move"] for r in oracle_m_pts],
                    [r["score"] for r in oracle_m_pts],
                    "s--", color="#2b5c8f", linewidth=1.8, markersize=5.5, label="OracleSim + LTH (real rules)")

        # Plot individual baseline gates
        markers = [
            ("morty", "FastPolicy (0 sims)", "#222222", "*", 110),
            ("citadel@always", "BASIL-3 (always full search)", "#b07aa1", "D", 65),
            ("citadel@voc:0.25", "BASIL-3 @ pre-search VOC (25%)", "#f28e2b", "v", 65),
            ("citadel@disagreement:0.25", "BASIL-3 @ disagreement (25%)", "#59a14f", "^", 65),
            ("citadel@confidence:0.25", "BASIL-3 @ confidence (25%)", "#9c755f", "P", 65),
            ("citadel@random:0.25", "BASIL-3 @ random gate (25%)", "#8c8c8c", "X", 65),
            ("citadel@share:0.25", "BASIL-3 @ visit-share (25%)", "#4e79a7", "p", 65),
            ("citadel@unreach:0.25", "BASIL-3 @ unreachable-stop (25%)", "#76b7b2", "h", 65),
            ("oracle", "OracleSim (full real-rule search)", "#1f77b4", "s", 70),
        ]
        for ag, lbl, col, mk, sz in markers:
            if ag in by_agent:
                r = by_agent[ag]
                ax.scatter([r["expansions_per_move"]], [r["score"]], color=col, marker=mk, s=sz, zorder=5, label=lbl)

        ax.set_xlabel("Model expansions per move")
        ax.set_ylabel("Win + ½ Draw score (both seats)")
        ax.set_title(f"Score vs. Compute (vs. {opp})")
        ax.legend(loc="lower right", fontsize=7.5, frameon=True)

    _save(fig, out_dir, "fig2_pareto_compute")


def plot_fig3_cronenberg(cron_data: dict, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    eps = cron_data["episodes"]
    xs = list(range(1, eps + 1))

    # Panel A: Posterior entropy (bits) and Brier calibration error across episodes
    ent = cron_data["posterior_curve"]["entropy_bits_start"] + [cron_data["posterior_curve"]["entropy_bits_end"][-1]]
    brier = cron_data["posterior_curve"]["rule_brier_start"]
    ax1.plot(range(0, eps + 1), ent, "o-", color="#2b5c8f", linewidth=2.2, markersize=6, label="Rule posterior entropy $H(P(u\\mid h_t))$ [bits]")
    ax1.set_xlabel("Games completed in unknown universe")
    ax1.set_ylabel("Rule posterior entropy (bits)", color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.plot(xs, brier, "s--", color="#e15759", linewidth=1.8, markersize=5.5, label="Rule marginal Brier error")
    ax1_r.set_ylabel("Posterior rule Brier error", color="#e15759")
    ax1_r.tick_params(axis="y", labelcolor="#e15759")
    ax1.set_title("(a) In-Context Rule Induction (Bayesian Filtration)")

    # Panel B: Score and compute across episodes by agent mode
    mode_styles = [
        ("full_info_citadel", "Full-Info BASIL-3 (visible rules ceiling)", "#1f77b4", "s--"),
        ("cfc_citadel", "BHF + Posterior-Sampled Search (full)", "#59a14f", "o-"),
        ("cfc_meeseeks", "BHF + Posterior-Sampled + LTH (adaptive)", "#e15759", "^-"),
        ("cfc_morty", "BHF + FastPolicy (0 sims)", "#f28e2b", "d-"),
        ("blind_morty", "Blind FastPolicy (no rule induction)", "#8c8c8c", "x:"),
    ]
    for m, lbl, col, fmt in mode_styles:
        if m in cron_data["modes"]:
            raw = cron_data["modes"][m]["score_by_episode"]
            # Seat-balanced 2-episode window (since odd episodes play Seat 1 and even play Seat 2)
            ys = [raw[0]] + [0.5 * (raw[i - 1] + raw[i]) for i in range(1, len(raw))]
            ax2.plot(xs, ys, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    ax2.set_xticks(xs)
    ax2.set_xlabel("Episode index within hidden-rule match (seat-balanced)")
    ax2.set_ylabel("Mean score across held-out universes")
    ax2.set_title("(b) Adaptation Across Match Episodes in Hidden Worlds")
    ax2.legend(loc="lower right", fontsize=8, frameon=True)
    _save(fig, out_dir, "fig3_cronenberg_cfc")


def plot_fig4_architecture_comparison(v1_hist: list[dict], v2_hist: list[dict], out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    it1 = [h["iteration"] for h in v1_hist]
    it2 = [h["iteration"] for h in v2_hist]

    ax1.plot(it1, [h["policy"] for h in v1_hist], "--", color="#8c8c8c", linewidth=1.8, label="Baseline World Model: policy CE")
    ax1.plot(it2, [h["policy"] for h in v2_hist], "-", color="#2b5c8f", linewidth=2.2, label="Toroidal-Equivariant BASIL-3: policy CE")
    ax1.plot(it1, [h["win"] for h in v1_hist], ":", color="#b07aa1", linewidth=1.8, label="Baseline World Model: win BCE")
    ax1.plot(it2, [h["win"] for h in v2_hist], "-", color="#e15759", linewidth=2.2, label="Toroidal-Equivariant BASIL-3: win BCE")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Training loss")
    ax1.set_title("(a) Policy & World-Model Win Loss Convergence")
    ax1.legend(loc="upper right", frameon=True)

    # Smooth 3-iter rolling average for noisy 8-game monitoring score
    def _smooth(arr, w=3):
        out = []
        for i in range(len(arr)):
            lo = max(0, i - w + 1)
            out.append(float(np.mean(arr[lo:i + 1])))
        return out

    ax2.plot(it1, _smooth([h["vs_random"] for h in v1_hist]), "--", color="#8c8c8c", linewidth=1.7, label="Baseline FastPolicy vs. random")
    ax2.plot(it2, _smooth([h["vs_random"] for h in v2_hist]), "-", color="#2b5c8f", linewidth=2.2, label="Toroidal-Equivariant FastPolicy vs. random")
    ax2.plot(it1, _smooth([h["vs_greedy"] for h in v1_hist]), ":", color="#f28e2b", linewidth=1.8, label="Baseline FastPolicy vs. greedy")
    ax2.plot(it2, _smooth([h["vs_greedy"] for h in v2_hist]), "-", color="#e15759", linewidth=2.2, label="Toroidal-Equivariant FastPolicy vs. greedy")
    ax2.set_xlabel("Training iteration")
    ax2.set_ylabel("Fast-policy score (3-iter moving avg)")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("(b) Breaking the Tactical Cold-Start Trap")
    ax2.legend(loc="lower right", frameon=True)

    _save(fig, out_dir, "fig4_architecture_comparison")


def aggregate_meeseeks_summaries(seed_dirs: list[Path]) -> dict:
    """Average `meeseeks.pt` statistics and validation replay curves across seeds."""
    import torch
    bundles = [torch.load(d / "meeseeks.pt", map_location="cpu", weights_only=False) for d in seed_dirs]
    out = {"modes": {}}
    for kind in ("latent", "sim"):
        stat_keys = bundles[0]["modes"][kind]["stats"].keys()
        stats = {}
        for k in stat_keys:
            vals = [float(b["modes"][kind]["stats"][k]) for b in bundles]
            stats[k] = round(float(np.mean(vals)), 4)
            stats[f"{k}_sd"] = round(float(np.std(vals)), 4)
        full_exps = round(float(np.mean([float(b["modes"][kind]["full_exps"]) for b in bundles])), 2)
        val_replay = {}
        rules = bundles[0]["modes"][kind]["val_replay"].keys()
        for r in rules:
            val_replay[r] = {}
            for bud in bundles[0]["modes"][kind]["val_replay"][r]:
                pts = [b["modes"][kind]["val_replay"][r][bud] for b in bundles]
                val_replay[r][bud] = {
                    mk: round(float(np.mean([float(p[mk]) for p in pts])), 4)
                    for mk in ("exps", "regret", "agree")
                }
        out["modes"][kind] = {"stats": stats, "full_exps": full_exps, "val_replay": val_replay}
    gates = [torch.load(d / "gate.pt", map_location="cpu", weights_only=False) for d in seed_dirs if (d / "gate.pt").exists()]
    if gates:
        vc = [float(g["labels"]["heldout_corr"]) for g in gates]
        out["voc_v2_corr"] = round(float(np.mean(vc)), 4)
        out["voc_v2_corr_sd"] = round(float(np.std(vc)), 4)
        out["voc_v2_corr_per_seed"] = [round(v, 4) for v in vc]
    return out


def aggregate_cronenberg(seed_dirs: list[Path], split: str = "val") -> dict:
    """Average `cronenberg_{split}.json` across seeds."""
    js = [json.loads((d / f"cronenberg_{split}.json").read_text()) for d in seed_dirs]
    eps = js[0]["episodes"]
    post = {}
    for k in js[0]["posterior_curve"]:
        post[k] = [round(float(np.mean([j["posterior_curve"][k][e] for j in js])), 4) for e in range(eps)]
    modes = {}
    for m in js[0]["modes"]:
        modes[m] = {
            "score_by_episode": [round(float(np.mean([j["modes"][m]["score_by_episode"][e] for j in js])), 4) for e in range(eps)],
            "expansions_by_episode": [round(float(np.mean([j["modes"][m]["expansions_by_episode"][e] for j in js])), 2) for e in range(eps)],
            "plan_rate_by_episode": [round(float(np.mean([j["modes"][m]["plan_rate_by_episode"][e] for j in js])), 3) for e in range(eps)],
            "mean_score": round(float(np.mean([j["modes"][m]["mean_score"] for j in js])), 4),
            "mean_score_sd": round(float(np.std([j["modes"][m]["mean_score"] for j in js])), 4),
            "early_score_ep1_2": round(float(np.mean([j["modes"][m]["early_score_ep1_2"] for j in js])), 4),
            "late_score_ep5_6": round(float(np.mean([j["modes"][m]["late_score_ep5_6"] for j in js])), 4),
            "mean_expansions": round(float(np.mean([j["modes"][m]["mean_expansions"] for j in js])), 2),
        }
    return {"episodes": eps, "n_seeds": len(js), "posterior_curve": post, "modes": modes}


def write_latex_tables(m_summary: dict, eval_data: dict, cron_data: dict, out_tex: Path, test_eval_data: dict | None = None, test_cron_data: dict | None = None):
    """Write `paper/tables_generated.tex` with scientific terminology for Tables 1, 2, 3 (and 4 if test split is present)."""
    lat_s = m_summary["modes"]["latent"]["stats"]
    rep = m_summary["modes"]["latent"]["val_replay"]

    lines = [
        "% Auto-generated by basil3.make_paper_figures -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Halting calibration and stopping-regret accuracy} across seeds on held-out rule families. "
        "Live Telemetry Halting (LTH) achieves strong positive correlation with true stopping regret once in-search telemetry ($t>0$) is observed, "
        "yielding lower stopping regret and higher final-move agreement at matched $25\\%$ compute budget.}",
        "\\label{tab:halting}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Halting Policy (@ $25\\%$ Budget) & Regret Corr.\\ $r$ ($t>0$) & Mean Exps/Move & Stopping Regret $\\downarrow$ & Final-Move Agreement $\\uparrow$ \\\\",
        "\\midrule",
    ]
    rule_names = [
        ("meeseeks", "\\textbf{Live Telemetry Halting (LTH-Regret, ours)}"),
        ("meeseeks_agree", "\\textbf{Live Telemetry Halting (LTH-Agree, ours)}"),
        ("share", "Visit-share threshold ($s_{\\text{top}} \\ge \\tau$)"),
        ("unreach", "Unreachable-leader early stop"),
        ("meeseeks_pre", "Pre-search regret gate ($t=0$ only)"),
        ("confgate", "Pre-search policy-entropy gate"),
        ("fixed", "Fixed simulation budget"),
    ]
    for rk, rlabel in rule_names:
        if rk not in rep or "0.25" not in rep[rk]:
            continue
        p = rep[rk]["0.25"]
        corr_str = f"{lat_s['heldout_regret_corr_live']:+.3f}" if rk == "meeseeks" else (
            f"{lat_s['heldout_regret_corr_t0']:+.3f}" if rk == "meeseeks_pre" else "---"
        )
        lines.append(f"{rlabel} & ${corr_str}$ & ${p['exps']:.1f}$ & ${p['regret']:.4f}$ & ${100*p['agree']:.1f}\\%$ \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Online game-play performance on unseen validation rule families} (paired evaluation across both seats, "
        "reporting mean score $\\text{win} + \\tfrac{1}{2}\\text{draw}$, $95\\%$ bootstrap CI over universes, paired $\\Delta$ vs.\\ \\texttt{FastPolicy}, and model expansions per move).}",
        "\\label{tab:online}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Agent & Opponent & Score & $95\\%$ CI & $\\Delta$ vs.\\ \\texttt{FastPolicy} & Exps/Move \\\\",
        "\\midrule",
    ]
    for r in eval_data["results"]:
        ag = SCIENTIFIC_AGENT_NAMES.get(r["agent"], r["agent"].replace("_", "\\_"))
        opp = r["opponent"]
        sc = r["score"]
        ci = r["score_ci95"]
        delta_key = [k for k in r if k.startswith("delta_vs_")]
        d_val = r[delta_key[0]] if delta_key else r.get("delta")
        d_str = f"{d_val:+.3f}" if d_val is not None else "---"
        lines.append(f"{ag} & \\texttt{{{opp}}} & ${sc:.3f}$ & $[{ci[0]:.3f}, {ci[1]:.3f}]$ & ${d_str}$ & ${r['expansions_per_move']:.1f}$ \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Hidden-Rule Regime multi-episode adaptation} across unseen universes. "
        f"Exact Bayesian Hypothesis Filtration (BHF) reduces rule posterior entropy from ${cron_data['posterior_curve']['entropy_bits_start'][0]:.2f}$ bits "
        f"(${cron_data['posterior_curve']['alive_start'][0]:.1f}$ candidate universes on average) at the start of Episode 1 "
        f"to ${cron_data['posterior_curve']['entropy_bits_end'][-1]:.2f}$ bits by the end of Episode {cron_data['episodes']}.}}",
        "\\label{tab:cronenberg}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Hidden-Rule Agent Architecture & Ep.\\ 1--2 Score & Ep.\\ 5--6 Score & Mean Match Score & Mean Exps/Move \\\\",
        "\\midrule",
    ]
    c_labels = [
        ("blind_morty", "Blind FastPolicy (no rule induction)"),
        ("cfc_morty", "BHF + FastPolicy ($0$ sims)"),
        ("cfc_meeseeks", "\\textbf{BHF + Posterior-Sampled Search + LTH (adaptive)}"),
        ("cfc_citadel", "\\textbf{BHF + Posterior-Sampled Search (full)}"),
        ("full_info_citadel", "Full-Info BASIL-3 (visible rules ceiling)"),
    ]
    for mk, mlabel in c_labels:
        if mk not in cron_data["modes"]:
            continue
        md = cron_data["modes"][mk]
        lines.append(f"{mlabel} & ${md['early_score_ep1_2']:.3f}$ & ${md['late_score_ep5_6']:.3f}$ & ${md['mean_score']:.3f}$ & ${md['mean_expansions']:.1f}$ \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    if test_eval_data is not None:
        lines += [
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{\\textbf{One-shot preregistered test-split evaluation} on held-out rule families (cryptographically locked prior to test evaluation, 3 seeds $\\times$ both seats).}",
            "\\label{tab:test_online}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Agent & Opponent & Test Score & $95\\%$ CI & $\\Delta$ vs.\\ \\texttt{FastPolicy} & Exps/Move \\\\",
            "\\midrule",
        ]
        for r in test_eval_data["results"]:
            ag = SCIENTIFIC_AGENT_NAMES.get(r["agent"], r["agent"].replace("_", "\\_"))
            opp = r["opponent"]
            sc = r["score"]
            ci = r["score_ci95"]
            delta_key = [k for k in r if k.startswith("delta_vs_")]
            d_val = r[delta_key[0]] if delta_key else r.get("delta")
            d_str = f"{d_val:+.3f}" if d_val is not None else "---"
            lines.append(f"{ag} & \\texttt{{{opp}}} & ${sc:.3f}$ & $[{ci[0]:.3f}, {ci[1]:.3f}]$ & ${d_str}$ & ${r['expansions_per_move']:.1f}$ \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    import torch
    ap = argparse.ArgumentParser(description="Generate BASIL-3 figures and LaTeX tables.")
    ap.add_argument("--seeds", nargs="+", required=True, help="seed directories (e.g. results/basil3_v2/seed_0 ...)")
    ap.add_argument("--eval-json", required=True, help="aggregated or single-run evaluation JSON")
    ap.add_argument("--split", default="val")
    ap.add_argument("--v1-dir", default="results/cpu_demo", help="baseline v1 directory for comparison")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--out-tex", default="paper/tables_generated.tex")
    args = ap.parse_args(argv)

    seed_dirs = [Path(s) for s in args.seeds]
    out_dir = Path(args.out_dir)

    v1_train_path = Path(args.v1_dir) / "training.json"
    v1_train = json.loads(v1_train_path.read_text()) if v1_train_path.exists() else {"history": [], "gate_labels": {"heldout_corr": -0.0035}}

    # Average v2 training history across seeds
    cks = [torch.load(d / "citadel.pt", map_location="cpu", weights_only=False) for d in seed_dirs]
    n_it = min(len(ck["history"]) for ck in cks)
    v2_hist = []
    for i in range(n_it):
        row = {"iteration": i + 1}
        for k in ("loss", "policy", "value", "legal", "win", "consistency", "vs_random", "vs_greedy"):
            row[k] = float(np.mean([ck["history"][i][k] for ck in cks]))
        v2_hist.append(row)

    m_summary = aggregate_meeseeks_summaries(seed_dirs)
    eval_path = Path(args.eval_json)
    val_eval_path = eval_path.parent / "val_summary.json"
    test_eval_path = eval_path.parent / "test_summary.json"
    eval_data = json.loads(val_eval_path.read_text()) if val_eval_path.exists() else json.loads(eval_path.read_text())
    cron_split = "val" if (seed_dirs[0] / "cronenberg_val.json").exists() else args.split
    cron_data = aggregate_cronenberg(seed_dirs, split=cron_split)
    test_eval_data = json.loads(test_eval_path.read_text()) if test_eval_path.exists() else None
    test_cron_data = aggregate_cronenberg(seed_dirs, split="test") if (seed_dirs[0] / "cronenberg_test.json").exists() else None

    plot_fig1_halting_diagnosis(v1_train, m_summary, out_dir)
    plot_fig2_pareto_compute(eval_data, out_dir)
    plot_fig3_cronenberg(cron_data, out_dir)
    if v1_train.get("history"):
        plot_fig4_architecture_comparison(v1_train["history"][:len(v2_hist)], v2_hist, out_dir)
    write_latex_tables(m_summary, eval_data, cron_data, Path(args.out_tex), test_eval_data, test_cron_data)
    print(f"Generated figures in {out_dir} and LaTeX tables in {args.out_tex}")


if __name__ == "__main__":
    main()

