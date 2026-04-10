import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

COLORS = {"Mamba": "#1B4F72", "GRU": "#E74C3C", "LSTM": "#F39C12", "MLP": "#95A5A6"}
MODE_COLORS = {"ppo_only": "#2ECC71", "thinking_only": "#E74C3C", "adaptive": "#1B4F72"}


def plot_world_model_comparison():
    with open("experiments/all_experiments.json") as f:
        data = json.load(f)
    wm = data["exp2_world_models"]

    models = ["Mamba", "GRU", "LSTM", "MLP"]
    ttt_1 = [wm[m]["ttt_1step"] for m in models]
    ttt_5 = [wm[m]["ttt_5step"] for m in models]
    c4_1 = [wm[m]["c4_1step"] for m in models]
    c4_5 = [wm[m]["c4_5step"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    w = 0.35

    axes[0].bar(x - w/2, ttt_1, w, label="1-step", color=[COLORS[m] for m in models], alpha=0.8)
    axes[0].bar(x + w/2, ttt_5, w, label="5-step", color=[COLORS[m] for m in models], alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].set_ylabel("MSE", fontsize=12)
    axes[0].set_title("TicTacToe — Prediction Error", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)

    for i, (v1, v5) in enumerate(zip(ttt_1, ttt_5)):
        axes[0].text(i - w/2, v1 + 0.02, f"{v1:.3f}", ha="center", fontsize=9)
        axes[0].text(i + w/2, v5 + 0.02, f"{v5:.3f}", ha="center", fontsize=9)

    axes[1].bar(x - w/2, c4_1, w, label="1-step", color=[COLORS[m] for m in models], alpha=0.8)
    axes[1].bar(x + w/2, c4_5, w, label="5-step", color=[COLORS[m] for m in models], alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=12)
    axes[1].set_ylabel("MSE", fontsize=12)
    axes[1].set_title("Connect4 — Prediction Error", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)

    for i, (v1, v5) in enumerate(zip(c4_1, c4_5)):
        axes[1].text(i - w/2, v1 + 0.05, f"{v1:.2f}", ha="center", fontsize=9)
        axes[1].text(i + w/2, v5 + 0.05, f"{v5:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/world_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/world_model_comparison.png")


def plot_planning_comparison():
    with open("experiments/final_comparison.json") as f:
        data = json.load(f)

    games = ["TicTacToe", "Connect4"]
    modes = ["ppo_only", "thinking_only", "adaptive"]
    labels = ["PPO Only", "Thinking Only", "Adaptive BASIL"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for gi, game in enumerate(games):
        win_rates = []
        for mode in modes:
            for r in data:
                if r["mode"] == mode and r["game"] == game:
                    win_rates.append(r["win_rate"] * 100)
                    break

        bars = axes[gi].bar(labels, win_rates,
                            color=[MODE_COLORS[m] for m in modes], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, win_rates):
            axes[gi].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                          f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

        axes[gi].set_ylim(0, 105)
        axes[gi].set_ylabel("Win Rate (%)", fontsize=12)
        axes[gi].set_title(f"{game} — Win Rate vs Random", fontsize=14, fontweight="bold")
        axes[gi].tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    plt.savefig("plots/planning_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/planning_comparison.png")


def plot_compute_tradeoff():
    with open("experiments/final_comparison.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    games = ["TicTacToe", "Connect4"]
    modes = ["ppo_only", "thinking_only", "adaptive"]
    labels = ["PPO Only", "Thinking Only", "Adaptive"]

    for gi, game in enumerate(games):
        sims = []
        wins = []
        for mode in modes:
            for r in data:
                if r["mode"] == mode and r["game"] == game:
                    sims.append(r["avg_sims_per_game"])
                    wins.append(r["win_rate"] * 100)
                    break

        for i, (s, w, label) in enumerate(zip(sims, wins, labels)):
            axes[gi].scatter(s, w, s=200, color=list(MODE_COLORS.values())[i],
                             edgecolor="black", linewidth=0.8, zorder=3)
            axes[gi].annotate(label, (s, w), textcoords="offset points",
                              xytext=(10, 8), fontsize=11, fontweight="bold")

        axes[gi].set_xlabel("Avg Simulations / Game", fontsize=12)
        axes[gi].set_ylabel("Win Rate (%)", fontsize=12)
        axes[gi].set_title(f"{game} — Win Rate vs Compute", fontsize=14, fontweight="bold")
        axes[gi].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/compute_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/compute_tradeoff.png")


def plot_fast_slow_split():
    with open("experiments/final_comparison.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    games = ["TicTacToe", "Connect4"]

    for gi, game in enumerate(games):
        for r in data:
            if r["mode"] == "adaptive" and r["game"] == game:
                fast = r["fast_pct"] * 100
                slow = r["slow_pct"] * 100
                break

        axes[gi].pie([fast, slow], labels=[f"Fast\n{fast:.1f}%", f"Slow\n{slow:.1f}%"],
                     colors=["#2ECC71", "#E74C3C"], autopct="", startangle=90,
                     textprops={"fontsize": 14, "fontweight": "bold"},
                     wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[gi].set_title(f"{game}\nAdaptive Path Split", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/fast_slow_split.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/fast_slow_split.png")


def plot_latency():
    with open("experiments/benchmark.json") as f:
        data = json.load(f)

    lat = data["pipeline_latency_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Fast Path\n(encode+policy)", "Fast + JSD\nCheck", "Slow Path\n(plan 10×5)"]
    values = [lat["fast_path"], lat["fast_with_jsd"], lat["slow_path"]]
    colors = ["#2ECC71", "#F39C12", "#E74C3C"]

    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)", fontsize=12)
    ax.set_title("Per-Move Latency — Fast vs Slow Path", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{val:.1f} ms", ha="center", fontsize=12, fontweight="bold")

    ax.text(0.95, 0.95, f"Speedup: {lat['speedup_fast_vs_slow']}x",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            fontweight="bold", color="#1B4F72",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8", alpha=0.8))

    plt.tight_layout()
    plt.savefig("plots/latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/latency.png")


def plot_model_sizes():
    with open("experiments/benchmark.json") as f:
        data = json.load(f)

    sizes = data["model_sizes"]
    names = list(sizes.keys())
    params = [sizes[n]["params"] / 1000 for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1B4F72", "#2E86C1", "#1B4F72", "#E74C3C", "#F39C12", "#95A5A6", "#27AE60"]
    bars = ax.barh(names, params, color=colors[:len(names)], alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, params):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}K", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Parameters (thousands)", fontsize=12)
    ax.set_title("Model Parameter Counts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/model_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/model_sizes.png")


def plot_throughput():
    with open("experiments/benchmark.json") as f:
        data = json.load(f)

    tp = data["throughput"]

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = list(tp.keys())
    labels = [k.replace("_", " ").title() for k in keys]
    gps = [tp[k]["games_per_sec"] for k in keys]
    wins = [tp[k]["win_rate"] * 100 for k in keys]
    colors = ["#2ECC71" if "fast" in k else "#1B4F72" for k in keys]

    bars = ax.bar(labels, gps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, g, w in zip(bars, gps, wins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{g:.1f}/s\n{w:.1f}% win", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Games per Second", fontsize=12)
    ax.set_title("Throughput — Fast vs Adaptive", fontsize=14, fontweight="bold")
    plt.xticks(rotation=15, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/throughput.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved plots/throughput.png")


if __name__ == "__main__":
    print("Generating plots...\n")
    plot_world_model_comparison()
    plot_planning_comparison()
    plot_compute_tradeoff()
    plot_fast_slow_split()
    plot_latency()
    plot_model_sizes()
    plot_throughput()
    print(f"\nAll plots saved to plots/")
    print(f"Files: {os.listdir('plots')}")
