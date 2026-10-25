"""Leak-proof evaluation of BASIL-3.

Each rule closes a specific leak or bias:
  1. Universes come from the requested split only, and their rule families must not appear among
     the checkpoint's training families (checked on every run, not assumed).
  2. Every agent plays the same openings of every universe, from both seats (no first-move or
     opening luck). Opponents draw their randomness from the same stream for every agent.
  3. Morty and Citadel get only (observation, legal mask). Only simulator agents (greedy, uct-N,
     oracle) get a copy of the real game.
  4. Gate thresholds come from gate.pt (set on validation positions that are not evaluation
     positions). Nothing is tuned here.
  5. Scores carry 95% intervals from a bootstrap over universes (games inside one universe are
     correlated), and every agent is compared with --reference on the same universes (paired).
  6. The test split is guarded: --preregister hashes the checkpoint, gate, code and arguments;
     a run with that hash can only be executed once in this run directory.
  7. Compute is reported: share of moves planned, search expansions per move, milliseconds per move.

Usage:
  python -m basil3.evaluate --run runs/full --split val --universes 30 --openings 4
  python -m basil3.evaluate --run runs/full --split test --universes 60 --openings 4 --preregister
  python -m basil3.evaluate --run runs/full --split test --universes 60 --openings 4
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .agents import (CitadelAgent, Gate, GreedyAgent, MortyAgent, OracleAgent, RandomAgent, UCTAgent,
                     VOCModel)
from .multiverse import N_ACTIONS, opening, universes
from .train import load_ckpt

_EVAL_SALT = 31_337
DEFAULT_AGENTS = ("morty,citadel@always,citadel@voc:0.25,citadel@disagreement:0.25,"
                  "citadel@confidence:0.25,citadel@random:0.25,oracle")


def make_player(spec, net, gate_ck, voc, args, rng, device):
    if spec == "morty":
        return MortyAgent(net, device)
    if spec == "oracle":
        return OracleAgent(net, args.sims, device)
    if spec == "random":
        return RandomAgent(rng)
    if spec == "greedy":
        return GreedyAgent(rng)
    if spec.startswith("uct-"):
        return UCTAgent(int(spec[4:]), rng)
    if spec.startswith("citadel@"):
        g = spec.split("@", 1)[1]
        if g in ("always", "never"):
            gate = Gate(g)
        else:
            kind, budget = g.split(":")
            budget = float(budget)
            if kind == "random":
                gate = Gate("random", rate=budget, rng=rng)
            else:
                if gate_ck is None:
                    raise FileNotFoundError("gate.pt missing: run python -m basil3.train --stage gate")
                thr = gate_ck["thresholds"][kind][f"{budget:g}"]
                gate = Gate(kind, threshold=thr, voc=voc if kind == "voc" else None)
        return CitadelAgent(net, gate, args.sims, beta=args.beta, device=device, rng=rng)
    raise ValueError(f"unknown player {spec}")


def play(agent, opp, u, opening_index, seat):
    """One game from a fixed opening. Returns (result for agent, illegal?, agent seconds, agent moves)."""
    g = opening(u, opening_index)
    secs, moves = 0.0, 0
    while not g.done:
        p = agent if g.to_move == seat else opp
        obs, legal = g.encode(), g.legal_mask()
        t0 = time.perf_counter()
        a = p.act(obs, legal, g.copy() if p.needs_simulator else None)
        if p is agent:
            secs += time.perf_counter() - t0
            moves += 1
        if not (0 <= a < N_ACTIONS and legal[a]):
            return (-1 if p is agent else 1), True, secs, moves
        g.step(a)
    return (0 if g.winner == 0 else (1 if g.winner == seat else -1)), False, secs, moves


def bootstrap_ci(per_universe, n_boot=2000, seed=0):
    x = np.asarray(per_universe, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def run_hash(args):
    h = hashlib.sha256()
    for name in ("citadel.pt", "gate.pt"):
        path = os.path.join(args.run, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(json.dumps({k: v for k, v in sorted(vars(args).items()) if k not in ("preregister", "out")}).encode())
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--universes", type=int, default=30)
    ap.add_argument("--openings", type=int, default=4)
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--opponents", default="greedy,uct-200")
    ap.add_argument("--reference", default="morty")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(2)

    h = run_hash(args)
    prereg = os.path.join(args.run, f"prereg_{h}.json")
    lock = os.path.join(args.run, f"test_{h}.done")
    if args.preregister:
        if args.split != "test":
            raise SystemExit("--preregister is only for the test split")
        with open(prereg, "w") as f:
            json.dump({"hash": h, "args": vars(args), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        raise SystemExit(f"Preregistered test run {h}. Commit {prereg} (or share its hash), then rerun without --preregister.")
    if args.split == "test":
        if not os.path.exists(prereg):
            raise SystemExit("Test split refused: no matching preregistration. Freeze everything, run with "
                             "--preregister first, then rerun the identical command without it.")
        if os.path.exists(lock):
            raise SystemExit(f"Test split refused: run {h} was already evaluated once ({lock}).")

    device = args.device
    net, ck = load_ckpt(os.path.join(args.run, "citadel.pt"), device)
    gate_path = os.path.join(args.run, "gate.pt")
    gate_ck = torch.load(gate_path, map_location="cpu", weights_only=False) if os.path.exists(gate_path) else None
    voc = None
    if gate_ck is not None:
        voc = VOCModel()
        voc.load_state_dict(gate_ck["voc"])
        voc.eval()

    unis = universes(args.split, args.universes)
    train_fams = set(ck["train_families"])
    for u in unis:                                         # leak guard, checked every run
        assert u.split() == args.split, f"universe {u.seed} is not in {args.split}"
        assert repr(u.family()) not in train_fams, f"LEAK: universe {u.seed} has a training rule family"
    if gate_ck is not None and args.split == "test":
        assert not set(gate_ck["val_seeds"]) & {u.seed for u in unis}, "LEAK: calibration universe in test"

    agents, opponents = args.agents.split(","), args.opponents.split(",")
    per_uni = {(a, o): {u.seed: [] for u in unis} for a in agents for o in opponents}
    stats = {(a, o): dict(w=0, d=0, l=0, illegal=0, secs=0.0, moves=0, planned=0,
                          expansions=0, probe_calls=0, search_calls=0) for a in agents for o in opponents}
    t0 = time.time()
    for ui, u in enumerate(unis):
        for k in range(args.openings):
            for seat in (1, 2):
                for o in opponents:
                    for ai, a in enumerate(agents):
                        rng_agent = np.random.default_rng([_EVAL_SALT, u.seed, k, seat, ai])
                        rng_opp = np.random.default_rng([_EVAL_SALT, u.seed, k, seat, 10_000])   # same for every agent
                        agent = make_player(a, net, gate_ck, voc, args, rng_agent, device)
                        opp = make_player(o, net, gate_ck, voc, args, rng_opp, device)
                        r, bad, secs, moves = play(agent, opp, u, k, seat)
                        st = stats[(a, o)]
                        st["w" if r > 0 else "d" if r == 0 else "l"] += 1
                        st["illegal"] += int(bad and r < 0)
                        st["secs"] += secs
                        st["moves"] += moves
                        st["planned"] += getattr(agent, "planned", 0)
                        st["expansions"] += getattr(agent, "expansions", 0)
                        st["probe_calls"] += getattr(agent, "probe_calls", 0)
                        st["search_calls"] += getattr(agent, "search_calls", 0)
                        per_uni[(a, o)][u.seed].append((r + 1) / 2)
        print(f"  universe {ui + 1}/{len(unis)} done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for o in opponents:
        ref = {s: np.mean(v) for s, v in per_uni.get((args.reference, o), {}).items()}
        for a in agents:
            means = {s: np.mean(v) for s, v in per_uni[(a, o)].items()}
            st = stats[(a, o)]
            n = st["w"] + st["d"] + st["l"]
            row = {"agent": a, "opponent": o, "games": n, "score": round(float(np.mean(list(means.values()))), 4),
                   "score_ci95": bootstrap_ci(list(means.values())), "wins": st["w"], "draws": st["d"], "losses": st["l"],
                   "plan_rate": round(st["planned"] / max(st["moves"], 1), 3),
                   "expansions_per_move": round(st["expansions"] / max(st["moves"], 1), 1),
                   "probes_per_move": round(st["probe_calls"] / max(st["moves"], 1), 2),
                   "imagination_calls_per_move": round((st["probe_calls"] + st["search_calls"]) / max(st["moves"], 1), 2),
                   "ms_per_move": round(1000 * st["secs"] / max(st["moves"], 1), 2), "illegal_moves": st["illegal"]}
            if ref and a != args.reference:
                diffs = [means[s] - ref[s] for s in means]
                row[f"delta_vs_{args.reference}"] = round(float(np.mean(diffs)), 4)
                row["delta_ci95"] = bootstrap_ci(diffs)
            rows.append(row)

    print(f"\n{args.split} | {len(unis)} universes x {args.openings} openings x 2 seats | sims {args.sims}")
    print(f"{'agent':<28}{'opponent':<10}{'score':>7}  {'95% CI':<17}{'delta':>8}  {'delta 95% CI':<18}{'plan':>6}{'exp/mv':>8}{'imag/mv':>9}{'ms/mv':>8}")
    for r in rows:
        d = r.get(f"delta_vs_{args.reference}")
        print(f"{r['agent']:<28}{r['opponent']:<10}{r['score']:>7.3f}  {str(r['score_ci95']):<17}"
              f"{'' if d is None else f'{d:+.3f}':>8}  {str(r.get('delta_ci95', '')):<18}"
              f"{r['plan_rate']:>6.2f}{r['expansions_per_move']:>8.1f}{r['imagination_calls_per_move']:>9.1f}{r['ms_per_move']:>8.1f}")
    out = args.out or os.path.join(args.run, f"eval_{args.split}_{h}.json")
    by_universe = [{"agent": a, "opponent": o,
                    "scores": [round(float(np.mean(per_uni[(a, o)][u.seed])), 6) for u in unis]}
                   for a in agents for o in opponents]
    with open(out, "w") as f:
        json.dump({"hash": h, "train_seed": ck["cfg"]["seed"], "args": vars(args),
                   "universes": [u.describe() for u in unis],
                   "per_universe": by_universe, "results": rows}, f, indent=1)
    if args.split == "test":
        open(lock, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
