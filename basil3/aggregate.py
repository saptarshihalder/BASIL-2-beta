"""Combine evaluation JSON files from independently trained seeds.

Run: python -m basil3.aggregate runs/full_s*/eval_test_*.json --out results/test_summary.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def interval(x, rng, repeats=2000):
    """Resample training seeds and rule families, keeping comparisons paired."""
    x = np.asarray(x, dtype=float)
    seeds, universes = x.shape
    draws = []
    for _ in range(repeats):
        s = rng.integers(seeds, size=seeds)
        u = rng.integers(universes, size=universes)
        draws.append(x[np.ix_(s, u)].mean())
    return [round(float(v), 4) for v in np.percentile(draws, (2.5, 97.5))]


def combine(paths, reference="morty"):
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    if len(runs) < 2:
        raise ValueError("Use at least two separately trained seeds")
    keys = ("split", "universes", "openings", "agents", "opponents", "sims", "beta")
    for run in runs[1:]:
        for key in keys:
            if run["args"].get(key) != runs[0]["args"].get(key):
                raise ValueError(f"Incompatible evaluation argument: {key}")
        if run["universes"] != runs[0]["universes"]:
            raise ValueError("These runs did not use the same test universes")
    seeds = [run.get("train_seed") for run in runs]
    if all(seed is not None for seed in seeds) and len(set(seeds)) != len(seeds):
        raise ValueError("Repeated training seed in the evaluation files")
    if not all("per_universe" in run for run in runs):
        raise ValueError("Re-evaluate with per-universe results enabled")
    grouped = defaultdict(list)
    meta = defaultdict(list)
    for run in runs:
        lookup = {(r["agent"], r["opponent"]): r for r in run["per_universe"]}
        assert len(lookup) == len(run["per_universe"])
        row_lookup = {(r["agent"], r["opponent"]): r for r in run["results"]}
        for key, rec in lookup.items():
            grouped[key].append(rec["scores"])
            meta[key].append(row_lookup[key])
    for key, scores in grouped.items():
        if len(scores) != len(runs):
            raise ValueError(f"Missing evaluation: {key}")

    rng = np.random.default_rng(2718)
    rows = []
    ordered_keys = [(r["agent"], r["opponent"]) for r in runs[0]["results"] if (r["agent"], r["opponent"]) in grouped]
    for k in sorted(grouped):
        if k not in ordered_keys:
            ordered_keys.append(k)
    for key in ordered_keys:
        agent, opponent = key
        x = np.array(grouped[key], dtype=float)
        row = {"agent": agent, "opponent": opponent, "seeds": len(runs),
               "score": round(float(x.mean()), 4), "score_ci95": interval(x, rng)}
        for field in ("plan_rate", "expansions_per_move", "imagination_calls_per_move", "ms_per_move", "illegal_moves"):
            row[field] = round(float(np.mean([r[field] for r in meta[key]])), 3)
        ref_key = (reference, opponent)
        if agent != reference:
            if ref_key not in grouped:
                raise ValueError(f"Missing reference {reference} vs {opponent}")
            diff = x - np.array(grouped[ref_key])
            row["delta"] = round(float(diff.mean()), 4)
            row[f"delta_vs_{reference}"] = row["delta"]
            row["delta_ci95"] = interval(diff, rng)
        rows.append(row)
    return {"reference": reference, "files": paths, "eval_args": {k: runs[0]["args"].get(k) for k in keys}, "results": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reference", default="morty")
    ap.add_argument("--out", default="results/basil3_summary.json")
    args = ap.parse_args(argv)
    data = combine(args.files, args.reference)
    for r in data["results"]:
        print(f"{r['agent']:<26} {r['opponent']:<10} {r['score']:.3f} {r['score_ci95']} "
              f"delta={r.get('delta', 0):+.3f} {r.get('delta_ci95', '')} "
              f"imag/mv={r['imagination_calls_per_move']:.1f}")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
