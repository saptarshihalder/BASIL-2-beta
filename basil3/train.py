"""BASIL-3 training. Touches TRAIN universes for learning and VAL universes for calibration only.

Stage "selfplay" (Expert Iteration):
  Each game picks a random train universe and plays itself with AlphaZero-style search on the real
  rules (the environment is available while learning; it is not available at decision time).
  Morty (h, f) learns move probabilities from visit counts and values from game outcomes.
  The Council of Ricks learns the rules: unrolled `unroll` steps along the real moves, each Rick
  predicts whether the move wins, which moves are legal next, and a latent matching the encoding of
  the real next position. Each Rick trains on its own bootstrap half of the positions.
Stage "gate":
  On train universes, estimate planning gain with a reference search that uses true transitions
  and Morty's value head. These are approximate labels, not exact game-theoretic values. Fit
  the VOC model to those labels, then
  pick every gate's threshold on VAL positions so each gate plans on the same share of moves.

Usage:
  python -m basil3.train --config configs/basil3_smoke.yaml --out runs/smoke --stage all
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .agents import FEATURES, VOCModel, gate_features
from .multiverse import (GENERATOR_VERSION, N_ACTIONS, Game, greedy_move, opening, universe_from_seed,
                         universes)
from .nets import build
from .search import LatentAdapter, SimAdapter, net_eval, search

DEFAULTS = dict(
    seed=0, device="cpu", threads=2, workers=1,
    ch=64, repr_blocks=3, dyn_blocks=2, n_ricks=4, prior_scale=1.0, bootstrap=True,
    train_universes=300, iterations=60, games_per_iter=64, sims=64, temp_moves=6,
    dir_alpha=0.3, dir_eps=0.25, replay_positions=100_000,
    train_steps=200, batch=128, unroll=3, lr=1e-3, weight_decay=1e-4,
    w_policy=1.0, w_value=1.0, w_legal=0.5, w_win=1.0, w_cons=2.0,
    gate_positions=2000, gate_sims=64, ref_sims=64, beta=0.5, gate_steps=600,
    budgets=[0.1, 0.25, 0.5], val_universes=30, calib_positions=1500,
)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return cfg


def save_ckpt(path, net, cfg, pool, iteration, history, opt=None):
    tmp = path + ".tmp"                      # write-then-rename: a crash never leaves a half-written file
    torch.save({"net": net.state_dict(), "net_config": net.config, "cfg": cfg, "iteration": iteration,
                "train_seeds": [u.seed for u in pool],
                "train_families": sorted({repr(u.family()) for u in pool}),
                "generator_version": GENERATOR_VERSION, "history": history,
                "opt": opt.state_dict() if opt is not None else None}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["generator_version"] != GENERATOR_VERSION:
        raise ValueError("checkpoint was trained on a different Multiverse generator version")
    net = build(ck["net_config"]).to(device)
    net.load_state_dict(ck["net"])
    return net.eval(), ck


# ----------------------------------------------------------------------------- self-play
def self_play_game(net, u, cfg, rng, device):
    g = Game(u)
    obs, legal, acts, pis, movers, wins = [], [], [], [], [], []
    while not g.done:
        o, m = g.encode(), g.legal_mask()
        visits, _, _ = search(SimAdapter(net, device), g, cfg["sims"],
                              dirichlet=(cfg["dir_alpha"], cfg["dir_eps"]), rng=rng)
        pi = visits / visits.sum()
        a = int(rng.choice(N_ACTIONS, p=pi)) if g.moves < cfg["temp_moves"] else int(np.argmax(visits))
        obs.append(o); legal.append(m); pis.append(pi); acts.append(a); movers.append(g.to_move)
        wins.append(g.step(a))
    obs.append(g.encode()); legal.append(np.zeros(N_ACTIONS, dtype=bool)); movers.append(g.to_move)
    z = [0.0 if g.winner == 0 else (1.0 if mv == g.winner else -1.0) for mv in movers]
    return dict(obs=np.stack(obs), legal=np.stack(legal), act=np.array(acts), pi=np.stack(pis).astype(np.float32),
                z=np.array(z, dtype=np.float32), win=np.array(wins, dtype=np.float32), universe=u.seed)


def _worker_games(job):
    """Self-play in a separate process (CPU, one thread). job = (weights, net config, cfg, [(universe seed, rng seed)])."""
    state, net_cfg, cfg, todo = job
    torch.set_num_threads(1)
    net = build(net_cfg)
    net.load_state_dict(state)
    net.eval()
    return [self_play_game(net, universe_from_seed(us), cfg, np.random.default_rng(rs), "cpu") for us, rs in todo]


def generate_games(net, pool, cfg, rng, device):
    jobs = [(pool[int(rng.integers(len(pool)))].seed, int(rng.integers(2**62))) for _ in range(cfg["games_per_iter"])]
    W = cfg["workers"]
    if W <= 1:
        return [self_play_game(net, universe_from_seed(us), cfg, np.random.default_rng(rs), device) for us, rs in jobs]
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    with mp.get_context("spawn").Pool(W) as workers:
        chunks = workers.map(_worker_games, [(state, net.config, cfg, jobs[i::W]) for i in range(W)])
    return [g for chunk in chunks for g in chunk]


class Replay:
    def __init__(self, capacity, n_ricks, rng, bootstrap=True):
        self.games, self.size, self.capacity, self.K, self.rng = [], 0, capacity, n_ricks, rng
        self.bootstrap = bootstrap

    def add(self, game):
        shape = (len(game["act"]), self.K)          # which Ricks learn from each position
        game["boot"] = self.rng.random(shape) < 0.5 if self.bootstrap else np.ones(shape, dtype=bool)
        self.games.append(game)
        self.size += len(game["act"])
        while self.size > self.capacity and len(self.games) > 1:
            self.size -= len(self.games.pop(0)["act"])

    def sample(self, B, U, device):
        lens = np.array([len(g["act"]) for g in self.games])
        gi = self.rng.choice(len(self.games), size=B, p=lens / lens.sum())
        z = lambda *s, t=np.float32: np.zeros(s, dtype=t)
        obs0, legal0, pi0, z0 = z(B, 14, 7, 7), z(B, N_ACTIONS, t=bool), z(B, N_ACTIONS), z(B)
        act, win, valid, has_pi, nz = z(B, U, t=np.int64), z(B, U), z(B, U), z(B, U), z(B, U)
        nobs, nlegal, npi, boot = z(B, U, 14, 7, 7), z(B, U, N_ACTIONS, t=bool), z(B, U, N_ACTIONS), z(B, self.K)
        for b, i in enumerate(gi):
            G = self.games[i]
            T = len(G["act"])
            t = int(self.rng.integers(T))
            obs0[b], legal0[b], pi0[b], z0[b], boot[b] = G["obs"][t], G["legal"][t], G["pi"][t], G["z"][t], G["boot"][t]
            for u in range(U):
                j = t + u
                if j >= T:
                    break
                valid[b, u], act[b, u], win[b, u] = 1.0, G["act"][j], G["win"][j]
                nobs[b, u], nlegal[b, u], nz[b, u] = G["obs"][j + 1], G["legal"][j + 1], G["z"][j + 1]
                if j + 1 < T:
                    npi[b, u], has_pi[b, u] = G["pi"][j + 1], 1.0
        to = lambda x: torch.as_tensor(x, device=device)
        return [to(x) for x in (obs0, legal0, pi0, z0, act, win, nobs, nlegal, npi, has_pi, nz, valid, boot)]


def masked_ce(logits, target, legal):
    logp = F.log_softmax(logits.masked_fill(~legal, -1e9), -1)
    return -(target * logp).sum(-1)


def train_step(net, opt, batch, cfg):
    net.train()
    obs0, legal0, pi0, z0, act, win, nobs, nlegal, npi, has_pi, nz, valid, boot = batch
    B, U = act.shape
    s0 = net.h(obs0)
    pol0, v0, leg0 = net.f(s0)
    parts = {"policy": masked_ce(pol0, pi0, legal0).mean(), "value": F.mse_loss(v0, z0),
             "legal": F.binary_cross_entropy_with_logits(leg0, legal0.float())}
    loss = cfg["w_policy"] * parts["policy"] + cfg["w_value"] * parts["value"] + cfg["w_legal"] * parts["legal"]
    with torch.no_grad():
        target = net.h(nobs.flatten(0, 1)).view(B, U, *s0.shape[1:])
    dyn = {"win": 0.0, "consistency": 0.0}
    K = net.n_ricks
    for k, rick in enumerate(net.ricks):
        s = s0
        for u in range(U):
            m = valid[:, u] * boot[:, k]
            if m.sum() == 0:
                break
            s = 0.5 * s + 0.5 * s.detach()                    # MuZero: halve gradients into dynamics
            s, wl = rick(s, act[:, u])
            pol, v, leg = net.f(s)
            l_win = F.binary_cross_entropy_with_logits(wl, win[:, u], reduction="none")
            l_cons = net.consistency(s, target[:, u])
            per = (cfg["w_win"] * l_win + cfg["w_cons"] * l_cons + cfg["w_value"] * (v - nz[:, u]) ** 2
                   + cfg["w_legal"] * F.binary_cross_entropy_with_logits(leg, nlegal[:, u].float(), reduction="none").mean(-1)
                   + cfg["w_policy"] * masked_ce(pol, npi[:, u], nlegal[:, u]) * has_pi[:, u])
            denom = m.sum().clamp(min=1.0)
            loss = loss + (per * m).sum() / denom / K
            dyn["win"] += ((l_win * m).sum() / denom).item() / K
            dyn["consistency"] += ((l_cons * m).sum() / denom).item() / K
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    return {"loss": loss.item(), **{k: v.item() for k, v in parts.items()}, **dyn}


def quick_check(net, pool, rng, device, games=8):
    """Monitoring only (train universes): Morty's score vs random and vs greedy, both seats."""
    out = {}
    for opp in ("random", "greedy"):
        score = 0.0
        for i in range(games):
            u = pool[int(rng.integers(len(pool)))]
            g, seat = opening(u, 900_000 + i), 1 + i % 2
            while not g.done:
                if g.to_move == seat:
                    p, _, _ = net_eval(net, g.encode(), g.legal_mask(), device)
                    a = int(np.argmax(p))
                else:
                    acts = g.legal_actions()
                    a = greedy_move(g, rng) if opp == "greedy" else acts[int(rng.integers(len(acts)))]
                g.step(a)
            score += 0.5 if g.winner == 0 else float(g.winner == seat)
        out[opp] = score / games
    return out


def selfplay_stage(cfg, out_dir, resume=False):
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng([cfg["seed"], 1])
    t0 = time.time()
    pool = universes("train", cfg["train_universes"])
    print(f"{len(pool)} train universes ready in {time.time() - t0:.0f}s, e.g. {pool[0].describe()}")
    net = build(cfg).to(device)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    replay = Replay(cfg["replay_positions"], net.n_ricks, rng, cfg["bootstrap"])
    history, start = [], 1
    path = os.path.join(out_dir, "citadel.pt")
    if resume and os.path.exists(path):          # continue an interrupted run (the replay buffer refills)
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck["train_seeds"] != [u.seed for u in pool]:
            raise ValueError("resume: this config builds a different training pool than the checkpoint")
        net.load_state_dict(ck["net"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        history, start = ck["history"], ck["iteration"] + 1
        rng = np.random.default_rng([cfg["seed"], 1, start])
        replay.rng = rng
        print(f"resuming after iteration {ck['iteration']}", flush=True)
    for it in range(start, cfg["iterations"] + 1):
        t0 = time.time()
        net.eval()
        for game in generate_games(net, pool, cfg, rng, device):
            replay.add(game)
        t_play = time.time() - t0
        logs = [train_step(net, opt, replay.sample(cfg["batch"], cfg["unroll"], device), cfg) for _ in range(cfg["train_steps"])]
        net.eval()
        mean = {k: round(float(np.mean([l[k] for l in logs])), 4) for k in logs[0]}
        check = quick_check(net, pool, rng, device)
        history.append({"iteration": it, "positions": replay.size, "play_s": round(t_play), "train_s": round(time.time() - t0 - t_play),
                        **mean, "vs_random": check["random"], "vs_greedy": check["greedy"]})
        print(json.dumps(history[-1]), flush=True)   # flush: progress shows up in redirected logs
        save_ckpt(path, net, cfg, pool, it, history, opt)
    return net


# ----------------------------------------------------------------------------------- gate
def q_reference(net, g, a, sims, device):
    """Reference-search estimate for action a, using true transitions and the learned value."""
    g2 = g.copy()
    if g2.step(a):
        return 1.0
    if g2.done:
        return 0.0
    visits, q, _ = search(SimAdapter(net, device), g2, sims)
    return -float((visits * q).sum() / max(visits.sum(), 1))


def positions(net, unis, n, rng, device, opening_offset):
    """Positions where Morty is to move, from Morty-vs-greedy games."""
    out = []
    while len(out) < n:
        u = unis[int(rng.integers(len(unis)))]
        g, seat = opening(u, opening_offset + int(rng.integers(10**6))), int(rng.integers(1, 3))
        while not g.done and len(out) < n:
            if g.to_move == seat:
                out.append(g.copy())
                p, _, _ = net_eval(net, g.encode(), g.legal_mask(), device)
                g.step(int(np.argmax(p)))
            else:
                g.step(greedy_move(g, rng))
    return out


def gate_stage(cfg, out_dir):
    device = cfg["device"]
    net, ck = load_ckpt(os.path.join(out_dir, "citadel.pt"), device)
    rng = np.random.default_rng([cfg["seed"], 2])
    train_pool = [universe_from_seed(s) for s in ck["train_seeds"]]

    X, Y = [], []
    t0 = time.time()
    for g in positions(net, train_pool, cfg["gate_positions"], rng, device, opening_offset=2_000_000):
        obs, legal = g.encode(), g.legal_mask()
        feats, fast, _ = gate_features(net, obs, legal, device)
        visits, _, _ = search(LatentAdapter(net, device, rng), (obs, legal), cfg["gate_sims"], beta=cfg["beta"])
        plan = int(np.argmax(visits))
        y = 0.0 if plan == fast else q_reference(net, g, plan, cfg["ref_sims"], device) - q_reference(net, g, fast, cfg["ref_sims"], device)
        X.append(feats)
        Y.append(y)
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    print(f"{len(Y)} labelled train positions in {time.time() - t0:.0f}s: search changed the move in "
          f"{np.mean(Y != 0):.0%}, mean reference-search gain {Y.mean():+.3f}, helped {np.mean(Y > 0):.0%}, hurt {np.mean(Y < 0):.0%}")

    voc = VOCModel()
    voc.mu.copy_(torch.as_tensor(X.mean(0)))
    voc.sd.copy_(torch.as_tensor(X.std(0) + 1e-6))
    idx = rng.permutation(len(Y))
    cut = int(0.8 * len(Y))
    tr, ho = idx[:cut], idx[cut:]
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    opt = torch.optim.Adam(voc.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(cfg["gate_steps"]):
        b = torch.as_tensor(rng.choice(tr, size=min(256, len(tr))))
        loss = F.smooth_l1_loss(voc(Xt[b]), Yt[b], beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = voc(Xt[ho]).numpy()
    corr = float(np.corrcoef(pred, Y[ho])[0, 1]) if len(ho) > 2 and Y[ho].std() > 0 and pred.std() > 0 else float("nan")
    print(f"VOC model held-out correlation with reference-search gain: {corr:.3f}")

    val_pool = universes("val", cfg["val_universes"])
    scores = {"confidence": [], "disagreement": [], "voc": []}
    for g in positions(net, val_pool, cfg["calib_positions"], rng, device, opening_offset=1_000_000):
        feats, _, sc = gate_features(net, g.encode(), g.legal_mask(), device)
        scores["confidence"].append(sc["confidence"])
        scores["disagreement"].append(sc["disagreement"])
        scores["voc"].append(voc.predict(feats))
    thresholds = {k: {f"{b:g}": float(np.quantile(v, 1 - b)) for b in cfg["budgets"]} for k, v in scores.items()}
    torch.save({"voc": voc.state_dict(), "thresholds": thresholds, "features": FEATURES,
                "labels": {"n": int(len(Y)), "changed": float(np.mean(Y != 0)), "mean_gain": float(Y.mean()),
                           "heldout_corr": corr},
                "val_seeds": [u.seed for u in val_pool], "calibration_positions": len(scores["voc"])},
               os.path.join(out_dir, "gate.pt"))
    print("thresholds (plan when score >= threshold):", json.dumps(thresholds))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["selfplay", "gate", "all"])
    ap.add_argument("--resume", action="store_true", help="continue an interrupted self-play run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(cfg["threads"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if args.stage in ("selfplay", "all"):
        selfplay_stage(cfg, args.out, resume=args.resume)
    if args.stage in ("gate", "all"):
        gate_stage(cfg, args.out)


if __name__ == "__main__":
    main()
