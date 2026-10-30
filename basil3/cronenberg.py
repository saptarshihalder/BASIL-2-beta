"""Cronenberg Mode & The Central Finite Curve: In-Context Bayesian Rule Induction + Multiverse Search.

In standard Multiverse, the 8 rule indicator planes of the observation (obs[5:13]: gravity,
wrap_h, wrap_v, and the 5 winning tetromino shapes I, O, T, S, L) are visible to the agent.
In Cronenberg Mode, the agent is dropped into a universe with HIDDEN rules (obs[5:13] masked out)
and plays a match of M consecutive games in that universe.

Because the hypothesis space of rules for a fixed board geometry (H, W, blocked) is finite
(at most 90 candidate rule combinations in the Central Finite Curve), we can maintain the exact
Bayesian posterior P(u | history) over candidate universes as the agent observes legal-move masks,
non-terminal board positions, and game outcomes across the match:
  * Turn 1 legal moves immediately identify whether gravity is active.
  * Every non-terminal board falsifies any rule hypothesis under which a player has already
    completed a winning shape.
  * Every game won by a stone placement falsifies any rule hypothesis under which that stone
    did not complete a winning shape.

During Multiverse Thompson Search (MultiverseAdapter), each of the K Ricks in Citadel samples its
own rule hypothesis u_k ~ P(u | history) from the Central Finite Curve and pins u_k's rule DNA
into its latent state s_k. The ensemble disagreement q_std therefore captures both epistemic
world-model uncertainty and rule-hypothesis uncertainty, naturally shrinking as the agent
discovers the laws of physics of the current universe across the match.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from .multiverse import N_ACTIONS, SHAPE_NAMES, Game, Universe, compile_rules
from .search import search


def all_shape_subsets():
    out = []
    for k in (1, 2):
        for comb in combinations(SHAPE_NAMES, k):
            out.append(tuple(comb))
    return tuple(out)


SHAPE_SUBSETS = all_shape_subsets()   # 5 singletons + 10 pairs = 15 shape hypotheses


def mask_rules(obs: np.ndarray) -> np.ndarray:
    """Return a copy of `obs` with the 8 rule planes (5..12) zeroed out (hidden-rule mode)."""
    out = obs.copy()
    out[5:13] = 0.0
    return out


def infer_geometry(obs: np.ndarray):
    """Extract observable board geometry (H, W, blocked flat indices) from planes 3 and 4 of `obs`."""
    on = obs[4] > 0.5
    H = int(on[:, 0].sum())
    W = int(on[0, :].sum())
    blk_grid = obs[3, :H, :W] > 0.5
    blocked = tuple(int(i) for i in np.flatnonzero(blk_grid.reshape(-1)))
    return H, W, blocked


def rule_planes_of(u: Universe, pad: int = 7) -> np.ndarray:
    """8 x pad x pad binary planes for (gravity, wrap_h, wrap_v, I, O, T, S, L)."""
    p = np.zeros((8, pad, pad), dtype=np.float32)
    p[0] = float(u.gravity)
    p[1] = float(u.wrap_h)
    p[2] = float(u.wrap_v)
    for j, name in enumerate(SHAPE_NAMES):
        p[3 + j] = float(name in u.shapes)
    return p


class CentralFiniteCurve:
    """Exact Bayesian posterior over the candidate rule universes for a given (H, W, blocked)."""

    def __init__(self, H: int, W: int, blocked: tuple):
        self.H, self.W, self.blocked = H, W, blocked
        self.candidates: list[Universe] = []
        self.pats: list[np.ndarray] = []
        for gravity in (False, True):
            for wrap_h in (False, True):
                for wrap_v in ((False,) if gravity else (False, True)):
                    for shapes in SHAPE_SUBSETS:
                        u = Universe(seed=-1, H=H, W=W, gravity=gravity, wrap_h=wrap_h,
                                     wrap_v=wrap_v, blocked=blocked, shapes=shapes)
                        pats, _ = compile_rules(u)
                        if len(pats) >= 8:
                            self.candidates.append(u)
                            self.pats.append(pats)
        self.alive = np.ones(len(self.candidates), dtype=bool)
        self.rule_matrix = np.stack([
            np.array([float(u.gravity), float(u.wrap_h), float(u.wrap_v)] +
                     [float(s in u.shapes) for s in SHAPE_NAMES], dtype=np.float32)
            for u in self.candidates
        ])  # (M, 8)

    def reset(self):
        self.alive[:] = True

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def posterior(self) -> np.ndarray:
        m = self.alive.astype(np.float64)
        s = m.sum()
        if s == 0:
            return np.full(len(self.candidates), 1.0 / len(self.candidates))
        return m / s

    def entropy_bits(self) -> float:
        n = max(self.n_alive, 1)
        return float(np.log2(n))

    def marginal_rules(self) -> np.ndarray:
        """Posterior marginal probability of each of the 8 rule flags: shape (8,)."""
        return (self.posterior()[:, None] * self.rule_matrix).sum(axis=0).astype(np.float32)

    def fill_obs(self, obs_blind: np.ndarray) -> np.ndarray:
        """Fill `obs_blind[5:13]` with the posterior marginal rule probabilities."""
        out = obs_blind.copy()
        m = self.marginal_rules()
        out[5:13] = m[:, None, None]
        return out

    def sample_universes(self, k: int, rng) -> list[Universe]:
        idx = np.flatnonzero(self.alive)
        if len(idx) == 0:
            idx = np.arange(len(self.candidates))
        pick = rng.choice(idx, size=k, replace=True)
        return [self.candidates[int(i)] for i in pick]

    def _board_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reconstruct the H x W board (1=mover, 2=opponent, 3=blocked, 0=empty) from `obs`."""
        b = np.zeros((self.H, self.W), dtype=np.int8)
        b[obs[0, :self.H, :self.W] > 0.5] = 1
        b[obs[1, :self.H, :self.W] > 0.5] = 2
        b[obs[3, :self.H, :self.W] > 0.5] = 3
        return b.reshape(-1)

    def observe_turn(self, obs_blind: np.ndarray, legal: np.ndarray):
        """Update posterior from a non-terminal position `(obs_blind, legal)`."""
        b = self._board_from_obs(obs_blind)
        # 1. Gravity check: does `legal` match free placement or gravity on `b`?
        free_cells = np.flatnonzero(b == 0)
        free_acts = {(c // self.W) * 7 + (c % self.W) for c in free_cells}
        actual_acts = set(np.flatnonzero(legal).tolist())
        if actual_acts != free_acts:
            # Must be gravity=True
            for i, u in enumerate(self.candidates):
                if not u.gravity:
                    self.alive[i] = False
        else:
            # If any column has an empty cell above another empty cell and all empty cells are legal, gravity is False
            b2d = b.reshape(self.H, self.W)
            has_floating_empty = bool(((b2d[:-1] == 0) & (b2d[1:] == 0)).any())
            if has_floating_empty:
                for i, u in enumerate(self.candidates):
                    if u.gravity:
                        self.alive[i] = False
        # 2. Non-terminal check: since the game is still ongoing, neither player (1 or 2) has covered a winning pattern
        for i in np.flatnonzero(self.alive):
            pats = self.pats[i]
            if ((b[pats] == 1).all(axis=1).any() or (b[pats] == 2).all(axis=1).any()):
                self.alive[i] = False

    def observe_game_end(self, final_board: np.ndarray, winner: int, last_cell: int | None = None):
        """Update posterior at the end of a game from the final H*W board and winner (0=draw, 1 or 2)."""
        for i in np.flatnonzero(self.alive):
            pats = self.pats[i]
            if winner == 0:
                if (final_board[pats] == 1).all(axis=1).any() or (final_board[pats] == 2).all(axis=1).any():
                    self.alive[i] = False
            else:
                won_pats = (final_board[pats] == winner).all(axis=1)
                if not won_pats.any():
                    self.alive[i] = False
                elif last_cell is not None:
                    # The winning stone at `last_cell` must be part of at least one completed winning pattern
                    if not (won_pats & (pats == last_cell).any(axis=1)).any():
                        self.alive[i] = False
        if not self.alive.any():
            # Safety fallback if noisy state was passed
            self.alive[:] = True


class MultiverseAdapter:
    """Latent MCTS adapter where each of the K Ricks imagines under its own Thompson-sampled
    rule hypothesis u_k ~ P(u | history) from the Central Finite Curve."""

    def __init__(self, net, cfc: CentralFiniteCurve, device="cpu", rng=None, legal_threshold=0.5):
        self.net, self.cfc, self.device, self.rng, self.thr = net, cfc, device, rng, legal_threshold

    @torch.no_grad()
    def root(self, root_input):
        obs_blind, legal = root_input
        K = self.net.n_ricks
        sampled_us = self.cfc.sample_universes(K, self.rng)
        obs_batch = np.repeat(obs_blind[None], K, axis=0)
        for k, u_k in enumerate(sampled_us):
            obs_batch[k, 5:13] = rule_planes_of(u_k)
        s_k = self.net.h(torch.as_tensor(obs_batch, device=self.device))   # (K, C, 7, 7)
        logits, v, _ = self.net.f(s_k)
        mask = torch.as_tensor(legal, device=self.device)
        probs = torch.softmax(logits.masked_fill(~mask[None], float("-inf")), -1).mean(0).cpu().numpy()
        return dict(state=s_k, priors=probs, legal=legal, value=float(v.mean()))

    @torch.no_grad()
    def step(self, s_k, a):
        """s_k has shape (K, C, 7, 7): each Rick transitions its own universe-conditioned latent."""
        K = self.net.n_ricks
        act_t = torch.tensor([a], device=self.device)
        next_lat, wl_list = [], []
        for k, rick in enumerate(self.net.ricks):
            sk, wlk = rick(s_k[k:k + 1], act_t)
            next_lat.append(sk[0])
            wl_list.append(wlk[0])
        lat = torch.stack(next_lat)                                        # (K, C, 7, 7)
        from .nets import N_STATIC, ballistic_legal, pack_hologram, scale_latent, unpack_hologram
        b_next = unpack_hologram(lat)
        dna = lat[:, -N_STATIC:]
        on = dna[:, 1:2]
        leg_next = ballistic_legal(b_next[:, 2:3], dna)
        s_h = self.net.h(torch.cat([b_next, dna, leg_next], dim=1))[:, :-N_STATIC]
        dyn = pack_hologram(scale_latent(0.85 * s_h + 0.15 * lat[:, :-N_STATIC], on), b_next, on)
        lat = torch.cat([dyn, dna], dim=1)
        win = torch.sigmoid(torch.stack(wl_list)).cpu().numpy()            # (K,)
        pol, val, leg = self.net.f(lat)
        val_np = val.cpu().numpy()
        q_k = win + (1.0 - win) * (-val_np)
        q, q_std, p_win = float(q_k.mean()), float(q_k.std()), float(win.mean())
        legal = (torch.sigmoid(leg).mean(0) > self.thr).cpu().numpy()
        if p_win >= 0.5:
            return dict(terminal=True, q=q, q_std=q_std)
        if not legal.any():
            return dict(terminal=True, q=p_win, q_std=float(win.std()))
        # Keep each Rick's universe DNA intact while syncing dynamic channels to one Thompson-sampled Rick
        idx = int(self.rng.integers(K))
        dyn = lat[idx:idx + 1, :-10].expand(K, -1, -1, -1)
        next_state = torch.cat([dyn, lat[:, -10:]], dim=1)
        priors = torch.softmax(pol, -1).mean(0).cpu().numpy()
        return dict(terminal=False, q=q, q_std=q_std, state=next_state, priors=priors, legal=legal,
                    value=float(val_np.mean()))


class CronenbergAgent:
    """Agent playing a multi-episode match in a universe with hidden rules.
    mode:
      * 'blind_morty': no posterior update; fills obs[5:13] = 0.5 and uses fast policy
      * 'cfc_morty': updates CentralFiniteCurve posterior; fills obs[5:13] with marginal rules; fast policy
      * 'cfc_citadel': updates CFC posterior; runs MultiverseAdapter search (full sims)
      * 'cfc_meeseeks': updates CFC posterior; runs MultiverseAdapter search with Meeseeks halting rule
    """
    needs_simulator = False

    def __init__(self, net, cfc: CentralFiniteCurve, mode="cfc_citadel", sims=48, beta=0.5,
                 rule=None, check_every=4, device="cpu", rng=None):
        self.net, self.cfc, self.mode, self.sims, self.beta = net, cfc, mode, sims, beta
        self.rule, self.check_every, self.device, self.rng = rule, check_every, device, rng
        self.moves = self.planned = self.expansions = 0

    def act(self, obs_blind: np.ndarray, legal: np.ndarray, game=None) -> int:
        from .meeseeks import CountingAdapter, Tracker
        from .search import net_eval

        if self.mode != "blind_morty":
            self.cfc.observe_turn(obs_blind, legal)

        self.moves += 1
        if self.mode == "blind_morty":
            obs = obs_blind.copy()
            obs[5:13] = 0.5
            p, _, _ = net_eval(self.net, obs, legal, self.device)
            return int(np.argmax(p))
        if self.mode == "cfc_morty":
            obs = self.cfc.fill_obs(obs_blind)
            p, _, _ = net_eval(self.net, obs, legal, self.device)
            return int(np.argmax(p))

        adapter = MultiverseAdapter(self.net, self.cfc, self.device, self.rng)
        if self.mode == "cfc_citadel" or self.rule is None:
            visits, _, n = search(adapter, (obs_blind, legal), self.sims, beta=self.beta)
            self.planned += 1
            self.expansions += n
            return int(np.argmax(visits))

        counter = CountingAdapter(adapter)
        tr = Tracker(counter, self.sims, self.check_every, rule=self.rule)
        _, _, n = search(counter, (obs_blind, legal), self.sims, beta=self.beta,
                         check_every=self.check_every, on_check=tr)
        self.expansions += n
        if n > 0:
            self.planned += 1
        return int(tr.trace[-1][2])


def play_cronenberg_match(net, u: Universe, episodes: int, mode: str, opp_spec: str,
                          sims: int = 48, beta: float = 0.5, rule=None, check_every: int = 4,
                          device: str = "cpu", seed: int = 0):
    """Play a match of `episodes` consecutive games in hidden-rule universe `u`."""
    from .agents import CitadelAgent, Gate, GreedyAgent, RandomAgent, UCTAgent
    from .multiverse import opening

    g0 = Game(u)
    H, W, blocked = infer_geometry(mask_rules(g0.encode()))
    cfc = CentralFiniteCurve(H, W, blocked)
    true_rules = np.array([float(u.gravity), float(u.wrap_h), float(u.wrap_v)] +
                          [float(s in u.shapes) for s in SHAPE_NAMES], dtype=np.float32)

    ep_records = []
    for ep in range(episodes):
        seat = 1 + (ep % 2)
        op_idx = ep // 2
        rng_agent = np.random.default_rng([99_001, seed, u.seed, ep, 1])
        rng_opp = np.random.default_rng([99_001, seed, u.seed, ep, 2])
        if opp_spec == "random":
            opp = RandomAgent(rng_opp)
        elif opp_spec == "greedy":
            opp = GreedyAgent(rng_opp)
        elif opp_spec.startswith("uct-"):
            opp = UCTAgent(int(opp_spec[4:]), rng_opp)
        else:
            raise ValueError(opp_spec)

        if mode == "full_info_citadel":
            agent = CitadelAgent(net, Gate("always"), sims, beta=beta, device=device, rng=rng_agent)
        else:
            agent = CronenbergAgent(net, cfc, mode=mode, sims=sims, beta=beta,
                                    rule=rule, check_every=check_every, device=device, rng=rng_agent)

        ent_start = cfc.entropy_bits()
        alive_start = cfc.n_alive
        marg = cfc.marginal_rules()
        brier = float(np.mean((marg - true_rules) ** 2))

        g = opening(u, op_idx)
        last_cell = None
        while not g.done:
            obs_full, legal = g.encode(), g.legal_mask()
            obs_blind = mask_rules(obs_full)
            if g.to_move == seat:
                if mode == "full_info_citadel":
                    cfc.observe_turn(obs_blind, legal)
                    a = agent.act(obs_full, legal)
                else:
                    a = agent.act(obs_blind, legal)
            else:
                if mode != "blind_morty":
                    cfc.observe_turn(obs_blind, legal)
                a = opp.act(obs_full, legal, g.copy() if opp.needs_simulator else None)
            last_cell = g.to_cell(a)
            g.step(a)

        if mode != "blind_morty":
            cfc.observe_game_end(g.board, g.winner, last_cell)

        score = 0.5 if g.winner == 0 else float(g.winner == seat)
        ep_records.append({
            "episode": ep + 1,
            "score": score,
            "entropy_bits_start": round(ent_start, 4),
            "entropy_bits_end": round(cfc.entropy_bits(), 4),
            "alive_start": alive_start,
            "alive_end": cfc.n_alive,
            "rule_brier_start": round(brier, 4),
            "expansions_per_move": round(agent.expansions / max(agent.moves, 1), 2),
            "plan_rate": round(agent.planned / max(agent.moves, 1), 3),
        })
    return ep_records


def evaluate_cronenberg(run_dir: str, split: str = "val", n_universes: int = 12,
                        episodes: int = 6, opponent: str = "greedy", sims: int = 48,
                        budget: str = "0.25", device: str = "cpu", out_path: str | None = None):
    """Evaluate hidden-rule Cronenberg matches across held-out universes and save summary JSON."""
    import json
    import os
    from .meeseeks import HaltModel, MeeseeksRule
    from .multiverse import universes
    from .train import load_ckpt

    net, ck = load_ckpt(os.path.join(run_dir, "citadel.pt"), device)
    seed = int(ck["cfg"]["seed"])
    beta = float(ck["cfg"].get("beta", 0.5))
    check_every = int(ck["cfg"].get("check_every", 4))

    rule = None
    m_path = os.path.join(run_dir, "meeseeks.pt")
    if os.path.exists(m_path):
        mb = torch.load(m_path, map_location="cpu", weights_only=False)
        mode_d = mb["modes"]["latent"]
        hm = HaltModel(len(mb["features"]))
        hm.load_state_dict(mode_d["model"])
        hm.eval()
        tau = float(mode_d["thresholds"]["meeseeks"].get(budget, 0.02))
        rule = MeeseeksRule(tau, int(mb["max_sims"]), hm, head="regret")

    unis = universes(split, n_universes)
    modes = ["blind_morty", "cfc_morty", "cfc_meeseeks", "cfc_citadel", "full_info_citadel"]
    summary_modes = {}
    post_curve = None

    for m in modes:
        all_eps = [
            play_cronenberg_match(net, u, episodes=episodes, mode=m, opp_spec=opponent,
                                  sims=sims, beta=beta, rule=rule if m == "cfc_meeseeks" else None,
                                  check_every=check_every, device=device, seed=seed)
            for u in unis
        ]
        score_by_ep = [round(float(np.mean([r[ep]["score"] for r in all_eps])), 4) for ep in range(episodes)]
        exps_by_ep = [round(float(np.mean([r[ep]["expansions_per_move"] for r in all_eps])), 2) for ep in range(episodes)]
        plan_by_ep = [round(float(np.mean([r[ep]["plan_rate"] for r in all_eps])), 3) for ep in range(episodes)]
        summary_modes[m] = {
            "score_by_episode": score_by_ep,
            "expansions_by_episode": exps_by_ep,
            "plan_rate_by_episode": plan_by_ep,
            "mean_score": round(float(np.mean(score_by_ep)), 4),
            "early_score_ep1_2": round(float(np.mean(score_by_ep[:2])), 4),
            "late_score_ep5_6": round(float(np.mean(score_by_ep[-2:])), 4),
            "mean_expansions": round(float(np.mean(exps_by_ep)), 2),
        }
        if m == "cfc_citadel":
            post_curve = {
                "entropy_bits_start": [round(float(np.mean([r[ep]["entropy_bits_start"] for r in all_eps])), 4) for ep in range(episodes)],
                "entropy_bits_end": [round(float(np.mean([r[ep]["entropy_bits_end"] for r in all_eps])), 4) for ep in range(episodes)],
                "alive_start": [round(float(np.mean([r[ep]["alive_start"] for r in all_eps])), 2) for ep in range(episodes)],
                "alive_end": [round(float(np.mean([r[ep]["alive_end"] for r in all_eps])), 2) for ep in range(episodes)],
                "rule_brier_start": [round(float(np.mean([r[ep]["rule_brier_start"] for r in all_eps])), 4) for ep in range(episodes)],
            }
        print(f"Cronenberg [{m:<18}] mean={summary_modes[m]['mean_score']:.3f} "
              f"ep1={score_by_ep[0]:.3f} -> ep{episodes}={score_by_ep[-1]:.3f} "
              f"exps/mv={summary_modes[m]['mean_expansions']:.1f}", flush=True)

    result = {
        "split": split,
        "n_universes": len(unis),
        "episodes": episodes,
        "opponent": opponent,
        "sims": sims,
        "budget": budget,
        "train_seed": seed,
        "posterior_curve": post_curve,
        "modes": summary_modes,
    }
    out = out_path or os.path.join(run_dir, f"cronenberg_{split}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Saved Cronenberg results -> {out}")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Cronenberg hidden-rule matches with Central Finite Curve.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--universes", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--budget", default="0.25")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    evaluate_cronenberg(args.run, split=args.split, n_universes=args.universes, episodes=args.episodes,
                        opponent=args.opponent, sims=args.sims, budget=args.budget, device=args.device,
                        out_path=args.out)


if __name__ == "__main__":
    main()


