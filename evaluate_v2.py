"""Fixed evaluation of the original BASIL pipeline (replaces main_basil.py and
run_final_comparison.py, which now just call this file).

What changed and why:
  * Per-game policy checkpoints (the old scripts played Connect4 with the TicTacToe policy).
  * The agent plays both seats; results are reported per seat and combined.
  * Paired, seeded games: game i of a given (game, opponent, seat) uses the same random
    numbers for every mode, so mode differences are not luck of the draw.
  * Opponents: random, greedy (win/block/centre) and perfect minimax for TicTacToe.
  * Gates are compared at the same planning rate. The JSD threshold is calibrated on separate
    calibration games (different seeds, random opponent) to hit --plan-rate; 'confidence_gate'
    (low max-prob) and 'random_gate' (coin flip) use the same rate, so any gain from 'adaptive'
    can be attributed to the ensemble signal rather than to planning more.
  * Wilson 95% intervals on win rate; world-model calls counted for compute accounting.

Usage:
  python evaluate_v2.py --games tictactoe,connect4 --opponents random,greedy,minimax --n 500
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.policy import MultiHeadPolicy
from models.planning import PlanningModule
from models.world_model_interface import load_world_model
from utils.uncertainty import compute_jsd, get_max_prob
from utils.common import GAMES, make_opponent, policy_path, set_seed, stable_seed
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    enc = Encoder(in_channels=3, latent_dim=64, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt = torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=True)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    return enc.eval()


def load_policy(game):
    path = policy_path(game)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run: python train_ppo.py --games {game}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if ckpt.get("game", game) != game:
        raise ValueError(f"{path} was trained on {ckpt['game']}, not {game}")
    pol = MultiHeadPolicy(latent_dim=ckpt["latent_dim"], hidden_dim=ckpt["hidden_dim"],
                          num_actions=ckpt["num_actions"], num_heads=ckpt["num_heads"]).to(DEVICE)
    pol.load_state_dict(ckpt["model_state_dict"])
    return pol.eval()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def paired_ci(a, b, seed=0, n_boot=2000):
    # Pair games by their seed/index, keeping both seats in the same bootstrap draw.
    n = min(len(a[1]), len(a[2]), len(b[1]), len(b[2]))
    if n == 0:
        return None
    delta = np.array([(a[1][i] + a[2][i] - b[1][i] - b[2][i]) / 2 for i in range(n)])
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(n, size=(n_boot, n))].mean(axis=1)
    return {"delta": round(float(delta.mean()), 4),
            "ci95": [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])],
            "pairs": n}


class Agent:
    def __init__(self, encoder, policy, planner, mode, thresholds, rate):
        self.encoder, self.policy, self.planner = encoder, policy, planner
        self.mode, self.thr, self.rate = mode, thresholds, rate
        self.fast = self.slow = 0
        self.rng = random.Random(0)

    def scores(self, env, state):
        x = torch.tensor(pad_state(state).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
        mask[0, env.get_valid_actions()] = True
        with torch.no_grad():
            z = self.encoder(x)
            action, _, _, logits, _ = self.policy.get_action_and_value(z, mask, deterministic=True)
            jsd = compute_jsd(logits, mask).item()
            maxp = get_max_prob(logits, mask).item()
        return z, action.item(), jsd, maxp

    def act(self, env, state):
        z, fast_action, jsd, maxp = self.scores(env, state)
        plan = {"ppo_only": False,
                "thinking_only": True,
                "adaptive": jsd >= self.thr["jsd"],
                "confidence_gate": maxp <= self.thr["maxp"],
                "random_gate": self.rng.random() < self.rate}[self.mode]
        if plan:
            self.slow += 1
            return self.planner.plan(z, env.get_valid_actions(), device=DEVICE)
        self.fast += 1
        return fast_action


def play(agent, opponent, env_class, seat):
    env = env_class()
    state, done = env.reset(), False
    while not done:
        a = agent.act(env, state) if env.current_player == seat else opponent(env)
        if a not in env.get_valid_actions():
            return -1, True  # illegal move = forfeit, counted and reported
        state, _, done, info = env.step(a)
    w = info["winner"]
    return (1 if w == seat else -1 if w == 3 - seat else 0), False


def calibrate(agent, env_class, rate, games, seed):
    """Thresholds that make the agent plan on `rate` of its moves, from calibration games only."""
    rng = random.Random(seed)
    opponent = make_opponent("random", rng)
    jsds, maxps = [], []
    for g in range(games):
        seat = 1 + g % 2
        env = env_class()
        state, done = env.reset(), False
        while not done:
            if env.current_player == seat:
                _, a, jsd, maxp = agent.scores(env, state)
                jsds.append(jsd)
                maxps.append(maxp)
            else:
                a = opponent(env)
            state, _, done, _ = env.step(a)
    return {"jsd": float(np.quantile(jsds, 1 - rate)), "maxp": float(np.quantile(maxps, rate)),
            "calibration_moves": len(jsds)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="tictactoe,connect4")
    ap.add_argument("--modes", default="ppo_only,thinking_only,adaptive,confidence_gate,random_gate")
    ap.add_argument("--opponents", default="random,greedy,minimax")
    ap.add_argument("--n", type=int, default=500, help="games per seat")
    ap.add_argument("--n-thinking", type=int, default=100, help="games per seat for thinking_only (slow)")
    ap.add_argument("--plan-rate", type=float, default=0.2)
    ap.add_argument("--calib-games", type=int, default=200)
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/eval_v2.json")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    encoder = load_encoder()
    wm = load_world_model("checkpoints/world_model.pt", str(DEVICE))
    rows = []
    for game in args.games.split(","):
        env_class, n_actions = GAMES[game]
        policy = load_policy(game)
        probe = Agent(encoder, policy, None, "ppo_only", {}, args.plan_rate)
        thr = calibrate(probe, env_class, args.plan_rate, args.calib_games, seed=stable_seed("calib", game, args.seed))
        print(f"\n[{game}] thresholds for plan rate {args.plan_rate:.0%}: JSD >= {thr['jsd']:.3f}, max-prob <= {thr['maxp']:.3f}")
        for opp_name in args.opponents.split(","):
            if opp_name == "minimax" and game != "tictactoe":
                continue
            game_scores, game_rows = {}, {}
            for mode in args.modes.split(","):
                planner = PlanningModule(wm, policy=policy, num_rollouts=args.rollouts,
                                         horizon=args.horizon, num_actions=n_actions)
                agent = Agent(encoder, policy, planner, mode, thr, args.plan_rate)
                n = args.n_thinking if mode == "thinking_only" else args.n
                seat_stats, illegal, seat_scores = {}, 0, {}
                for seat in (1, 2):
                    wdl, outcomes = [0, 0, 0], []
                    for i in range(n):
                        key = stable_seed(game, opp_name, seat, i, args.seed)  # same for every mode
                        set_seed(key)
                        agent.rng = random.Random(key + 1)  # gate coin flips: own stream
                        r, bad = play(agent, make_opponent(opp_name, random.Random(key)), env_class, seat)
                        illegal += bad
                        outcomes.append((r + 1) / 2)
                        wdl[{1: 0, 0: 1, -1: 2}[r]] += 1
                    seat_scores[seat] = outcomes
                    seat_stats[f"seat{seat}"] = {"wins": wdl[0], "draws": wdl[1], "losses": wdl[2]}
                w = sum(s["wins"] for s in seat_stats.values())
                d = sum(s["draws"] for s in seat_stats.values())
                total = 2 * n
                moves = agent.fast + agent.slow
                row = {"game": game, "opponent": opp_name, "mode": mode, "games": total,
                       "win_rate": round(w / total, 4), "win_ci95": wilson(w, total),
                       "score": round((w + 0.5 * d) / total, 4), "loss_rate": round((total - w - d) / total, 4),
                       "plan_rate": round(agent.slow / max(moves, 1), 4),
                       "wm_calls_per_game": round(planner.simulations / total, 1),
                       "illegal_moves": illegal, "thresholds": thr, **seat_stats}
                rows.append(row)
                game_scores[mode], game_rows[mode] = seat_scores, row
                print(f"  {opp_name:>8} {mode:>16}: win {row['win_rate']:.1%} {row['win_ci95']}  "
                      f"loss {row['loss_rate']:.1%}  plan {row['plan_rate']:.1%}  calls/game {row['wm_calls_per_game']}")
            for mode, row in game_rows.items():
                for ref in ("ppo_only", "random_gate"):
                    if mode != ref and ref in game_scores:
                        row[f"paired_vs_{ref}"] = paired_ci(game_scores[mode], game_scores[ref],
                                                              seed=stable_seed(game, opp_name, mode, ref, args.seed))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
