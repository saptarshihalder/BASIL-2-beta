"""PPO training of the 5-head policy, one checkpoint per game.

Fixes against the original:
  * --game: Connect4 gets its own policy (the evaluation scripts used to load the
    TicTacToe-only policy for Connect4).
  * The agent alternates seats during training, so it can be evaluated moving second too.
  * Bootstrapped heads: each head learns from its own random half of every batch, so their
    disagreement (JSD) is a real uncertainty signal. Before, the loss only saw the heads'
    average, so all heads got identical gradients and their differences never changed.
  * Seeds, and an optional stronger training opponent (--opponent greedy|mixed).

Usage:  python train_ppo.py --games tictactoe,connect4 --episodes 3000 --seed 0
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.policy import MultiHeadPolicy
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS
from utils.common import GAMES, make_opponent, policy_path, set_seed

LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 5
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
UPDATE_STEPS = 4
BATCH_SIZE = 64
MIN_STEPS = 256
EVAL_INTERVAL = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt_path = "checkpoints/autoencoder.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Run train_autoencoder.py first (a random encoder makes results meaningless).")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def encode(encoder, state):
    padded = pad_state(state)
    x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(x).squeeze(0)


def get_valid_mask(env):
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
    for a in env.get_valid_actions():
        mask[a] = True
    return mask.unsqueeze(0)


def collect_episode(encoder, policy, env_class, seat, opponent):
    """One game with the agent in `seat` (1 or 2). Rewards are from the agent's side."""
    env = env_class()
    state = env.reset()
    traj = {"states": [], "actions": [], "log_probs": [], "values": [], "masks": []}
    done = False
    while not done:
        if env.current_player == seat:
            z = encode(encoder, state)
            valid_mask = get_valid_mask(env)
            with torch.no_grad():
                action, log_prob, value, _, _ = policy.get_action_and_value(z.unsqueeze(0), valid_mask)
            action_int = action.item()
            traj["states"].append(z)
            traj["actions"].append(action.squeeze())
            traj["log_probs"].append(log_prob.squeeze())
            traj["values"].append(value.item())
            traj["masks"].append(valid_mask.squeeze(0))
        else:
            action_int = opponent(env)
        state, _, done, info = env.step(action_int)

    winner = info["winner"]
    final_reward = 1.0 if winner == seat else (-1.0 if winner == 3 - seat else 0.0)
    n = len(traj["states"])
    rewards, dones = [0.0] * n, [False] * n
    if n > 0:
        rewards[-1], dones[-1] = final_reward, True
    return traj, rewards, dones, info


def compute_gae(rewards, values, dones):
    advantages, gae = [], 0.0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] * (1 - float(dones[t])) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(policy, optimizer, states, actions, old_log_probs, advantages, returns, masks):
    states_t = torch.stack(states).to(DEVICE)
    actions_t = torch.stack(actions).to(DEVICE)
    old_lp_t = torch.stack(old_log_probs).to(DEVICE).detach()
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    masks_t = torch.stack(masks).to(DEVICE)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # Bootstrap: head k trains only on samples where boot[k, i] == 1 (fixed for this batch).
    boot = (torch.rand(policy.num_heads, len(states_t), device=DEVICE) < 0.5).float()

    for _ in range(UPDATE_STEPS):
        idx = torch.randperm(len(states_t), device=DEVICE)
        for start in range(0, len(states_t), BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            lp, ent, new_val = policy.evaluate_actions_per_head(states_t[b], actions_t[b], masks_t[b])
            ratio = (lp - old_lp_t[b]).exp()  # each head vs the behaviour (averaged) policy
            surr = torch.min(ratio * adv_t[b], torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[b])
            w = boot[:, b]
            denom = w.sum().clamp(min=1.0)
            loss = (-(surr * w).sum() / denom
                    + VALUE_COEF * F.mse_loss(new_val, ret_t[b])
                    - ENTROPY_COEF * (ent * w).sum() / denom)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


def evaluate(encoder, policy, env_class, num_games=200, seed=12345):
    """Fast-path score vs random, half the games in each seat."""
    rng = random.Random(seed)
    opponent = make_opponent("random", rng)
    wins = 0
    for g in range(num_games):
        seat = 1 + g % 2
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == seat:
                z = encode(encoder, state)
                with torch.no_grad():
                    action, _, _, _, _ = policy.get_action_and_value(z.unsqueeze(0), get_valid_mask(env), deterministic=True)
                a = action.item()
            else:
                a = opponent(env)
            state, _, done, info = env.step(a)
        wins += info["winner"] == seat
    return wins / num_games


def train_game(game, episodes, seed, opponent_name="random"):
    env_class, _ = GAMES[game]
    set_seed(seed)
    rng = random.Random(seed)
    encoder = load_encoder()
    policy = MultiHeadPolicy(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                             num_actions=MAX_ACTIONS, num_heads=NUM_HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    opponents = {"random": [make_opponent("random", rng)], "greedy": [make_opponent("greedy", rng)],
                 "mixed": [make_opponent("random", rng), make_opponent("greedy", rng)]}[opponent_name]

    print(f"\n  Training {game} for {episodes} episodes (opponent: {opponent_name}, seed {seed})")
    buf = {k: [] for k in ("s", "a", "lp", "v", "r", "d", "m")}
    for ep in range(1, episodes + 1):
        traj, rewards, dones, _ = collect_episode(encoder, policy, env_class, seat=1 + ep % 2,
                                                  opponent=rng.choice(opponents))
        if not traj["states"]:
            continue
        buf["s"] += traj["states"]; buf["a"] += traj["actions"]; buf["lp"] += traj["log_probs"]
        buf["v"] += traj["values"]; buf["r"] += rewards; buf["d"] += dones; buf["m"] += traj["masks"]
        if len(buf["s"]) >= MIN_STEPS:
            adv, ret = compute_gae(buf["r"], buf["v"], buf["d"])
            ppo_update(policy, optimizer, buf["s"], buf["a"], buf["lp"], adv, ret, buf["m"])
            buf = {k: [] for k in buf}
        if ep % EVAL_INTERVAL == 0:
            print(f"  {ep:6d} | fast-path win rate vs random (both seats): {evaluate(encoder, policy, env_class):.1%}")

    os.makedirs("checkpoints", exist_ok=True)
    path = policy_path(game)
    torch.save({"model_state_dict": policy.state_dict(), "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
                "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS, "game": game, "seed": seed,
                "episodes": episodes, "opponent": opponent_name}, path)
    print(f"  Saved -> {path}")
    return policy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="tictactoe,connect4")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponent", default="random", choices=["random", "greedy", "mixed"])
    args = ap.parse_args(argv)
    for game in args.games.split(","):
        train_game(game, args.episodes, args.seed, args.opponent)


if __name__ == "__main__":
    main()
