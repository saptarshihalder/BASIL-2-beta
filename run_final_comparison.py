import os
import sys
import random
import json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.policy import MultiHeadPolicy
from models.planning import PlanningModule
from models.world_model_interface import load_world_model
from utils.uncertainty import should_plan
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 5
JSD_THRESHOLD = 1.5
MAX_PROB_THRESHOLD = 0.3
NUM_ROLLOUTS = 10
HORIZON = 5
NUM_GAMES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt = torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_policy(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    policy = MultiHeadPolicy(
        latent_dim=ckpt.get("latent_dim", LATENT_DIM),
        hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
        num_actions=ckpt.get("num_actions", MAX_ACTIONS),
        num_heads=ckpt.get("num_heads", NUM_HEADS),
    ).to(DEVICE)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    return policy


def encode(encoder, state):
    padded = pad_state(state)
    x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(x)


def get_valid_mask(env):
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
    for a in env.get_valid_actions():
        mask[a] = True
    return mask.unsqueeze(0)


def run_eval(encoder, policy, planner, env_class, num_games, mode, game_name):
    wins, losses, draws = 0, 0, 0
    fast_moves, slow_moves = 0, 0
    total_sims = 0

    for _ in tqdm(range(num_games), desc=f"  {mode:>10s} {game_name}", ncols=65):
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                z = encode(encoder, state)
                valid_mask = get_valid_mask(env)
                valid_actions = env.get_valid_actions()

                with torch.no_grad():
                    action, _, _, logits_list, _ = policy.get_action_and_value(
                        z, valid_mask, deterministic=True
                    )

                if mode == "ppo_only":
                    action_int = action.item()
                    fast_moves += 1

                elif mode == "thinking_only":
                    action_int = planner.plan(z, valid_actions, device=DEVICE)
                    slow_moves += 1
                    total_sims += NUM_ROLLOUTS

                elif mode == "adaptive":
                    use_plan, jsd, max_p = should_plan(
                        logits_list, valid_mask, JSD_THRESHOLD, MAX_PROB_THRESHOLD
                    )
                    if use_plan.item() and planner is not None:
                        action_int = planner.plan(z, valid_actions, device=DEVICE)
                        slow_moves += 1
                        total_sims += NUM_ROLLOUTS
                    else:
                        action_int = action.item()
                        fast_moves += 1
                else:
                    action_int = action.item()
                    fast_moves += 1
            else:
                action_int = random.choice(env.get_valid_actions())

            state, _, done, info = env.step(action_int)

        if info["winner"] == 1:
            wins += 1
        elif info["winner"] == 2:
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    total_moves = fast_moves + slow_moves
    return {
        "mode": mode,
        "game": game_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "fast_moves": fast_moves,
        "slow_moves": slow_moves,
        "fast_pct": fast_moves / total_moves if total_moves > 0 else 0,
        "slow_pct": slow_moves / total_moves if total_moves > 0 else 0,
        "total_sims": total_sims,
        "avg_sims_per_game": total_sims / num_games,
    }


def main():
    print("=" * 65)
    print("  BASIL — Final Comparison: PPO vs Thinking vs Adaptive")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Games per test: {NUM_GAMES}")
    print(f"  JSD threshold: {JSD_THRESHOLD}, Max prob: {MAX_PROB_THRESHOLD}")
    print()

    encoder = load_encoder()
    policy_path = "checkpoints/policy_ensemble.pt"
    if not os.path.exists(policy_path):
        policy_path = "checkpoints/policy_basic.pt"
    policy = load_policy(policy_path)
    print(f"  Policy: {policy_path}")

    wm = load_world_model("checkpoints/world_model.pt", str(DEVICE))
    planner = PlanningModule(wm, policy=policy, num_rollouts=NUM_ROLLOUTS, horizon=HORIZON)
    print(f"  Planner: {NUM_ROLLOUTS} rollouts x {HORIZON} steps")

    all_results = []

    for game_name, env_class in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        print(f"\n  --- {game_name} ---")
        n = NUM_GAMES

        for mode in ["ppo_only", "thinking_only", "adaptive"]:
            games = 200 if mode == "thinking_only" else n
            r = run_eval(encoder, policy, planner, env_class, games, mode, game_name)
            all_results.append(r)

    print("\n" + "=" * 65)
    print(f"  {'Mode':>14s} | {'Game':>5s} | {'Win%':>6s} | {'Fast%':>6s} | {'Slow%':>6s} | {'Sims/Game':>9s}")
    print("  " + "-" * 55)
    for r in all_results:
        print(f"  {r['mode']:>14s} | {r['game']:>5s} | {r['win_rate']:>5.1%} | "
              f"{r['fast_pct']:>5.1%} | {r['slow_pct']:>5.1%} | {r['avg_sims_per_game']:>9.1f}")
    print("=" * 65)

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/final_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved -> experiments/final_comparison.json")


if __name__ == "__main__":
    main()
