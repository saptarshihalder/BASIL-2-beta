import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.policy import MultiHeadPolicy
from utils.uncertainty import should_plan
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS
from train_ppo import (
    load_encoder, encode, get_valid_mask,
    collect_self_play_episode, compute_gae, ppo_update, evaluate,
    LATENT_DIM, HIDDEN_DIM, GAMMA, GAE_LAMBDA, CLIP_EPS,
    ENTROPY_COEF, VALUE_COEF, PPO_EPOCHS, BATCH_SIZE, DEVICE, LR,
)

NUM_HEADS = 5
MAX_EPISODES = 1000
EVAL_INTERVAL = 50
JSD_THRESHOLD = 1.0
MAX_PROB_THRESHOLD = 0.6


def evaluate_with_uncertainty(encoder, policy, env_class, num_games=100):
    wins = 0
    fast = 0
    slow = 0
    for _ in range(num_games):
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                z = encode(encoder, state)
                valid_mask = get_valid_mask(env, DEVICE)
                with torch.no_grad():
                    action, _, _, logits_list = policy.get_action_and_value(z, valid_mask, deterministic=True)
                    use_plan, jsd, max_p = should_plan(logits_list, valid_mask, JSD_THRESHOLD, MAX_PROB_THRESHOLD)
                if use_plan.item():
                    slow += 1
                else:
                    fast += 1
                action_int = action.item()
            else:
                valid = env.get_valid_actions()
                action_int = random.choice(valid)
            state, reward, done, info = env.step(action_int)
        if info["winner"] == 1:
            wins += 1
    total = fast + slow
    return wins / num_games, fast / total if total > 0 else 0, slow / total if total > 0 else 0


def main():
    print("=" * 58)
    print("  BASIL — Ensemble PPO Training (Step 4)")
    print("=" * 58)
    print(f"  Device: {DEVICE}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  JSD threshold: {JSD_THRESHOLD}")
    print()

    encoder = load_encoder()
    policy = MultiHeadPolicy(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
        num_actions=MAX_ACTIONS, num_heads=NUM_HEADS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    print("Training on TicTacToe vs random (with ensemble)...")
    print(f"{'Episode':>8} | {'Win Rate':>10} | {'Fast %':>8} | {'Slow %':>8}")
    print("-" * 45)

    all_states, all_actions, all_lps, all_values, all_rewards, all_dones, all_masks = [], [], [], [], [], [], []

    for ep in range(1, MAX_EPISODES + 1):
        p1_data, rewards_p1, info = collect_self_play_episode(encoder, policy, TicTacToe)

        if len(p1_data["states"]) == 0:
            continue

        dones = [False] * len(rewards_p1)
        dones[-1] = True
        advantages, returns = compute_gae(rewards_p1, p1_data["values"], dones)

        all_states.extend(p1_data["states"])
        all_actions.extend(p1_data["actions"])
        all_lps.extend(p1_data["log_probs"])
        all_values.extend(p1_data["values"])
        all_rewards.extend(rewards_p1)
        all_dones.extend(dones)
        all_masks.extend(p1_data["masks"])

        if ep % 10 == 0 and len(all_states) >= BATCH_SIZE:
            adv_all, ret_all = compute_gae(all_rewards, all_values, all_dones)
            ppo_update(policy, optimizer, all_states, all_actions, all_lps, adv_all, ret_all, all_masks)
            all_states, all_actions, all_lps, all_values, all_rewards, all_dones, all_masks = [], [], [], [], [], [], []

        if ep % EVAL_INTERVAL == 0:
            wr, fast_pct, slow_pct = evaluate_with_uncertainty(encoder, policy, TicTacToe)
            print(f"  {ep:5d}   | {wr:>9.1%} | {fast_pct:>7.1%} | {slow_pct:>7.1%}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS,
    }, "checkpoints/policy_ensemble.pt")
    print("Saved -> checkpoints/policy_ensemble.pt")


if __name__ == "__main__":
    main()
