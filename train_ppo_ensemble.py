import os
import sys
import random
import numpy as np
import torch
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
    collect_rollout, compute_gae, ppo_update, evaluate,
    LATENT_DIM, HIDDEN_DIM, GAMMA, GAE_LAMBDA, CLIP_EPS,
    ENTROPY_COEF, VALUE_COEF, UPDATE_STEPS, ROLLOUT_STEPS,
    DEVICE, LR,
)

NUM_HEADS = 5
EPOCHS = 500
EVAL_INTERVAL = 50
JSD_THRESHOLD = 0.5
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
                valid_mask = get_valid_mask(env)
                with torch.no_grad():
                    action, _, _, logits_list, _ = policy.get_action_and_value(z.unsqueeze(0), valid_mask, deterministic=True)
                    use_plan, jsd, max_p = should_plan(logits_list, valid_mask, JSD_THRESHOLD, MAX_PROB_THRESHOLD)
                if use_plan.item():
                    slow += 1
                else:
                    fast += 1
                action_int = action.item()
            else:
                action_int = random.choice(env.get_valid_actions())
            state, _, done, info = env.step(action_int)
        if info["winner"] == 1:
            wins += 1
    total = fast + slow
    return wins / num_games, fast / total if total > 0 else 0, slow / total if total > 0 else 0


def main():
    print("=" * 55)
    print("  BASIL — Ensemble PPO Training (Merged)")
    print("=" * 55)
    print(f"  Device: {DEVICE}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  JSD threshold: {JSD_THRESHOLD}")
    print(f"  Epochs: {EPOCHS}")
    print()

    encoder = load_encoder()
    policy = MultiHeadPolicy(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
        num_actions=MAX_ACTIONS, num_heads=NUM_HEADS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    print("  Training on TicTacToe vs random (ensemble)...")
    print(f"  {'Epoch':>7} | {'Win Rate':>10} | {'Fast %':>8} | {'Slow %':>8}")
    print("  " + "-" * 42)

    env = TicTacToe()
    env.reset()

    for epoch in range(EPOCHS):
        states, actions, log_probs, values, rewards, dones, masks = collect_rollout(
            env, encoder, policy
        )
        advantages, returns = compute_gae(rewards, values, dones)
        ppo_update(policy, optimizer, states, actions, log_probs, advantages, returns, masks)

        if (epoch + 1) % EVAL_INTERVAL == 0:
            wr, fast_pct, slow_pct = evaluate_with_uncertainty(encoder, policy, TicTacToe)
            print(f"  {epoch+1:5d}   | {wr:>9.1%} | {fast_pct:>7.1%} | {slow_pct:>7.1%}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS,
    }, "checkpoints/policy_ensemble.pt")
    print("  Saved -> checkpoints/policy_ensemble.pt")


if __name__ == "__main__":
    main()
