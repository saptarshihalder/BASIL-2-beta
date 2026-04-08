import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.policy import MultiHeadPolicy
from utils.uncertainty import should_plan
from envs.tictactoe import TicTacToe
from train_ppo import (
    load_encoder, encode, get_valid_mask,
    collect_p1_episode, compute_gae, ppo_update, evaluate,
    LATENT_DIM, HIDDEN_DIM, DEVICE, LR, MIN_STEPS, MAX_ACTIONS,
)

NUM_HEADS = 5
EPOCHS = 1000
EVAL_INTERVAL = 50
JSD_THRESHOLD = 0.5
MAX_PROB_THRESHOLD = 0.6


def evaluate_with_uncertainty(encoder, policy, env_class, num_games=100):
    wins, fast, slow = 0, 0, 0
    for _ in range(num_games):
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                z = encode(encoder, state)
                valid_mask = get_valid_mask(env)
                with torch.no_grad():
                    action, _, _, logits_list, _ = policy.get_action_and_value(
                        z.unsqueeze(0), valid_mask, deterministic=True
                    )
                    use_plan, _, _ = should_plan(logits_list, valid_mask, JSD_THRESHOLD, MAX_PROB_THRESHOLD)
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
    print("  BASIL — Ensemble PPO Training")
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

    print("  Training on TicTacToe vs random...")
    print(f"  {'Epoch':>7} | {'Win Rate':>10} | {'Fast %':>8} | {'Slow %':>8}")
    print("  " + "-" * 42)

    buf_s, buf_a, buf_lp, buf_v, buf_r, buf_d, buf_m = [], [], [], [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        p1, rewards, dones, info = collect_p1_episode(encoder, policy, TicTacToe)
        if len(p1["states"]) == 0:
            continue

        buf_s.extend(p1["states"])
        buf_a.extend(p1["actions"])
        buf_lp.extend(p1["log_probs"])
        buf_v.extend(p1["values"])
        buf_r.extend(rewards)
        buf_d.extend(dones)
        buf_m.extend(p1["masks"])

        if len(buf_s) >= MIN_STEPS:
            adv, ret = compute_gae(buf_r, buf_v, buf_d)
            ppo_update(policy, optimizer, buf_s, buf_a, buf_lp, adv, ret, buf_m)
            buf_s, buf_a, buf_lp, buf_v, buf_r, buf_d, buf_m = [], [], [], [], [], [], []

        if epoch % EVAL_INTERVAL == 0:
            wr, fp, sp = evaluate_with_uncertainty(encoder, policy, TicTacToe)
            print(f"  {epoch:5d}   | {wr:>9.1%} | {fp:>7.1%} | {sp:>7.1%}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS,
    }, "checkpoints/policy_ensemble.pt")
    print("  Saved -> checkpoints/policy_ensemble.pt")


if __name__ == "__main__":
    main()
