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
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 1
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64
MAX_EPISODES = 500
EVAL_INTERVAL = 50
TARGET_WINRATE = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt_path = "checkpoints/autoencoder.pt"
    if os.path.exists(ckpt_path):
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
        return encoder(x)


def get_valid_mask(env, device):
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=device)
    for a in env.get_valid_actions():
        mask[a] = True
    return mask.unsqueeze(0)


def collect_episode(encoder, policy, env_class):
    env = env_class()
    state = env.reset()
    states, actions, log_probs, values, rewards, dones, masks = [], [], [], [], [], [], []

    done = False
    while not done:
        z = encode(encoder, state)
        valid_mask = get_valid_mask(env, DEVICE)

        with torch.no_grad():
            action, log_prob, value, _ = policy.get_action_and_value(z, valid_mask)

        action_int = action.item()

        if env.current_player == 1:
            next_state, reward, done, info = env.step(action_int)
        else:
            next_state, reward, done, info = env.step(action_int)
            reward = -reward

        states.append(z.squeeze(0))
        actions.append(action_int)
        log_probs.append(log_prob.item())
        values.append(value.item())
        rewards.append(reward)
        dones.append(done)
        masks.append(valid_mask.squeeze(0))

        state = next_state

    return states, actions, log_probs, values, rewards, dones, masks, info


def collect_self_play_episode(encoder, policy, env_class):
    env = env_class()
    state = env.reset()

    p1_data = {"states": [], "actions": [], "log_probs": [], "values": [], "masks": []}
    all_rewards_p1 = []

    done = False
    while not done:
        z = encode(encoder, state)
        valid_mask = get_valid_mask(env, DEVICE)

        if env.current_player == 1:
            with torch.no_grad():
                action, log_prob, value, _ = policy.get_action_and_value(z, valid_mask)
            action_int = action.item()
            p1_data["states"].append(z.squeeze(0))
            p1_data["actions"].append(action_int)
            p1_data["log_probs"].append(log_prob.item())
            p1_data["values"].append(value.item())
            p1_data["masks"].append(valid_mask.squeeze(0))
        else:
            valid = env.get_valid_actions()
            action_int = random.choice(valid)

        state, reward, done, info = env.step(action_int)

    if info["winner"] == 1:
        final_reward = 1.0
    elif info["winner"] == 2:
        final_reward = -1.0
    else:
        final_reward = 0.0

    n_moves = len(p1_data["states"])
    rewards_p1 = [0.0] * n_moves
    if n_moves > 0:
        rewards_p1[-1] = final_reward

    return p1_data, rewards_p1, info


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t] if t < len(dones) else True)) - values[t]
        gae = delta + gamma * lam * (1.0 - float(dones[t] if t < len(dones) else True)) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(policy, optimizer, states, actions, old_log_probs, advantages, returns, masks):
    states_t = torch.stack(states).to(DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
    old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32).to(DEVICE)
    adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
    ret_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    masks_t = torch.stack(masks).to(DEVICE)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(len(states_t))
        for start in range(0, len(states_t), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(states_t))
            idx = indices[start:end]

            new_lp, entropy, new_val = policy.evaluate_actions(
                states_t[idx], actions_t[idx], masks_t[idx]
            )

            ratio = (new_lp - old_lp_t[idx]).exp()
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_val, ret_t[idx])
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def evaluate(encoder, policy, env_class, num_games=100):
    wins = 0
    for _ in range(num_games):
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                z = encode(encoder, state)
                valid_mask = get_valid_mask(env, DEVICE)
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(z, valid_mask, deterministic=True)
                action_int = action.item()
            else:
                valid = env.get_valid_actions()
                action_int = random.choice(valid)
            state, reward, done, info = env.step(action_int)
        if info["winner"] == 1:
            wins += 1
    return wins / num_games


def main():
    print("=" * 58)
    print("  BASIL — PPO Training (Step 3: Basic PPO)")
    print("=" * 58)
    print(f"  Device: {DEVICE}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  Target: {TARGET_WINRATE:.0%} win rate on TTT")
    print()

    encoder = load_encoder()
    policy = MultiHeadPolicy(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
        num_actions=MAX_ACTIONS, num_heads=NUM_HEADS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    print("Training on TicTacToe vs random...")
    print(f"{'Episode':>8} | {'Win Rate':>10} | {'Status':>10}")
    print("-" * 35)

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
            wr = evaluate(encoder, policy, TicTacToe)
            status = "PASS" if wr >= TARGET_WINRATE else ""
            print(f"  {ep:5d}   | {wr:>9.1%} | {status}")
            if wr >= TARGET_WINRATE:
                print(f"\n  Target reached at episode {ep}!")
                break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS,
    }, "checkpoints/policy_basic.pt")
    print("Saved -> checkpoints/policy_basic.pt")


if __name__ == "__main__":
    main()
