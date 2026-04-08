import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.policy import MultiHeadPolicy
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

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
EPOCHS = 500
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
        return encoder(x).squeeze(0)


def get_valid_mask(env):
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
    for a in env.get_valid_actions():
        mask[a] = True
    return mask.unsqueeze(0)


def collect_p1_episode(encoder, policy, env_class):
    env = env_class()
    state = env.reset()
    p1 = {"states": [], "actions": [], "log_probs": [], "values": [], "masks": []}
    done = False

    while not done:
        if env.current_player == 1:
            z = encode(encoder, state)
            valid_mask = get_valid_mask(env)
            with torch.no_grad():
                action, log_prob, value, _, _ = policy.get_action_and_value(
                    z.unsqueeze(0), valid_mask
                )
            action_int = action.item()
            p1["states"].append(z)
            p1["actions"].append(action.squeeze())
            p1["log_probs"].append(log_prob.squeeze())
            p1["values"].append(value.item())
            p1["masks"].append(valid_mask.squeeze(0))
        else:
            action_int = random.choice(env.get_valid_actions())

        state, reward, done, info = env.step(action_int)

    if info["winner"] == 1:
        final_reward = 1.0
    elif info["winner"] == 2:
        final_reward = -1.0
    else:
        final_reward = 0.0

    n = len(p1["states"])
    rewards = [0.0] * n
    dones = [False] * n
    if n > 0:
        rewards[-1] = final_reward
        dones[-1] = True

    return p1, rewards, dones, info


def compute_gae(rewards, values, dones):
    advantages = []
    gae = 0.0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - float(dones[t])) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(policy, optimizer, states, actions, old_log_probs, advantages, returns, masks):
    states_t = torch.stack(states).to(DEVICE)
    actions_t = torch.stack(actions).to(DEVICE)
    old_lp_t = torch.stack(old_log_probs).to(DEVICE).detach()
    adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
    ret_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
    masks_t = torch.stack(masks).to(DEVICE)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(UPDATE_STEPS):
        idx = torch.randperm(len(states_t))
        for start in range(0, len(states_t), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(states_t))
            b = idx[start:end]

            new_lp, ent, new_val = policy.evaluate_actions(states_t[b], actions_t[b], masks_t[b])
            ratio = (new_lp - old_lp_t[b]).exp()
            surr1 = ratio * adv_t[b]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[b]

            loss = (
                -torch.min(surr1, surr2).mean()
                + VALUE_COEF * F.mse_loss(new_val, ret_t[b])
                + ENTROPY_COEF * -ent.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
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
                valid_mask = get_valid_mask(env)
                with torch.no_grad():
                    action, _, _, _, _ = policy.get_action_and_value(
                        z.unsqueeze(0), valid_mask, deterministic=True
                    )
                action_int = action.item()
            else:
                action_int = random.choice(env.get_valid_actions())
            state, _, done, info = env.step(action_int)
        if info["winner"] == 1:
            wins += 1
    return wins / num_games


def train_game(env_class, save_path):
    encoder = load_encoder()
    policy = MultiHeadPolicy(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
        num_actions=MAX_ACTIONS, num_heads=NUM_HEADS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    print(f"\n  Training on {env_class.__name__}...")
    print(f"  {'Epoch':>7} | {'Win Rate':>10} | {'Status':>8}")
    print("  " + "-" * 32)

    buf_s, buf_a, buf_lp, buf_v, buf_r, buf_d, buf_m = [], [], [], [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        p1, rewards, dones, info = collect_p1_episode(encoder, policy, env_class)

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
            wr = evaluate(encoder, policy, env_class)
            status = "PASS" if wr >= TARGET_WINRATE else ""
            print(f"  {epoch:5d}   | {wr:>9.1%} | {status}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
        "num_actions": MAX_ACTIONS, "num_heads": NUM_HEADS,
    }, save_path)
    print(f"  Saved -> {save_path}")
    return policy


def main():
    print("=" * 50)
    print("  BASIL — PPO Training")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  Min buffer: {MIN_STEPS} steps")
    print(f"  Epochs: {EPOCHS}")

    train_game(TicTacToe, "checkpoints/policy_ttt.pt")
    train_game(ConnectFour, "checkpoints/policy_c4.pt")


if __name__ == "__main__":
    main()
