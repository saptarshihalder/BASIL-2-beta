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
ROLLOUT_STEPS = 256
EPOCHS = 200
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


def collect_rollout(env, encoder, policy):
    states, actions, log_probs, values, rewards, dones, masks = [], [], [], [], [], [], []
    state = env.reset() if env.done else env.get_state()

    for _ in range(ROLLOUT_STEPS):
        z = encode(encoder, state)
        valid_mask = get_valid_mask(env)

        with torch.no_grad():
            action, log_prob, value, _, _ = policy.get_action_and_value(
                z.unsqueeze(0), valid_mask
            )

        if env.current_player == 1:
            action_int = action.item()
        else:
            action_int = random.choice(env.get_valid_actions())

        next_state, reward, done, info = env.step(action_int)

        if env.current_player == 2:
            reward = -reward

        states.append(z)
        actions.append(action.squeeze())
        log_probs.append(log_prob.squeeze())
        values.append(value.item())
        rewards.append(reward)
        dones.append(done)
        masks.append(valid_mask.squeeze(0))

        if done:
            state = env.reset()
        else:
            state = next_state

    return states, actions, log_probs, values, rewards, dones, masks


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
        for start in range(0, len(states_t), 64):
            end = min(start + 64, len(states_t))
            b = idx[start:end]

            new_lp, ent, new_val = policy.evaluate_actions(states_t[b], actions_t[b], masks_t[b])
            ratio = (new_lp - old_lp_t[b]).exp()
            surr1 = ratio * adv_t[b]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_val, ret_t[b])
            entropy_loss = -ent.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
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
                    action, _, _, _, _ = policy.get_action_and_value(z.unsqueeze(0), valid_mask, deterministic=True)
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
    print(f"  {'Epoch':>7} | {'Reward':>8} | {'Win Rate':>10}")
    print("  " + "-" * 32)

    env = env_class()
    env.reset()

    for epoch in range(EPOCHS):
        states, actions, log_probs, values, rewards, dones, masks = collect_rollout(
            env, encoder, policy
        )
        advantages, returns = compute_gae(rewards, values, dones)
        ppo_update(policy, optimizer, states, actions, log_probs, advantages, returns, masks)

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == 0:
            wr = evaluate(encoder, policy, env_class)
            print(f"  {epoch+1:5d}   | {np.mean(rewards):>7.3f} | {wr:>9.1%}")

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
    print("  BASIL — PPO Training (Merged)")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  Rollout: {ROLLOUT_STEPS} steps/epoch")
    print(f"  Epochs: {EPOCHS}")

    train_game(TicTacToe, "checkpoints/policy_ttt.pt")
    train_game(ConnectFour, "checkpoints/policy_c4.pt")


if __name__ == "__main__":
    main()
