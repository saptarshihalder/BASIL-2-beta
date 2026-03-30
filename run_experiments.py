import os
import sys
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.mamba_world_model import MambaWorldModel
from models.baselines import GRUWorldModel, LSTMWorldModel, MLPWorldModel
from models.policy import MultiHeadPolicy
from models.planning import PlanningModule
from models.world_model_interface import WorldModelInterface
from utils.uncertainty import should_plan
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

LATENT_DIM = 64
ACTION_DIM = 16
HIDDEN_DIM = 128
STATE_DIM = 16
NUM_BLOCKS = 2
BATCH_SIZE = 64
TRAIN_EPOCHS = 100
LR = 3e-4
NUM_GAMES = 10000
ROLLOUT_H = 5
NUM_EVAL_GAMES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt = torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def encode_state(encoder, state):
    padded = pad_state(state)
    x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(x).squeeze(0).cpu()


def collect_trajectories(encoder, env_class, num_games, min_len=6):
    trajectories = []
    collected = 0
    for _ in tqdm(range(num_games), desc=f"  {env_class.__name__}", ncols=50):
        env = env_class()
        state = env.reset()
        z_list = [encode_state(encoder, state)]
        a_list, r_list, d_list = [], [], []
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = random.choice(valid)
            a_list.append(action)
            state, reward, done, _ = env.step(action)
            z_list.append(encode_state(encoder, state))
            r_list.append(reward)
            d_list.append(1.0 if done else 0.0)
        if len(a_list) >= min_len:
            trajectories.append({
                "z": torch.stack(z_list),
                "a": torch.tensor(a_list, dtype=torch.long),
                "r": torch.tensor(r_list, dtype=torch.float32),
                "d": torch.tensor(d_list, dtype=torch.float32),
            })
            collected += 1
    return trajectories


def extract_windows(trajectories, window_size):
    z_win, a_win, r_win, d_win = [], [], [], []
    for traj in trajectories:
        T = len(traj["a"])
        for t in range(T - window_size + 1):
            z_win.append(traj["z"][t:t+window_size+1])
            a_win.append(traj["a"][t:t+window_size])
            r_win.append(traj["r"][t:t+window_size])
            d_win.append(traj["d"][t:t+window_size])
    return torch.stack(z_win), torch.stack(a_win), torch.stack(r_win), torch.stack(d_win)


def train_world_model(model, train_loader, val_loader, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    done_loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = None

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for z_win, a_win, r_win, d_win in train_loader:
            z_win, a_win = z_win.to(DEVICE), a_win.to(DEVICE)
            r_win, d_win = r_win.to(DEVICE), d_win.to(DEVICE)
            B = z_win.size(0)
            model.reset_hidden(B, DEVICE)
            total_loss = torch.tensor(0.0, device=DEVICE)
            z_curr = z_win[:, 0]
            for t in range(ROLLOUT_H):
                z_pred, r_pred, d_logit, _ = model(z_curr, a_win[:, t])
                step_loss = (
                    state_loss_fn(z_pred, z_win[:, t+1])
                    + 0.5 * reward_loss_fn(r_pred, r_win[:, t].unsqueeze(-1))
                    + done_loss_fn(d_logit, d_win[:, t].unsqueeze(-1))
                )
                total_loss = total_loss + step_loss
                z_curr = z_pred
            avg_loss = total_loss / ROLLOUT_H
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for z_win, a_win, r_win, d_win in val_loader:
                z_win, a_win = z_win.to(DEVICE), a_win.to(DEVICE)
                r_win, d_win = r_win.to(DEVICE), d_win.to(DEVICE)
                B = z_win.size(0)
                model.reset_hidden(B, DEVICE)
                total_loss = 0.0
                z_curr = z_win[:, 0]
                for t in range(ROLLOUT_H):
                    z_pred, r_pred, d_logit, _ = model(z_curr, a_win[:, t])
                    total_loss += state_loss_fn(z_pred, z_win[:, t+1]).item()
                    z_curr = z_pred
                val_loss += (total_loss / ROLLOUT_H) * B
                val_n += B
        val_avg = val_loss / val_n
        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"    {name:6s} epoch {epoch+1:3d}/{TRAIN_EPOCHS}  val_loss={val_avg:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return best_val


def test_multistep(model, encoder, env_class, num_traj=50):
    model.eval()
    mse_1, mse_5 = [], []
    for _ in range(num_traj):
        env = env_class()
        state = env.reset()
        states_list = [state.copy()]
        actions_list = []
        done = False
        while not done and len(actions_list) < 20:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = random.choice(valid)
            actions_list.append(action)
            state, _, done, _ = env.step(action)
            states_list.append(state.copy())
        if len(actions_list) < ROLLOUT_H + 1:
            continue

        z_real = []
        for s in states_list:
            z_real.append(encode_state(encoder, s).to(DEVICE))
        z_real = torch.stack(z_real)

        for t in range(len(actions_list) - ROLLOUT_H):
            model.reset_hidden(1, DEVICE)
            a_t = torch.tensor([actions_list[t]], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                z_pred, _, _ = model.predict(z_real[t:t+1], a_t)
            mse_1.append(nn.functional.mse_loss(z_pred, z_real[t+1:t+2]).item())

            model.reset_hidden(1, DEVICE)
            z_curr = z_real[t:t+1]
            for step in range(ROLLOUT_H):
                a_s = torch.tensor([actions_list[t+step]], dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    z_curr, _, _ = model.predict(z_curr, a_s)
            mse_5.append(nn.functional.mse_loss(z_curr, z_real[t+ROLLOUT_H:t+ROLLOUT_H+1]).item())

    return np.mean(mse_1) if mse_1 else 0, np.mean(mse_5) if mse_5 else 0


def evaluate_agent(encoder, policy, planner, env_class, num_games, mode="adaptive"):
    wins = 0
    fast_moves = 0
    slow_moves = 0
    for _ in range(num_games):
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                z = encode_state(encoder, state).unsqueeze(0).to(DEVICE)
                valid_mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
                for a in env.get_valid_actions():
                    valid_mask[a] = True
                valid_mask = valid_mask.unsqueeze(0)
                with torch.no_grad():
                    action, _, _, logits_list = policy.get_action_and_value(z, valid_mask, deterministic=True)
                    use_plan, _, _ = should_plan(logits_list, valid_mask)

                if mode == "always" and planner is not None:
                    action_int = planner.plan(z, env.get_valid_actions(), device=DEVICE)
                    slow_moves += 1
                elif mode == "never":
                    action_int = action.item()
                    fast_moves += 1
                elif mode == "adaptive":
                    if use_plan.item() and planner is not None:
                        action_int = planner.plan(z, env.get_valid_actions(), device=DEVICE)
                        slow_moves += 1
                    else:
                        action_int = action.item()
                        fast_moves += 1
                else:
                    action_int = action.item()
                    fast_moves += 1
            else:
                valid = env.get_valid_actions()
                action_int = random.choice(valid)
            state, _, done, info = env.step(action_int)
        if info["winner"] == 1:
            wins += 1
    total_moves = fast_moves + slow_moves
    return {
        "win_rate": wins / num_games,
        "fast_pct": fast_moves / total_moves if total_moves > 0 else 0,
        "slow_pct": slow_moves / total_moves if total_moves > 0 else 0,
        "total_sims": slow_moves * 10,
    }


def exp2_world_model_comparison(encoder, train_loader, val_loader):
    print("\n" + "=" * 60)
    print("  EXP 2: World Model Comparison (Mamba vs GRU vs LSTM vs MLP)")
    print("=" * 60)

    models_to_test = {
        "Mamba": MambaWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM, STATE_DIM, NUM_BLOCKS),
        "GRU": GRUWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM),
        "LSTM": LSTMWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM),
        "MLP": MLPWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM),
    }

    results = {}
    for name, model in models_to_test.items():
        print(f"\n  Training {name}...")
        model = model.to(DEVICE)
        best_val = train_world_model(model, train_loader, val_loader, name)

        print(f"  Testing {name}...")
        ttt_1, ttt_5 = test_multistep(model, encoder, TicTacToe)
        c4_1, c4_5 = test_multistep(model, encoder, ConnectFour)
        params = sum(p.numel() for p in model.parameters())

        results[name] = {
            "params": params,
            "val_loss": best_val,
            "ttt_1step": ttt_1, "ttt_5step": ttt_5,
            "c4_1step": c4_1, "c4_5step": c4_5,
        }
        print(f"    TTT  1-step={ttt_1:.4f}  5-step={ttt_5:.4f}")
        print(f"    C4   1-step={c4_1:.4f}  5-step={c4_5:.4f}")

    print(f"\n  {'Model':>6} | {'Params':>8} | {'TTT 1s':>8} | {'TTT 5s':>8} | {'C4 1s':>8} | {'C4 5s':>8}")
    print("  " + "-" * 58)
    for name, r in results.items():
        print(f"  {name:>6} | {r['params']:>8,} | {r['ttt_1step']:>8.4f} | {r['ttt_5step']:>8.4f} | {r['c4_1step']:>8.4f} | {r['c4_5step']:>8.4f}")

    return results


def exp3_planning_comparison(encoder, policy, planner):
    print("\n" + "=" * 60)
    print("  EXP 3: Always Plan vs Adaptive vs Never Plan")
    print("=" * 60)

    results = {}
    for mode in ["never", "adaptive", "always"]:
        print(f"\n  Mode: {mode}")
        for name, env_class in [("TTT", TicTacToe), ("C4", ConnectFour)]:
            n = 200 if mode == "always" else NUM_EVAL_GAMES
            print(f"    {name} ({n} games)...", end=" ", flush=True)
            r = evaluate_agent(encoder, policy, planner, env_class, n, mode)
            key = f"{mode}_{name}"
            results[key] = r
            print(f"win={r['win_rate']:.1%}  fast={r['fast_pct']:.1%}  slow={r['slow_pct']:.1%}  sims={r['total_sims']}")

    print(f"\n  {'Mode':>10} | {'Game':>4} | {'Win%':>6} | {'Fast%':>6} | {'Slow%':>6} | {'Sims':>6}")
    print("  " + "-" * 48)
    for key, r in results.items():
        mode, game = key.split("_")
        print(f"  {mode:>10} | {game:>4} | {r['win_rate']:>5.1%} | {r['fast_pct']:>5.1%} | {r['slow_pct']:>5.1%} | {r['total_sims']:>6}")

    return results


def main():
    print("=" * 60)
    print("  BASIL — Running All Experiments")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    encoder = load_encoder()
    print("Loaded encoder\n")

    print("Collecting trajectories for world model training...")
    ttt_traj = collect_trajectories(encoder, TicTacToe, NUM_GAMES, min_len=ROLLOUT_H+1)
    c4_traj = collect_trajectories(encoder, ConnectFour, NUM_GAMES, min_len=ROLLOUT_H+1)
    all_traj = ttt_traj + c4_traj
    random.shuffle(all_traj)

    z_win, a_win, r_win, d_win = extract_windows(all_traj, ROLLOUT_H)
    n_train = int(0.9 * len(z_win))
    perm = torch.randperm(len(z_win))
    z_win, a_win, r_win, d_win = z_win[perm], a_win[perm], r_win[perm], d_win[perm]
    train_ds = TensorDataset(z_win[:n_train], a_win[:n_train], r_win[:n_train], d_win[:n_train])
    val_ds = TensorDataset(z_win[n_train:], a_win[n_train:], r_win[n_train:], d_win[n_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"  Windows: {len(z_win)} (train={n_train}, val={len(z_win)-n_train})")

    exp2_results = exp2_world_model_comparison(encoder, train_loader, val_loader)

    policy_path = "checkpoints/policy_ensemble.pt"
    if not os.path.exists(policy_path):
        policy_path = "checkpoints/policy_basic.pt"
    ckpt = torch.load(policy_path, map_location=DEVICE, weights_only=True)
    policy = MultiHeadPolicy(
        latent_dim=ckpt.get("latent_dim", 64),
        hidden_dim=ckpt.get("hidden_dim", 128),
        num_actions=ckpt.get("num_actions", MAX_ACTIONS),
        num_heads=ckpt.get("num_heads", 5),
    ).to(DEVICE)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    from models.world_model_interface import load_world_model
    wm = load_world_model("checkpoints/world_model.pt", str(DEVICE))
    planner = PlanningModule(wm, num_rollouts=10, horizon=5)

    exp3_results = exp3_planning_comparison(encoder, policy, planner)

    os.makedirs("experiments", exist_ok=True)
    all_results = {"exp2_world_models": exp2_results, "exp3_planning": exp3_results}
    with open("experiments/all_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> experiments/all_experiments.json")

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
