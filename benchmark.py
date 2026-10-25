import os
import sys
import time
import random
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder, BoardAutoEncoder
from models.mamba_world_model import MambaWorldModel
from models.baselines import GRUWorldModel, LSTMWorldModel, MLPWorldModel
from models.policy import MultiHeadPolicy
from models.planning import PlanningModule
from models.world_model_interface import load_world_model
from utils.uncertainty import should_plan, compute_jsd
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS
from utils.common import GAMES, policy_path, set_seed

set_seed(0)
JSD_THRESHOLD = 0.5  # normalised JSD; only used for latency/throughput, not for reported results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64
ACTION_DIM = 16
HIDDEN_DIM = 128
STATE_DIM = 16
NUM_BLOCKS = 2
NUM_HEADS = 5


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_inference(fn, name, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / runs) * 1000
    return avg_ms


def main():
    print("=" * 65)
    print("  BASIL — Full System Benchmark")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    results = {}

    print("-" * 65)
    print("  1. MODEL SIZES")
    print("-" * 65)

    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    autoencoder = BoardAutoEncoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    mamba = MambaWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM, STATE_DIM, NUM_BLOCKS).to(DEVICE)
    gru = GRUWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM).to(DEVICE)
    lstm = LSTMWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM).to(DEVICE)
    mlp = MLPWorldModel(LATENT_DIM, ACTION_DIM, MAX_ACTIONS, HIDDEN_DIM).to(DEVICE)
    policy = MultiHeadPolicy(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=MAX_ACTIONS, num_heads=NUM_HEADS).to(DEVICE)

    models = {
        "Encoder": encoder,
        "AutoEncoder": autoencoder,
        "Mamba WM": mamba,
        "GRU WM": gru,
        "LSTM WM": lstm,
        "MLP WM": mlp,
        "Policy (5 heads)": policy,
    }

    print(f"  {'Model':<18s} | {'Params':>10s} | {'Size (MB)':>10s}")
    print("  " + "-" * 44)
    size_results = {}
    for name, model in models.items():
        p = count_params(model)
        s = model_size_mb(model)
        print(f"  {name:<18s} | {p:>10,} | {s:>10.3f}")
        size_results[name] = {"params": p, "size_mb": round(s, 3)}
    results["model_sizes"] = size_results

    print()
    print("-" * 65)
    print("  2. INFERENCE SPEED (single sample)")
    print("-" * 65)

    z = torch.randn(1, LATENT_DIM, device=DEVICE)
    a = torch.randint(0, MAX_ACTIONS, (1,), device=DEVICE)
    board = torch.randn(1, 3, PAD_H, PAD_W, device=DEVICE)
    actions_seq = torch.randint(0, MAX_ACTIONS, (1, 5), device=DEVICE)
    valid_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool, device=DEVICE)

    encoder.eval()
    mamba.eval()
    gru.eval()
    lstm.eval()
    mlp.eval()
    policy.eval()

    speed_results = {}

    def enc_fn():
        with torch.no_grad():
            encoder(board)
    t = benchmark_inference(enc_fn, "Encoder")
    print(f"  Encoder forward:           {t:>8.3f} ms")
    speed_results["encoder"] = round(t, 3)

    def policy_fn():
        with torch.no_grad():
            policy.get_action_and_value(z, valid_mask)
    t = benchmark_inference(policy_fn, "Policy")
    print(f"  Policy forward:            {t:>8.3f} ms")
    speed_results["policy"] = round(t, 3)

    def jsd_fn():
        with torch.no_grad():
            logits_list, _ = policy(z, valid_mask)
            compute_jsd(logits_list, valid_mask)
    t = benchmark_inference(jsd_fn, "JSD")
    print(f"  JSD computation:           {t:>8.3f} ms")
    speed_results["jsd"] = round(t, 3)

    for wm_name, wm_model in [("Mamba", mamba), ("GRU", gru), ("LSTM", lstm), ("MLP", mlp)]:
        wm_model.eval()
        def wm_fn(m=wm_model):
            with torch.no_grad():
                m.reset_hidden(1, DEVICE)
                m.predict(z, a)
        t = benchmark_inference(wm_fn, wm_name)
        print(f"  {wm_name:6s} 1-step predict:   {t:>8.3f} ms")
        speed_results[f"{wm_name.lower()}_1step"] = round(t, 3)

    for wm_name, wm_model in [("Mamba", mamba), ("GRU", gru), ("LSTM", lstm), ("MLP", mlp)]:
        wm_model.eval()
        def wm_fn(m=wm_model):
            with torch.no_grad():
                m.multi_step_rollout(z, actions_seq)
        t = benchmark_inference(wm_fn, wm_name)
        print(f"  {wm_name:6s} 5-step rollout:   {t:>8.3f} ms")
        speed_results[f"{wm_name.lower()}_5step"] = round(t, 3)

    results["inference_speed_ms"] = speed_results

    print()
    print("-" * 65)
    print("  3. FULL PIPELINE LATENCY (per move)")
    print("-" * 65)

    ckpt_path = "checkpoints/autoencoder.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])

    pol_path = policy_path("tictactoe")  # latency is measured on TicTacToe; throughput reloads per game
    if os.path.exists(pol_path):
        ckpt = torch.load(pol_path, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(ckpt["model_state_dict"])

    wm_path = "checkpoints/world_model.pt"
    planner = None
    if os.path.exists(wm_path):
        wm_interface = load_world_model(wm_path, str(DEVICE))
        planner = PlanningModule(wm_interface, policy=policy, num_rollouts=10, horizon=5)

    encoder.eval()
    policy.eval()

    env = TicTacToe()
    state = env.reset()

    def fast_path():
        padded = pad_state(state)
        x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_enc = encoder(x)
            mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
            action, _, _, _, _ = policy.get_action_and_value(z_enc, mask, deterministic=True)
        return action

    t_fast = benchmark_inference(fast_path, "Fast path", warmup=20, runs=200)
    print(f"  Fast path (encode+policy):         {t_fast:>8.3f} ms")

    def fast_with_jsd():
        padded = pad_state(state)
        x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_enc = encoder(x)
            mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
            action, _, _, logits_list, _ = policy.get_action_and_value(z_enc, mask, deterministic=True)
            use_plan, jsd, max_p = should_plan(logits_list, mask, JSD_THRESHOLD)
        return action

    t_jsd = benchmark_inference(fast_with_jsd, "Fast+JSD", warmup=20, runs=200)
    print(f"  Fast path + JSD check:             {t_jsd:>8.3f} ms")

    if planner is not None:
        def slow_path():
            padded = pad_state(state)
            x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z_enc = encoder(x)
            valid = env.get_valid_actions()
            planner.plan(z_enc, valid, device=DEVICE)

        t_slow = benchmark_inference(slow_path, "Slow path", warmup=5, runs=20)
        print(f"  Slow path (encode+plan 10x5):      {t_slow:>8.3f} ms")
        results["pipeline_latency_ms"] = {
            "fast_path": round(t_fast, 3),
            "fast_with_jsd": round(t_jsd, 3),
            "slow_path": round(t_slow, 3),
            "speedup_fast_vs_slow": round(t_slow / t_fast, 1),
        }
        print(f"  Speedup (fast vs slow):            {t_slow/t_fast:>8.1f}x")
    else:
        results["pipeline_latency_ms"] = {
            "fast_path": round(t_fast, 3),
            "fast_with_jsd": round(t_jsd, 3),
        }

    print()
    print("-" * 65)
    print("  4. THROUGHPUT (games per second)")
    print("-" * 65)

    throughput = {}
    for game_name, env_class in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        game_key = game_name.lower()
        if os.path.exists(policy_path(game_key)):  # each game uses its own policy
            policy.load_state_dict(torch.load(policy_path(game_key), map_location=DEVICE, weights_only=True)["model_state_dict"])
        if planner is not None:
            planner = PlanningModule(wm_interface, policy=policy, num_rollouts=10, horizon=5,
                                     num_actions=GAMES[game_key][1])
        for mode in ["fast", "adaptive"]:
            n_games = 500 if mode == "fast" else 200
            start = time.perf_counter()
            wins = 0
            for game_idx in range(n_games):
                seat = 1 + game_idx % 2  # both seats
                env = env_class()
                state_g = env.reset()
                done = False
                while not done:
                    if env.current_player == seat:
                        padded = pad_state(state_g)
                        x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            z_enc = encoder(x)
                            mask = get_valid_mask_bench(env)
                            action, _, _, logits_list, _ = policy.get_action_and_value(z_enc, mask, deterministic=True)

                        if mode == "adaptive" and planner is not None:
                            use_plan, _, _ = should_plan(logits_list, mask, JSD_THRESHOLD)
                            if use_plan.item():
                                action_int = planner.plan(z_enc, env.get_valid_actions(), device=DEVICE)
                            else:
                                action_int = action.item()
                        else:
                            action_int = action.item()
                    else:
                        action_int = random.choice(env.get_valid_actions())
                    state_g, _, done, info = env.step(action_int)
                if info["winner"] == seat:
                    wins += 1
            elapsed = time.perf_counter() - start
            gps = n_games / elapsed
            wr = wins / n_games
            key = f"{mode}_{game_name}"
            throughput[key] = {"games_per_sec": round(gps, 2), "win_rate": round(wr, 3), "total_time": round(elapsed, 1)}
            print(f"  {mode:>8s} {game_name:<12s}  {gps:>6.1f} games/s  win={wr:.1%}  ({elapsed:.1f}s for {n_games})")

    results["throughput"] = throughput

    print()
    print("-" * 65)
    print("  5. MEMORY USAGE")
    print("-" * 65)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        padded = pad_state(TicTacToe().reset())
        x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_enc = encoder(x)
            mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
            policy.get_action_and_value(z_enc, mask)

        peak_fast = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak GPU memory (fast path):   {peak_fast:>8.1f} MB")

        if planner is not None:
            torch.cuda.reset_peak_memory_stats()
            planner.plan(z_enc, [0,1,2], device=DEVICE)
            peak_slow = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  Peak GPU memory (slow path):   {peak_slow:>8.1f} MB")
            results["memory_mb"] = {"fast_path": round(peak_fast, 1), "slow_path": round(peak_slow, 1)}
        else:
            results["memory_mb"] = {"fast_path": round(peak_fast, 1)}
    else:
        print("  (GPU memory tracking not available on CPU)")

    print()
    print("=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved -> experiments/benchmark.json")


def get_valid_mask_bench(env):
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=DEVICE)
    for a in env.get_valid_actions():
        mask[a] = True
    return mask.unsqueeze(0)


if __name__ == "__main__":
    main()
