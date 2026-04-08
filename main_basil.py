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
MAX_PROB_THRESHOLD = 0.6
NUM_ROLLOUTS = 10
HORIZON = 5
NUM_EVAL_GAMES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt = torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_policy(path="checkpoints/policy_ensemble.pt"):
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


def basil_step(encoder, policy, planner, env, state):
    z = encode(encoder, state)
    valid_mask = get_valid_mask(env)
    valid_actions = env.get_valid_actions()

    with torch.no_grad():
        action, _, _, logits_list, _ = policy.get_action_and_value(z, valid_mask, deterministic=True)
        use_plan, jsd, max_p = should_plan(logits_list, valid_mask, JSD_THRESHOLD, MAX_PROB_THRESHOLD)

    if use_plan.item() and planner is not None:
        planned_action = planner.plan(z, valid_actions, device=DEVICE)
        return planned_action, True, jsd.item()
    else:
        return action.item(), False, jsd.item()


def evaluate_basil(encoder, policy, planner, env_class, num_games=NUM_EVAL_GAMES, game_name=""):
    wins, losses, draws = 0, 0, 0
    fast_moves, slow_moves = 0, 0
    jsd_values, game_lengths = [], []

    for _ in tqdm(range(num_games), desc=f"  {game_name}", ncols=60):
        env = env_class()
        state = env.reset()
        done = False
        moves = 0
        while not done:
            if env.current_player == 1:
                action, used_planning, jsd_val = basil_step(encoder, policy, planner, env, state)
                if used_planning:
                    slow_moves += 1
                else:
                    fast_moves += 1
                jsd_values.append(jsd_val)
            else:
                action = random.choice(env.get_valid_actions())
            state, _, done, info = env.step(action)
            moves += 1
        game_lengths.append(moves)
        if info["winner"] == 1:
            wins += 1
        elif info["winner"] == 2:
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    total_moves = fast_moves + slow_moves
    return {
        "num_games": num_games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / total, "loss_rate": losses / total, "draw_rate": draws / total,
        "fast_moves": fast_moves, "slow_moves": slow_moves,
        "fast_pct": fast_moves / total_moves if total_moves > 0 else 0,
        "slow_pct": slow_moves / total_moves if total_moves > 0 else 0,
        "avg_jsd": float(np.mean(jsd_values)) if jsd_values else 0,
        "avg_game_length": float(np.mean(game_lengths)),
    }


def print_results(results, label):
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Games:       {results['num_games']}")
    print(f"  Wins:        {results['wins']}  ({results['win_rate']:.1%})")
    print(f"  Losses:      {results['losses']}  ({results['loss_rate']:.1%})")
    print(f"  Draws:       {results['draws']}  ({results['draw_rate']:.1%})")
    print(f"  Avg length:  {results['avg_game_length']:.1f} moves")
    print(f"  Fast moves:  {results['fast_moves']}  ({results['fast_pct']:.1%})")
    print(f"  Slow moves:  {results['slow_moves']}  ({results['slow_pct']:.1%})")
    print(f"  Avg JSD:     {results['avg_jsd']:.4f}")
    print(f"{'=' * 55}")


def main():
    print("=" * 55)
    print("  BASIL — Full System Evaluation (Merged)")
    print("=" * 55)
    print(f"  Device:        {DEVICE}")
    print(f"  JSD threshold: {JSD_THRESHOLD}")
    print(f"  Rollouts:      {NUM_ROLLOUTS} x {HORIZON} steps")
    print(f"  Eval games:    {NUM_EVAL_GAMES}")
    print()

    encoder = load_encoder()
    print("Loaded encoder")

    policy_path = "checkpoints/policy_ensemble.pt"
    if not os.path.exists(policy_path):
        policy_path = "checkpoints/policy_basic.pt"
    policy = load_policy(policy_path)
    print(f"Loaded policy from {policy_path}")

    planner = None
    wm_path = "checkpoints/world_model.pt"
    if os.path.exists(wm_path):
        wm = load_world_model(wm_path, str(DEVICE))
        planner = PlanningModule(wm, policy=policy, num_rollouts=NUM_ROLLOUTS, horizon=HORIZON)
        print("Loaded world model + policy-guided planner")
    else:
        print("No world model — fast path only")

    os.makedirs("experiments", exist_ok=True)

    for name, env_class in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        print(f"\nEvaluating {name} ({NUM_EVAL_GAMES} games)...")
        results = evaluate_basil(encoder, policy, planner, env_class, NUM_EVAL_GAMES, name)
        print_results(results, f"BASIL vs Random — {name}")
        save_path = f"experiments/basil_merged_{name.lower()}.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {save_path}")


if __name__ == "__main__":
    main()
