"""End-to-end smoke run of the fixed original pipeline with tiny settings.

Runs in a temporary directory, so your real checkpoints/ and experiments/ are never touched.
Takes a few minutes on a CPU. The numbers it prints mean nothing; it only proves every stage runs.

Usage (from the repo root):  python tests/smoke_original_pipeline.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
work = tempfile.mkdtemp(prefix="basil_smoke_")
os.chdir(work)
print(f"working in {work}")

import train_autoencoder  # noqa: E402
train_autoencoder.NUM_STATES_PER_GAME, train_autoencoder.EPOCHS = 400, 1
train_autoencoder.main()

import train_ppo  # noqa: E402
train_ppo.EVAL_INTERVAL = 40
train_ppo.main(["--games", "tictactoe,connect4", "--episodes", "40", "--seed", "0"])

import train_world_model_multistep as wm  # noqa: E402
wm.NUM_GAMES, wm.EPOCHS = 40, 1
wm.main()

import evaluate_v2  # noqa: E402
evaluate_v2.main(["--n", "3", "--n-thinking", "2", "--calib-games", "6", "--rollouts", "2", "--horizon", "3",
                  "--opponents", "random,greedy,minimax"])

import run_experiments  # noqa: E402
run_experiments.NUM_GAMES, run_experiments.TRAIN_EPOCHS, run_experiments.SEEDS = 40, 1, [0, 1]
run_experiments.main()
print("\nSMOKE RUN OK")
