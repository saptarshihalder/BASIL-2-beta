"""Kept for backward compatibility: the ensemble training now lives in train_ppo.py.

The old version trained on TicTacToe only and saved one checkpoint that every evaluation
script then also used for Connect4. Now each game gets its own bootstrapped 5-head policy.

Usage:  python train_ppo_ensemble.py --games tictactoe,connect4 --episodes 3000 --seed 0
"""
from train_ppo import main

if __name__ == "__main__":
    main()
