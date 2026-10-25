"""Shared helpers for the fixed original BASIL pipeline.

Seeding, config, one policy checkpoint per game, stable seeds for paired games,
and the non-learning opponents that any learned agent has to beat.
"""
import copy
import hashlib
import os
import random
from functools import lru_cache

import numpy as np
import torch
import yaml

from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = {"tictactoe": (TicTacToe, 9), "connect4": (ConnectFour, 7)}  # env class, action count


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(*key) -> int:
    """Seed derived from a key with SHA-256 (Python's hash() changes between runs)."""
    return int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)


def load_cfg(path=None):
    with open(path or os.path.join(ROOT, "configs", "default.yaml")) as f:
        return yaml.safe_load(f)


def policy_path(game: str) -> str:
    """One checkpoint per game. The old scripts played Connect4 with the TicTacToe policy."""
    return os.path.join("checkpoints", f"policy_ensemble_{game}.pt")  # run scripts from the repo root


# ---------------------------------------------------------------- opponents
def _wins_if(env, action, player):
    e = copy.deepcopy(env)
    e.current_player = player
    _, _, done, info = e.step(action)
    return done and info["winner"] == player


def greedy_action(env, rng: random.Random):
    """Win if possible, else block, else centre-first. Beats random ~96% (TTT) / ~99% (C4)."""
    acts = env.get_valid_actions()
    me = env.current_player
    for who in (me, 3 - me):
        for a in acts:
            if _wins_if(env, a, who):
                return a
    if isinstance(env, TicTacToe):
        for pref in ([4], [0, 2, 6, 8], [1, 3, 5, 7]):
            opts = [a for a in pref if a in acts]
            if opts:
                return rng.choice(opts)
    return min(acts, key=lambda c: (abs(c - 3), rng.random()))


_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _ttt_winner(b):
    for x, y, z in _LINES:
        if b[x] and b[x] == b[y] == b[z]:
            return b[x]
    return 0


@lru_cache(maxsize=None)
def _ttt_value(b, p):
    """Exact game value for player p to move on TicTacToe board tuple b (+1 win, 0 draw, -1 loss)."""
    w = _ttt_winner(b)
    if w:
        return 1 if w == p else -1
    moves = [i for i in range(9) if b[i] == 0]
    if not moves:
        return 0
    return max(-_ttt_value(b[:i] + (p,) + b[i + 1:], 3 - p) for i in moves)


def minimax_action(env, rng: random.Random):
    """Perfect TicTacToe play (random among equally good moves)."""
    assert isinstance(env, TicTacToe), "minimax opponent is TicTacToe-only"
    b = tuple(int(v) for v in env.board.reshape(-1))
    p = env.current_player
    acts = env.get_valid_actions()
    vals = {a: -_ttt_value(b[:a] + (p,) + b[a + 1:], 3 - p) for a in acts}
    best = max(vals.values())
    return rng.choice([a for a in acts if vals[a] == best])


def make_opponent(name: str, rng: random.Random):
    if name == "random":
        return lambda env: rng.choice(env.get_valid_actions())
    if name == "greedy":
        return lambda env: greedy_action(env, rng)
    if name == "minimax":
        return lambda env: minimax_action(env, rng)
    raise ValueError(f"unknown opponent {name}")
