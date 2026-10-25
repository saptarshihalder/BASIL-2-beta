import torch

from train_world_model_multistep import extract_windows, split_trajectories
from evaluate_v2 import paired_ci
from envs.connect4 import ConnectFour
from envs.tictactoe import TicTacToe


def test_windows_do_not_cross_games():
    games = []
    for game_id in range(20):
        games.append({
            "z": torch.full((9, 2), game_id, dtype=torch.float32),
            "a": torch.zeros(8, dtype=torch.long),
            "r": torch.zeros(8),
            "d": torch.zeros(8),
        })
    train, val = split_trajectories(games, 0.2)
    tr_z, *_ = extract_windows(train, 5)
    va_z, *_ = extract_windows(val, 5)
    assert set(tr_z[:, 0, 0].tolist()).isdisjoint(set(va_z[:, 0, 0].tolist()))
    assert len(tr_z) == 16 * 4
    assert len(va_z) == 4 * 4


def test_paired_interval_is_on_differences():
    a = {1: [1, 1, 1], 2: [0.5, 0.5, 0.5]}
    b = {1: [0, 0, 0], 2: [0.5, 0.5, 0.5]}
    ci = paired_ci(a, b, seed=7)
    assert ci == {"delta": 0.5, "ci95": [0.5, 0.5], "pairs": 3}


def test_invalid_moves_are_rejected():
    for env, bad in ((TicTacToe(), [-1, 9, 1.5]),
                     (ConnectFour(), [-1, 7, 1.5])):
        for action in bad:
            try:
                env.step(action)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {action}")
