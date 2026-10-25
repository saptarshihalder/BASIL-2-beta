import torch

from train_world_model import make_dataloaders


def test_single_step_validation_uses_different_games():
    ids = torch.arange(20).repeat_interleave(7)
    n = len(ids)
    data = {
        "z_t": ids.float().unsqueeze(1),
        "actions": torch.zeros(n, dtype=torch.long),
        "z_next": ids.float().unsqueeze(1),
        "rewards": torch.zeros(n),
        "dones": torch.zeros(n),
        "game_ids": ids,
    }
    train, val = make_dataloaders(data, 0.2)
    a = set(train.dataset.tensors[0].squeeze(1).tolist())
    b = set(val.dataset.tensors[0].squeeze(1).tolist())
    assert a.isdisjoint(b)
    assert len(a) == 16 and len(b) == 4
