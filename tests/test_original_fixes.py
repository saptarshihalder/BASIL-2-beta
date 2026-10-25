"""Regression tests for the fixes to the original BASIL pipeline. Run:  python -m pytest -q"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.planning import PlanningModule  # noqa: E402
from models.policy import MultiHeadPolicy  # noqa: E402
from utils.uncertainty import compute_jsd  # noqa: E402


class FakeWorldModel:
    """Our move (step 0) changes nothing; on the next step the opponent wins if we chose action 0,
    and the game is drawn if we chose action 1."""

    def reset(self, batch_size=1):
        self.t, self.first = 0, None

    def predict(self, z, a):
        if self.t == 0:
            self.first = int(a.item())
        self.t += 1
        r = 1.0 if (self.t == 2 and self.first == 0) else 0.0
        return z, torch.tensor([r]), torch.tensor([1.0 if self.t == 2 else 0.0])


def test_planner_counts_opponent_wins_against_us():
    planner = PlanningModule(FakeWorldModel(), policy=None, num_rollouts=3, horizon=5, num_actions=2)
    assert planner.plan(torch.zeros(1, 4), [0, 1]) == 1     # the old planner picked 0 (opponent's win = +1)


def test_planner_samples_only_this_games_actions():
    planner = PlanningModule(FakeWorldModel(), policy=None, num_actions=7)
    assert all(int(planner._pick_action(torch.zeros(1, 4), "cpu")) < 7 for _ in range(200))


def test_jsd_is_normalised():
    logits = [torch.tensor([[10.0, 0, 0]]), torch.tensor([[0, 10.0, 0]]), torch.tensor([[0, 0, 10.0]])]
    j = compute_jsd(logits, torch.tensor([[True, True, True]])).item()
    assert 0.95 <= j <= 1.0
    same = compute_jsd([logits[0]] * 3, torch.tensor([[True, True, True]])).item()
    assert same < 1e-4


def _head_gaps(policy):
    w = [h.weight.detach().clone() for h in policy.policy_heads]
    return torch.stack([w[k] - w[0] for k in range(1, len(w))])


def test_bootstrapped_heads_actually_diverge():
    import train_ppo
    torch.manual_seed(0)
    policy = MultiHeadPolicy(latent_dim=8, hidden_dim=16, num_actions=train_ppo.MAX_ACTIONS, num_heads=3)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    n = 64
    states = [torch.randn(8) for _ in range(n)]
    masks = [torch.ones(train_ppo.MAX_ACTIONS, dtype=torch.bool) for _ in range(n)]
    with torch.no_grad():
        acts, lps = [], []
        for s, m in zip(states, masks):
            a, lp, _, _, _ = policy.get_action_and_value(s.unsqueeze(0), m.unsqueeze(0))
            acts.append(a.squeeze())
            lps.append(lp.squeeze())
    before = _head_gaps(policy)
    adv = torch.randn(n).tolist()
    train_ppo.ppo_update(policy, opt, states, acts, lps, adv, torch.randn(n).tolist(), masks)
    # with the old averaged loss every head got the same gradient, so these gaps never changed
    assert (_head_gaps(policy) - before).abs().max() > 1e-4
