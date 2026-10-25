import torch


class PlanningModule:
    """Random-shooting planner over the learned world model.

    Fixes against the original version:
      1. Sign. The environment gives +1 to whoever wins, and the world model learned exactly
         that. Imagined steps alternate between our move (even t) and the opponent's (odd t),
         so a predicted win on an odd step is the opponent's win and now counts as -1.
         The old code added every win as +1, so it could not tell winning from losing.
      2. Action range. Imagined moves are drawn only from this game's actions (7 in Connect4),
         not from all 9 slots of the shared action embedding.
      3. One pass per rollout (the old code simulated every rollout twice).
    Limitation kept on purpose: this world model has no legality head, so deeper imagined moves
    can still be illegal. BASIL-3 (basil3/) adds a legality head.
    """

    def __init__(self, world_model_interface, policy=None, num_rollouts=10, horizon=5,
                 gamma=0.99, num_actions=9):
        self.wm = world_model_interface
        self.policy = policy
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.num_actions = num_actions
        self.simulations = 0  # world-model calls, for compute accounting

    def _pick_action(self, z, device):
        n = self.num_actions
        if self.policy is not None:
            with torch.no_grad():
                logits_list, _ = self.policy(z)
                avg = torch.stack(logits_list, dim=0).mean(dim=0)[:, :n]
                return torch.distributions.Categorical(logits=avg).sample()
        return torch.randint(0, n, (z.size(0),), device=device)

    @staticmethod
    def perspective(t: int) -> float:
        """+1 for imagined step t made by us (even t), -1 for the opponent's steps (odd t)."""
        return 1.0 if t % 2 == 0 else -1.0

    def plan(self, z_t, valid_actions, device="cpu"):
        if len(valid_actions) == 0:
            return 0
        best_action, best_value = valid_actions[0], float("-inf")
        for action in valid_actions:
            total = 0.0
            for _ in range(self.num_rollouts):
                self.wm.reset(1)
                z = z_t.clone()
                a = torch.tensor([action], dtype=torch.long, device=device)
                ret, discount = 0.0, 1.0
                for t in range(self.horizon):
                    z, r, d = self.wm.predict(z, a)
                    self.simulations += 1
                    ret += discount * self.perspective(t) * r.item()
                    discount *= self.gamma
                    if d.item() > 0.5:
                        break
                    a = self._pick_action(z, device)
                total += ret
            value = total / self.num_rollouts
            if value > best_value:
                best_value, best_action = value, action
        return best_action
