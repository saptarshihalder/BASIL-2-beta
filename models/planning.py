import torch
import torch.nn.functional as F


class PlanningModule:
    def __init__(self, world_model_interface, num_rollouts=10, horizon=5, gamma=0.99):
        self.wm = world_model_interface
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.gamma = gamma

    def plan(self, z_t, valid_actions, device="cpu"):
        if len(valid_actions) == 0:
            return 0

        best_action = valid_actions[0]
        best_value = float('-inf')

        for action in valid_actions:
            total_return = 0.0

            for _ in range(self.num_rollouts):
                self.wm.reset(1)
                z_curr = z_t.clone()
                rollout_return = 0.0
                discount = 1.0

                a_tensor = torch.tensor([action], dtype=torch.long, device=device)
                z_next, reward, done = self.wm.predict(z_curr, a_tensor)
                rollout_return += discount * reward.item()
                discount *= self.gamma
                z_curr = z_next

                if done.item() > 0.5:
                    total_return += rollout_return
                    continue

                for step in range(1, self.horizon):
                    random_action = torch.randint(0, 9, (1,), device=device)
                    z_next, reward, done = self.wm.predict(z_curr, random_action)
                    rollout_return += discount * reward.item()
                    discount *= self.gamma
                    z_curr = z_next
                    if done.item() > 0.5:
                        break

                total_return += rollout_return

            avg_return = total_return / self.num_rollouts
            if avg_return > best_value:
                best_value = avg_return
                best_action = action

        return best_action
