import torch
import torch.nn.functional as F


class PlanningModule:
    def __init__(self, world_model_interface, policy=None,
                 num_rollouts=10, horizon=5, gamma=0.99):
        self.wm = world_model_interface
        self.policy = policy
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.gamma = gamma

    def _pick_action(self, z, device):
        if self.policy is not None:
            with torch.no_grad():
                logits_list, _ = self.policy(z)
                avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
                probs = F.softmax(avg_logits, dim=-1)
                return torch.distributions.Categorical(probs).sample()
        return torch.randint(0, 9, (z.size(0),), device=device)

    def plan(self, z_t, valid_actions, device="cpu"):
        if len(valid_actions) == 0:
            return 0

        best_action = valid_actions[0]
        best_value = float('-inf')

        for action in valid_actions:
            actions_batch = []
            for _ in range(self.num_rollouts):
                seq = [action]
                z_curr = z_t.clone()
                self.wm.reset(1)
                a_t = torch.tensor([action], dtype=torch.long, device=device)
                z_next, _, _ = self.wm.predict(z_curr, a_t)
                z_curr = z_next
                for step in range(1, self.horizon):
                    a_next = self._pick_action(z_curr, device)
                    seq.append(a_next.item())
                    z_next, _, _ = self.wm.predict(z_curr, a_next)
                    z_curr = z_next
                actions_batch.append(seq)

            total_return = 0.0
            for seq in actions_batch:
                self.wm.reset(1)
                actions_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                z_states, rewards, dones = self.wm.rollout(z_t, actions_tensor)
                rewards = rewards.squeeze(0)
                dones = dones.squeeze(0)
                discount = 1.0
                ret = 0.0
                for t in range(self.horizon):
                    ret += discount * rewards[t].item()
                    discount *= self.gamma
                    if dones[t].item() > 0.5:
                        break
                total_return += ret

            avg_return = total_return / self.num_rollouts
            if avg_return > best_value:
                best_value = avg_return
                best_action = action

        return best_action
