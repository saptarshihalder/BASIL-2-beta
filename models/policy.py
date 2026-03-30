import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadPolicy(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, num_actions=9, num_heads=5):
        super().__init__()
        self.num_heads = num_heads
        self.num_actions = num_actions

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_actions) for _ in range(num_heads)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z, valid_mask=None):
        h = self.shared(z)
        logits_list = []
        for head in self.policy_heads:
            logits = head(h)
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask, float('-inf'))
            logits_list.append(logits)
        value = self.value_head(h).squeeze(-1)
        return logits_list, value

    def get_action_and_value(self, z, valid_mask=None, deterministic=False):
        logits_list, value = self.forward(z, valid_mask)
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        probs = F.softmax(avg_logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.distributions.Categorical(probs).log_prob(action)
        return action, log_prob, value, logits_list

    def evaluate_actions(self, z, actions, valid_mask=None):
        logits_list, value = self.forward(z, valid_mask)
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        probs = F.softmax(avg_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
