"""Citadel networks.

    observation --h--> latent s --f--> (move logits, value, legal-move logits)      "Morty"
    (s, a) --g_k--> (next latent s', P(move a wins)),  k = 1..K                    "Council of Ricks"

* Every value is from the point of view of the player to move (the observation is encoded that
  way), so the sign of a backed-up value flips at every ply. This removes the old sign bug by
  construction.
* Each Rick is a separate dynamics network. Ricks see different bootstrap halves of the data and
  carry a fixed random "prior" network on their win head (randomised priors, Osband et al. 2018),
  so they agree where the data pins the rules down and disagree where it does not. That
  disagreement is the uncertainty signal used by the search and the gate.
* f also predicts which moves are legal, so imagined lines never use illegal moves and
  imagined positions with no legal move are treated as draws.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiverse import N_ACTIONS, N_PLANES, PAD


def _norm(ch):
    return nn.GroupNorm(min(8, ch), ch)


class Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), _norm(ch), nn.ReLU(),
                                 nn.Conv2d(ch, ch, 3, padding=1), _norm(ch))

    def forward(self, x):
        return F.relu(x + self.net(x))


def scale_latent(s):
    """Min-max scale each latent to [0, 1] (MuZero) so imagined latents stay in range."""
    flat = s.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    return ((flat - lo) / (hi - lo + 1e-5)).view_as(s)


class Representation(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, ch, 3, padding=1), _norm(ch), nn.ReLU())
        self.body = nn.Sequential(*[Res(ch) for _ in range(blocks)])

    def forward(self, x):
        return scale_latent(self.body(self.stem(x)))


class Prediction(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.policy = nn.Sequential(nn.Conv2d(ch, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * PAD * PAD, N_ACTIONS))
        self.value = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(2 * PAD * PAD, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.legal = nn.Conv2d(ch, 1, 1)

    def forward(self, s):
        return self.policy(s), self.value(s).squeeze(-1), self.legal(s).flatten(1)


class Rick(nn.Module):
    """One member of the Council: (latent, action) -> (next latent, win logit)."""

    def __init__(self, ch, blocks, prior_scale):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(ch + 1, ch, 3, padding=1), _norm(ch), nn.ReLU())
        self.body = nn.Sequential(*[Res(ch) for _ in range(blocks)])
        self.win = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * PAD * PAD, 1))
        self.prior = nn.Sequential(nn.Conv2d(ch + 1, 4, 3, padding=1), nn.Tanh(), nn.Flatten(), nn.Linear(4 * PAD * PAD, 1))
        for p in self.prior.parameters():          # fixed random function, never trained
            p.requires_grad_(False)
        self.prior_scale = prior_scale

    def forward(self, s, a):
        plane = F.one_hot(a, N_ACTIONS).float().view(-1, 1, PAD, PAD)
        x = torch.cat([s, plane], 1)
        h = self.body(self.inp(x))
        win_logit = self.win(h).squeeze(-1) + self.prior_scale * self.prior(x).squeeze(-1)
        return scale_latent(h), win_logit


class Citadel(nn.Module):
    def __init__(self, ch=64, repr_blocks=3, dyn_blocks=2, n_ricks=4, prior_scale=1.0):
        super().__init__()
        self.config = dict(ch=ch, repr_blocks=repr_blocks, dyn_blocks=dyn_blocks, n_ricks=n_ricks,
                           prior_scale=prior_scale)
        self.h = Representation(ch, repr_blocks)
        self.f = Prediction(ch)
        self.ricks = nn.ModuleList([Rick(ch, dyn_blocks, prior_scale) for _ in range(n_ricks)])

    @property
    def n_ricks(self):
        return len(self.ricks)

    @staticmethod
    def consistency(s_pred, s_target):
        """Imagined latent vs the encoding of the real next position (no gradient into the target):
        minus their correlation, so -1 means identical up to scale. There is no trainable projection
        head, so the loss cannot be met by collapsing a projector to a constant; h stays anchored by
        the policy, value and legality losses."""
        p, t = s_pred.flatten(1), s_target.flatten(1).detach()
        p = F.normalize(p - p.mean(1, keepdim=True), dim=-1)
        t = F.normalize(t - t.mean(1, keepdim=True), dim=-1)
        return -(p * t).sum(-1)

    @torch.no_grad()
    def imagine(self, s, a):
        """Run every Rick on (s, a) and Morty on every imagined latent.
        Returns latents (K,B,C,7,7), win prob (K,B), policy logits (K,B,A), value (K,B), legal logits (K,B,A)."""
        lat, wl = zip(*[r(s, a) for r in self.ricks])
        lat = torch.stack(lat)
        K, B = lat.shape[:2]
        pol, val, leg = self.f(lat.flatten(0, 1))
        return lat, torch.sigmoid(torch.stack(wl)), pol.view(K, B, -1), val.view(K, B), leg.view(K, B, -1)


def build(cfg: dict) -> Citadel:
    keys = ("ch", "repr_blocks", "dyn_blocks", "n_ricks", "prior_scale")
    return Citadel(**{k: cfg[k] for k in keys if k in cfg})
