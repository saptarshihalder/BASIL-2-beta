"""Players and the planning gate.

The evaluator gives every agent the observation and the legal-move mask. Only agents with
needs_simulator = True (classic opponents and the perfect-model reference) also get a copy of the
real game. Morty and the Citadel agents never see it, so they cannot plan with the true rules.

The gate ("Meeseeks": summoned for one hard move, gone afterwards) decides when Morty's instinct
is not enough and Portal search is worth its cost:
  never / always / random  baselines
  confidence               plan when Morty's top move probability is low
  disagreement             plan when the Ricks disagree about the top moves' outcomes
  voc                      plan when a small learned model predicts that searching will gain the
                           most value per move (value of computation; the model is trained on
                           train universes against outcomes checked with the real rules)
Score gates plan when score >= threshold. Thresholds are chosen on validation positions so every
gate plans on the same fraction of moves (its budget); an online quantile controller keeps that
rate in deployment.
"""
import numpy as np
import torch
import torch.nn as nn

from .multiverse import N_ACTIONS, greedy_move
from .search import LatentAdapter, SimAdapter, net_eval, search

FEATURES = ["max_prob", "entropy", "margin", "value", "abs_value", "lookahead_regret",
            "council_std_mean", "council_std_max", "win_prob_max", "legal_frac"]


@torch.no_grad()
def gate_features(net, obs, legal, device="cpu", top_m=3):
    """Cheap features for the gate: Morty's confidence plus one imagined step by every Rick on
    Morty's top moves. Returns (features, Morty's move, raw scores for the simple gates)."""
    p, v, s = net_eval(net, obs, legal, device)
    acts = np.flatnonzero(legal)
    probs = p[acts]
    order = acts[np.argsort(-probs, kind="stable")][:top_m]
    top = np.sort(probs)[::-1]
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(max(len(acts), 2)))
    _, win, _, val, _ = net.imagine(s.expand(len(order), -1, -1, -1), torch.as_tensor(order, device=device))
    q = (win + (1 - win) * (-val)).cpu().numpy()            # (Ricks, top moves), mover's view
    qm, qs = q.mean(0), q.std(0)
    margin = top[0] - (top[1] if len(top) > 1 else 0.0)
    feats = np.array([top[0], entropy, margin, v, abs(v), qm.max() - qm[0], qs.mean(), qs.max(),
                      float(win.mean(0).max()), len(acts) / N_ACTIONS], dtype=np.float32)
    scores = {"confidence": float(1 - top[0]), "disagreement": float(qs.mean())}
    return feats, int(order[0]), scores


class VOCModel(nn.Module):
    """Predicts how much value searching would add over Morty's move (can be negative)."""

    def __init__(self, hidden=64):
        super().__init__()
        n = len(FEATURES)
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sd", torch.ones(n))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)

    @torch.no_grad()
    def predict(self, feats):
        return float(self(torch.as_tensor(np.asarray(feats, dtype=np.float32)[None])))


class Gate:
    def __init__(self, kind, threshold=None, rate=None, rng=None, voc=None, online_budget=None, lr=0.01):
        self.kind, self.threshold, self.rate, self.rng, self.voc = kind, threshold, rate, rng, voc
        self.online_budget, self.lr = online_budget, lr

    def score(self, feats, scores):
        return self.voc.predict(feats) if self.kind == "voc" else scores[self.kind]

    def decide(self, feats, scores):
        if self.kind in ("never", "always"):
            return self.kind == "always"
        if self.kind == "random":
            return bool(self.rng.random() < self.rate)
        plan = self.score(feats, scores) >= self.threshold
        if self.online_budget is not None:        # Robbins-Monro tracking of the (1 - budget) quantile
            self.threshold += self.lr * (float(plan) - self.online_budget)
        return bool(plan)


# ------------------------------------------------------------------------------- agents
class RandomAgent:
    needs_simulator = False

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, legal, game=None):
        acts = np.flatnonzero(legal)
        return int(acts[self.rng.integers(len(acts))])


class GreedyAgent:
    needs_simulator = True

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, legal, game=None):
        return greedy_move(game, self.rng)


class UCTAgent:
    """Classic MCTS with random playouts on the real rules. Knows nothing learned."""
    needs_simulator = True

    def __init__(self, sims, rng):
        self.sims, self.rng = sims, rng

    def act(self, obs, legal, game=None):
        visits, _, _ = search(SimAdapter(None, rng=self.rng), game, self.sims, c_puct=1.4)
        return int(np.argmax(visits))


class OracleAgent:
    """Morty + search on the real rules: what Citadel could do with a perfect world model."""
    needs_simulator = True

    def __init__(self, net, sims, device="cpu"):
        self.net, self.sims, self.device = net, sims, device
        self.moves = self.planned = self.expansions = 0

    def act(self, obs, legal, game=None):
        visits, _, n = search(SimAdapter(self.net, self.device), game, self.sims)
        self.moves += 1
        self.planned += 1
        self.expansions += n
        return int(np.argmax(visits))


class MortyAgent:
    needs_simulator = False

    def __init__(self, net, device="cpu"):
        self.net, self.device = net, device
        self.moves = self.planned = self.expansions = 0

    def act(self, obs, legal, game=None):
        p, _, _ = net_eval(self.net, obs, legal, self.device)
        self.moves += 1
        return int(np.argmax(p))


class CitadelAgent:
    """Morty by default; Portal search inside the Council's imagination when the gate says so."""
    needs_simulator = False

    def __init__(self, net, gate, sims, beta=0.5, device="cpu", rng=None):
        self.net, self.gate, self.sims, self.beta, self.device, self.rng = net, gate, sims, beta, device, rng
        self.moves = self.planned = self.expansions = 0
        self.probe_calls = self.search_calls = 0

    def act(self, obs, legal, game=None):
        self.moves += 1
        if self.gate.kind == "always":
            fast, plan = None, True
        elif self.gate.kind == "never":
            fast, _, _ = net_eval(self.net, obs, legal, self.device)
            return int(np.argmax(fast))
        else:
            feats, fast, scores = gate_features(self.net, obs, legal, self.device)
            # The random control pays for the same probes as the learned gates.
            self.probe_calls += min(3, int(legal.sum())) * self.net.n_ricks
            plan = self.gate.decide(feats, scores)
        if not plan:
            return fast
        self.planned += 1
        visits, _, n = search(LatentAdapter(self.net, self.device, self.rng), (obs, legal), self.sims, beta=self.beta)
        self.expansions += n
        self.search_calls += n * self.net.n_ricks
        return int(np.argmax(visits))
