"""One PUCT tree search, three kinds of model behind it.

  SimAdapter(None)  classic MCTS on the real rules with random playouts (an opponent; no learning)
  SimAdapter(net)   AlphaZero-style search on the real rules with Morty's priors and values
                    (used to generate training data, and as the perfect-model upper bound)
  LatentAdapter     "Portal search": search inside the Council's imagination. After the root it
                    never touches the real rules. Each expansion asks every Rick; the child latent
                    comes from one Rick drawn at random (Thompson sampling over models), and the
                    backed-up value is mean - beta * std across Ricks, so lines the Council
                    disagrees about are trusted less.

Values are always from the point of view of the player who chose the move into a node, and they
flip sign at every ply (negamax).
"""
import numpy as np
import torch

from .multiverse import N_ACTIONS, Game, PAD
from .nets import torus_tactical_field


class Node:
    __slots__ = ("state", "P", "legal", "children", "N", "W", "v", "terminal", "tv", "sd")

    def __init__(self):
        self.children, self.N, self.W = {}, 0, 0.0
        self.terminal, self.tv, self.v, self.sd = False, 0.0, 0.0, 0.0
        self.state = self.P = self.legal = None


def _fill(node, out):
    node.state = out["state"]
    node.legal = np.flatnonzero(out["legal"])
    p = np.asarray(out["priors"], dtype=np.float64)[node.legal]
    node.P = p / p.sum() if p.sum() > 0 else np.full(len(node.legal), 1.0 / max(len(node.legal), 1))
    node.v = float(out["value"])


def _select(node, c_puct):
    Q = np.full(len(node.legal), node.v)   # unvisited moves start at the node's own value estimate
    n = np.zeros(len(node.legal))
    for i, a in enumerate(node.legal):
        ch = node.children.get(int(a))
        if ch is not None and ch.N > 0:
            Q[i], n[i] = ch.W / ch.N, ch.N
    return int(np.argmax(Q + c_puct * node.P * np.sqrt(node.N + 1) / (1 + n)))


def search(adapter, root_input, sims, c_puct=1.5, dirichlet=None, rng=None, beta=0.0,
           check_every=None, on_check=None):
    """Returns (visit counts over actions, mean value per action, number of model expansions).

    Anytime mode: if `on_check` is given, it is called as on_check(root, sims_done) before the
    first simulation and after every `check_every` simulations; returning True stops the search.
    A search stopped at t simulations is exactly the first t simulations of the uninterrupted
    search with the same random stream, so halting rules can be replayed offline from one trace.
    """
    root = Node()
    _fill(root, adapter.root(root_input))
    if len(root.legal) == 0:
        raise ValueError("no legal move at the root")
    if dirichlet is not None:
        alpha, eps = dirichlet
        root.P = (1 - eps) * root.P + eps * rng.dirichlet([alpha] * len(root.legal))
    expansions = 0
    for i in range(sims):
        if on_check is not None and i % check_every == 0 and on_check(root, i):
            break
        node, path = root, [root]
        while True:
            a = int(node.legal[_select(node, c_puct)])
            child = node.children.get(a)
            if child is None:
                out = adapter.step(node.state, a)
                expansions += 1
                child = Node()
                child.sd = float(out.get("q_std", 0.0))      # model disagreement on this edge
                value = out["q"] - beta * child.sd
                if out["terminal"]:
                    child.terminal, child.tv = True, value
                else:
                    _fill(child, out)
                node.children[a] = child
                path.append(child)
                break
            path.append(child)
            if child.terminal:
                value = child.tv
                break
            node = child
        for n in reversed(path[1:]):      # negamax backup
            n.N += 1
            n.W += value
            value = -value
        root.N += 1
    else:
        if on_check is not None:                  # final look at the completed search
            on_check(root, sims)
    visits, q = np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)
    for a, ch in root.children.items():
        visits[a], q[a] = ch.N, ch.W / max(ch.N, 1)
    return visits, q, expansions


def rollout_value(g: Game, rng) -> float:
    """Random playout; +1 if the player to move at g wins, -1 if they lose, 0 for a draw."""
    g, me = g.copy(), g.to_move
    while not g.done:
        acts = g.legal_actions()
        g.step(acts[int(rng.integers(len(acts)))])
    return 0.0 if g.winner == 0 else (1.0 if g.winner == me else -1.0)


@torch.no_grad()
def net_eval(net, obs, legal, device):
    """Morty's move probabilities (illegal moves zeroed) and value for one observation."""
    s = net.h(torch.as_tensor(obs[None], device=device))
    logits, v, _ = net.f(s)
    logits = logits[0].masked_fill(~torch.as_tensor(legal, device=device), float("-inf"))
    return torch.softmax(logits, -1).cpu().numpy(), float(v[0]), s


class SimAdapter:
    """Search on the real rules. Needs the Game object, so imagination-only agents never use it."""

    def __init__(self, net=None, device="cpu", rng=None):
        self.net, self.device, self.rng = net, device, rng

    def _eval(self, g, tactical=False):
        legal = g.legal_mask()
        if self.net is None:
            return legal.astype(np.float64), rollout_value(g, self.rng), legal
        obs = g.encode()
        if not tactical:
            p, v, _ = net_eval(self.net, obs, legal, self.device)
            return p, v, legal
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None], device=self.device)
            s = self.net.h(obs_t)
            logits, v, _ = self.net.f(s)
            v_tac, tac_w, pol_boost = torus_tactical_field(
                obs_t[:, 0:3], obs_t[:, 13:14], obs_t[:, 3:13]
            )
            vt, tw, vn = float(v_tac[0]), float(tac_w[0]), float(v[0])
            v_out = tw * (0.85 * vt + 0.15 * vn) + (1.0 - tw) * (0.65 * vn + 0.35 * vt)
            logits = (logits[0] + pol_boost[0]).masked_fill(~torch.as_tensor(legal, device=self.device), float("-inf"))
            return torch.softmax(logits, -1).cpu().numpy(), float(np.clip(v_out, -1.0, 1.0)), legal

    def root(self, g):
        p, v, legal = self._eval(g, tactical=False)
        return dict(state=g, priors=p, legal=legal, value=v)

    def step(self, g, a):
        g2 = g.copy()
        if g2.step(a):
            return dict(terminal=True, q=1.0)
        if g2.done:
            return dict(terminal=True, q=0.0)
        p, v, legal = self._eval(g2, tactical=True)
        return dict(terminal=False, q=-v, state=g2, priors=p, legal=legal, value=v)


class LatentAdapter:
    """Portal search: plan with the Council only. Root input is (observation, legal mask)."""

    def __init__(self, net, device="cpu", rng=None, legal_threshold=0.5):
        self.net, self.device, self.rng, self.thr = net, device, rng, legal_threshold

    def root(self, root_input):
        obs, legal = root_input
        p, v, s = net_eval(self.net, obs, legal, self.device)
        return dict(state=s, priors=p, legal=legal, value=v)

    @torch.no_grad()
    def step(self, s, a):
        lat, win, pol, val, leg = self.net.imagine(s, torch.tensor([a], device=self.device))
        win, val = win[:, 0].cpu().numpy(), val[:, 0].cpu().numpy()
        q_k = win + (1.0 - win) * (-val)            # value of move a for the player who made it
        q, q_std, p_win = float(q_k.mean()), float(q_k.std()), float(win.mean())
        legal = (torch.sigmoid(leg[:, 0]).mean(0) > self.thr).cpu().numpy()
        if p_win >= 0.5:
            return dict(terminal=True, q=q, q_std=q_std)
        if not legal.any():                           # imagined full board: draw
            return dict(terminal=True, q=p_win, q_std=float(win.std()))
        k = int(self.rng.integers(len(win)))           # Thompson: follow one Rick below this edge
        priors = torch.softmax(pol[:, 0], -1).mean(0).cpu().numpy()
        return dict(terminal=False, q=q, q_std=q_std, state=lat[k], priors=priors, legal=legal,
                    value=float(val.mean()))
