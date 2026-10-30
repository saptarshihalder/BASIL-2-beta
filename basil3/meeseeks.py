"""Mr. Meeseeks: learned anytime halting for tree search ("exists only until the problem is solved").

A pre-search gate decides once, before thinking, whether to think. Meeseeks instead watches the
search while it runs and, every `check_every` simulations, decides whether more thinking is still
worth it. It works with either search in this package:

    kind="latent"   search inside the learned world model (Citadel; no simulator at decision time)
    kind="sim"      search on the real rules with the learned priors and values (Oracle)

Training targets are self-supervised and dense. For one position we run the full-budget search
once and record root statistics at every checkpoint t. Because a search stopped after t simulations
is exactly the prefix of the full search, the label for checkpoint t is known from the same trace:

    regret_t = Q_T(a_T) - Q_T(a_t)      (Q_T: root action values after the full search,
                                          a_t: the move we would play if we stopped at t,
                                          a_T: the full-budget move)
    agree_t  = [a_t == a_T]

so every searched position gives T / check_every labels, all measured by one search instead of
the difference of two noisy reference searches (the label noise that sank the pre-search VOC gate).
At t = 0 the move is the fast policy's, so the same model also decides whether to think at all.

Every halting rule here (fixed budgets, classic early stopping, the confidence pre-gate and
Meeseeks) is a function of the live features, so all of them are calibrated to the same average
compute on validation traces by exact offline replay, then run online in games.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search import LatentAdapter, SimAdapter, search

LIVE_FEATURES = [
    "frac_sims", "log_sims", "prior_max", "prior_entropy", "net_value", "abs_net_value", "legal_frac",
    "share_top", "gap_top2", "visit_entropy", "q_best", "q_gap_top2", "q_max_minus_best",
    "root_value", "dv_prior", "dv_last", "best_is_prior", "prior_of_best", "changes_rate",
    "since_change", "sd_best", "sd_mean", "unreachable", "expanded_frac",
]
_IDX = {k: i for i, k in enumerate(LIVE_FEATURES)}


class CountingAdapter:
    """Wraps a search adapter and counts model expansions (the compute we report)."""

    def __init__(self, inner):
        self.inner, self.n = inner, 0

    def root(self, x):
        return self.inner.root(x)

    def step(self, s, a):
        self.n += 1
        return self.inner.step(s, a)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class Tracker:
    """The on_check callback: computes live features at every checkpoint, keeps a trace, and asks
    an optional halting rule whether to stop."""

    def __init__(self, counter, max_sims, check_every, rule=None):
        self.counter, self.max_sims, self.check_every, self.rule = counter, max_sims, check_every, rule
        self.trace = []                    # (sims_done, expansions, best_move, features)
        self.prev_best, self.prev_value, self.changes, self.last_change = None, None, 0, 0
        self.final = None                  # (actions, Q, N) of the root when the search ends

    def features(self, root, t):
        legal = root.legal
        P = root.P
        nl = max(len(legal), 1)
        N = np.array([root.children[int(a)].N if int(a) in root.children else 0 for a in legal], dtype=np.float64)
        W = np.array([root.children[int(a)].W if int(a) in root.children else 0.0 for a in legal])
        Q = np.where(N > 0, W / np.maximum(N, 1), np.nan)
        prior_best = int(np.argmax(P))
        if N.sum() > 0:
            order = np.lexsort((-P, -N))               # most visits, ties broken by prior
            b, s2 = order[0], (order[1] if len(order) > 1 else order[0])
            dist = N / N.sum()
        else:
            b, s2 = prior_best, prior_best
            dist = P
        best = int(legal[b])
        q_best = Q[b] if N[b] > 0 else root.v
        q_s2 = Q[s2] if N[s2] > 0 else q_best
        q_max = np.nanmax(Q) if np.any(N > 0) else q_best
        root_value = float(W.sum() / N.sum()) if N.sum() > 0 else root.v
        if self.prev_best is not None and best != self.prev_best:
            self.changes += 1
            self.last_change = len(self.trace)
        n_checks = max(len(self.trace), 1)
        sds = [root.children[int(a)].sd for a in legal if int(a) in root.children]
        sd_best = root.children[best].sd if best in root.children else 0.0
        n_sorted = np.sort(N)[::-1]
        gap = n_sorted[0] - (n_sorted[1] if len(n_sorted) > 1 else 0.0)
        f = np.array([
            t / self.max_sims, np.log1p(t) / np.log1p(self.max_sims), float(P.max()), _entropy(P) / np.log(max(nl, 2)),
            root.v, abs(root.v), nl / 49.0,
            float(N[b] / max(N.sum(), 1)), float(gap / max(N.sum(), 1)), _entropy(dist) / np.log(max(nl, 2)),
            float(q_best), float(q_best - q_s2), float(q_max - q_best),
            root_value, root_value - root.v, root_value - (self.prev_value if self.prev_value is not None else root.v),
            float(b == prior_best), float(P[b]), self.changes / n_checks, (len(self.trace) - self.last_change) / max(self.max_sims / self.check_every, 1),
            float(sd_best), float(np.mean(sds)) if sds else 0.0, float(gap > self.max_sims - t), len(sds) / nl,
        ], dtype=np.float32)
        self.prev_best, self.prev_value = best, root_value
        return best, f

    def __call__(self, root, t):
        best, f = self.features(root, t)
        self.trace.append((t, self.counter.n, best, f))
        if t >= self.max_sims:
            legal = root.legal
            self.final = (np.array(legal), np.array([root.children[int(a)].W / root.children[int(a)].N
                                                     if int(a) in root.children and root.children[int(a)].N > 0 else np.nan
                                                     for a in legal]),
                          np.array([root.children[int(a)].N if int(a) in root.children else 0 for a in legal]))
            return True
        return bool(self.rule is not None and self.rule.stop(f, t))


def make_adapter(kind, net, device, rng):
    if kind == "latent":
        return LatentAdapter(net, device, rng)
    if kind == "sim":
        return SimAdapter(net, device)
    raise ValueError(kind)


def root_input(kind, game):
    return (game.encode(), game.legal_mask()) if kind == "latent" else game


def run_trace(net, kind, game, max_sims, check_every, beta, rng, device="cpu"):
    """Full-budget search with a checkpoint trace. Returns a dict of per-checkpoint arrays + labels."""
    counter = CountingAdapter(make_adapter(kind, net, device, rng))
    tr = Tracker(counter, max_sims, check_every)
    search(counter, root_input(kind, game), max_sims, beta=beta if kind == "latent" else 0.0,
           check_every=check_every, on_check=tr)
    acts, Qf, Nf = tr.final
    a_final = int(acts[np.lexsort((-np.nan_to_num(Qf, nan=-9), -Nf))[0]])
    qmap = {int(a): q for a, q in zip(acts, Qf)}
    q_floor = float(np.nanmin(Qf)) if np.any(~np.isnan(Qf)) else 0.0
    qa = lambda a: q_floor if np.isnan(qmap.get(a, np.nan)) else float(qmap[a])
    sims = np.array([t for t, _, _, _ in tr.trace])
    exps = np.array([e for _, e, _, _ in tr.trace])
    best = np.array([b for _, _, b, _ in tr.trace])
    feats = np.stack([f for _, _, _, f in tr.trace])
    regret = np.array([qa(a_final) - qa(int(b)) for b in best], dtype=np.float32)
    agree = (best == a_final).astype(np.float32)
    return dict(sims=sims, exps=exps, best=best, feats=feats, regret=regret, agree=agree, final=a_final)


# --------------------------------------------------------------------------------- model
class HaltModel(nn.Module):
    """Predicts the regret of stopping now and the chance that the current move is final."""

    def __init__(self, n_in=len(LIVE_FEATURES), hidden=128, mask=None):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.regret = nn.Linear(hidden, 1)
        self.agree = nn.Linear(hidden, 1)
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.register_buffer("mask", torch.ones(n_in) if mask is None else torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, x):
        h = self.body((x - self.mu) / self.sd * self.mask)
        return self.regret(h).squeeze(-1), self.agree(h).squeeze(-1)

    @torch.no_grad()
    def score(self, x, head="regret"):
        """Higher = more reason to keep thinking."""
        r, a = self(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32))
        return (r if head == "regret" else -torch.sigmoid(a) + 1.0).numpy()


def fit_halt_model(X, R, A, groups, rng, steps=3000, mask=None, hidden=128, lr=1e-3):
    """Fit on (features, regret, agree). `groups` (position ids) keeps the held-out split by position."""
    model = HaltModel(X.shape[1], hidden, mask)
    model.mu.copy_(torch.as_tensor(X.mean(0)))
    model.sd.copy_(torch.as_tensor(X.std(0) + 1e-6))
    ug = np.unique(groups)
    ho_groups = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False).tolist())
    ho = np.array([g in ho_groups for g in groups])
    tr_idx, ho_idx = np.flatnonzero(~ho), np.flatnonzero(ho)
    Xt, Rt, At = (torch.as_tensor(v, dtype=torch.float32) for v in (X, R, A))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.as_tensor(rng.choice(tr_idx, size=min(512, len(tr_idx))))
        r, a = model(Xt[b])
        loss = F.smooth_l1_loss(r, Rt[b], beta=0.05) + 0.5 * F.binary_cross_entropy_with_logits(a, At[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        r, a = model(Xt[ho_idx])
    r, p = r.numpy(), torch.sigmoid(a).numpy()
    R_ho, A_ho = R[ho_idx], A[ho_idx]
    corr = float(np.corrcoef(r, R_ho)[0, 1]) if R_ho.std() > 0 and r.std() > 0 else float("nan")
    t0_mask = X[ho_idx, 0] == 0.0
    live_mask = X[ho_idx, 0] > 0.0
    corr_t0 = (float(np.corrcoef(r[t0_mask], R_ho[t0_mask])[0, 1])
               if t0_mask.sum() > 2 and R_ho[t0_mask].std() > 0 and r[t0_mask].std() > 0 else float("nan"))
    corr_live = (float(np.corrcoef(r[live_mask], R_ho[live_mask])[0, 1])
                 if live_mask.sum() > 2 and R_ho[live_mask].std() > 0 and r[live_mask].std() > 0 else float("nan"))
    acc = float(((p > 0.5) == (A_ho > 0.5)).mean())
    acc_live = float(((p[live_mask] > 0.5) == (A_ho[live_mask] > 0.5)).mean()) if live_mask.any() else acc
    return model, {
        "heldout_regret_corr": corr,
        "heldout_regret_corr_t0": corr_t0,
        "heldout_regret_corr_live": corr_live,
        "heldout_agree_acc": acc,
        "heldout_agree_acc_live": acc_live,
        "heldout_agree_base": float(A_ho.mean()),
        "n_train": int(len(tr_idx)),
        "n_heldout": int(len(ho_idx)),
    }


# --------------------------------------------------------------------------------- rules
class Rule:
    """stop(features, sims_done) -> bool. One scalar `threshold` sets the compute budget."""
    name = "rule"

    def __init__(self, threshold, max_sims):
        self.threshold, self.max_sims = threshold, max_sims

    def stop(self, f, t):
        raise NotImplementedError


class FixedRule(Rule):
    name = "fixed"                                   # threshold = number of simulations

    def stop(self, f, t):
        return t >= self.threshold


class ShareRule(Rule):
    """Stop once the most-visited move holds `threshold` of the visits (after a minimum of 8 sims)."""
    name = "share"

    def stop(self, f, t):
        return t >= 8 and f[_IDX["share_top"]] >= self.threshold


class UnreachableRule(Rule):
    """Classic early stopping: stop when the runner-up can no longer catch the leader within the
    budget. The threshold is the budget itself (an integer number of simulations)."""
    name = "unreach"

    def stop(self, f, t):
        if t >= self.threshold:
            return True
        n_top_gap = f[_IDX["gap_top2"]] * max(t, 1)
        return t > 0 and n_top_gap > self.threshold - t


class ConfidenceGate(Rule):
    """Pre-search gate: think the full budget only if the fast policy's top probability is below
    the threshold (decided once, at t = 0)."""
    name = "confgate"

    def stop(self, f, t):
        return f[_IDX["prior_max"]] >= self.threshold if t == 0 else t >= self.max_sims


class MeeseeksRule(Rule):
    """Stop as soon as the predicted regret of stopping falls below the threshold."""
    name = "meeseeks"

    def __init__(self, threshold, max_sims, model, head="regret", presearch_only=False):
        super().__init__(threshold, max_sims)
        self.model, self.head, self.presearch_only = model, head, presearch_only

    def stop(self, f, t):
        if self.presearch_only and t > 0:            # ablation: may only skip thinking at t = 0
            return t >= self.max_sims
        return float(self.model.score(f, self.head)[0]) <= self.threshold


def replay(traces, rule):
    """Offline replay of a halting rule on full traces -> (mean expansions, mean regret, agree rate)."""
    exps, reg, agr = [], [], []
    for tr in traces:
        k = len(tr["sims"]) - 1
        for i, (t, f) in enumerate(zip(tr["sims"], tr["feats"])):
            if rule.stop(f, int(t)):
                k = i
                break
        exps.append(tr["exps"][k])
        reg.append(tr["regret"][k])
        agr.append(tr["agree"][k])
    return float(np.mean(exps)), float(np.mean(reg)), float(np.mean(agr))


def _batched_scores(traces, model, head):
    for tr in traces:
        if "_score" not in tr or tr.get("_score_key") != (id(model), head):
            tr["_score"] = model.score(tr["feats"], head)
            tr["_score_key"] = (id(model), head)


def replay_meeseeks(traces, threshold, model, head="regret", presearch_only=False):
    """Fast exact replay for MeeseeksRule using precomputed scores."""
    _batched_scores(traces, model, head)
    exps, reg, agr = [], [], []
    for tr in traces:
        s = tr["_score"]
        k = len(s) - 1
        if presearch_only:
            k = 0 if s[0] <= threshold else k
        else:
            hit = np.flatnonzero(s <= threshold)
            if len(hit):
                k = int(hit[0])
        exps.append(tr["exps"][k])
        reg.append(tr["regret"][k])
        agr.append(tr["agree"][k])
    return float(np.mean(exps)), float(np.mean(reg)), float(np.mean(agr))


def calibrate(evaluate_fn, candidates, budget):
    """Pick the candidate threshold whose replayed mean expansions is closest to `budget`.
    `evaluate_fn(threshold) -> mean expansions` must be monotone in the threshold."""
    best, best_err = None, float("inf")
    for c in candidates:
        e = evaluate_fn(c)
        if abs(e - budget) < best_err:
            best, best_err = c, abs(e - budget)
    return float(best)


def calibrate_all_rules(val_traces, model, max_sims, budgets):
    """Calibrate thresholds for Meeseeks and anytime baselines on validation traces for each
    fractional compute budget in `budgets` (as a share of full-budget expansions)."""
    full_exps = float(np.mean([tr["exps"][-1] for tr in val_traces]))
    all_feats = np.concatenate([tr["feats"] for tr in val_traces], axis=0)
    t0_feats = np.stack([tr["feats"][0] for tr in val_traces], axis=0)
    reg_scores = model.score(all_feats, "regret")
    agr_scores = model.score(all_feats, "agree")
    pre_scores = model.score(t0_feats, "regret")
    prior_max = t0_feats[:, _IDX["prior_max"]]

    q_grid = np.linspace(0.0, 1.0, 161)
    reg_cands = np.unique(np.quantile(reg_scores, q_grid))
    agr_cands = np.unique(np.quantile(agr_scores, q_grid))
    pre_cands = np.unique(np.quantile(pre_scores, q_grid))
    conf_cands = np.unique(np.quantile(prior_max, q_grid))
    share_cands = np.linspace(0.20, 0.99, 80)
    unreach_cands = np.arange(0, max_sims + 1, 2)
    fixed_cands = val_traces[0]["sims"]

    thresholds = {k: {} for k in ("meeseeks", "meeseeks_agree", "meeseeks_pre", "share", "unreach", "confgate", "fixed")}
    curves = {k: {} for k in thresholds}
    for b in budgets:
        key = f"{b:g}"
        target = b * full_exps

        thr = calibrate(lambda c: replay_meeseeks(val_traces, c, model, "regret", False)[0], reg_cands, target)
        thresholds["meeseeks"][key] = thr
        curves["meeseeks"][key] = dict(zip(("exps", "regret", "agree"),
                                           replay_meeseeks(val_traces, thr, model, "regret", False)))

        thr = calibrate(lambda c: replay_meeseeks(val_traces, c, model, "agree", False)[0], agr_cands, target)
        thresholds["meeseeks_agree"][key] = thr
        curves["meeseeks_agree"][key] = dict(zip(("exps", "regret", "agree"),
                                                 replay_meeseeks(val_traces, thr, model, "agree", False)))

        thr = calibrate(lambda c: replay_meeseeks(val_traces, c, model, "regret", True)[0], pre_cands, target)
        thresholds["meeseeks_pre"][key] = thr
        curves["meeseeks_pre"][key] = dict(zip(("exps", "regret", "agree"),
                                               replay_meeseeks(val_traces, thr, model, "regret", True)))

        thr = calibrate(lambda c: replay(val_traces, ShareRule(c, max_sims))[0], share_cands, target)
        thresholds["share"][key] = thr
        curves["share"][key] = dict(zip(("exps", "regret", "agree"), replay(val_traces, ShareRule(thr, max_sims))))

        thr = calibrate(lambda c: replay(val_traces, UnreachableRule(c, max_sims))[0], unreach_cands, target)
        thresholds["unreach"][key] = thr
        curves["unreach"][key] = dict(zip(("exps", "regret", "agree"), replay(val_traces, UnreachableRule(thr, max_sims))))

        thr = calibrate(lambda c: replay(val_traces, ConfidenceGate(c, max_sims))[0], conf_cands, target)
        thresholds["confgate"][key] = thr
        curves["confgate"][key] = dict(zip(("exps", "regret", "agree"), replay(val_traces, ConfidenceGate(thr, max_sims))))

        thr = calibrate(lambda c: replay(val_traces, FixedRule(c, max_sims))[0], fixed_cands, target)
        thresholds["fixed"][key] = thr
        curves["fixed"][key] = dict(zip(("exps", "regret", "agree"), replay(val_traces, FixedRule(thr, max_sims))))

    return {"full_exps": full_exps, "thresholds": thresholds, "val_replay": curves}


def build_anytime_rule(rule_name, budget_str, max_sims, m_ck, model):
    if rule_name == "fixed_sims":
        return FixedRule(float(budget_str), max_sims)
    thr = m_ck["thresholds"][rule_name][budget_str]
    if rule_name == "meeseeks":
        return MeeseeksRule(thr, max_sims, model, head="regret", presearch_only=False)
    if rule_name == "meeseeks_agree":
        return MeeseeksRule(thr, max_sims, model, head="agree", presearch_only=False)
    if rule_name == "meeseeks_pre":
        return MeeseeksRule(thr, max_sims, model, head="regret", presearch_only=True)
    if rule_name == "share":
        return ShareRule(thr, max_sims)
    if rule_name == "unreach":
        return UnreachableRule(thr, max_sims)
    if rule_name == "confgate":
        return ConfidenceGate(thr, max_sims)
    if rule_name == "fixed":
        return FixedRule(thr, max_sims)
    raise ValueError(f"unknown anytime rule {rule_name}")


# --------------------------------------------------------------------------------- agent
class AnytimeAgent:
    """Fast policy + anytime search (latent or real rules) stopped by a halting rule."""

    def __init__(self, net, kind, rule, max_sims, check_every, beta=0.5, device="cpu", rng=None):
        self.net, self.kind, self.rule, self.max_sims, self.check_every = net, kind, rule, max_sims, check_every
        self.beta, self.device, self.rng = beta, device, rng
        self.needs_simulator = kind == "sim"
        self.moves = self.planned = self.expansions = 0
        self.search_calls = 0

    def act(self, obs, legal, game=None):
        counter = CountingAdapter(make_adapter(self.kind, self.net, self.device, self.rng))
        tr = Tracker(counter, self.max_sims, self.check_every, rule=self.rule)
        inp = (obs, legal) if self.kind == "latent" else game
        visits, _, n = search(counter, inp, self.max_sims, beta=self.beta if self.kind == "latent" else 0.0,
                              check_every=self.check_every, on_check=tr)
        self.moves += 1
        self.expansions += n
        self.search_calls += n * (self.net.n_ricks if self.kind == "latent" else 1)
        if n > 0:
            self.planned += 1
        # The tracker's best move at the moment the search stopped: most visits, ties broken by the
        # prior, and the fast policy's move when no simulation ran. Identical to the offline replay.
        return int(tr.trace[-1][2])
