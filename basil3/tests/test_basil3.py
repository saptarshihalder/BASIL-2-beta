"""BASIL-3 tests. Run from the repo root:  python -m pytest -q"""
import numpy as np
import torch

from basil3.agents import CitadelAgent, Gate
from basil3.multiverse import (ORIENTATIONS, PAD, Game, Universe, _family_from_seed, canonical_family,
                               compile_rules, opening, split_of_family, split_of_seed, universes)
from basil3.nets import Citadel
from basil3.search import SimAdapter, search


def U(**kw):
    base = dict(seed=-1, H=5, W=5, gravity=False, wrap_h=False, wrap_v=False, blocked=(), shapes=("I",))
    base.update(kw)
    return Universe(**base)


def act(r, c):
    return r * PAD + c


def test_orientation_counts():
    assert {k: len(v) for k, v in ORIENTATIONS.items()} == {"I": 2, "O": 1, "T": 4, "S": 4, "L": 8}


def test_line_win():
    g = Game(U())
    for c in range(3):
        assert g.step(act(0, c)) == 0.0
        g.step(act(4, c))
    assert g.step(act(0, 3)) == 1.0 and g.done and g.winner == 1


def test_gravity_landing():
    g = Game(U(gravity=True))
    assert sorted(g.legal_actions()) == [act(4, c) for c in range(5)]
    g.step(act(4, 2))
    assert act(3, 2) in g.legal_actions() and act(4, 2) not in g.legal_actions()
    try:
        g.step(act(0, 0))
    except ValueError:
        return
    raise AssertionError("a floating stone was accepted")


def test_portal_wrap_win_and_no_wrap_control():
    for wrap, expected in ((True, 1.0), (False, 0.0)):
        g = Game(U(wrap_h=wrap))
        for c in (3, 4, 0):
            g.step(act(0, c))
            g.step(act(2, c))
        assert g.step(act(0, 1)) == expected      # cells 3,4,0,1 only connect through the portal


def test_blocked_cells_never_legal():
    g = Game(U(blocked=(0, 6)))
    assert act(0, 0) not in g.legal_actions() and act(1, 1) not in g.legal_actions()


def test_random_games_end_consistently():
    rng = np.random.default_rng(0)
    for u in universes("train", 5):
        pats, _ = compile_rules(u)
        for _ in range(10):
            g = Game(u)
            while not g.done:
                acts = g.legal_actions()
                g.step(acts[int(rng.integers(len(acts)))])
            if g.winner:
                assert any((g.board[p] == g.winner).all() for p in pats)
            else:
                assert not g.legal_actions()
                assert not any((g.board[p] == pl).all() for p in pats for pl in (1, 2))


def test_encoding_is_from_the_movers_view():
    g = Game(U())
    g.step(act(2, 2))
    x = g.encode()
    assert x[1, 2, 2] == 1 and x[0, 2, 2] == 0     # player 2 to move: player 1's stone is "opponent"
    assert x[13, 2, 2] == 0 and x[13].sum() == 24


def _position(own, opp, empty):
    """5x5 board: checkerboard filler (no four in a row anywhere), then the given cells."""
    g = Game(U())
    for r in range(5):
        for c in range(5):
            g.board[r * 5 + c] = 1 if (r + c) % 2 == 0 else 2
    for cells, v in ((own, 1), (opp, 2), (empty, 0)):
        for r, c in cells:
            g.board[r * 5 + c] = v
    g.to_move = 1
    return g


def test_search_takes_the_win():
    g = _position(own=[(0, 0), (0, 1), (0, 2)], opp=[], empty=[(0, 3), (2, 2), (4, 4)])
    assert g.completes(3, 1)
    visits, _, _ = search(SimAdapter(None, rng=np.random.default_rng(0)), g, 300)
    assert int(np.argmax(visits)) == act(0, 3)


def test_search_blocks_the_loss():
    """The original BASIL planner scored the opponent's wins as its own; this is the regression test."""
    g = _position(own=[], opp=[(4, 0), (4, 1), (4, 2)], empty=[(4, 3), (0, 0), (0, 4)])
    A, B, C = 4 * 5 + 3, 0, 4
    assert g.completes(A, 2) and not g.completes(B, 2) and not g.completes(C, 2)
    assert not any(g.completes(x, 1) for x in (A, B, C))
    visits, _, _ = search(SimAdapter(None, rng=np.random.default_rng(0)), g, 300)
    assert int(np.argmax(visits)) == act(4, 3)


def test_splits_disjoint_and_symmetry_safe():
    fams = {}
    for seed in range(1500):
        f = canonical_family(*_family_from_seed(seed))
        fams.setdefault(f, set()).add(split_of_seed(seed))
    assert all(len(s) == 1 for s in fams.values())
    a = canonical_family(5, 7, False, True, False, ("I",))
    b = canonical_family(7, 5, False, False, True, ("I",))      # the transposed game
    assert a == b and split_of_family(a) == split_of_family(b)
    train = {u.family() for u in universes("train", 30)}
    assert not train & {u.family() for u in universes("val", 8)}
    assert not train & {u.family() for u in universes("test", 8)}


def test_determinism():
    a = [u.describe() for u in universes("val", 3)]
    assert a == [u.describe() for u in universes("val", 3)]
    u = universes("val", 1)[0]
    assert (opening(u, 5).board == opening(u, 5).board).all()


def test_imagination_agent_only_plays_legal_moves():
    torch.manual_seed(0)
    net = Citadel(ch=16, repr_blocks=1, dyn_blocks=1, n_ricks=2).eval()
    rng = np.random.default_rng(0)
    for u in universes("val", 3):
        g = opening(u, 0)
        for kind in ("always", "never"):
            a = CitadelAgent(net, Gate(kind), sims=8, rng=rng).act(g.encode(), g.legal_mask())
            assert g.legal_mask()[a]


def test_pinned_universe_dna_persists_across_imagination():
    torch.manual_seed(0)
    net = Citadel(ch=16, repr_blocks=1, dyn_blocks=1, n_ricks=2).eval()
    g = opening(universes("val", 1)[0], 0)
    obs = torch.as_tensor(g.encode()[None])
    s0 = net.h(obs)
    assert torch.allclose(s0[:, -10:], obs[:, 3:13])
    s1, _ = net.ricks[0](s0, torch.tensor([g.legal_actions()[0]]))
    assert torch.allclose(s1[:, -10:], obs[:, 3:13])


def test_meeseeks_trace_fit_and_anytime_agent():
    from basil3.meeseeks import AnytimeAgent, calibrate_all_rules, fit_halt_model, build_anytime_rule, run_trace
    torch.manual_seed(0)
    net = Citadel(ch=16, repr_blocks=1, dyn_blocks=1, n_ricks=2).eval()
    rng = np.random.default_rng(0)
    unis = universes("val", 2)
    traces = [run_trace(net, "latent", opening(u, k), max_sims=12, check_every=4, beta=0.5, rng=rng)
              for u in unis for k in range(3)]
    X = np.concatenate([t["feats"] for t in traces], 0)
    R = np.concatenate([t["regret"] for t in traces], 0)
    A = np.concatenate([t["agree"] for t in traces], 0)
    G = np.concatenate([np.full(len(t["sims"]), i) for i, t in enumerate(traces)])
    model, stats = fit_halt_model(X, R, A, G, rng, steps=50, hidden=32)
    assert "heldout_regret_corr_live" in stats
    cal = calibrate_all_rules(traces, model, max_sims=12, budgets=[0.25, 0.5])
    rule = build_anytime_rule("meeseeks", "0.25", 12, cal, model)
    g = opening(unis[0], 0)
    a = AnytimeAgent(net, "latent", rule, 12, 4, rng=rng).act(g.encode(), g.legal_mask())
    assert g.legal_mask()[a]


def test_central_finite_curve_never_eliminates_true_universe():
    from basil3.cronenberg import CentralFiniteCurve, infer_geometry, mask_rules, play_cronenberg_match
    torch.manual_seed(0)
    net = Citadel(ch=16, repr_blocks=1, dyn_blocks=1, n_ricks=2).eval()
    for u in universes("val", 3):
        g = Game(u)
        H, W, blk = infer_geometry(mask_rules(g.encode()))
        cfc = CentralFiniteCurve(H, W, blk)
        n0 = cfc.n_alive
        cfc.observe_turn(mask_rules(g.encode()), g.legal_mask())
        assert 1 <= cfc.n_alive < n0
        assert any(
            (c.gravity, c.wrap_h, c.wrap_v, c.shapes) == (u.gravity, u.wrap_h, u.wrap_v, u.shapes)
            for c, ok in zip(cfc.candidates, cfc.alive) if ok
        )
    recs = play_cronenberg_match(net, universes("val", 1)[0], episodes=3, mode="cfc_citadel", opp_spec="random", sims=8)
    assert len(recs) == 3 and recs[-1]["entropy_bits_end"] <= recs[0]["entropy_bits_start"]


if __name__ == "__main__":
    import sys
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    sys.exit(0)
