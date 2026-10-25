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


if __name__ == "__main__":
    import sys
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    sys.exit(0)
