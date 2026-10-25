"""Multiverse: a procedurally generated family of two-player connection games.

Every universe is a different game. Rules vary along five axes:
  * board size  H x W in {5, 6, 7}^2 (all boards are padded to 7x7)
  * gravity     pieces fall to the lowest free cell of a column (Connect-4 style) or not
  * portals     the left/right edges (and, without gravity, top/bottom) wrap around
  * blocked     0-3 cells that nobody can use
  * win shapes  one or two tetromino shapes (I, O, T, S, L, any rotation or mirror image);
                the first player to cover one placement of a winning shape wins

Why a generated family instead of TicTacToe/Connect4:
  * no human strategy, opening book or published heuristic exists for these games, so nothing
    the designer knows can leak into the agent;
  * test games use rule *combinations* never seen in training (the split hashes the canonical
    family: size, gravity, portals, shapes), so the score measures planning, not memory;
  * universes whose first player wins or draws almost always are rejected before use.

Nothing in this file imports torch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PAD = 7
N_ACTIONS = PAD * PAD          # action = padded cell index r * 7 + c
N_PLANES = 14
SHAPE_NAMES = ("I", "O", "T", "S", "L")
GENERATOR_VERSION = "basil3-multiverse-v1"   # change it and every split changes; never do it mid-study
_GEN_SALT, _OPEN_SALT, _BAL_SALT = 9_137, 4_421, 7_771
SPLIT_PERCENT = {"train": 70, "val": 15, "test": 15}

_BASE = {
    "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "L": ((0, 0), (1, 0), (2, 0), (2, 1)),
}


def _orientations(cells):
    out, pts = set(), list(cells)
    for _ in range(4):
        pts = [(c, -r) for r, c in pts]                       # rotate 90 degrees
        for shape in (pts, [(r, -c) for r, c in pts]):        # and its mirror image
            mr, mc = min(r for r, _ in shape), min(c for _, c in shape)
            out.add(tuple(sorted((r - mr, c - mc) for r, c in shape)))
    return sorted(out)


ORIENTATIONS = {k: _orientations(v) for k, v in _BASE.items()}


def canonical_family(H, W, gravity, wrap_h, wrap_v, shapes):
    """Rules without blocked cells. A gravity-free board and its transpose are the same game,
    so both map to one key and always land in the same split."""
    key = (H, W, gravity, wrap_h, wrap_v, tuple(shapes))
    if not gravity:
        key = min(key, (W, H, False, wrap_v, wrap_h, tuple(shapes)))
    return key


def split_of_family(family) -> str:
    h = int.from_bytes(hashlib.sha256(f"{GENERATOR_VERSION}|{family}".encode()).digest()[:4], "big") % 100
    if h < SPLIT_PERCENT["train"]:
        return "train"
    return "val" if h < SPLIT_PERCENT["train"] + SPLIT_PERCENT["val"] else "test"


@dataclass(frozen=True)
class Universe:
    seed: int
    H: int
    W: int
    gravity: bool
    wrap_h: bool
    wrap_v: bool
    blocked: tuple          # flat indices r * W + c on the H x W board
    shapes: tuple

    def family(self):
        return canonical_family(self.H, self.W, self.gravity, self.wrap_h, self.wrap_v, self.shapes)

    def split(self) -> str:
        return split_of_family(self.family())

    def describe(self) -> str:
        bits = [f"{self.H}x{self.W}", "gravity" if self.gravity else "free placement"]
        if self.wrap_h or self.wrap_v:
            bits.append("portals " + "+".join(n for n, f in (("left-right", self.wrap_h), ("top-bottom", self.wrap_v)) if f))
        bits.append(f"{len(self.blocked)} blocked")
        bits.append("wins: " + ",".join(self.shapes))
        return f"universe {self.seed}: " + ", ".join(bits)


@lru_cache(maxsize=8192)
def compile_rules(u: Universe):
    """All placements of every winning shape, plus for each cell the placements that use it."""
    blocked, pats = set(u.blocked), set()
    for name in u.shapes:
        for ori in ORIENTATIONS[name]:
            for r in range(u.H):
                for c in range(u.W):
                    cells = []
                    for dr, dc in ori:
                        rr, cc = r + dr, c + dc
                        if u.wrap_v:
                            rr %= u.H
                        if u.wrap_h:
                            cc %= u.W
                        if not (0 <= rr < u.H and 0 <= cc < u.W):
                            break
                        cells.append(rr * u.W + cc)
                    else:
                        if len(set(cells)) == 4 and not blocked.intersection(cells):
                            pats.add(tuple(sorted(cells)))
    pats = np.array(sorted(pats), dtype=np.int64).reshape(-1, 4)
    by_cell = tuple(np.flatnonzero((pats == i).any(axis=1)) for i in range(u.H * u.W))
    return pats, by_cell


class Game:
    """Mutable state. board[i] for i = r * W + c: 0 empty, 1 / 2 stones, 3 blocked."""

    __slots__ = ("u", "board", "to_move", "winner", "done", "moves", "_pats", "_by_cell")

    def __init__(self, u: Universe):
        self.u = u
        self._pats, self._by_cell = compile_rules(u)
        self.board = np.zeros(u.H * u.W, dtype=np.int8)
        self.board[list(u.blocked)] = 3
        self.to_move, self.winner, self.done, self.moves = 1, 0, False, 0

    def copy(self) -> "Game":
        g = Game.__new__(Game)
        g.u, g._pats, g._by_cell = self.u, self._pats, self._by_cell
        g.board = self.board.copy()
        g.to_move, g.winner, g.done, g.moves = self.to_move, self.winner, self.done, self.moves
        return g

    def legal_cells(self):
        if self.done:
            return []
        b, H, W = self.board, self.u.H, self.u.W
        if not self.u.gravity:
            return np.flatnonzero(b == 0).tolist()
        out = []
        for c in range(W):                 # a piece falls until it hits a stone, a block or the floor
            land = -1
            for r in range(H):
                if b[r * W + c] != 0:
                    break
                land = r * W + c
            if land >= 0:
                out.append(land)
        return out

    def to_action(self, cell: int) -> int:
        return (cell // self.u.W) * PAD + cell % self.u.W

    def to_cell(self, action: int) -> int:
        r, c = divmod(int(action), PAD)
        return r * self.u.W + c if (r < self.u.H and c < self.u.W) else -1

    def legal_actions(self):
        return [self.to_action(i) for i in self.legal_cells()]

    def legal_mask(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[self.legal_actions()] = True
        return m

    def completes(self, cell: int, player: int) -> bool:
        """Would `player` win by occupying empty `cell`? (Does not change the board.)"""
        idx = self._by_cell[cell]
        if len(idx) == 0:
            return False
        pats = self._pats[idx]
        return bool(((self.board[pats] == player) | (pats == cell)).all(axis=1).any())

    def step(self, action: int) -> float:
        """Play for the player to move. Returns 1.0 if this move wins, else 0.0."""
        cell = self.to_cell(action)
        if self.done or cell < 0 or cell not in self.legal_cells():
            raise ValueError(f"illegal action {action} in {self.u.describe()}")
        p = self.to_move
        won = self.completes(cell, p)
        self.board[cell] = p
        self.moves += 1
        if won:
            self.done, self.winner = True, p
        elif not self.legal_cells():
            self.done = True                # board full: draw (winner stays 0)
        self.to_move = 3 - p
        return 1.0 if won else 0.0

    def encode(self) -> np.ndarray:
        """14 x 7 x 7 planes from the point of view of the player to move:
        0 my stones, 1 opponent stones, 2 empty, 3 blocked, 4 on-board, 5 gravity,
        6 left-right portals, 7 top-bottom portals, 8-12 winning shapes I,O,T,S,L, 13 legal moves."""
        u = self.u
        x = np.zeros((N_PLANES, PAD, PAD), dtype=np.float32)
        b = self.board.reshape(u.H, u.W)
        x[0, :u.H, :u.W] = b == self.to_move
        x[1, :u.H, :u.W] = b == 3 - self.to_move
        x[2, :u.H, :u.W] = b == 0
        x[3, :u.H, :u.W] = b == 3
        x[4, :u.H, :u.W] = 1.0
        x[5], x[6], x[7] = float(u.gravity), float(u.wrap_h), float(u.wrap_v)
        for j, name in enumerate(SHAPE_NAMES):
            x[8 + j] = float(name in u.shapes)
        x[13].flat[self.legal_actions()] = 1.0
        return x


# ----------------------------------------------------------------------------- generator
def greedy_move(g: Game, rng) -> int:
    """Win if possible, else block the opponent's immediate win, else random."""
    cells = g.legal_cells()
    for who in (g.to_move, 3 - g.to_move):
        for i in cells:
            if g.completes(i, who):
                return g.to_action(i)
    return g.to_action(cells[int(rng.integers(len(cells)))])


def _family_from_seed(seed: int):
    rng = np.random.default_rng([_GEN_SALT, seed])
    H, W = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    gravity = bool(rng.random() < 0.5)
    wrap_h = bool(rng.random() < 0.3)
    wrap_v = bool(rng.random() < 0.3) and not gravity
    k = int(rng.integers(1, 3))
    shapes = tuple(sorted(rng.choice(SHAPE_NAMES, size=k, replace=False).tolist(), key=SHAPE_NAMES.index))
    return H, W, gravity, wrap_h, wrap_v, shapes


def split_of_seed(seed: int) -> str:
    return split_of_family(canonical_family(*_family_from_seed(seed)))


def balance_stats(u: Universe, games: int = 30, seed: int = 0):
    rng = np.random.default_rng([_BAL_SALT, u.seed, seed])
    p1 = draws = 0
    lengths = []
    for _ in range(games):
        g = Game(u)
        while not g.done:
            g.step(greedy_move(g, rng))
        p1 += g.winner == 1
        draws += g.winner == 0
        lengths.append(g.moves)
    return p1 / games, draws / games, float(np.mean(lengths))


def is_playable(u: Universe) -> bool:
    """Reject degenerate universes: too few ways to win, or greedy self-play that is one-sided,
    mostly drawn or over almost immediately."""
    if len(compile_rules(u)[0]) < 8:
        return False
    p1, draw, length = balance_stats(u)
    return 0.25 <= p1 <= 0.80 and draw <= 0.6 and length >= 7


@lru_cache(maxsize=None)
def universe_from_seed(seed: int):
    """Deterministic. The seed fixes the family; only the blocked cells are re-drawn until the
    universe is playable. Returns None if no layout of this family is playable."""
    H, W, gravity, wrap_h, wrap_v, shapes = _family_from_seed(seed)
    for attempt in range(1, 11):
        rng = np.random.default_rng([_GEN_SALT, seed, attempt])
        n_block = int(rng.integers(0, 4))
        blocked = tuple(sorted(int(i) for i in rng.choice(H * W, size=n_block, replace=False)))
        u = Universe(seed, H, W, gravity, wrap_h, wrap_v, blocked, shapes)
        if is_playable(u):
            return u
    return None


def universes(split: str, n: int, start: int = 0):
    """The first n playable universes of a split, scanning seeds from `start`. Deterministic."""
    if split not in SPLIT_PERCENT:
        raise ValueError(split)
    out, seed = [], start
    while len(out) < n:
        if split_of_seed(seed) == split:
            u = universe_from_seed(seed)
            if u is not None:
                out.append(u)
        seed += 1
        if seed - start > 200 * n + 1000:
            raise RuntimeError("generator exhausted; check SPLIT_PERCENT")
    return out


def opening(u: Universe, index: int, plies: int = 2) -> Game:
    """Evaluation opening `index`: `plies` uniformly random moves from a seed that depends only on
    (universe, index). Every agent sees the same openings, from both seats."""
    for attempt in range(1000):
        rng = np.random.default_rng([_OPEN_SALT, u.seed, index, attempt])
        g = Game(u)
        for _ in range(plies):
            acts = g.legal_actions()
            g.step(acts[int(rng.integers(len(acts)))])
            if g.done:
                break
        if not g.done:
            return g
    raise RuntimeError(f"no opening for {u.describe()}")
