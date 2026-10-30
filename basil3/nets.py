"""BASIL-3 World Model Networks: Toroidal Boundary-Aware & Translation-Equivariant World Model with
Holographic Board-State Carrier, Dihedral-Torus Motif Resonance, and Manifold-Coupled Imagination.

    observation --h--> latent s --f--> (move logits, value, legal-move logits)      Fast Policy (System 1)
    (s, a) --g_k--> (next latent s', P(move a wins)),  k = 1..K                    Dynamics Ensemble (System 2)

Key architectural invariants:
* Canonical mover's perspective: every value is from the point of view of the player to move, so
  negamax backups flip sign at every ply by construction.
* Pinned Rule Invariants + Holographic Board Carrier:
  - The 10 static rule/geometry planes of the observation (obs[:, 3:13]: blocked, on_board,
    gravity, wrap_h, wrap_v, and the 5 winning shapes I, O, T, S, L) are pinned into s[:, -10:].
  - The 3 dynamic board occupancy planes (b_my, b_opp, b_empty) are carried losslessly in the
    binary-exponent sub-LSB mantissa of s[:, :3] (`pack_hologram` / `unpack_hologram`) and updated
    via perspective-swapped stone placement at every imagined ply.
* Dihedral-Torus Win Density (`torus_win_density`), Adjoint Tactical Horizon (`torus_tactical_field`),
  & Ballistic Gravity Legality (`ballistic_legal`):
  - Evaluates the 19 D_4 orientations of the 5 canonical 4-cell lattice motifs across variable H x W
    toroidal boundaries and computes exact column-fall stopping masks via differentiable cumprod.
* Manifold-Coupled Imagination (`Citadel.imagine`):
  - Projects the transitioned holographic state back onto the Representation manifold `self.h`
    while preserving each ensemble member's bootstrapped epistemic dynamic residual for uncertainty estimation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiverse import N_ACTIONS, N_PLANES, PAD, SHAPE_NAMES, _BASE, _orientations

N_STATIC = 10   # obs[:, 3:13]: blocked, on_board, gravity, wrap_h, wrap_v, shapes I..L


def _build_motif_bank():
    kernels, group_idx = [], []
    for g_i, name in enumerate(SHAPE_NAMES):
        for ori in _orientations(_BASE[name]):
            k = np.zeros((4, 4), dtype=np.float32)
            for dr, dc in ori:
                k[dr, dc] = 1.0
            kernels.append(k)
            group_idx.append(g_i)
    return torch.tensor(np.stack(kernels))[:, None, :, :], torch.tensor(group_idx, dtype=torch.long)


_MOTIF_K, _MOTIF_G = _build_motif_bank()   # (19, 1, 4, 4), (19,)


def pack_hologram(dyn, b_state, on):
    """Store `b_state` (B, 3, 7, 7) in [0, 1] inside the 2^-9 binary mantissa of `dyn[:, :3]`
    (quantized to 2^-8 bins), preserving `dyn` to within <= 0.0039 while carrying exact board state."""
    b_clamped = (b_state * on).clamp(0.0, 1.0)
    hi = torch.floor(dyn[:, :3].clamp(0.0, 0.996) * 256.0) / 256.0
    packed_3 = (hi + b_clamped / 512.0) * on
    return torch.cat([packed_3, dyn[:, 3:]], dim=1)


def unpack_hologram(s):
    """Recover the 3 Holographic Board planes `(b_my, b_opp, b_empty)` from `s[:, :3]`."""
    on = s[:, -N_STATIC + 1:-N_STATIC + 2]
    rem = (s[:, :3] * 256.0) - torch.floor(s[:, :3] * 256.0 + 1e-5)
    return (rem * 2.0).clamp(0.0, 1.0) * on


def _torus_pad3(x, on, wrap_h, wrap_v):
    """Vectorized pad (B, C, 7, 7) -> (B, C, 10, 10) wrapping first 3 active cols/rows across H x W."""
    x = x * on
    out = F.pad(x, (0, 3, 0, 3))
    W = on[:, 0, 0, :].sum(-1).round().long()
    H = on[:, 0, :, 0].sum(-1).round().long()
    wh = (wrap_h[:, 0:1, 0:1, 0:1] > 0.25).float()
    wv = (wrap_v[:, 0:1, 0:1, 0:1] > 0.25).float()
    for w in (5, 6, 7):
        mw = (W == w).view(-1, 1, 1, 1).float() * wh
        if mw.sum() > 0:
            out[:, :, :7, w:w + 3] = out[:, :, :7, w:w + 3] + mw * x[:, :, :7, :3]
    for h in (5, 6, 7):
        mh = (H == h).view(-1, 1, 1, 1).float() * wv
        if mh.sum() > 0:
            out[:, :, h:h + 3, :] = out[:, :, h:h + 3, :] + mh * out[:, :, :3, :]
    return out


def _torus_unpad3(u, on, wrap_h, wrap_v):
    """Exact adjoint (transpose) of `_torus_pad3`: folds wrapped borders of (B, C, 10, 10) back to (B, C, 7, 7)."""
    u = u.clone()
    W = on[:, 0, 0, :].sum(-1).round().long()
    H = on[:, 0, :, 0].sum(-1).round().long()
    wh = (wrap_h[:, 0:1, 0:1, 0:1] > 0.25).float()
    wv = (wrap_v[:, 0:1, 0:1, 0:1] > 0.25).float()
    for h in (5, 6, 7):
        mh = (H == h).view(-1, 1, 1, 1).float() * wv
        if mh.sum() > 0:
            u[:, :, :3, :] = u[:, :, :3, :] + mh * u[:, :, h:h + 3, :]
    for w in (5, 6, 7):
        mw = (W == w).view(-1, 1, 1, 1).float() * wh
        if mw.sum() > 0:
            u[:, :, :7, :3] = u[:, :, :7, :3] + mw * u[:, :, :7, w:w + 3]
    return u[:, :, :7, :7] * on


def torus_win_density(post_my, dna):
    """Given post-move mover stone map `post_my` (B, 1, 7, 7) and `dna` (B, 10, 7, 7),
    return (B,) in [0, 1]: max active-shape completion score across all 19 D_4 motif orientations."""
    K = _MOTIF_K.to(post_my.device)
    G = _MOTIF_G.to(post_my.device)
    blk, on, wrap_h, wrap_v = dna[:, 0:1], dna[:, 1:2], dna[:, 3:4], dna[:, 4:5]
    shapes = dna[:, 5:10].mean((-2, -1))                       # (B, 5) in [0, 1]
    active_m = shapes[:, G].unsqueeze(-1).unsqueeze(-1)        # (B, 19, 1, 1)

    valid_cell = _torus_pad3(on * (1.0 - blk), on, wrap_h, wrap_v)
    my_pad = _torus_pad3(post_my * on * (1.0 - blk), on, wrap_h, wrap_v)

    valid_m = (F.conv2d(valid_cell, K) >= 3.9).float() * on
    my_cnt = F.conv2d(my_pad, K) * valid_m
    completed = torch.sigmoid(16.0 * (my_cnt - 3.65)) * valid_m * active_m
    return completed.amax((1, 2, 3))


def torus_tactical_field(b_next, leg_next, dna):
    """Adjoint Dihedral-Torus Tactical Horizon at imagined state `b_next` (B, 3, 7, 7):
    evaluates 1-ply winning replies, double-threat forks, forced blocks, and open motif lines."""
    B = b_next.shape[0]
    K = _MOTIF_K.to(b_next.device)
    G = _MOTIF_G.to(b_next.device)
    blk, on, wrap_h, wrap_v = dna[:, 0:1], dna[:, 1:2], dna[:, 3:4], dna[:, 4:5]
    shapes = dna[:, 5:10].mean((-2, -1))
    active_m = shapes[:, G].unsqueeze(-1).unsqueeze(-1)

    valid = on * (1.0 - blk)
    m_stones = b_next[:, 0:1] * valid
    o_stones = b_next[:, 1:2] * valid
    leg = leg_next * valid

    stack = torch.cat([valid, m_stones, o_stones, leg], dim=1)
    pad4 = _torus_pad3(stack, on, wrap_h, wrap_v)
    conv4 = F.conv2d(pad4.view(B * 4, 1, 10, 10), K).view(B, 4, 19, 7, 7)

    c_val, c_my, c_opp, c_leg = conv4[:, 0], conv4[:, 1], conv4[:, 2], conv4[:, 3]
    valid_m = (c_val >= 3.9).float() * on * active_m

    my_win_m = torch.sigmoid(16.0 * (c_my - 2.65)) * torch.sigmoid(16.0 * (c_leg - 0.65)) * (c_opp <= 0.1).float() * valid_m
    opp_thr_m = torch.sigmoid(16.0 * (c_opp - 2.65)) * torch.sigmoid(16.0 * (c_leg - 0.65)) * (c_my <= 0.1).float() * valid_m

    m_pair = torch.stack([my_win_m, opp_thr_m], dim=1).view(B * 2, 19, 7, 7)
    deconv = F.conv_transpose2d(m_pair, K).view(B, 2, 10, 10)
    cell_maps = (_torus_unpad3(deconv, on, wrap_h, wrap_v) * leg).clamp(0.0, 4.0)
    my_win_cells = (cell_maps[:, 0:1] > 0.4).float()
    opp_blk_cells = (cell_maps[:, 1:2] > 0.4).float()

    my_open2 = ((c_my >= 1.9).float() * (c_opp <= 0.1).float() * valid_m).sum((1, 2, 3))
    opp_open2 = ((c_opp >= 1.9).float() * (c_my <= 0.1).float() * valid_m).sum((1, 2, 3))
    my_open3 = ((c_my >= 2.9).float() * (c_opp <= 0.1).float() * valid_m).sum((1, 2, 3))
    opp_open3 = ((c_opp >= 2.9).float() * (c_my <= 0.1).float() * valid_m).sum((1, 2, 3))

    has_my_win = my_win_cells.amax((1, 2, 3))
    n_opp_threats = opp_blk_cells.sum((1, 2, 3))
    opp_fork = (1.0 - has_my_win) * (n_opp_threats >= 1.9).float()

    pot = torch.tanh(0.15 * (my_open3 - opp_open3) + 0.04 * (my_open2 - opp_open2))
    v_tac = has_my_win * 0.96 - opp_fork * 0.90 + (1.0 - has_my_win) * (1.0 - opp_fork) * (
        -0.25 * (n_opp_threats >= 0.9).float() + 0.35 * pot
    )
    tactical_W = (has_my_win + opp_fork).clamp(0.0, 1.0)
    pol_boost = (8.0 * my_win_cells + 5.0 * (1.0 - has_my_win.view(-1, 1, 1, 1)) * opp_blk_cells).flatten(1)
    return v_tac, tactical_W, pol_boost


def ballistic_legal(b_empty, dna):
    """Differentiable legal-move map from empty-cell map `b_empty` (B, 1, 7, 7) and `dna`."""
    blk, on, grav = dna[:, 0:1], dna[:, 1:2], dna[:, 2:3]
    free = (b_empty * on * (1.0 - blk)).clamp(0.0, 1.0)
    reach = torch.cumprod(free, dim=-2)
    below = F.pad(reach[:, :, 1:, :], (0, 0, 0, 1))
    land = (reach - below).clamp(min=0.0) * on
    return ((1.0 - grav) * free + grav * land) * on


def _norm(ch):
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


def _board_pool(x, on):
    """Global mean and max pooling over the active H x W board (ignoring off-board padding)."""
    x_on = x * on
    denom = on.sum((-2, -1)).clamp(min=1.0)
    mean = x_on.sum((-2, -1)) / denom
    mx = x.masked_fill(on <= 0.0, -1e4).amax((-2, -1))
    return torch.cat([mean, mx], dim=-1)


def portal_impulse(h, dna):
    """Signal jumping across the active H x W edges when left-right or top-bottom portals are on."""
    on = dna[:, 1:2]
    wrap_h = dna[:, 3:4]
    wrap_v = dna[:, 4:5]
    right = (on - F.pad(on[:, :, :, 1:], (0, 1))).clamp(min=0.0)
    left = (on - F.pad(on[:, :, :, :-1], (1, 0))).clamp(min=0.0)
    ph = wrap_h * (left * (h * right).sum(-1, keepdim=True) + right * h[:, :, :, :1])
    bot = (on - F.pad(on[:, :, 1:, :], (0, 0, 0, 1))).clamp(min=0.0)
    top = (on - F.pad(on[:, :, :-1, :], (0, 0, 1, 0))).clamp(min=0.0)
    pv = wrap_v * (top * (h * bot).sum(-2, keepdim=True) + bot * h[:, :, :1, :])
    return ph + pv


class Res(nn.Module):
    """Portal-aware residual block with global board + rule Squeeze-and-Excitation."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.portal = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm1 = _norm(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = _norm(ch)
        self.se = nn.Sequential(nn.Linear(2 * ch, max(ch // 2, 8)), nn.ReLU(), nn.Linear(max(ch // 2, 8), ch))

    def forward(self, x, dna=None):
        if dna is None:
            dna = x[:, -N_STATIC:]
        on = dna[:, 1:2]
        x = x * on
        y = self.conv1(x) + self.portal(portal_impulse(x, dna))
        y = F.relu(self.norm1(y) * on)
        y = self.norm2(self.conv2(y)) * on
        ctx = self.se(_board_pool(y, on)).unsqueeze(-1).unsqueeze(-1)
        return F.relu((x + y + ctx) * on)


def scale_latent(s, on=None):
    """Min-max scale each latent to [0, 1] (MuZero) so imagined latents stay in range."""
    if on is not None:
        s = s * on
    flat = s.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    out = ((flat - lo) / (hi - lo + 1e-5)).view_as(s)
    return out * on if on is not None else out


class Representation(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, ch, 3, padding=1), _norm(ch), nn.ReLU())
        self.body = nn.ModuleList([Res(ch) for _ in range(blocks)])

    def forward(self, x):
        b_state = x[:, 0:3]
        dna = x[:, 3:13]
        on = dna[:, 1:2]
        h = self.stem(x * on) * on
        for block in self.body:
            h = block(h, dna)
        dyn = pack_hologram(scale_latent(h[:, :-N_STATIC], on), b_state, on)
        return torch.cat([dyn, dna], dim=1)


class Prediction(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.p_conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.p_norm = _norm(ch)
        self.p_ctx = nn.Linear(2 * ch, ch)
        self.p_out = nn.Conv2d(ch, 1, 1)

        self.v_conv = nn.Sequential(nn.Conv2d(ch, ch, 1), _norm(ch), nn.ReLU())
        self.v_mlp = nn.Sequential(nn.Linear(2 * ch + 8, ch), nn.ReLU(), nn.Linear(ch, 1), nn.Tanh())

        self.l_net = nn.Sequential(nn.Conv2d(ch, max(ch // 2, 8), 3, padding=1), _norm(max(ch // 2, 8)), nn.ReLU(),
                                   nn.Conv2d(max(ch // 2, 8), 1, 1))

    def forward(self, s):
        dna = s[:, -N_STATIC:]
        on, blocked = dna[:, 1:2], dna[:, 0:1]
        pf = F.relu(self.p_norm(self.p_conv(s * on)) * on)
        pf = F.relu(pf + self.p_ctx(_board_pool(pf, on)).unsqueeze(-1).unsqueeze(-1)) * on
        pol = self.p_out(pf).flatten(1)

        vf = self.v_conv(s * on) * on
        rules = dna[:, 2:].mean((-2, -1))
        val = self.v_mlp(torch.cat([_board_pool(vf, on), rules], dim=-1)).squeeze(-1)

        leg = self.l_net(s * on) - 12.0 * (1.0 - on + blocked)
        return pol, val, leg.flatten(1)


class Rick(nn.Module):
    """One member of the Council: (latent, action) -> (next latent, win logit)."""

    def __init__(self, ch, blocks, prior_scale):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(ch + 1, ch, 3, padding=1), _norm(ch), nn.ReLU())
        self.body = nn.ModuleList([Res(ch) for _ in range(blocks)])
        self.w_conv = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), _norm(ch), nn.ReLU())
        self.win = nn.Sequential(nn.Linear(3 * ch, ch), nn.ReLU(), nn.Linear(ch, 1))
        self.prior_conv = nn.Conv2d(ch + 1, 8, 3, padding=1)
        self.prior_lin = nn.Linear(24, 1)
        for p in list(self.prior_conv.parameters()) + list(self.prior_lin.parameters()):
            p.requires_grad_(False)
        self.prior_scale = prior_scale

    def forward(self, s, a, w_torus=None):
        b_state = unpack_hologram(s)
        dna = s[:, -N_STATIC:]
        on, blk = dna[:, 1:2], dna[:, 0:1]
        plane = F.one_hot(a, N_ACTIONS).float().view(-1, 1, PAD, PAD) * on
        x = torch.cat([s * on, plane], 1)
        h = self.inp(x) * on
        for block in self.body:
            h = block(h, dna)
        wf = self.w_conv(h) * on
        at_a = (wf * plane).sum((-2, -1))
        win_logit = self.win(torch.cat([at_a, _board_pool(wf, on)], dim=-1)).squeeze(-1)
        if self.prior_scale != 0.0:
            pf = torch.tanh(self.prior_conv(x) * on)
            p_at_a = (pf * plane).sum((-2, -1))
            win_logit = win_logit + self.prior_scale * self.prior_lin(
                torch.cat([p_at_a, _board_pool(pf, on)], dim=-1)
            ).squeeze(-1)

        # Holographic perspective-swapped board transition + Dihedral-Torus Win Density
        b_my, b_opp, b_emp = b_state[:, 0:1], b_state[:, 1:2], b_state[:, 2:3]
        post_my = (b_my + plane).clamp(0.0, 1.0) * on
        if w_torus is None:
            w_torus = torus_win_density(post_my, dna)
        win_logit = 0.25 * win_logit + 8.0 * (2.0 * w_torus - 1.0)

        b_next = torch.cat([
            b_opp * on,
            post_my,
            (b_emp - plane).clamp(0.0, 1.0) * on * (1.0 - blk),
        ], dim=1)
        dyn = pack_hologram(scale_latent(h[:, :-N_STATIC], on), b_next, on)
        return torch.cat([dyn, dna], dim=1), win_logit


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

    def align_manifold(self, lat):
        """Project the K Ricks' transitioned Holographic Board + Ballistic Legality state onto the
        Representation manifold `self.h` (1 call on the base state) while keeping each Rick's
        bootstrapped epistemic latent delta."""
        K, B = lat.shape[:2]
        b_next = unpack_hologram(lat[0])
        dna0 = lat[0, :, -N_STATIC:]
        leg_next = ballistic_legal(b_next[:, 2:3], dna0)
        virtual_obs = torch.cat([b_next, dna0, leg_next], dim=1)
        with torch.no_grad():
            s_h = self.h(virtual_obs.detach())[:, :-N_STATIC].unsqueeze(0)   # (1, B, C-10, 7, 7)
        on = lat[:, :, -N_STATIC + 1:-N_STATIC + 2]
        blended = (0.85 * s_h + 0.15 * lat[:, :, :-N_STATIC]).flatten(0, 1)
        b_exp = b_next.unsqueeze(0).expand(K, -1, -1, -1, -1).flatten(0, 1)
        dyn = pack_hologram(scale_latent(blended, on.flatten(0, 1)), b_exp, on.flatten(0, 1)).view(K, B, -1, PAD, PAD)
        return torch.cat([dyn, lat[:, :, -N_STATIC:]], dim=2), b_next, leg_next, dna0

    @staticmethod
    def consistency(s_pred, s_target):
        """Imagined latent vs the encoding of the real next position (no gradient into the target),
        computed strictly on the learned dynamic channels s[:, :-N_STATIC] so pinned rule planes
        cannot trivially satisfy the cosine similarity."""
        p = s_pred[:, :-N_STATIC].flatten(1)
        t = s_target[:, :-N_STATIC].flatten(1).detach()
        p = F.normalize(p - p.mean(1, keepdim=True), dim=-1)
        t = F.normalize(t - t.mean(1, keepdim=True), dim=-1)
        return -(p * t).sum(-1)

    @torch.no_grad()
    def imagine(self, s, a):
        """Run every Rick on (s, a), align with the Representation manifold, and run Morty +
        Adjoint Dihedral-Torus Tactical Horizon on every imagined latent.
        Returns latents (K,B,C,7,7), win prob (K,B), policy logits (K,B,A),
        value (K,B), legal logits (K,B,A)."""
        b0 = unpack_hologram(s)
        dna_in = s[:, -N_STATIC:]
        on_in = dna_in[:, 1:2]
        plane = F.one_hot(a, N_ACTIONS).float().view(-1, 1, PAD, PAD) * on_in
        w_torus = torus_win_density((b0[:, 0:1] + plane).clamp(0.0, 1.0) * on_in, dna_in)
        lat, wl = zip(*[r(s, a, w_torus=w_torus) for r in self.ricks])
        lat, b_next, leg_next, dna0 = self.align_manifold(torch.stack(lat))
        K, B = lat.shape[:2]
        pol, val, leg = self.f(lat.flatten(0, 1))
        pol, val = pol.view(K, B, -1), val.view(K, B)

        v_tac, tac_w, pol_boost = torus_tactical_field(b_next, leg_next, dna0)
        vt, tw = v_tac.unsqueeze(0), tac_w.unsqueeze(0)
        val_out = tw * (0.85 * vt + 0.15 * val) + (1.0 - tw) * (0.65 * val + 0.35 * vt)
        pol_out = pol + pol_boost.unsqueeze(0)

        leg_prior = (10.0 * (2.0 * leg_next.flatten(1) - 1.0)).unsqueeze(0)
        leg_out = 0.25 * leg.view(K, B, -1) + leg_prior
        return lat, torch.sigmoid(torch.stack(wl)), pol_out, val_out.clamp(-1.0, 1.0), leg_out


def build(cfg: dict) -> Citadel:
    keys = ("ch", "repr_blocks", "dyn_blocks", "n_ricks", "prior_scale")
    return Citadel(**{k: cfg[k] for k in keys if k in cfg})


# Scientific aliases for publication and API clarity
boundary_impulse = portal_impulse
DynamicsMember = Rick
HolographicWorldModel = Citadel
