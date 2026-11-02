"""#!/usr/bin/env python3"""
"""
CPA-VO-SP2-SA (corrected): Prioritized multi-robot trajectory planning with
Forward-CPA / COLREGs stern-passing bypass ordering, VO-style trajectory scoring,
and space-time simulated annealing.

What this code IS (state it this way in the paper):
  * A CENTRALIZED, PRIORITIZED, OFFLINE planner. Robot i plans against the full
    planned trajectories of robots 1..i-1 (no sensor model, no reactive control).
  * Forward CPA prediction (closest point of approach, as in maritime COLREGs
    practice) is used to ORDER lateral bypass candidates (side + magnitude).
    Dragonfly interception (Mischiati et al., Nature 2015) is inspiration only.
  * Velocity-obstacle (VO) cone penalty and clearance/heading scoring are
    borrowed in spirit from Zhong et al. (2021) but this is NOT a dynamic window
    approach: whole trajectories are scored offline, no velocity sampling per step.
  * Smoothing: quartic corner blend with zero curvature at its joints
    (curvature-continuous). It is NOT minimum-snap.

Kinematics (enforced in Traj, verified by Traj.kin and --selftest):
  * Start and end at rest.
  * |v| in [v_min_dyn, v_max_dyn]
  * |dv/dt| <= a_v_max  (speed profile uses a backward + forward pass, so the
    robot brakes BEFORE a corner)
  * |omega| = |kappa| * v <= w_max  (v <= w_max / kappa along the path)
  * |d omega/dt| <= a_w_max  (enforced iteratively: violating zones are slowed
    and re-simulated; the result is verified, not proven)
  * Curvature is estimated on the 0.05-cell resampled polyline (running-max
    filter for the speed limit, moving average for the omega/alpha check).
  * The terminal stop step is excluded from the acceleration check.

Usage:
    python3 sp2_sa_V5.py --selftest
    python3 sp2_sa_V5.py --scenario static --num-seeds 10
    python3 sp2_sa_V5.py --scenario static --ablation --num-seeds 10
    python3 sp2_sa_V5.py --scenario static --paper-homotopy
    python3 sp2_sa_V5.py --scenario static --v-max-dyn 1.08   # allow boost
"""
import argparse
import heapq
import math
import os
import time
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt, maximum_filter1d, uniform_filter1d


# =============================================================================
# 1. CONFIGURATION, SCENARIOS & ENVIRONMENT
# =============================================================================
@dataclass
class Cfg:
    V: float = 1.0            # nominal cruise speed [cells/s] (= 10 paper units/s)
    v_max_dyn: float = 1.0    # max speed. Default == V so no method gets a free speed advantage
    v_min_dyn: float = 0.0
    w_max: float = 2.5        # max angular velocity [rad/s]
    a_v_max: float = 1.5      # max linear acceleration [cells/s^2]
    a_w_max: float = 1.5      # max angular acceleration [rad/s^2]
    dt: float = 0.1
    tmax: float = 250.0
    d_safe: float = 2.0       # reporting safety distance [cells] (= 20 paper units)
    plan_margin: float = 0.05 # planner uses d_safe + plan_margin to avoid knife-edge solutions
    margin: float = 0.3       # obstacle inflation [cells]
    sp_buf: float = 0.08
    d_c: float = 0.8
    lam_turn: float = 0.05
    lam_clr: float = 0.2
    lam_wp: float = 0.01
    k_iter: int = 50
    alpha: float = 0.9
    r0: float = 2.0
    p_acc0: float = 0.8
    stall_max: int = 3
    round_r: float = 0.95
    tau: float = 3.0
    st_sa_steps: int = 30
    coop_tol: float = 1.2
    # CPA / COLREGs parameters
    use_cpa: bool = True      # False = ablation (no CPA-based ordering of bypass candidates)
    t_pred_df: float = 6.5
    # VO-style scoring parameters
    preview_dist: float = 3.5
    w_head: float = 6.0
    w_dist: float = 6.0
    w_vel: float = 8.0
    w_vo: float = -100.0
    w_df: float = 12.0
    hold_goal: bool = True
    max_aw_iters: int = 6     # iterations for angular-acceleration slow-down repair
    kin_tol: float = 1.02     # tolerance when verifying kinematic limits


NAMES = ['a', 'b', 'c', 'd', 'e']
COLORS = ['#1f3fbf', '#e0262d', '#2ca02c', '#b03aa8', '#e6c200']

FREE_ROBOTS = {
    'a': ((0.5, 0.5), (35.5, 38.5)),
    'b': ((0.5, 25.5), (39.5, 16.5)),
    'c': ((13.5, 37.5), (27.5, 1.5)),
    'd': ((36.5, 31.5), (2.5, 11.5)),
    'e': ((35.5, 8.5), (3.5, 38.5)),
}

STATIC_ROBOTS = {
    'a': ((1.5, 38.5), (27.5, 8.5)),
    'b': ((2.5, 4.5), (36.5, 34.5)),
    'c': ((18.5, 0.5), (18.5, 38.5)),
    'd': ((2.5, 22.5), (31.5, 10.5)),
    'e': ((36.5, 15.5), (2.5, 14.5)),
}

STATIC_RECTS = [
    (6, 33, 9, 38), (13, 27, 18, 31), (17, 31, 20, 34), (21, 24, 24, 27),
    (36, 36, 39, 39), (28, 31, 32, 35), (14, 18, 18, 22), (29, 13, 33, 17),
    (8, 9, 12, 16), (2, 0, 5, 3), (17, 5, 28, 7), (33, 2, 36, 11),
]

SCENARIOS = {'free': ([], FREE_ROBOTS), 'static': (STATIC_RECTS, STATIC_ROBOTS)}

STATIC_HOMOTOPY_GUIDES = {
    'a': [(1.5, 38.5), (5.2, 32.2), (12.5, 25.5), (19.2, 22.6), (23.2, 13.8), (27.5, 8.5)],
    'b': [(2.5, 4.5), (13.2, 8.0), (20.2, 16.2), (26.5, 22.0), (34.2, 28.5), (36.5, 34.5)],
    'c': [(18.5, 0.5), (16.0, 4.5), (16.0, 7.8), (19.5, 17.5), (25.0, 23.2),
          (25.0, 28.0), (21.2, 34.8), (18.5, 38.5)],
    'd': [(2.5, 22.5), (18.8, 22.8), (23.5, 20.5), (27.8, 12.2), (31.5, 10.5)],
    'e': [(36.5, 15.5), (33.8, 17.8), (28.2, 17.8), (24.5, 17.3), (18.5, 15.2),
          (12.8, 16.8), (7.2, 16.8), (2.5, 14.5)],
}


def dsp(cfg):
    """Distance used for PLANNING (slightly larger than the reported d_safe)."""
    return cfg.d_safe + cfg.plan_margin


class Env:
    def __init__(self, rects, cfg, size=40, res=0.1):
        self.size, self.res, self.n, self.rects = size, res, int(round(size / res)), rects
        n = self.n
        raw = np.zeros((n, n), bool)
        for x0, y0, x1, y1 in rects:
            raw[int(round(y0 / res)):int(round(y1 / res)),
                int(round(x0 / res)):int(round(x1 / res))] = True
        self.raw = raw
        self.dist = distance_transform_edt(~raw) * res if raw.any() else np.full((n, n), 99.0)
        self.mask = self.dist < cfg.margin
        self.mask_sp = self.dist < (cfg.margin + cfg.sp_buf)

    def _idx(self, P):
        P = np.atleast_2d(P)
        ix = np.floor(P[:, 0] / self.res).astype(int)
        iy = np.floor(P[:, 1] / self.res).astype(int)
        out = (ix < 0) | (ix >= self.n) | (iy < 0) | (iy >= self.n)
        return np.clip(ix, 0, self.n - 1), np.clip(iy, 0, self.n - 1), out

    def blocked(self, P, mask=None):
        m = self.mask if mask is None else mask
        ix, iy, out = self._idx(P)
        return out | m[iy, ix]

    def los(self, a, b, mask=None):
        m = self.mask if mask is None else mask
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dist = math.hypot(bx - ax, by - ay)
        n = max(2, int(dist / 0.08) + 1)
        t = np.linspace(0.0, 1.0, n)
        ix = np.floor((ax + (bx - ax) * t) / self.res).astype(int)
        iy = np.floor((ay + (by - ay) * t) / self.res).astype(int)
        if (ix < 0).any() or (ix >= self.n).any() or (iy < 0).any() or (iy >= self.n).any():
            return False
        return not bool(m[iy, ix].any())

    def clearance(self, P):
        ix, iy, _ = self._idx(P)
        return self.dist[iy, ix]


def densify(poly, ds=0.05):
    if poly is None or len(poly) == 0:
        return np.zeros((0, 2))
    if len(poly) == 1:
        return poly.copy()
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    poly, s = poly[keep], s[keep]
    if len(poly) < 2 or s[-1] < 1e-9:
        return poly
    q = np.unique(np.append(np.arange(0.0, s[-1], ds), s[-1]))
    return np.column_stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])])


def total_turn_deg(P):
    d = np.diff(P, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    d = d[seg > 1e-6]
    if len(d) < 2:
        return 0.0
    n = np.hypot(d[:, 0], d[:, 1])
    c = np.clip((d[:-1] * d[1:]).sum(1) / (n[:-1] * n[1:] + 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)).sum())


# =============================================================================
# 2. THETA* & WAYPOINT SIMULATED ANNEALING
# =============================================================================
def theta_star(env, start, goal, mask=None):
    mask = env.mask if mask is None else mask
    N = env.size
    cen = lambda n: np.array([n[0] + 0.5, n[1] + 0.5])
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
    B = env.blocked(np.column_stack([cx.ravel(), cy.ravel()]), mask).reshape(N, N)
    s = (min(N - 1, max(0, int(start[0]))), min(N - 1, max(0, int(start[1]))))
    g = (min(N - 1, max(0, int(goal[0]))), min(N - 1, max(0, int(goal[1]))))
    B[s] = False
    B[g] = False
    h = lambda n: math.hypot(n[0] - g[0], n[1] - g[1])
    gc = {s: 0.0}
    par = {s: s}
    closed = set()
    heap = [(h(s), s)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == g:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < N and 0 <= nb[1] < N) or B[nb] or nb in closed:
                    continue
                if not env.los(cen(cur), cen(nb), mask):
                    continue
                p = par[cur]
                use = p if (p != cur and env.los(cen(p), cen(nb), mask)) else cur
                cand = gc[use] + math.hypot(use[0] - nb[0], use[1] - nb[1])
                if cand < gc.get(nb, 1e18):
                    gc[nb] = cand
                    par[nb] = use
                    heapq.heappush(heap, (cand + h(nb), nb))
    if g not in closed:
        return None
    chain = [g]
    while chain[-1] != s:
        chain.append(par[chain[-1]])
    chain = chain[::-1]
    pts = [np.array(start, float)] + [cen(n) for n in chain[1:-1]] + [np.array(goal, float)]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not env.los(pts[i], pts[j], mask):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out, float)


def chain_is_valid(env, X, mask=None):
    if X is None or len(X) < 2:
        return False
    m = env.mask if mask is None else mask
    if env.blocked(X, m).any():
        return False
    for i in range(len(X) - 1):
        if not env.los(X[i], X[i + 1], m):
            return False
    return True


def validate_or_repair_chain(env, X_seed, start, goal, mask=None):
    m = env.mask if mask is None else mask
    if X_seed is None or len(X_seed) < 2:
        return theta_star(env, start, goal, m)
    if chain_is_valid(env, X_seed, m):
        return X_seed.copy()
    pts = [np.asarray(start, float)]
    for k in range(1, len(X_seed) - 1):
        if not env.blocked(X_seed[k], m)[0]:
            pts.append(np.asarray(X_seed[k], float))
    pts.append(np.asarray(goal, float))
    repaired = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = repaired[-1], pts[i + 1]
        if env.los(a, b, m):
            repaired.append(b)
        else:
            sub = theta_star(env, a, b, m)
            if sub is None:
                return theta_star(env, start, goal, m)
            repaired.extend(sub[1:])
    out = np.array(repaired, float)
    return out if chain_is_valid(env, out, m) else theta_star(env, start, goal, m)


def energy(env, X, cfg):
    d = np.diff(X, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    E = float(n.sum()) + cfg.lam_wp * len(X)
    if len(X) > 2:
        c = np.clip((d[:-1] * d[1:]).sum(1) / (n[:-1] * n[1:] + 1e-12), -1.0, 1.0)
        E += cfg.lam_turn * float((np.arccos(c) ** 2).sum())
        E += cfg.lam_clr * float((np.maximum(0.0, cfg.d_c - env.clearance(X[1:-1])) ** 2).sum())
    return E


def propose(env, X, r, mask, rng):
    m, u = len(X), rng.random()
    if u < 0.6 and m > 2:
        j = int(rng.integers(1, m - 1))
        P = X[j] + rng.uniform(-r, r, 2)
        if env.blocked(P, mask)[0] or not (env.los(X[j - 1], P, mask) and env.los(P, X[j + 1], mask)):
            return None
        Xn = X.copy()
        Xn[j] = P
        return Xn
    if u < 0.8:
        j = int(rng.integers(0, m - 1))
        P = 0.5 * (X[j] + X[j + 1]) + rng.uniform(-r, r, 2)
        if env.blocked(P, mask)[0] or not (env.los(X[j], P, mask) and env.los(P, X[j + 1], mask)):
            return None
        return np.insert(X, j + 1, P, axis=0)
    if m > 2:
        j = int(rng.integers(1, m - 1))
        if env.los(X[j - 1], X[j + 1], mask):
            return np.delete(X, j, axis=0)
    return None


def _raw_sp2_sa(env, start, goal, cfg, rng, mask=None, X_seed=None):
    mask = env.mask if mask is None else mask
    X = validate_or_repair_chain(env, X_seed, start, goal, mask)
    if X is None or len(X) <= 2:
        return X
    E = energy(env, X, cfg)
    bestX, bestE = X.copy(), E
    deltas = [abs(energy(env, Xn, cfg) - E) for Xn in
              (propose(env, X, cfg.r0, mask, rng) for _ in range(45)) if Xn is not None]
    Tm = max(-float(np.mean(deltas)) / math.log(cfg.p_acc0), 1e-3) if deltas else 1.0
    T0, T, stall = 0.01 * Tm, Tm, 0
    while T > T0 and stall < cfg.stall_max:
        improved = False
        r = max(0.1, cfg.r0 * T / Tm)
        for _ in range(cfg.k_iter):
            Xn = propose(env, X, r, mask, rng)
            if Xn is None:
                continue
            En = energy(env, Xn, cfg)
            dE = En - E
            if dE < 0.0 or rng.random() < math.exp(-dE / T):
                X, E = Xn, En
                if E < bestE - 1e-9:
                    bestX, bestE, improved = X.copy(), E, True
        T *= cfg.alpha
        stall = 0 if improved else stall + 1
    return bestX


def sp2_sa(env, start, goal, cfg, rng, mask=None):
    use_mask = env.mask_sp if mask is None else mask
    X_theta = theta_star(env, start, goal, use_mask)
    if X_theta is None or len(X_theta) <= 2:
        return X_theta
    X_sa = _raw_sp2_sa(env, start, goal, cfg, rng, mask=use_mask, X_seed=X_theta)
    if X_sa is None:
        return X_theta
    poly_t = smooth_bezier(env, X_theta, cfg)
    poly_s = smooth_bezier(env, X_sa, cfg)
    len_t = float(np.hypot(*np.diff(poly_t, axis=0).T).sum()) if len(poly_t) > 1 else 1e9
    len_s = float(np.hypot(*np.diff(poly_s, axis=0).T).sum()) if len(poly_s) > 1 else 1e9
    if len_t <= len_s + 0.08:
        return X_theta
    return X_sa


# =============================================================================
# 3. SMOOTHING: BEZIER BASELINE & CURVATURE-CONTINUOUS QUARTIC CORNER BLEND
# =============================================================================
def smooth_bezier(env, X, cfg, mask=None):
    if X is None or len(X) == 0:
        return np.zeros((0, 2))
    if len(X) <= 2:
        return densify(X)
    use_mask = env.mask if mask is None else mask
    pieces = [X[0:1]]
    t = np.linspace(0.0, 1.0, 15)[:, None]
    for j in range(1, len(X) - 1):
        P, a, b = X[j], X[j - 1], X[j + 1]
        la, lb = float(np.linalg.norm(a - P)), float(np.linalg.norm(b - P))
        if la < 1e-9 or lb < 1e-9:
            continue
        u1, u2 = (a - P) / la, (b - P) / lb
        d, curve = min(cfg.round_r, 0.48 * la, 0.48 * lb), None
        while d > 0.05:
            A, B = P + u1 * d, P + u2 * d
            c = (1.0 - t) ** 2 * A + 2.0 * (1.0 - t) * t * P + t ** 2 * B
            if not env.blocked(c, use_mask).any():
                curve = c
                break
            d *= 0.5
        pieces.append(curve if curve is not None else P[None])
    pieces.append(X[-1:])
    return densify(np.vstack(pieces))


def smooth_c2_quartic(env, X, cfg, mask=None):
    """
    Quartic corner blend (curvature-continuous at its joints; NOT minimum-snap):
      w0(t) = (1-t)^3 (1+t),  w2(t) = t^3 (2-t),  w1 = 1 - w0 - w2
    w0''(0)=w0''(1)=w2''(0)=w2''(1)=0, so the blend has zero second derivative at
    both joints with the adjoining straight segments.
    If no collision-free blend exists, the sharp corner is kept; Traj then slows
    the robot to a very low speed there (v <= w_max / kappa).
    """
    if X is None or len(X) == 0:
        return np.zeros((0, 2))
    use_mask = env.mask if mask is None else mask
    if len(X) <= 2:
        return densify(X)
    seg = np.hypot(*np.diff(X, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-6])
    Xc = X[keep]
    if len(Xc) <= 2:
        return densify(Xc)
    pieces = [Xc[0:1]]
    tau = np.linspace(0.0, 1.0, 21)[:, None]
    w0 = (1.0 - tau) ** 3 * (1.0 + tau)
    w2 = (tau ** 3) * (2.0 - tau)
    w1 = 1.0 - w0 - w2
    for j in range(1, len(Xc) - 1):
        P, a, b = Xc[j], Xc[j - 1], Xc[j + 1]
        la, lb = float(np.linalg.norm(a - P)), float(np.linalg.norm(b - P))
        if la < 1e-9 or lb < 1e-9:
            continue
        u1, u2 = (a - P) / la, (b - P) / lb
        d = min(cfg.round_r * 1.45, 0.48 * la, 0.48 * lb)
        curve = None
        while d > 0.05:
            A, B = P + u1 * d, P + u2 * d
            c = w0 * A + w1 * P + w2 * B
            if not env.blocked(c, use_mask).any():
                curve = c
                break
            d *= 0.6
        pieces.append(curve if curve is not None else P[None])
    pieces.append(Xc[-1:])
    poly = densify(np.vstack(pieces), ds=0.05)
    if not env.blocked(poly, use_mask).any():
        return poly
    return smooth_bezier(env, Xc, cfg, mask=use_mask)


# =============================================================================
# 4. KINEMATICALLY CONSTRAINED TRAJECTORY
# =============================================================================
def _discrete_curvature_profile(poly):
    """|kappa| [rad/cell] at every vertex of poly."""
    n = len(poly)
    kappa = np.zeros(n, float)
    if n < 3:
        return kappa
    d = np.diff(poly, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    valid = (seg[:-1] > 1e-7) & (seg[1:] > 1e-7)
    cos_th = np.ones(n - 2, float)
    cos_th[valid] = np.clip(
        (d[:-1][valid] * d[1:][valid]).sum(1) / (seg[:-1][valid] * seg[1:][valid]), -1.0, 1.0)
    dtheta = np.arccos(cos_th)
    ds = 0.5 * (seg[:-1] + seg[1:]) + 1e-9
    kappa[1:-1] = dtheta / ds
    kappa[0], kappa[-1] = kappa[1], kappa[-2]
    return kappa


def _accel_limited_profile(s, v_lim, a):
    """Backward (braking) then forward (acceleration) pass; v(0)=v(L)=0."""
    n = len(s)
    v = v_lim.astype(float).copy()
    v[0] = 0.0
    v[-1] = 0.0
    ds = np.diff(s)
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a * ds[i]))
    for i in range(n - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * a * ds[i]))
    return v


def _integrate_profile(s, vb, v_des, cfg, L):
    """Time-domain rollout under the speed cap vb(s), the accel window and v_des(t)."""
    dv_max = cfg.a_v_max * cfg.dt
    sa = [0.0]
    v_curr = 0.0
    for k in range(len(v_des)):
        s_curr = sa[-1]
        v_low = max(cfg.v_min_dyn, v_curr - dv_max)
        v_up = min(cfg.v_max_dyn, v_curr + dv_max)
        v_try = min(max(float(v_des[k]), v_low), v_up)
        s_look = min(s_curr + v_try * cfg.dt, L - 0.005)
        v_cap = float(np.interp(s_look, s, vb))
        # accel window wins over the look-ahead cap; the 0.85*a_v_max braking margin absorbs it
        v_new = max(min(v_try, v_cap), v_low)
        s_next = s_curr + v_new * cfg.dt
        if L - s_next < 0.005:
            sa.append(L)
            break
        sa.append(s_next)
        v_curr = v_new
    return np.array(sa, float)


def _kinematics(sa, s, kappa_sm, dt):
    """Returns (max|omega|, max|alpha|, max|a_v|, alpha_series, s_of_alpha). Starts at rest."""
    if len(sa) < 3:
        return 0.0, 0.0, 0.0, np.zeros(0), np.zeros(0)
    v = np.diff(sa) / dt
    s_mid = 0.5 * (sa[:-1] + sa[1:])
    w = np.interp(s_mid, s, kappa_sm) * v
    v0 = np.concatenate([[0.0], v])
    w0 = np.concatenate([[0.0], w])
    a = np.diff(v0) / dt
    al = np.diff(w0) / dt
    # exclude the terminal stop step from the acceleration checks
    a_chk, al_chk = a[:-1], al[:-1]
    s_al = s_mid[:len(al_chk)]
    return (float(np.max(np.abs(w))) if len(w) else 0.0,
            float(np.max(np.abs(al_chk))) if len(al_chk) else 0.0,
            float(np.max(np.abs(a_chk))) if len(a_chk) else 0.0,
            al_chk, s_al)


class Traj:
    """
    Speed profile built from: v <= w_max/kappa(s), v <= v_max_dyn, braking-feasible
    (backward pass with 0.85*a_v_max), start/end at rest, plus optional time-domain
    modulation mods (('boost', t0, t1, v) or ('gauss', t_c, sigma, min_fac)).
    Angular acceleration is repaired iteratively. Check self.kin afterwards.
    """
    def __init__(self, poly, mods, cfg):
        self.poly, self.mods = poly, mods
        self.kin = {'max_w': 0.0, 'max_alpha': 0.0, 'max_a': 0.0}
        if poly is None or len(poly) < 2:
            self.s = np.array([0.0])
            self.length = 0.0
            self.sa = np.array([0.0])
            self.pos = np.zeros((1, 2)) if poly is None or len(poly) == 0 else poly.copy()
            self.done = False
            self.T = 0.0
            return

        seg = np.hypot(*np.diff(poly, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(self.s[-1])
        self.length = L

        K = int(cfg.tmax / cfg.dt)
        v_des = self._desired_speed(mods, K, cfg)

        if L < 0.04:
            sa = np.array([0.0, L])
        else:
            kappa_raw = _discrete_curvature_profile(poly)
            kappa_lim = maximum_filter1d(kappa_raw, size=7, mode='nearest')
            kappa_sm = uniform_filter1d(kappa_raw, size=9, mode='nearest')
            v_lim = np.minimum(cfg.v_max_dyn, cfg.w_max / np.maximum(kappa_lim, 1e-9))
            v_lim = np.maximum(v_lim, 0.02)
            a_eff = 0.85 * cfg.a_v_max
            sa = None
            for it in range(cfg.max_aw_iters + 1):
                vb = _accel_limited_profile(self.s, v_lim, a_eff)
                sa = _integrate_profile(self.s, vb, v_des, cfg, L)
                mw, mal, ma, al_series, s_al = _kinematics(sa, self.s, kappa_sm, cfg.dt)
                self.kin = {'max_w': mw, 'max_alpha': mal, 'max_a': ma}
                if mal <= cfg.a_w_max * cfg.kin_tol or it == cfg.max_aw_iters:
                    break
                bad = np.abs(al_series) > cfg.a_w_max * cfg.kin_tol
                zone = np.zeros(len(self.s), bool)
                for sv in s_al[bad]:
                    zone |= np.abs(self.s - sv) < 0.5
                v_lim = np.where(zone, np.maximum(v_lim * 0.85, 0.02), v_lim)

        self.sa = sa
        self.done = bool(sa[-1] >= L - 1e-6)
        self.pos = np.column_stack([
            np.interp(self.sa, self.s, poly[:, 0]),
            np.interp(self.sa, self.s, poly[:, 1]),
        ])
        self.T = (len(self.pos) - 1) * cfg.dt

    @staticmethod
    def _desired_speed(mods, K, cfg):
        if not mods:
            return np.full(K, min(cfg.V, cfg.v_max_dyn), float)
        t = np.arange(K) * cfg.dt
        f = np.ones(K, float)
        for mod in mods:
            if mod[0] == 'gauss':
                _, t_c, sigma_t, min_fac = mod
                g_prof = 1.0 - (1.0 - min_fac) * np.exp(-0.5 * ((t - t_c) / max(sigma_t, 0.1)) ** 2)
                in_win = np.abs(t - t_c) <= 2.0 * cfg.tau
                f[in_win] = np.minimum(f[in_win], g_prof[in_win])
            elif mod[0] == 'boost':
                _, t0, t1, boost_v = mod
                w = (t >= t0) & (t < t1)
                f[w] = np.maximum(f[w], min(boost_v, cfg.v_max_dyn) / cfg.V)
        return np.clip(cfg.V * f, cfg.v_min_dyn, cfg.v_max_dyn)

    def kin_ok(self, cfg):
        tol = cfg.kin_tol
        return bool(self.kin['max_w'] <= cfg.w_max * tol and
                    self.kin['max_alpha'] <= cfg.a_w_max * tol and
                    self.kin['max_a'] <= cfg.a_v_max * tol)


def _pad(pos, K, hold):
    if len(pos) >= K:
        return pos[:K]
    pad = np.repeat(pos[-1:], K - len(pos), axis=0) if hold else np.full((K - len(pos), 2), np.nan)
    return np.vstack([pos, pad])


def velocity_at(pos, k, dt):
    """Instantaneous velocity v(k) = (pos[k+1] - pos[k]) / dt, zero once stopped."""
    if len(pos) < 2 or k >= len(pos) - 1:
        return np.zeros(2, float)
    k0 = max(0, k)
    return (pos[k0 + 1] - pos[k0]) / dt


def heading(pos, k):
    k_cl = min(max(k, 0), len(pos) - 1)
    v = pos[min(k_cl + 3, len(pos) - 1)] - pos[max(k_cl - 3, 0)]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def conflict_metrics(pos, other_list, d_safe, hold_goal=True):
    if not other_list:
        return 0, 0.0, None, 99.0
    tot_steps = 0
    pen_cost = 0.0
    first = None
    min_d = 99.0
    for hi, ph in enumerate(other_list):
        if hold_goal:
            K = max(len(pos), len(ph))
            p_self = _pad(pos, K, True)
            p_oth = _pad(ph, K, True)
            d = np.hypot(p_self[:, 0] - p_oth[:, 0], p_self[:, 1] - p_oth[:, 1])
        else:
            n = min(len(pos), len(ph))
            if n == 0:
                continue
            d = np.hypot(pos[:n, 0] - ph[:n, 0], pos[:n, 1] - ph[:n, 1])
        md = float(np.min(d))
        if md < min_d:
            min_d = md
        bad = d < d_safe
        if bad.any():
            tot_steps += int(bad.sum())
            pen = d_safe - d[bad]
            pen_cost += float(np.dot(pen, pen) + 5.0 * np.max(pen))
            k = int(np.argmax(bad))
            if first is None or k < first[0]:
                first = (k, hi)
    return tot_steps, pen_cost, first, min_d


# -----------------------------------------------------------------------------
# Forward CPA prediction + COLREGs stern-passing preference
# -----------------------------------------------------------------------------
def forward_cpa_prediction(pos_self, k_idx, pos_higher, cfg):
    """
    Closest point of approach from current positions and instantaneous velocities.
    Returns t_cpa, d_cpa, and a COLREGs-style stern-passing side preference
    (pass behind the higher-priority robot).
    """
    k_s = min(max(0, k_idx), len(pos_self) - 1)
    k_p = min(max(0, k_idx), len(pos_higher) - 1)
    p_i = pos_self[k_s]
    p_h = pos_higher[k_p]
    v_i = velocity_at(pos_self, k_s, cfg.dt)
    if np.linalg.norm(v_i) < 1e-6:
        v_i = heading(pos_self, k_s) * cfg.V
    v_h = velocity_at(pos_higher, k_p, cfg.dt)

    r_ih = p_i - p_h
    v_ih = v_i - v_h
    v_rel_sq = float(np.dot(v_ih, v_ih)) + 1e-6
    t_cpa = float(np.clip(-np.dot(r_ih, v_ih) / v_rel_sq, 0.0, cfg.t_pred_df))
    d_cpa = float(np.linalg.norm((p_i + v_i * t_cpa) - (p_h + v_h * t_cpa)))

    u_self = heading(pos_self, k_s)
    n_left = np.array([-u_self[1], u_self[0]])
    target_vec = -v_h if np.linalg.norm(v_h) > 0.05 else (p_i - p_h)
    preferred_sign = 1.0 if float(np.dot(n_left, target_vec)) >= 0.0 else -1.0
    return {'t_cpa': t_cpa, 'd_cpa': d_cpa, 'n_left': n_left, 'preferred_sign': preferred_sign}


# -----------------------------------------------------------------------------
# VO-style trajectory scoring (in the spirit of Zhong et al., 2021; offline)
# -----------------------------------------------------------------------------
def select_stage_target(global_poly, current_pos, preview_dist):
    dists = np.hypot(global_poly[:, 0] - current_pos[0], global_poly[:, 1] - current_pos[1])
    idx_closest = int(np.argmin(dists))
    seg = np.hypot(*np.diff(global_poly[idx_closest:], axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx_rel = int(np.searchsorted(cum, preview_dist, side='left'))
    idx_target = min(len(global_poly) - 1, idx_closest + idx_rel)
    return global_poly[idx_target]


def in_velocity_obstacle_cone(p_a, v_a, p_b, v_b, r_safe, t_horizon=4.5):
    v_ab = v_a - v_b
    rel_pos = p_b - p_a
    dist = float(np.linalg.norm(rel_pos))
    if dist <= r_safe:
        return True, 0.0
    v_norm_sq = float(np.dot(v_ab, v_ab))
    if v_norm_sq < 1e-8:
        return False, dist
    t_star = float(np.dot(rel_pos, v_ab) / v_norm_sq)
    if t_star < 0.0 or t_star > t_horizon:
        return False, dist
    d_min = float(np.linalg.norm(rel_pos - v_ab * t_star))
    return (d_min < r_safe), d_min


def vo_score_trajectory(env, tr_cand, global_poly, higher_pos, cfg):
    """Heading + clearance + speed + VO penalty + CPA clearance. All higher-priority
    trajectories are known (centralized planner), so no sensor range is modelled."""
    pos = tr_cand.pos
    K = len(pos)
    vo_hits = 0
    min_d_cpa = 99.0
    for hi_pos in higher_pos:
        K_max = max(K, len(hi_pos)) if cfg.hold_goal else min(K, len(hi_pos))
        if K_max < 2:
            continue
        p_self = _pad(pos, K_max, True) if cfg.hold_goal else pos[:K_max]
        p_hi = _pad(hi_pos, K_max, True) if cfg.hold_goal else hi_pos[:K_max]
        d_vec = np.hypot(p_self[:, 0] - p_hi[:, 0], p_self[:, 1] - p_hi[:, 1])
        k_min = int(np.argmin(d_vec))
        k_check = min(max(0, k_min - 8), K - 1)
        p_a = pos[k_check]
        v_a = velocity_at(pos, k_check, cfg.dt)
        p_b = hi_pos[min(k_check, len(hi_pos) - 1)]
        v_b = velocity_at(hi_pos, k_check, cfg.dt)
        in_vo, _ = in_velocity_obstacle_cone(p_a, v_a, p_b, v_b, cfg.d_safe)
        if in_vo:
            vo_hits += 1
        min_d_cpa = min(min_d_cpa, forward_cpa_prediction(pos, k_check, hi_pos, cfg)['d_cpa'])

    sample_idx = np.linspace(0, K - 1, min(20, K)).astype(int)
    head_align = 0.0
    for k in sample_idx[:-1]:
        st_pt = select_stage_target(global_poly, pos[k], cfg.preview_dist)
        vec_st = st_pt - pos[k]
        u_st = vec_st / (np.linalg.norm(vec_st) + 1e-9)
        head_align += float(np.dot(heading(pos, k), u_st))
    norm_head = head_align / max(1, len(sample_idx) - 1)
    norm_dist = float(np.clip(np.min(env.clearance(pos)) / 2.0, 0.0, 1.0))
    norm_vel = float(tr_cand.length / max(tr_cand.T * cfg.v_max_dyn, 1e-6))
    norm_vo = 1.0 if vo_hits > 0 else 0.0
    norm_df = float(np.clip(min_d_cpa / (cfg.d_safe + 1.0), 0.0, 1.0))
    return (cfg.w_head * norm_head + cfg.w_dist * norm_dist + cfg.w_vel * norm_vel +
            cfg.w_vo * norm_vo + cfg.w_df * norm_df)


# =============================================================================
# 5. BYPASS CANDIDATES & SPACE-TIME SA
# =============================================================================
def project_point_to_chain_arclength(X_chain, pt):
    seg_vecs = np.diff(X_chain, axis=0)
    seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    seg_dists = np.concatenate([[0.0], np.cumsum(seg_lens)])
    best_dist_sq, best_s = 1e18, 0.0
    for i in range(len(seg_lens)):
        L_i = float(seg_lens[i])
        if L_i < 1e-12:
            continue
        t_proj = float(np.clip(np.dot(pt - X_chain[i], seg_vecs[i]) / (L_i * L_i), 0.0, 1.0))
        proj = X_chain[i] + t_proj * seg_vecs[i]
        d_sq = float(np.sum((pt - proj) ** 2))
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_s = float(seg_dists[i] + t_proj * L_i)
    return best_s, seg_dists


def insert_waypoint_on_chain(env, X, P):
    best_X, best_inc = None, 1e18
    for j in range(len(X) - 1):
        if env.los(X[j], P) and env.los(P, X[j + 1]):
            inc = (np.linalg.norm(P - X[j]) + np.linalg.norm(X[j + 1] - P)
                   - np.linalg.norm(X[j + 1] - X[j]))
            if inc < best_inc:
                best_inc = inc
                best_X = np.insert(X, j + 1, P, axis=0)
    return best_X


def prune_chain_around(env, X, keep_idx):
    if len(X) <= 3:
        return X

    def prune_sub(sub):
        if len(sub) <= 2:
            return sub
        out = [sub[0]]
        i = 0
        while i < len(sub) - 1:
            j = len(sub) - 1
            while j > i + 1 and not env.los(sub[i], sub[j]):
                j -= 1
            out.append(sub[j])
            i = j
        return np.array(out, float)

    pre = prune_sub(X[:keep_idx + 1])
    post = prune_sub(X[keep_idx:])
    return np.vstack([pre[:-1], post])


def cpa_anchored_bypass_candidates(env, X_base, tr_base, k_conf, hi_pos, cfg):
    """Bypass candidates. With cfg.use_cpa the side ordering comes from
    forward CPA + stern passing; with use_cpa=False (ablation) it is a fixed order."""
    cands = []
    pos = tr_base.pos
    k_cl = min(k_conf, len(pos) - 1)
    u_dir = heading(pos, k_cl)
    n_perp = np.array([-u_dir[1], u_dir[0]])
    p_conf = pos[k_cl]

    if cfg.use_cpa:
        df = forward_cpa_prediction(pos, max(0, k_cl - 15), hi_pos, cfg)
        pref = df['preferred_sign']
    else:
        pref = 1.0

    # Fixed offset magnitudes. (Scaling them by the CPA deficit was tested and dropped:
    # it removed the small offsets that are often the only feasible ones.)
    base_mags = [0.65, 1.05, 1.50, 2.05, 2.75]
    lat_pool = [pref * m for m in base_mags] + [-pref * m for m in base_mags]

    for lat in lat_pool:
        for lon in (-1.4, 0.0, 1.4):
            P = p_conf + lat * n_perp + lon * u_dir
            if not env.blocked(P)[0]:
                Xc = insert_waypoint_on_chain(env, X_base, P)
                if Xc is not None:
                    cands.append(Xc)
                    idx_p = int(np.argmin(np.linalg.norm(Xc - P, axis=1)))
                    if 0 < idx_p < len(Xc) - 1:
                        Xp = prune_chain_around(env, Xc, idx_p)
                        if len(Xp) < len(Xc):
                            cands.append(Xp)
                if len(X_base) > 2:
                    j_cl = 1 + int(np.argmin(np.linalg.norm(X_base[1:-1] - p_conf, axis=1)))
                    if np.linalg.norm(X_base[j_cl] - p_conf) < 7.5:
                        if env.los(X_base[j_cl - 1], P) and env.los(P, X_base[j_cl + 1]):
                            Xs = X_base.copy()
                            Xs[j_cl] = P
                            cands.append(Xs)

    s_conf_raw, seg_dists = project_point_to_chain_arclength(X_base, p_conf)
    L_raw = float(seg_dists[-1])
    interp_X = lambda s_val: np.array([
        np.interp(s_val, seg_dists, X_base[:, 0]),
        np.interp(s_val, seg_dists, X_base[:, 1]),
    ])

    for ds_pre in (3.5, 7.0):
        s_pre = max(0.0, s_conf_raw - ds_pre)
        P_pre = interp_X(s_pre)
        if env.blocked(P_pre)[0]:
            continue
        idx_pre = max(0, min(int(np.searchsorted(seg_dists, s_pre, side='right')) - 1, len(X_base) - 2))
        if not env.los(X_base[idx_pre], P_pre):
            continue
        for lat in lat_pool[:8]:
            for lon in (-0.8, 0.8):
                P_mid = p_conf + lat * n_perp + lon * u_dir
                if env.blocked(P_mid)[0] or not env.los(P_pre, P_mid):
                    continue
                for idx_post in range(idx_pre + 1, len(X_base)):
                    if env.los(P_mid, X_base[idx_post]):
                        pts = list(X_base[:idx_pre + 1])
                        if np.linalg.norm(P_pre - pts[-1]) > 0.4:
                            pts.append(P_pre)
                        pts.append(P_mid)
                        pts.extend(X_base[idx_post:])
                        cands.append(np.array(pts, float))
                        break
                for ds_post in (4.0, 7.5):
                    s_post = min(L_raw, s_conf_raw + ds_post)
                    P_post = interp_X(s_post)
                    if env.blocked(P_post)[0] or not env.los(P_mid, P_post):
                        continue
                    idx_p2 = min(int(np.searchsorted(seg_dists, s_post, side='right')), len(X_base) - 1)
                    if env.los(P_post, X_base[idx_p2]):
                        pts = list(X_base[:idx_pre + 1])
                        if np.linalg.norm(P_pre - pts[-1]) > 0.4:
                            pts.append(P_pre)
                        pts.append(P_mid)
                        if np.linalg.norm(P_post - X_base[idx_p2]) > 0.4:
                            pts.append(P_post)
                        pts.extend(X_base[idx_p2:])
                        cands.append(np.array(pts, float))
                        break
    return cands


def plan_single_cpa_vo_sa(env, X_init, start, goal, higher, cfg, rng, sa_steps=None,
                          smoother=smooth_c2_quartic):
    """Single-robot planner against fixed higher-priority trajectories."""
    if sa_steps is None:
        sa_steps = cfg.st_sa_steps
    d_plan = dsp(cfg)
    X_init = validate_or_repair_chain(env, X_init, start, goal)
    poly0 = smoother(env, X_init, cfg)
    traj0 = Traj(poly0, [], cfg)
    nc0, pen0, fst0, _ = conflict_metrics(traj0.pos, higher, d_plan, cfg.hold_goal)

    if nc0 == 0 and traj0.done and not env.blocked(poly0).any():
        if cfg.v_max_dyn > cfg.V + 1e-4:
            tr_fast = Traj(poly0, [('boost', 0.0, traj0.T, cfg.v_max_dyn)], cfg)
            if (tr_fast.done and not env.blocked(tr_fast.pos).any() and
                    conflict_metrics(tr_fast.pos, higher, d_plan, cfg.hold_goal)[0] == 0):
                return tr_fast, X_init
        return traj0, X_init

    best_cf_traj, best_cf_X, best_cf_cost = None, None, 1e18
    best_any_traj, best_any_X, best_any_score = traj0, X_init.copy(), (nc0, pen0, traj0.T)

    def eval_cand(X_cand, mods=None):
        nonlocal best_cf_traj, best_cf_X, best_cf_cost, best_any_traj, best_any_X, best_any_score
        if mods is None:
            mods = []
        if not mods and best_cf_traj is not None:
            raw_L = float(np.hypot(*np.diff(X_cand, axis=0).T).sum())
            if raw_L / cfg.v_max_dyn > best_cf_traj.T + 0.45:
                return None, 9999, 9999.0, None
        poly = smoother(env, X_cand, cfg)
        if len(poly) < 2 or env.blocked(poly).any():
            return None, 9999, 9999.0, None
        tr = Traj(poly, mods, cfg)
        if not tr.done:
            return None, 9999, 9999.0, None
        nc_s, p_s, _, _ = conflict_metrics(tr.pos, higher, d_plan, cfg.hold_goal)
        nc_b, p_b, fst_b, _ = conflict_metrics(tr.pos, higher, d_plan + 0.02, cfg.hold_goal)
        if nc_s == 0:
            turn_tr = total_turn_deg(tr.pos)
            vo_G = vo_score_trajectory(env, tr, poly0, higher, cfg)
            cost_tr = tr.T + 0.0015 * turn_tr - 0.015 * vo_G
            if cost_tr < best_cf_cost - 1e-6:
                best_cf_traj, best_cf_X, best_cf_cost = tr, X_cand.copy(), cost_tr
        sc = (nc_s, p_s, tr.T)
        if sc < best_any_score:
            best_any_traj, best_any_X, best_any_score = tr, X_cand.copy(), sc
        return tr, nc_b, p_b, fst_b

    # Phase 1: CPA-guided bypass seeding
    if fst0 is not None:
        hi_conflict_pos = higher[fst0[1]]
        cands1 = cpa_anchored_bypass_candidates(env, X_init, traj0, fst0[0], hi_conflict_pos, cfg)
        scored1 = []
        for Xc in cands1:
            tr, ncb, pb, fstb = eval_cand(Xc)
            if tr is not None:
                vo_score = vo_score_trajectory(env, tr, poly0, higher, cfg)
                scored1.append((ncb, pb, tr.T - 0.02 * vo_score, Xc, tr, fstb))
        scored1.sort(key=lambda x: (x[0], x[1], x[2]))
        for ncb, pb, _, Xc1, tr1, fstb1 in scored1[:3]:
            if fstb1 is not None:
                hi_pos2 = higher[fstb1[1]]
                for Xc2 in cpa_anchored_bypass_candidates(env, Xc1, tr1, fstb1[0], hi_pos2, cfg)[:12]:
                    eval_cand(Xc2)

    # Phase 2: space-time SA
    if sa_steps > 0 and fst0 is not None:
        X_curr = best_cf_X.copy() if best_cf_X is not None else best_any_X.copy()
        if len(X_curr) == 2:
            hi_conflict_pos = higher[fst0[1]]
            if cfg.use_cpa:
                df0 = forward_cpa_prediction(traj0.pos, max(0, fst0[0] - 10), hi_conflict_pos, cfg)
                sign, n_left = df0['preferred_sign'], df0['n_left']
            else:
                u0 = heading(traj0.pos, min(fst0[0], len(traj0.pos) - 1))
                sign, n_left = 1.0, np.array([-u0[1], u0[0]])
            P_mid = traj0.pos[min(fst0[0], len(traj0.pos) - 1)] + sign * 1.2 * n_left
            if not env.blocked(P_mid)[0]:
                Xc = insert_waypoint_on_chain(env, X_curr, P_mid)
                if Xc is not None:
                    X_curr = Xc
        tr_curr, ncb_curr, pb_curr, _ = eval_cand(X_curr)
        if tr_curr is None:
            tr_curr, ncb_curr, pb_curr = best_any_traj, best_any_score[0], best_any_score[1]
        c_curr = tr_curr.T + 0.002 * total_turn_deg(tr_curr.pos) + 80.0 * pb_curr + 25.0 * ncb_curr

        T_sa = 0.8
        for step in range(sa_steps):
            T_sa *= 0.93
            r_step = max(0.12, 1.6 * (0.94 ** step))
            m = len(X_curr)
            u = rng.random()
            X_new = None
            if u < 0.72 and m > 2:
                j = int(rng.integers(1, m - 1))
                P = X_curr[j] + rng.normal(0, r_step, 2)
                if not env.blocked(P)[0] and env.los(X_curr[j - 1], P) and env.los(P, X_curr[j + 1]):
                    X_new = X_curr.copy()
                    X_new[j] = P
            elif u < 0.86 and m < 7:
                _, _, fst_c, _ = conflict_metrics(tr_curr.pos, higher, d_plan + 0.05, cfg.hold_goal)
                if fst_c is not None:
                    p_c = tr_curr.pos[min(fst_c[0], len(tr_curr.pos) - 1)]
                    P = p_c + rng.normal(0, r_step, 2)
                    if not env.blocked(P)[0]:
                        X_new = insert_waypoint_on_chain(env, X_curr, P)
            elif m > 2:
                j = int(rng.integers(1, m - 1))
                if env.los(X_curr[j - 1], X_curr[j + 1]):
                    X_new = np.delete(X_curr, j, axis=0)
            if X_new is None:
                continue
            tr_new, ncb_new, pb_new, _ = eval_cand(X_new)
            if tr_new is None:
                continue
            c_new = tr_new.T + 0.002 * total_turn_deg(tr_new.pos) + 80.0 * pb_new + 25.0 * ncb_new
            if c_new < c_curr or rng.random() < math.exp(-(c_new - c_curr) / max(T_sa, 1e-3)):
                X_curr, tr_curr, c_curr = X_new, tr_new, c_new

    # Phase 3: speed modulation (boost only has an effect if v_max_dyn > V)
    cands_for_mod = [X_init, best_any_X] + ([best_cf_X] if best_cf_X is not None else [])
    for cand_X in cands_for_mod:
        poly = smoother(env, cand_X, cfg)
        if len(poly) < 2 or env.blocked(poly).any():
            continue
        tr_b = Traj(poly, [], cfg)
        nc_b0, _, fst, _ = conflict_metrics(tr_b.pos, higher, d_plan, cfg.hold_goal)
        if nc_b0 == 0:
            if cfg.v_max_dyn > cfg.V + 1e-4:
                eval_cand(cand_X, [('boost', 0.0, tr_b.T, cfg.v_max_dyn)])
            continue
        if fst is None:
            continue
        ts = fst[0] * cfg.dt
        if cfg.v_max_dyn > cfg.V + 1e-4:
            eval_cand(cand_X, [('boost', max(0.0, ts - 2.0 * cfg.tau), ts + 0.5, cfg.v_max_dyn)])
        found_mod = False
        for sigma_t in (1.2, 2.0):
            for min_fac in (0.65, 0.45, 0.25, 0.0):
                tr_g, _, _, _ = eval_cand(cand_X, [('gauss', max(0.5, ts - 1.5), sigma_t, min_fac)])
                if tr_g is not None and conflict_metrics(tr_g.pos, higher, d_plan, cfg.hold_goal)[0] == 0:
                    found_mod = True
                    break
            if found_mod:
                break

    if best_cf_traj is not None:
        initial_cf_T = best_cf_traj.T
        changed = True
        while changed and len(best_cf_X) > 2:
            changed = False
            for j in range(1, len(best_cf_X) - 1):
                if env.los(best_cf_X[j - 1], best_cf_X[j + 1]):
                    X_try = np.delete(best_cf_X, j, axis=0)
                    poly_try = smoother(env, X_try, cfg)
                    if len(poly_try) >= 2 and not env.blocked(poly_try).any():
                        tr_try = Traj(poly_try, best_cf_traj.mods, cfg)
                        if tr_try.done and conflict_metrics(tr_try.pos, higher, d_plan, cfg.hold_goal)[0] == 0:
                            if tr_try.T <= initial_cf_T + 0.02:
                                best_cf_traj, best_cf_X, changed = tr_try, X_try, True
                                break
        return best_cf_traj, best_cf_X
    return best_any_traj, best_any_X


# =============================================================================
# 6. MASTER SOLVER
# =============================================================================
def solve_cpa_vo_sp2_sa(env, robots, cfg, rng, use_homotopy=False):
    order = list(robots.keys())
    d_plan = dsp(cfg)
    X_init, tr_init, ptime = {}, {}, {}
    for r in order:
        t0 = time.perf_counter()
        s, g = robots[r]
        if use_homotopy and r in STATIC_HOMOTOPY_GUIDES:
            X_seed = np.array(STATIC_HOMOTOPY_GUIDES[r], float)
            X_init[r] = _raw_sp2_sa(env, s, g, cfg, rng, X_seed=X_seed)
        else:
            X_init[r] = sp2_sa(env, s, g, cfg, rng)
        tr_init[r] = Traj(smooth_c2_quartic(env, X_init[r], cfg), [], cfg)
        ptime[r] = time.perf_counter() - t0

    trajs, X_sol = {}, {}
    for idx, r in enumerate(order):
        t0 = time.perf_counter()
        s, g = robots[r]
        higher_names = order[:idx]
        higher_pos = [trajs[m].pos for m in higher_names]
        tr_uni, X_uni = plan_single_cpa_vo_sa(
            env, X_init[r], s, g, higher_pos, cfg, rng, smoother=smooth_c2_quartic)
        uni_ok = conflict_metrics(tr_uni.pos, higher_pos, d_plan, cfg.hold_goal)[0] == 0

        # Cooperative priority-preserving micro-nudge
        nc0, _, fst0, _ = conflict_metrics(tr_init[r].pos, higher_pos, d_plan, cfg.hold_goal)
        best_coop = None
        if fst0 is not None and ((not uni_ok) or tr_uni.T > tr_init[r].T + 0.35):
            k_conf, hi_idx = fst0
            h_name = higher_names[hi_idx]
            others_for_h = [trajs[m].pos for m in higher_names if m != h_name]
            h_cands = cpa_anchored_bypass_candidates(
                env, X_sol[h_name], trajs[h_name], k_conf, tr_init[r].pos, cfg)
            valid_h = []
            for X_h_cand in h_cands:
                poly_h = smooth_c2_quartic(env, X_h_cand, cfg)
                if len(poly_h) < 2 or env.blocked(poly_h).any():
                    continue
                tr_h_cand = Traj(poly_h, [], cfg)
                if not tr_h_cand.done or tr_h_cand.T > trajs[h_name].T + cfg.coop_tol:
                    continue
                if conflict_metrics(tr_h_cand.pos, others_for_h, d_plan, cfg.hold_goal)[0] > 0:
                    continue
                new_higher_pos = [tr_h_cand.pos if m == h_name else trajs[m].pos for m in higher_names]
                nc_r0, pen_r0, _, _ = conflict_metrics(tr_init[r].pos, new_higher_pos, d_plan, cfg.hold_goal)
                nc_rem = sum(conflict_metrics(tr_init[m].pos, [tr_h_cand.pos], d_plan, cfg.hold_goal)[0]
                             for m in order[idx + 1:])
                valid_h.append((nc_r0, 0.2 * nc_rem + pen_r0, tr_h_cand.T, tr_h_cand, X_h_cand, new_higher_pos))
            valid_h.sort(key=lambda x: (x[0], x[1], x[2]))
            for nc_r0, _, _, tr_h_cand, X_h_cand, new_higher_pos in valid_h[:3]:
                if nc_r0 == 0 and not env.blocked(tr_init[r].pos).any():
                    tr_r_cand, X_r_cand = tr_init[r], X_init[r]
                else:
                    tr_r_cand, X_r_cand = plan_single_cpa_vo_sa(
                        env, X_init[r], s, g, new_higher_pos, cfg, rng, sa_steps=16,
                        smoother=smooth_c2_quartic)
                if (tr_r_cand.done and not env.blocked(tr_r_cand.pos).any() and
                        conflict_metrics(tr_r_cand.pos, new_higher_pos, d_plan, cfg.hold_goal)[0] == 0):
                    team_T_new = tr_h_cand.T + tr_r_cand.T
                    team_T_old = trajs[h_name].T + tr_uni.T
                    # Feasibility first: a conflict-free cooperative plan beats a colliding one.
                    better = (not uni_ok) or (team_T_new < team_T_old - 0.15)
                    if better and (best_coop is None or team_T_new < best_coop[0]):
                        best_coop = (team_T_new, h_name, tr_h_cand, X_h_cand, tr_r_cand, X_r_cand)
                        if nc_r0 == 0:
                            break

        if best_coop is not None:
            _, h_name, tr_h_new, X_h_new, tr_r_new, X_r_new = best_coop
            trajs[h_name], X_sol[h_name] = tr_h_new, X_h_new
            trajs[r], X_sol[r] = tr_r_new, X_r_new
        else:
            trajs[r], X_sol[r] = tr_uni, X_uni
        ptime[r] += time.perf_counter() - t0

    tot = sum(
        conflict_metrics(trajs[n].pos, [trajs[m].pos for m in order[:i]], cfg.d_safe, cfg.hold_goal)[0]
        for i, n in enumerate(order))
    return tot, trajs, X_init, ptime, order


# =============================================================================
# 7. METRICS, TABLES & PLOTS
# =============================================================================
def collect_metrics(env, trajs, ptime, cfg):
    rows = []
    for idx, r in enumerate(NAMES):
        tr = trajs[r]
        others = [trajs[o].pos for o in NAMES if o != r]
        nc_all, _, _, md_cells = conflict_metrics(tr.pos, others, cfg.d_safe, cfg.hold_goal)  # reported at d_safe
        obs_hit = bool(env.blocked(tr.pos).any())
        kin_ok = tr.kin_ok(cfg)
        success = bool(tr.done and not obs_hit and nc_all == 0 and kin_ok)
        rows.append({
            'priority': idx + 1, 'name': r,
            'length': tr.length * 10.0, 'min_dist': md_cells * 10.0, 'time_s': tr.T,
            'turn_deg': total_turn_deg(tr.pos), 'cpu_s': ptime[r], 'success': float(success),
            'max_w': tr.kin['max_w'], 'max_alpha': tr.kin['max_alpha'], 'max_a': tr.kin['max_a'],
        })
    return rows


def print_tables(env_key, rows, label, cfg, seed_stats=None):
    env_title = 'Barrier-free environment' if env_key == 'free' else 'Static environment'
    print('\n' + '=' * 118)
    print(f'  PER-ROBOT RESULTS - {env_title.upper()} ({label})')
    print('=' * 118)
    print(f'{"Algorithm":28} | {"Robot":8} | {"Priority":>8} | {"Path length":>11} | '
          f'{"Min dist":>10} | {"Move time(s)":>12} | {"Turn(deg)":>10} | {"CPU(s)":>8}')
    print('-' * 118)
    for row in rows:
        alg_col = label[:28] if row['priority'] == 1 else ''
        robot_lbl = f"Robot {row['priority']}"
        print(f"{alg_col:28} | {robot_lbl:8} | {row['priority']:8d} | "
              f"{row['length']:11.2f} | {row['min_dist']:10.2f} | {row['time_s']:12.2f} | "
              f"{row['turn_deg']:10.2f} | {row['cpu_s']:8.3f}")
    print('-' * 118)
    aL = np.mean([r['length'] for r in rows])
    mL = np.max([r['length'] for r in rows])
    aT = np.mean([r['time_s'] for r in rows])
    aTurn = np.mean([r['turn_deg'] for r in rows])
    aCpu = np.mean([r['cpu_s'] for r in rows])
    sr = 100.0 * np.mean([r['success'] for r in rows])
    print(f'  SUMMARY: avg len {aL:.2f} | max len {mL:.2f} | avg time {aT:.2f}s | '
          f'avg turn {aTurn:.1f} deg | avg CPU {aCpu:.2f}s | success {sr:.0f}%')
    if seed_stats is not None and seed_stats['n'] > 1:
        print(f"  OVER {seed_stats['n']} SEEDS: avg time {seed_stats['T_mean']:.2f} +/- {seed_stats['T_std']:.2f} s | "
              f"avg length {seed_stats['L_mean']:.2f} +/- {seed_stats['L_std']:.2f} | "
              f"success {seed_stats['succ']*100:.0f}% of robot-runs")
    print(f"  KINEMATICS (max over robots): |w| {max(r['max_w'] for r in rows):.2f}/{cfg.w_max} rad/s | "
          f"|alpha| {max(r['max_alpha'] for r in rows):.2f}/{cfg.a_w_max} rad/s^2 | "
          f"|a| {max(r['max_a'] for r in rows):.2f}/{cfg.a_v_max} cells/s^2")
    print('  Note: CPU time is single-thread Python wall-clock; not comparable to C++ baselines.')
    print('-' * 118)


def draw_map(ax, env, title):
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks(range(0, 41, 5))
    ax.grid(True, lw=0.4, color='#b0b0b0', alpha=0.6)
    ax.set_title(title, fontsize=10.5, weight='bold', pad=8)
    for x0, y0, x1, y1 in env.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color='k', zorder=2))


def plot_results(env, robots, trajs, name, outdir, cfg, tag):
    env_label = 'Barrier-free environment' if name == 'free' else 'Static environment'
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2))
    draw_map(axs[0], env, f'(a) CPA-VO-SP2-SA - {env_label}')
    for i, r in enumerate(NAMES):
        tr, c = trajs[r], COLORS[i]
        st, go = robots[r]
        axs[0].plot(tr.pos[:, 0], tr.pos[:, 1], color=c, lw=2.3, zorder=3,
                    label=f'robot {r} (L={tr.length*10:.1f}, T={tr.T:.2f}s)')
        axs[0].plot(*st, 's', color='limegreen', mec='k', ms=7.5, zorder=4)
        axs[0].plot(*go, 's', color='red', mec='k', ms=7.5, zorder=4)
        for k in range(0, len(tr.pos), 100):
            axs[0].plot(*tr.pos[k], 'o', color=c, ms=3.2, mec='w', mew=0.4, zorder=4)
        j = max(1, len(tr.pos) // 2)
        axs[0].annotate('', xy=tr.pos[min(j + 3, len(tr.pos) - 1)], xytext=tr.pos[j],
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.7), zorder=4)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=True)

    ax1 = axs[1]
    ax1.set_title(f'(b) Inter-Robot Distance vs. Time - {env_label}', fontsize=10.5, weight='bold', pad=8)
    for a in range(len(NAMES)):
        for b in range(a + 1, len(NAMES)):
            A, B = trajs[NAMES[a]].pos, trajs[NAMES[b]].pos
            K = max(len(A), len(B)) if cfg.hold_goal else min(len(A), len(B))
            pA = _pad(A, K, True) if cfg.hold_goal else A[:K]
            pB = _pad(B, K, True) if cfg.hold_goal else B[:K]
            ax1.plot(np.arange(K) * cfg.dt, np.linalg.norm(pA - pB, axis=1) * 10.0, lw=1.25,
                     label=f'distance of robot {NAMES[a]} and {NAMES[b]}')
    ax1.axhline(cfg.d_safe * 10.0, ls='--', color='k', lw=1.2, label='safety threshold (20.0)')
    ax1.set_xlabel('Time (s)', fontsize=9.5)
    ax1.set_ylabel('Distance (paper units, x10)', fontsize=9.5)
    ax1.set_ylim(bottom=0)
    ax1.legend(ncol=2, fontsize=7, loc='upper right', frameon=True)
    ax1.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{name}_{tag}_results.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. SELF-TEST & MAIN
# =============================================================================
def selftest(cfg):
    """Checks that the speed profile respects w_max, a_v_max, a_w_max on test paths."""
    env = Env([], cfg)
    cases = {
        'straight': np.array([[2.0, 2.0], [20.0, 2.0]]),
        'right-angle': np.array([[2.0, 2.0], [15.0, 2.0], [15.0, 15.0]]),
        'zigzag': np.array([[2.0, 2.0], [10.0, 2.0], [14.0, 8.0], [22.0, 8.0], [26.0, 14.0]]),
    }
    ok_all = True
    for nm, X in cases.items():
        poly = smooth_c2_quartic(env, X, cfg)
        tr = Traj(poly, [], cfg)
        ok = tr.done and tr.kin_ok(cfg)
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {nm:12} T={tr.T:6.2f}s  |w|={tr.kin['max_w']:.2f}/{cfg.w_max}  "
              f"|alpha|={tr.kin['max_alpha']:.2f}/{cfg.a_w_max}  |a|={tr.kin['max_a']:.2f}/{cfg.a_v_max}")
    # sharp corner (no blend): must slow to a crawl instead of violating w_max
    X = np.array([[2.0, 2.0], [10.0, 2.0], [10.0, 10.0]])
    tr = Traj(densify(X), [], cfg)
    ok = tr.done and tr.kin['max_w'] <= cfg.w_max * cfg.kin_tol
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] sharp corner  T={tr.T:6.2f}s  |w|={tr.kin['max_w']:.2f}/{cfg.w_max}")
    # starts at rest
    v0 = float(np.linalg.norm(tr.pos[1] - tr.pos[0]) / cfg.dt)
    ok = v0 <= cfg.a_v_max * cfg.dt * 1.02
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] starts at rest, first-step speed {v0:.3f} <= {cfg.a_v_max*cfg.dt:.3f}")
    print('SELFTEST', 'PASSED' if ok_all else 'FAILED')
    return ok_all


def run_scenario(name, cfg, seed, outdir, use_homotopy=False, num_seeds=1, ablation=False):
    rects, robots = SCENARIOS[name]
    env = Env(rects, cfg)
    variants = [('CPA-VO-SP2-SA (proposed)', 'proposed', cfg)]
    if ablation:
        variants.append(('Ablation: no CPA ordering', 'no_cpa', replace(cfg, use_cpa=False)))

    for label, tag, vcfg in variants:
        runs, first_trajs = [], None
        for s_offset in range(num_seeds):
            _, trajs, _, pt, _ = solve_cpa_vo_sp2_sa(
                env, robots, vcfg, np.random.default_rng(seed + s_offset),
                use_homotopy=(use_homotopy and name == 'static'))
            runs.append(collect_metrics(env, trajs, pt, vcfg))
            if first_trajs is None:
                first_trajs = trajs
        avg_rows = []
        for i in range(len(NAMES)):
            avg_rows.append({
                'priority': i + 1, 'name': NAMES[i],
                'length': float(np.mean([r[i]['length'] for r in runs])),
                'min_dist': float(np.min([r[i]['min_dist'] for r in runs])),
                'time_s': float(np.mean([r[i]['time_s'] for r in runs])),
                'turn_deg': float(np.mean([r[i]['turn_deg'] for r in runs])),
                'cpu_s': float(np.mean([r[i]['cpu_s'] for r in runs])),
                'success': float(np.mean([r[i]['success'] for r in runs])),
                'max_w': float(np.max([r[i]['max_w'] for r in runs])),
                'max_alpha': float(np.max([r[i]['max_alpha'] for r in runs])),
                'max_a': float(np.max([r[i]['max_a'] for r in runs])),
            })
        per_seed_T = [np.mean([row['time_s'] for row in run]) for run in runs]
        per_seed_L = [np.mean([row['length'] for row in run]) for run in runs]
        stats = {'n': num_seeds, 'T_mean': float(np.mean(per_seed_T)), 'T_std': float(np.std(per_seed_T)),
                 'L_mean': float(np.mean(per_seed_L)), 'L_std': float(np.std(per_seed_L)),
                 'succ': float(np.mean([row['success'] for run in runs for row in run]))}
        print_tables(name, avg_rows, label, vcfg, stats)
        plot_results(env, robots, first_trajs, name, outdir, vcfg, tag)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='static', choices=['free', 'static', 'both'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-seeds', type=int, default=1, help='use >= 10 for paper results')
    ap.add_argument('--out', default='results_cpa_vo')
    ap.add_argument('--paper-homotopy', action='store_true')
    ap.add_argument('--ablation', action='store_true', help='also run with CPA ordering disabled')
    ap.add_argument('--v-max-dyn', type=float, default=None,
                    help='max speed (default = V = 1.0; >V enables the speed boost)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    cfg = Cfg()
    if args.v_max_dyn is not None:
        cfg = replace(cfg, v_max_dyn=args.v_max_dyn)
    if args.selftest:
        raise SystemExit(0 if selftest(cfg) else 1)

    os.makedirs(args.out, exist_ok=True)
    for sc in (['free', 'static'] if args.scenario == 'both' else [args.scenario]):
        run_scenario(sc, cfg, args.seed, args.out, use_homotopy=args.paper_homotopy,
                     num_seeds=args.num_seeds, ablation=args.ablation)
    print(f'\nFigures saved in ./{args.out}/')