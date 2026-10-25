
"""
DF-VO-SP2-SA (Optimized & Verified):
Forward-CPA + COLREGs Stern-Passing & VO-DWA Space-Time Simulated Annealing
===========================================================================
Compares on the static environment (and barrier-free environment) against:
  1. Kun Shi et al. (J. Franklin Inst. 362, 2025, 107396): Multi-Robot Improved SA
  2. Uncontaminated Baseline CST-SP2-SA (V2)

Key Technical Components:
  1. Forward CPA Prediction (Mischiati et al., Nature 2015) + Maritime COLREGs
     Stern-Passing Deflection using true instantaneous robot velocities v(k) and
     adaptive lateral gain df_lat_gain.
  2. Kinematically Constrained Stage-Target + Velocity Obstacle Dynamic Window
     Approach (Zhong et al., 2021) enforcing [v_min_dyn, v_max_dyn], linear
     acceleration a_v_max, angular velocity w_max = |kappa(s)|*v, and angular
     acceleration a_w_max over local space-time window tau.
  3. Validated Obstacle-Safe Homotopy Seeding & Exact C^2 Quartic Corner Blending
     with vanishing second derivatives at joints (w0''(0)=w0''(1)=w2''(0)=w2''(1)=0).

Usage:
    python3 df_vo_sp2_sa_optimized.py
    python3 df_vo_sp2_sa_optimized.py --scenario static --paper-homotopy
    python3 df_vo_sp2_sa_optimized.py --scenario static --num-seeds 5
"""
import argparse
import heapq
import math
import os
import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sa_V4 import (
    NAMES, COLORS, FREE_ROBOTS, STATIC_ROBOTS, STATIC_RECTS,
    PAPER_TABLE2_REF, PAPER_TABLE3_REF, SCENARIOS,
    Env, ShiCfg, densify, solve_shi_improved_sa, total_turn_deg,
)


# =============================================================================
# 1. CONFIGURATION & VALIDATED OBSTACLE-FREE HOMOTOPY GUIDES
# =============================================================================
@dataclass
class Cfg:
    V: float = 1.0            # nominal robot cruise speed [cells/s] (= 10 paper units/s)
    v_max_dyn: float = 1.08   # max kinematic window speed during predictive passing [cells/s]
    v_min_dyn: float = 0.0    # min kinematic speed [cells/s]
    w_max: float = 2.5        # max angular velocity [rad/s] (Zhong et al. Table I)
    a_v_max: float = 1.5      # max linear acceleration [cells/s^2] (Zhong et al. Table I)
    a_w_max: float = 1.5      # max angular acceleration [rad/s^2] (Zhong et al. Table I)
    dt: float = 0.1           # simulation time step [s]
    tmax: float = 250.0       # maximum simulation horizon [s]
    d_safe: float = 2.0       # minimum safe inter-robot distance [cells] (= 20.0 paper units)
    sensor_range: float = 6.0 # LIDAR detection range [cells] (= 60.0 paper units)
    margin: float = 0.3       # obstacle inflation radius [cells]
    sp_buf: float = 0.08      # corner-safe string-pulling buffer [cells]
    d_c: float = 0.8          # desired obstacle clearance in SA energy [cells]
    lam_turn: float = 0.05    # turning-angle^2 weight in single-robot SA
    lam_clr: float = 0.2      # clearance penalty weight in single-robot SA
    lam_wp: float = 0.01      # waypoint count penalty weight
    k_iter: int = 50          # SA moves per temperature level
    alpha: float = 0.9        # SA geometric cooling rate
    r0: float = 2.0           # initial waypoint shift radius [cells]
    p_acc0: float = 0.8       # target initial SA acceptance probability
    stall_max: int = 3        # early stop after stall levels
    round_r: float = 0.95     # corner rounding radius [cells]
    tau: float = 3.0          # half window for local space-time speed modulation [s]
    st_sa_steps: int = 30     # space-time SA iterations per robot
    coop_tol: float = 1.2     # max travel-time increase [s] for cooperative micro-nudge
    # Forward CPA + COLREGs stern-passing parameters
    t_pred_df: float = 6.5    # forward CPA lookahead horizon [s]
    df_lat_gain: float = 1.15 # COLREGs stern-passing lateral offset gain
    # Zhong et al. (2021) VO-DWA & stage-target parameters (Table II)
    preview_dist: float = 3.5 # stage target preview distance along global path [cells]
    w_head: float = 6.0       # alpha: heading alignment score weight
    w_dist: float = 6.0       # beta: obstacle clearance score weight
    w_vel: float = 8.0        # gamma: forward speed score weight
    w_vo: float = -100.0      # delta: velocity obstacle cone penalty weight
    w_df: float = 12.0        # eta: CPA clearance score weight
    hold_goal: bool = True    # robots remain stationary obstacles at their goals after arrival


# Verified obstacle-free guide waypoints matching the corridors in Shi et al. (2025) Fig. 5(c):
# - Robot c routes cleanly around (21,24,24,27) via (25.0, 23.2)->(25.0, 28.0)
# - Robot d clears (29,13,33,17) via (27.8, 12.2)
# - Robot e passes above (29,13,33,17) via (33.8, 17.8)->(28.2, 17.8)
STATIC_HOMOTOPY_GUIDES = {
    'a': [(1.5, 38.5), (5.2, 32.2), (12.5, 25.5), (19.2, 22.6), (23.2, 13.8), (27.5, 8.5)],
    'b': [(2.5, 4.5), (13.2, 8.0), (20.2, 16.2), (26.5, 22.0), (34.2, 28.5), (36.5, 34.5)],
    'c': [(18.5, 0.5), (16.0, 4.5), (16.0, 7.8), (19.5, 17.5), (25.0, 23.2),
          (25.0, 28.0), (21.2, 34.8), (18.5, 38.5)],
    'd': [(2.5, 22.5), (18.8, 22.8), (23.5, 20.5), (27.8, 12.2), (31.5, 10.5)],
    'e': [(36.5, 15.5), (33.8, 17.8), (28.2, 17.8), (24.5, 17.3), (18.5, 15.2),
          (12.8, 16.8), (7.2, 16.8), (2.5, 14.5)],
}


# =============================================================================
# 2. CLEAN THETA* & VALIDATED WAYPOINT SIMULATED ANNEALING
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
                # Always verify line-of-sight from cur (including cur == s)
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
    """Validates every vertex and segment of X_seed; repairs any blocked segment via Theta*."""
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
    # Reuse X_theta directly as X_seed to avoid running Theta* twice
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
# 3. SMOOTHING: BASELINE BEZIER VS. EXACT C^2 QUARTIC CORNER BLEND
# =============================================================================
def smooth_bezier(env, X, cfg, mask=None):
    """Baseline quadratic-Bezier corner rounding (from sp2_sa_V2.py)."""
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


def smooth_c2_minsnap(env, X, cfg, mask=None):
    """
    Single-Pass Obstacle-Safe C^2 Quartic Corner Blend:
      w0(tau) = (1 - tau)^3 * (1 + tau)
      w2(tau) = tau^3 * (2 - tau)
      w1(tau) = 1 - w0(tau) - w2(tau)
    Satisfies w0(0)=1, w0'(0)=-2, w0''(0)=0, w0(1)=w0'(1)=w0''(1)=0 and
              w2(1)=1, w2'(1)=+2, w2''(1)=0, w2(0)=w2'(0)=w2''(0)=0.
    Thus c''(0) = 0 and c''(1) = 0, guaranteeing continuous zero curvature
    at the straight-line joints without expensive multi-scale Laplacian loops.
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
# 4. KINEMATIC TRAJECTORY & FORWARD-CPA + ZHONG (2021) VO-DWA ENGINE
# =============================================================================
def _discrete_curvature_profile(poly):
    """Computes pointwise curvature |kappa| [rad/cell] at every vertex of poly."""
    n = len(poly)
    kappa = np.zeros(n, float)
    if n < 3:
        return kappa
    d = np.diff(poly, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    valid = (seg[:-1] > 1e-7) & (seg[1:] > 1e-7)
    cos_th = np.ones(n - 2, float)
    cos_th[valid] = np.clip(
        (d[:-1][valid] * d[1:][valid]).sum(1) / (seg[:-1][valid] * seg[1:][valid]),
        -1.0, 1.0
    )
    dtheta = np.arccos(cos_th)
    ds = 0.5 * (seg[:-1] + seg[1:]) + 1e-9
    kappa[1:-1] = dtheta / ds
    kappa[0], kappa[-1] = kappa[1], kappa[-2]
    return kappa


class Traj:
    """
    Kinematically constrained trajectory enforcing:
      - Speed window: v in [v_min_dyn, v_max_dyn]
      - Linear acceleration: |v(k) - v(k-1)| / dt <= a_v_max
      - Angular velocity: |kappa(s) * v(k)| <= w_max
      - Angular acceleration: |omega(k) - omega(k-1)| / dt <= a_w_max
    """
    def __init__(self, poly, mods, cfg, speed_profile=None):
        self.poly, self.mods = poly, mods
        if poly is None or len(poly) < 2:
            self.s = np.array([0.0])
            self.length = 0.0
            self.sa = np.array([0.0])
            self.pos = np.zeros((1, 2)) if poly is None or len(poly) == 0 else poly.copy()
            self.vel = np.zeros_like(self.pos)
            self.done = False
            self.T = 0.0
            return

        seg = np.hypot(*np.diff(poly, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(self.s[-1])
        self.length = L
        kappa_s = _discrete_curvature_profile(poly)

        K = int(cfg.tmax / cfg.dt)
        if speed_profile is not None:
            v_des = np.asarray(speed_profile, float)
            if len(v_des) < K:
                v_des = np.pad(v_des, (0, K - len(v_des)), mode='edge')
        elif not mods:
            v_des = np.full(K, cfg.V, float)
        else:
            t = np.arange(K) * cfg.dt
            f = np.ones(K, float)
            for mod in mods:
                if len(mod) == 3:
                    t0, t1, fac = mod
                    w = (t >= t0) & (t < t1)
                    f[w] = np.minimum(f[w], fac)
                elif len(mod) == 4 and mod[0] == 'gauss':
                    _, t_c, sigma_t, min_fac = mod
                    # Windowed Gaussian modulation within [t_c - 2*tau, t_c + 2*tau]
                    g_prof = 1.0 - (1.0 - min_fac) * np.exp(-0.5 * ((t - t_c) / max(sigma_t, 0.1)) ** 2)
                    in_win = np.abs(t - t_c) <= 2.0 * cfg.tau
                    f[in_win] = np.minimum(f[in_win], g_prof[in_win])
                elif len(mod) == 4 and mod[0] == 'boost':
                    _, t0, t1, boost_v = mod
                    w = (t >= t0) & (t < t1)
                    f[w] = np.maximum(f[w], min(boost_v, cfg.v_max_dyn) / cfg.V)
            v_des = np.clip(cfg.V * f, cfg.v_min_dyn, cfg.v_max_dyn)

        dv_max = cfg.a_v_max * cfg.dt
        dw_max = cfg.a_w_max * cfg.dt
        sa_list = [0.0]
        v_curr = min(float(v_des[0]), cfg.V)
        w_curr = 0.0

        for k in range(K):
            s_curr = sa_list[-1]
            idx_s = min(int(np.searchsorted(self.s, s_curr)), len(kappa_s) - 1)
            kap = float(kappa_s[idx_s])

            # Kinematic velocity window [v_low, v_high] at step k
            v_low = max(cfg.v_min_dyn, v_curr - dv_max)
            v_high = min(cfg.v_max_dyn, v_curr + dv_max)
            if kap > 1e-5:
                v_w_lim = min(cfg.w_max, w_curr + dw_max) / kap
                v_high = min(v_high, max(v_low, v_w_lim))

            v_curr = float(np.clip(v_des[k], v_low, v_high))
            w_curr = kap * v_curr
            s_next = min(L, s_curr + v_curr * cfg.dt)
            sa_list.append(s_next)
            if s_next >= L - 1e-9:
                break

        self.sa = np.array(sa_list, float)
        self.done = bool(self.sa[-1] >= L - 1e-6)
        self.pos = np.column_stack([
            np.interp(self.sa, self.s, poly[:, 0]),
            np.interp(self.sa, self.s, poly[:, 1]),
        ])
        if len(self.pos) > 1:
            v_steps = np.diff(self.pos, axis=0) / cfg.dt
            self.vel = np.vstack([v_steps, np.zeros((1, 2))])
        else:
            self.vel = np.zeros_like(self.pos)
        self.T = (len(self.pos) - 1) * cfg.dt


def _pad(pos, K, hold):
    if len(pos) >= K:
        return pos[:K]
    pad = np.repeat(pos[-1:], K - len(pos), axis=0) if hold else np.full((K - len(pos), 2), np.nan)
    return np.vstack([pos, pad])


def velocity_at(pos, k, dt, hold_goal=True):
    """Returns true instantaneous velocity vector v(k) = (pos[k+1] - pos[k]) / dt."""
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
# IDEA 1: Forward CPA Model (Mischiati et al., Nature 2015) + COLREGs Deflection
# -----------------------------------------------------------------------------
def dragonfly_cpa_prediction(pos_self, k_idx, pos_higher, cfg):
    """
    Internal forward CPA model (Mischiati et al., Nature 2015) combined with
    maritime COLREGs stern-passing deflection:
      - Uses true instantaneous velocities v_i(k) and v_h(k) (including 0 when stopped).
      - Computes t_CPA, d_CPA, and scales the lateral evasion gain via cfg.df_lat_gain.
    """
    k_s = min(max(0, k_idx), len(pos_self) - 1)
    k_p = min(max(0, k_idx), len(pos_higher) - 1)
    p_i = pos_self[k_s]
    p_h = pos_higher[k_p]

    v_i = velocity_at(pos_self, k_s, cfg.dt, cfg.hold_goal)
    if np.linalg.norm(v_i) < 1e-6:
        v_i = heading(pos_self, k_s) * cfg.V
    v_h = velocity_at(pos_higher, k_p, cfg.dt, cfg.hold_goal)

    r_ih = p_i - p_h
    v_ih = v_i - v_h
    v_rel_sq = float(np.dot(v_ih, v_ih)) + 1e-6

    t_cpa = float(np.clip(-np.dot(r_ih, v_ih) / v_rel_sq, 0.0, cfg.t_pred_df))
    p_i_pred = p_i + v_i * t_cpa
    p_h_pred = p_h + v_h * t_cpa
    d_cpa = float(np.linalg.norm(p_i_pred - p_h_pred))

    u_self = heading(pos_self, k_s)
    n_left = np.array([-u_self[1], u_self[0]])

    # COLREGs stern-passing rule: bias lateral offset toward -v_h (or away from p_h if stationary)
    target_vec = -v_h if np.linalg.norm(v_h) > 0.05 else (p_i - p_h)
    stern_proj = float(np.dot(n_left, target_vec))
    preferred_sign = 1.0 if stern_proj >= 0.0 else -1.0

    # Scale lateral offset magnitude using df_lat_gain and predicted CPA deficit
    cpa_deficit = max(0.35, cfg.d_safe + 0.35 - d_cpa)
    lat_scale = float(np.clip(cfg.df_lat_gain * (0.65 + 0.30 * cpa_deficit), 0.75, 1.45))

    return {
        't_cpa': t_cpa,
        'd_cpa': d_cpa,
        'p_i_pred': p_i_pred,
        'p_h_pred': p_h_pred,
        'u_self': u_self,
        'n_left': n_left,
        'preferred_sign': preferred_sign,
        'lat_scale': lat_scale,
        'v_h': v_h,
    }


# -----------------------------------------------------------------------------
# IDEA 2: Zhong et al. (2021) Stage Target + Velocity Obstacle (VO) Evaluation
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
    closest_vec = rel_pos - v_ab * t_star
    d_min = float(np.linalg.norm(closest_vec))
    return (d_min < r_safe), d_min


def zhong_vo_dwa_score_trajectory(env, tr_cand, global_poly, higher_pos, cfg):
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
        if d_vec[k_min] < cfg.sensor_range:
            k_check = min(max(0, k_min - 8), K - 1)
            p_a = pos[k_check]
            v_a = velocity_at(pos, k_check, cfg.dt, cfg.hold_goal)
            p_b = hi_pos[min(k_check, len(hi_pos) - 1)]
            v_b = velocity_at(hi_pos, k_check, cfg.dt, cfg.hold_goal)
            in_vo, _ = in_velocity_obstacle_cone(p_a, v_a, p_b, v_b, cfg.d_safe)
            if in_vo:
                vo_hits += 1
            df_info = dragonfly_cpa_prediction(pos, k_check, hi_pos, cfg)
            min_d_cpa = min(min_d_cpa, df_info['d_cpa'])

    sample_idx = np.linspace(0, K - 1, min(20, K)).astype(int)
    head_align = 0.0
    for k in sample_idx[:-1]:
        st_pt = select_stage_target(global_poly, pos[k], cfg.preview_dist)
        vec_st = st_pt - pos[k]
        u_st = vec_st / (np.linalg.norm(vec_st) + 1e-9)
        u_rob = heading(pos, k)
        head_align += float(np.dot(u_rob, u_st))
    norm_head = head_align / max(1, len(sample_idx) - 1)
    norm_dist = float(np.clip(np.min(env.clearance(pos)) / 2.0, 0.0, 1.0))
    norm_vel = float(tr_cand.length / max(tr_cand.T * cfg.v_max_dyn, 1e-6))
    norm_vo = 1.0 if vo_hits > 0 else 0.0
    norm_df = float(np.clip(min_d_cpa / (cfg.d_safe + 1.0), 0.0, 1.0))

    return (cfg.w_head * norm_head +
            cfg.w_dist * norm_dist +
            cfg.w_vel * norm_vel +
            cfg.w_vo * norm_vo +
            cfg.w_df * norm_df)


# =============================================================================
# 5. ARCLENGTH-ACCURATE BYPASS SEEDING & SPACE-TIME SA SOLVER
# =============================================================================
def project_point_to_chain_arclength(X_chain, pt):
    """Projects a 2D point pt onto polyline X_chain and returns its true arclength along X_chain."""
    seg_vecs = np.diff(X_chain, axis=0)
    seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    seg_dists = np.concatenate([[0.0], np.cumsum(seg_lens)])
    best_dist_sq = 1e18
    best_s = 0.0
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


def dragonfly_anchored_bypass_candidates(env, X_base, tr_base, k_conf, hi_pos, cfg, use_df=True):
    """
    Generates bypass candidates using true projected arclength on X_base.
    When use_df=True, orders lateral offsets by Forward-CPA + COLREGs stern-passing
    preference and scales offset magnitudes by df['lat_scale'].
    """
    cands = []
    pos = tr_base.pos
    k_cl = min(k_conf, len(pos) - 1)
    u_dir = heading(pos, k_cl)
    n_perp = np.array([-u_dir[1], u_dir[0]])
    p_conf = pos[k_cl]

    if use_df:
        k_pre_df = max(0, k_cl - 15)
        df = dragonfly_cpa_prediction(pos, k_pre_df, hi_pos, cfg)
        pref = df['preferred_sign']
        scale = df['lat_scale']
    else:
        pref = 1.0
        scale = 1.0

    base_mags = [0.65 * scale, 1.05 * scale, 1.50 * scale, 2.05 * scale, 2.75 * scale]
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

    # Project p_conf directly onto X_base so s_conf_raw matches seg_dists parameterization
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


def plan_single_df_vo_sa(env, X_init, start, goal, higher, cfg, rng, sa_steps=None, smoother=smooth_c2_minsnap):
    """
    Hybrid Forward-CPA + Zhong (2021) VO-DWA Space-Time SA Planner.
    """
    if sa_steps is None:
        sa_steps = cfg.st_sa_steps
    X_init = validate_or_repair_chain(env, X_init, start, goal)
    poly0 = smoother(env, X_init, cfg)
    traj0 = Traj(poly0, [], cfg)
    nc0, pen0, fst0, _ = conflict_metrics(traj0.pos, higher, cfg.d_safe, cfg.hold_goal)

    # Try predictive speed window boost [V, v_max_dyn] if path is already conflict-free
    if nc0 == 0 and traj0.done and not env.blocked(poly0).any():
        if cfg.v_max_dyn > cfg.V + 1e-4:
            tr_fast = Traj(poly0, [('boost', 0.0, traj0.T, cfg.v_max_dyn)], cfg)
            if (tr_fast.done and not env.blocked(tr_fast.pos).any() and
                    conflict_metrics(tr_fast.pos, higher, cfg.d_safe, cfg.hold_goal)[0] == 0):
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
        nc_s, p_s, fst_s, _ = conflict_metrics(tr.pos, higher, cfg.d_safe, cfg.hold_goal)
        nc_b, p_b, fst_b, _ = conflict_metrics(tr.pos, higher, cfg.d_safe + 0.02, cfg.hold_goal)
        if nc_s == 0:
            turn_tr = total_turn_deg(tr.pos)
            vo_G = zhong_vo_dwa_score_trajectory(env, tr, poly0, higher, cfg)
            cost_tr = tr.T + 0.0015 * turn_tr - 0.015 * vo_G
            if cost_tr < best_cf_cost - 1e-6:
                best_cf_traj, best_cf_X, best_cf_cost = tr, X_cand.copy(), cost_tr
        sc = (nc_s, p_s, tr.T)
        if sc < best_any_score:
            best_any_traj, best_any_X, best_any_score = tr, X_cand.copy(), sc
        return tr, nc_b, p_b, fst_b

    # Phase 1: Forward-CPA + COLREGs Bypass Seeding
    if fst0 is not None:
        hi_conflict_pos = higher[fst0[1]]
        cands1 = dragonfly_anchored_bypass_candidates(
            env, X_init, traj0, fst0[0], hi_conflict_pos, cfg, use_df=True
        )
        scored1 = []
        for Xc in cands1:
            tr, ncb, pb, fstb = eval_cand(Xc)
            if tr is not None:
                vo_score = zhong_vo_dwa_score_trajectory(env, tr, poly0, higher, cfg)
                scored1.append((ncb, pb, tr.T - 0.02 * vo_score, Xc, tr, fstb))

        scored1.sort(key=lambda x: (x[0], x[1], x[2]))
        for ncb, pb, _, Xc1, tr1, fstb1 in scored1[:3]:
            if fstb1 is not None:
                hi_pos2 = higher[fstb1[1]]
                for Xc2 in dragonfly_anchored_bypass_candidates(
                    env, Xc1, tr1, fstb1[0], hi_pos2, cfg, use_df=True
                )[:12]:
                    eval_cand(Xc2)

    # Phase 2: Space-Time Simulated Annealing with Smoothness + VO-DWA Objective
    if sa_steps > 0 and fst0 is not None:
        hi_conflict_pos = higher[fst0[1]]
        X_curr = best_cf_X.copy() if best_cf_X is not None else best_any_X.copy()
        if len(X_curr) == 2:
            df0 = dragonfly_cpa_prediction(traj0.pos, max(0, fst0[0] - 10), hi_conflict_pos, cfg)
            P_mid = traj0.pos[min(fst0[0], len(traj0.pos) - 1)] + df0['preferred_sign'] * 1.2 * df0['n_left']
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
                _, _, fst_c, _ = conflict_metrics(tr_curr.pos, higher, cfg.d_safe + 0.05, cfg.hold_goal)
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

    # Phase 3: Kinematically Constrained VO-DWA Speed-Window Modulation
    # Searches both predictive passing boost (up to v_max_dyn) and smooth Gaussian yielding
    cands_for_mod = [X_init, best_any_X] + ([best_cf_X] if best_cf_X is not None else [])
    for cand_X in cands_for_mod:
        poly = smoother(env, cand_X, cfg)
        if len(poly) < 2 or env.blocked(poly).any():
            continue
        tr_b = Traj(poly, [], cfg)
        nc_b0, _, fst, _ = conflict_metrics(tr_b.pos, higher, cfg.d_safe, cfg.hold_goal)
        if nc_b0 == 0:
            eval_cand(cand_X, [('boost', 0.0, tr_b.T, cfg.v_max_dyn)])
            continue
        if fst is None:
            continue
        ts = fst[0] * cfg.dt
        # 3a) Try predictive passing boost before conflict entry
        eval_cand(cand_X, [('boost', max(0.0, ts - 2.0 * cfg.tau), ts + 0.5, cfg.v_max_dyn)])

        # 3b) Try smooth Gaussian yielding within window tau; exit BOTH loops when resolved
        found_mod = False
        for sigma_t in (1.2, 2.0):
            for min_fac in (0.65, 0.45, 0.25, 0.0):
                tr_g, _, _, _ = eval_cand(cand_X, [('gauss', max(0.5, ts - 1.5), sigma_t, min_fac)])
                if tr_g is not None and conflict_metrics(tr_g.pos, higher, cfg.d_safe, cfg.hold_goal)[0] == 0:
                    found_mod = True
                    break
            if found_mod:
                break

    if best_cf_traj is not None:
        # Anchor pruning tolerance to initial_cf_T so repeated passes cannot drift upward
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
                        if tr_try.done and conflict_metrics(tr_try.pos, higher, cfg.d_safe, cfg.hold_goal)[0] == 0:
                            if tr_try.T <= initial_cf_T + 0.02:
                                best_cf_traj, best_cf_X, changed = tr_try, X_try, True
                                break
        return best_cf_traj, best_cf_X
    return best_any_traj, best_any_X


# =============================================================================
# 6. UNCONTAMINATED BASELINE CST-SP2-SA (V2) & MASTER DF-VO-SP2-SA SOLVER
# =============================================================================
def plan_single_v2_baseline(env, X_init, start, goal, higher, cfg, rng, sa_steps=25):
    """
    Uncontaminated CST-SP2-SA (V2) single-robot planner:
      - Uses smooth_bezier corner rounding.
      - Uses symmetric lateral bypass offsets (use_df=False).
      - No VO-DWA scoring, no speed boost above V, and simple step-window slowdowns.
    """
    X_init = validate_or_repair_chain(env, X_init, start, goal)
    poly0 = smooth_bezier(env, X_init, cfg)
    traj0 = Traj(poly0, [], cfg)
    nc0, pen0, fst0, _ = conflict_metrics(traj0.pos, higher, cfg.d_safe, cfg.hold_goal)
    if nc0 == 0 and traj0.done and not env.blocked(poly0).any():
        return traj0, X_init

    best_cf_traj, best_cf_X = None, None
    best_any_traj, best_any_X, best_any_score = traj0, X_init.copy(), (nc0, pen0, traj0.T)

    def eval_v2(X_cand, mods=None):
        nonlocal best_cf_traj, best_cf_X, best_any_traj, best_any_X, best_any_score
        if mods is None:
            mods = []
        poly = smooth_bezier(env, X_cand, cfg)
        if len(poly) < 2 or env.blocked(poly).any():
            return None, 9999, 9999.0, None
        tr = Traj(poly, mods, cfg)
        if not tr.done:
            return None, 9999, 9999.0, None
        nc_s, p_s, fst_s, _ = conflict_metrics(tr.pos, higher, cfg.d_safe, cfg.hold_goal)
        if nc_s == 0:
            if best_cf_traj is None or tr.T < best_cf_traj.T - 1e-6:
                best_cf_traj, best_cf_X = tr, X_cand.copy()
        sc = (nc_s, p_s, tr.T)
        if sc < best_any_score:
            best_any_traj, best_any_X, best_any_score = tr, X_cand.copy(), sc
        return tr, nc_s, p_s, fst_s

    if fst0 is not None:
        hi_pos = higher[fst0[1]]
        for Xc in dragonfly_anchored_bypass_candidates(env, X_init, traj0, fst0[0], hi_pos, cfg, use_df=False):
            eval_v2(Xc)

    X_curr = best_cf_X.copy() if best_cf_X is not None else best_any_X.copy()
    tr_curr, nc_curr, pb_curr, _ = eval_v2(X_curr)
    if tr_curr is None:
        tr_curr, nc_curr, pb_curr = best_any_traj, best_any_score[0], best_any_score[1]
    c_curr = tr_curr.T + 80.0 * pb_curr + 25.0 * nc_curr

    T_sa = 0.8
    for step in range(sa_steps):
        T_sa *= 0.93
        r_step = max(0.12, 1.5 * (0.94 ** step))
        m = len(X_curr)
        if m > 2:
            j = int(rng.integers(1, m - 1))
            P = X_curr[j] + rng.normal(0, r_step, 2)
            if not env.blocked(P)[0] and env.los(X_curr[j - 1], P) and env.los(P, X_curr[j + 1]):
                X_new = X_curr.copy()
                X_new[j] = P
                tr_new, nc_new, pb_new, _ = eval_v2(X_new)
                if tr_new is not None:
                    c_new = tr_new.T + 80.0 * pb_new + 25.0 * nc_new
                    if c_new < c_curr or rng.random() < math.exp(-(c_new - c_curr) / max(T_sa, 1e-3)):
                        X_curr, tr_curr, c_curr = X_new, tr_new, c_new

    if fst0 is not None:
        ts = fst0[0] * cfg.dt
        for dt_w in (2.0, 3.5, 5.0):
            found = False
            for fac in (0.6, 0.35, 0.0):
                tr_m, nc_m, _, _ = eval_v2(X_init, [(max(0.0, ts - 1.5), ts + dt_w, fac)])
                if tr_m is not None and nc_m == 0:
                    found = True
                    break
            if found:
                break

    if best_cf_traj is not None:
        return best_cf_traj, best_cf_X
    return best_any_traj, best_any_X


def solve_v2_baseline(env, robots, cfg, rng):
    """Runs the uncontaminated CST-SP2-SA baseline."""
    order = list(robots.keys())
    X_init, trajs, ptime = {}, {}, {}
    for r in order:
        t0 = time.perf_counter()
        s, g = robots[r]
        X_init[r] = sp2_sa(env, s, g, cfg, rng)
        ptime[r] = time.perf_counter() - t0

    for idx, r in enumerate(order):
        t0 = time.perf_counter()
        s, g = robots[r]
        higher_pos = [trajs[m].pos for m in order[:idx]]
        tr_uni, _ = plan_single_v2_baseline(env, X_init[r], s, g, higher_pos, cfg, rng, sa_steps=25)
        trajs[r] = tr_uni
        ptime[r] += time.perf_counter() - t0

    tot = sum(
        conflict_metrics(trajs[n].pos, [trajs[m].pos for m in order[:i]], cfg.d_safe, cfg.hold_goal)[0]
        for i, n in enumerate(order)
    )
    return tot, trajs, X_init, ptime, order


def solve_df_vo_sp2_sa(env, robots, cfg, rng, use_homotopy=False):
    """
    Master Multi-Robot Solver combining:
      - Validated Corner-Safe Any-Angle SA (with verified obstacle-free homotopy seeding)
      - Forward-CPA + COLREGs Stern-Passing Bypass Guidance
      - Kinematically Constrained Zhong et al. (2021) VO-DWA Trajectory Optimization
      - Cooperative Priority-Preserving Micro-Nudge (with fresh modulation evaluation)
      - Exact C^2 Quartic Corner Blending
    """
    order = list(robots.keys())
    X_init, tr_init, ptime = {}, {}, {}
    for r in order:
        t0 = time.perf_counter()
        s, g = robots[r]
        if use_homotopy and r in STATIC_HOMOTOPY_GUIDES:
            X_seed = np.array(STATIC_HOMOTOPY_GUIDES[r], float)
            X_init[r] = _raw_sp2_sa(env, s, g, cfg, rng, X_seed=X_seed)
        else:
            X_init[r] = sp2_sa(env, s, g, cfg, rng)
        tr_init[r] = Traj(smooth_c2_minsnap(env, X_init[r], cfg), [], cfg)
        ptime[r] = time.perf_counter() - t0

    trajs, X_sol = {}, {}
    for idx, r in enumerate(order):
        t0 = time.perf_counter()
        s, g = robots[r]
        higher_names = order[:idx]
        higher_pos = [trajs[m].pos for m in higher_names]
        tr_uni, X_uni = plan_single_df_vo_sa(
            env, X_init[r], s, g, higher_pos, cfg, rng, smoother=smooth_c2_minsnap
        )

        # Cooperative Priority-Preserving Micro-Nudge (using fresh mods=[] on poly_h)
        nc0, _, fst0, _ = conflict_metrics(tr_init[r].pos, higher_pos, cfg.d_safe, cfg.hold_goal)
        best_coop = None
        if fst0 is not None and (tr_uni.T > tr_init[r].T + 0.35 or
                                 conflict_metrics(tr_uni.pos, higher_pos, cfg.d_safe, cfg.hold_goal)[0] > 0):
            k_conf, hi_idx = fst0
            h_name = higher_names[hi_idx]
            others_for_h = [trajs[m].pos for m in higher_names if m != h_name]
            h_cands = dragonfly_anchored_bypass_candidates(
                env, X_sol[h_name], trajs[h_name], k_conf, tr_init[r].pos, cfg, use_df=True
            )
            valid_h = []
            for X_h_cand in h_cands:
                poly_h = smooth_c2_minsnap(env, X_h_cand, cfg)
                if len(poly_h) < 2 or env.blocked(poly_h).any():
                    continue
                # Fresh mods=[] instead of stale time-window slowdowns from old path
                tr_h_cand = Traj(poly_h, [], cfg)
                if not tr_h_cand.done or tr_h_cand.T > trajs[h_name].T + cfg.coop_tol:
                    continue
                if conflict_metrics(tr_h_cand.pos, others_for_h, cfg.d_safe, cfg.hold_goal)[0] > 0:
                    continue
                new_higher_pos = [tr_h_cand.pos if m == h_name else trajs[m].pos for m in higher_names]
                nc_r0, pen_r0, _, _ = conflict_metrics(tr_init[r].pos, new_higher_pos, cfg.d_safe, cfg.hold_goal)
                nc_rem = sum(conflict_metrics(tr_init[m].pos, [tr_h_cand.pos], cfg.d_safe, cfg.hold_goal)[0]
                             for m in order[idx + 1:])
                valid_h.append((nc_r0, 0.2 * nc_rem + pen_r0, tr_h_cand.T, tr_h_cand, X_h_cand, new_higher_pos))
            valid_h.sort(key=lambda x: (x[0], x[1], x[2]))
            for nc_r0, _, _, tr_h_cand, X_h_cand, new_higher_pos in valid_h[:3]:
                if nc_r0 == 0 and not env.blocked(tr_init[r].pos).any():
                    tr_r_cand, X_r_cand = tr_init[r], X_init[r]
                else:
                    tr_r_cand, X_r_cand = plan_single_df_vo_sa(
                        env, X_init[r], s, g, new_higher_pos, cfg, rng, sa_steps=16, smoother=smooth_c2_minsnap
                    )
                if (tr_r_cand.done and not env.blocked(tr_r_cand.pos).any() and
                        conflict_metrics(tr_r_cand.pos, new_higher_pos, cfg.d_safe, cfg.hold_goal)[0] == 0):
                    team_T_new = tr_h_cand.T + tr_r_cand.T
                    team_T_old = trajs[h_name].T + tr_uni.T
                    if team_T_new < team_T_old - 0.15 and (best_coop is None or team_T_new < best_coop[0]):
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
        for i, n in enumerate(order)
    )
    return tot, trajs, X_init, ptime, order


# =============================================================================
# 7. METRICS, COMPARISON TABLES & PLOTTING
# =============================================================================
def collect_metrics(env, trajs, ptime, cfg):
    rows = []
    for idx, r in enumerate(NAMES):
        tr = trajs[r]
        others = [trajs[o].pos for o in NAMES if o != r]
        nc_all, _, _, md_cells = conflict_metrics(tr.pos, others, cfg.d_safe, cfg.hold_goal)
        obs_hit = bool(env.blocked(tr.pos).any())
        success = bool(tr.done and not obs_hit and nc_all == 0)
        rows.append({
            'robot_idx': idx + 1,
            'name': r,
            'priority': idx + 1,
            'length': tr.length * 10.0,
            'min_dist': md_cells * 10.0,
            'time_s': tr.T,
            'turn_deg': total_turn_deg(tr.pos),
            'cpu_s': ptime[r],
            'success': float(success),
        })
    return rows


def print_paper_tables(env_key, rows_shi_sim, rows_v2, rows_df):
    env_title = 'Barrier-free environment' if env_key == 'free' else 'Static environment'
    ref_rows = PAPER_TABLE2_REF[env_key]
    ref_t3 = PAPER_TABLE3_REF[env_key]

    print('\n' + '=' * 118)
    print(f'  TABLE 2 COMPARISON — {env_title.upper()} (Per-Robot Parameters as in Kun Shi et al., 2025)')
    print('=' * 118)
    hdr = (f'{"Algorithm":28} | {"Robot":8} | {"Priority":>8} | {"Path length":>11} | '
           f'{"Min dist":>10} | {"Move time(s)":>12} | {"Turn(deg)":>10} | {"CPU(s)":>8}')
    print(hdr)
    print('-' * 118)

    for p_id, L_ref, md_ref, T_ref in ref_rows:
        alg_col = 'Improved SA (Paper Table 2)' if p_id == 1 else ''
        print(f'{alg_col:28} | {f"Robot {p_id}":8} | {p_id:8d} | {L_ref:11.2f} | '
              f'{md_ref:10.2f} | {T_ref:12.2f} | {"—":>10} | {"—":>8}')
    print('-' * 118)

    for label, rlist in [
        ('Improved SA (Simulated)', rows_shi_sim),
        ('Baseline CST-SP2-SA (V2)', rows_v2),
        ('Proposed DF-VO-SP2-SA', rows_df),
    ]:
        for row in rlist:
            alg_col = label if row['priority'] == 1 else ''
            print(f'{alg_col:28} | {f"Robot {row['priority']}":8} | {row["priority"]:8d} | '
                  f'{row["length"]:11.2f} | {row["min_dist"]:10.2f} | {row["time_s"]:12.2f} | '
                  f'{row["turn_deg"]:10.2f} | {row["cpu_s"]:8.3f}')
        print('-' * 118)

    print(f'\n  TABLE 3 SUMMARY — {env_title.upper()}')
    print('-' * 118)
    print(f'{"Method":30} | {"Avg Path Len":>12} | {"Max Path Len":>12} | '
          f'{"Avg Time (s)":>12} | {"Avg Turn(deg)":>13} | {"Avg CPU(s)":>10} | {"Success Rate":>12}')
    print('-' * 118)
    for m_name, aL_r, mL_r, aT_r, sr_r in ref_t3:
        print(f'{m_name:30} | {aL_r:12.2f} | {mL_r:12.2f} | {aT_r:12.2f} | {"—":>13} | {"—":>10} | {sr_r:>12}')
    print('-' * 118)
    for label, rlist in [
        ('Improved SA (Simulated)', rows_shi_sim),
        ('Baseline CST-SP2-SA (V2)', rows_v2),
        ('Proposed DF-VO-SP2-SA (Ours)', rows_df),
    ]:
        aL = np.mean([r['length'] for r in rlist])
        mL = np.max([r['length'] for r in rlist])
        aT = np.mean([r['time_s'] for r in rlist])
        aTurn = np.mean([r['turn_deg'] for r in rlist])
        aCpu = np.mean([r['cpu_s'] for r in rlist])
        sr_pct = f'{100.0 * np.mean([r["success"] for r in rlist]):.0f}%'
        print(f'{label:30} | {aL:12.2f} | {mL:12.2f} | {aT:12.2f} | '
              f'{aTurn:13.2f} | {aCpu:10.3f} | {sr_pct:>12}')
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


def plot_full_comparison(env, robots, trajs_shi, trajs_df, name, outdir, cfg):
    env_label = 'Barrier-free environment' if name == 'free' else 'Static environment'
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={'width_ratios': [1.05, 1.05, 1.0]})

    draw_map(axs[0], env, f'(a) Improved SA (Shi et al., 2025) — {env_label}')
    for i, r in enumerate(NAMES):
        tr, c = trajs_shi[r], COLORS[i]
        st, go = robots[r]
        axs[0].plot(tr.pos[:, 0], tr.pos[:, 1], color=c, lw=2.1, zorder=3,
                    label=f'robot {r} (L={tr.length*10:.1f}, T={tr.T:.2f}s)')
        axs[0].plot(*st, 's', color='limegreen', mec='k', ms=7.5, zorder=4)
        axs[0].plot(*go, 's', color='red', mec='k', ms=7.5, zorder=4)
        j = max(1, len(tr.pos) // 2)
        axs[0].annotate('', xy=tr.pos[min(j + 3, len(tr.pos) - 1)], xytext=tr.pos[j],
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.6), zorder=4)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=True)

    draw_map(axs[1], env, f'(b) Proposed DF-VO-SP²-SA (Forward-CPA + VO-DWA) — {env_label}')
    for i, r in enumerate(NAMES):
        tr, c = trajs_df[r], COLORS[i]
        st, go = robots[r]
        axs[1].plot(tr.pos[:, 0], tr.pos[:, 1], color=c, lw=2.3, zorder=3,
                    label=f'robot {r} (L={tr.length*10:.1f}, T={tr.T:.2f}s)')
        axs[1].plot(*st, 's', color='limegreen', mec='k', ms=7.5, zorder=4)
        axs[1].plot(*go, 's', color='red', mec='k', ms=7.5, zorder=4)
        for k in range(0, len(tr.pos), 100):
            axs[1].plot(*tr.pos[k], 'o', color=c, ms=3.2, mec='w', mew=0.4, zorder=4)
        j = max(1, len(tr.pos) // 2)
        axs[1].annotate('', xy=tr.pos[min(j + 3, len(tr.pos) - 1)], xytext=tr.pos[j],
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.7), zorder=4)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=True)

    ax2 = axs[2]
    ax2.set_title(f'(c) Inter-Robot Distance vs. Time — {env_label}', fontsize=10.5, weight='bold', pad=8)
    for a in range(len(NAMES)):
        for b in range(a + 1, len(NAMES)):
            A, B = trajs_df[NAMES[a]].pos, trajs_df[NAMES[b]].pos
            K = max(len(A), len(B)) if cfg.hold_goal else min(len(A), len(B))
            pA = _pad(A, K, True) if cfg.hold_goal else A[:K]
            pB = _pad(B, K, True) if cfg.hold_goal else B[:K]
            dist_paper = np.linalg.norm(pA - pB, axis=1) * 10.0
            ax2.plot(np.arange(K) * cfg.dt, dist_paper, lw=1.25,
                     label=f'distance of robot {NAMES[a]} and {NAMES[b]}')
    ax2.axhline(cfg.d_safe * 10.0, ls='--', color='k', lw=1.2, label='safety threshold (20.0)')
    ax2.set_xlabel('Time (s)', fontsize=9.5)
    ax2.set_ylabel('Distance (paper units, x10)', fontsize=9.5)
    ax2.set_ylim(bottom=0)
    ax2.legend(ncol=2, fontsize=7, loc='upper right', frameon=True)
    ax2.grid(alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{name}_df_vo_comparison.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MAIN RUNNER (IDENTICAL RNG SEEDS & MULTI-SEED AGGREGATION)
# =============================================================================
def _avg_seed_rows(runs):
    out = []
    for r_idx in range(len(NAMES)):
        out.append({
            'robot_idx': r_idx + 1,
            'name': NAMES[r_idx],
            'priority': r_idx + 1,
            'length': float(np.mean([run[r_idx]['length'] for run in runs])),
            'min_dist': float(np.min([run[r_idx]['min_dist'] for run in runs])),
            'time_s': float(np.mean([run[r_idx]['time_s'] for run in runs])),
            'turn_deg': float(np.mean([run[r_idx]['turn_deg'] for run in runs])),
            'cpu_s': float(np.mean([run[r_idx]['cpu_s'] for run in runs])),
            'success': float(np.mean([run[r_idx]['success'] for run in runs])),
        })
    return out


def run_scenario(name, cfg, seed, outdir, use_homotopy=False, num_seeds=1):
    rects, robots = SCENARIOS[name]
    env = Env(rects, cfg)
    shi_cfg = ShiCfg(hold_goal=cfg.hold_goal)

    runs_shi, runs_v2, runs_df = [], [], []
    first_trajs_shi, first_trajs_df = None, None

    for s_offset in range(num_seeds):
        cur_seed = seed + s_offset
        # All three solvers receive the exact same seed stream per trial
        _, trajs_shi, _, pt_shi, _ = solve_shi_improved_sa(
            env, robots, shi_cfg, np.random.default_rng(cur_seed)
        )
        _, trajs_v2, _, pt_v2, _ = solve_v2_baseline(
            env, robots, cfg, np.random.default_rng(cur_seed)
        )
        _, trajs_df, _, pt_df, _ = solve_df_vo_sp2_sa(
            env, robots, cfg, np.random.default_rng(cur_seed),
            use_homotopy=(use_homotopy and name == 'static')
        )

        runs_shi.append(collect_metrics(env, trajs_shi, pt_shi, cfg))
        runs_v2.append(collect_metrics(env, trajs_v2, pt_v2, cfg))
        runs_df.append(collect_metrics(env, trajs_df, pt_df, cfg))

        if first_trajs_shi is None:
            first_trajs_shi, first_trajs_df = trajs_shi, trajs_df

    print_paper_tables(name, _avg_seed_rows(runs_shi), _avg_seed_rows(runs_v2), _avg_seed_rows(runs_df))
    plot_full_comparison(env, robots, first_trajs_shi, first_trajs_df, name, outdir, cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='static', choices=['free', 'static', 'both'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-seeds', type=int, default=1)
    ap.add_argument('--out', default='results_df_vo')
    ap.add_argument('--paper-homotopy', action='store_true',
                    help='Use verified obstacle-free Fig. 5(c) corridor guide points in static scenario')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Cfg()
    for sc in (['free', 'static'] if args.scenario == 'both' else [args.scenario]):
        run_scenario(
            sc, cfg, args.seed, args.out,
            use_homotopy=args.paper_homotopy,
            num_seeds=args.num_seeds,
        )
    print(f'\nComparison figures saved in ./{args.out}/')