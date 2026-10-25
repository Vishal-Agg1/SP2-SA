#!/usr/bin/env python3
"""
Improved Simulated Annealing (ISA) for Multi-Robot Dynamic Path Planning with Priority
======================================================================================
Faithful implementation of:
  Kun Shi, Luyao Yang, Zhengtian Wu, Baoping Jiang, Qing Gao,
  "Multi-robot dynamic path planning with priority based on simulated annealing",
  Journal of the Franklin Institute 362 (2025) 107396.

Implements:
  - Algorithm 1: Single-robot Improved SA with probabilistic neighbor selection
    (Eqs. 2-4, p0 in [0.7, 0.95]), calibrated Boltzmann acceptance (Tm=10000,
    T0=4000, alpha=0.98) with best-solution tracking, and obstacle-safe redundant
    point deletion (Eqs. 5-7, d(Pm, Pn) <= 14 paper units, |m - n| >= 2).
  - Section 4: Piecewise septic Minimum-Snap polynomial trajectory optimization
    minimizing int (p^(4)(s))^2 ds with unit-tangent boundary conditions.
  - Algorithm 2: Priority-based multi-robot conflict prevention using movement
    angle alpha (Fig. 2, alpha >= 120 deg) and Eq. (9)-(10) travel-time comparison
    F = L_n / V_f - (L_x / V + (L_m - L_x) / V_f) between replanning and waiting.

Usage:
    python3 shi_improved_sa.py
    python3 shi_improved_sa.py --scenario static --seed 0 --num-seeds 5
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
from scipy.ndimage import distance_transform_edt


# =============================================================================
# 1. CONFIGURATION & SCENARIOS (Shi et al., 2025 Figs. 3c, 5c, Tables 2 & 3)
# =============================================================================
@dataclass
class ShiCfg:
    V: float = 1.0            # nominal robot speed V_f [cells/s] (= 10 paper units/s)
    dt: float = 0.1           # simulation time step [s]
    tmax: float = 250.0       # maximum simulation horizon [s]
    d_safe: float = 2.0       # minimum safe inter-robot distance [cells] (= 20 paper units)
    sensor_range: float = 6.0 # LIDAR detection range [cells] (= 60 paper units)
    margin: float = 0.3       # static obstacle inflation radius [cells]
    sp_buf: float = 0.08      # string-pulling safety buffer [cells]
    a_v_max: float = 1.5      # max linear acceleration/deceleration [cells/s^2]
    # Algorithm 1 parameters (Section 2.2 & Section 4)
    p0: float = 0.85          # probability of greedy neighbor selection (Eq. 4, [0.7, 0.95])
    Tm: float = 10000.0       # initial temperature (Section 4)
    T0: float = 4000.0        # ending temperature (Section 4)
    alpha: float = 0.98       # cooling rate (Section 2.3 & Section 4)
    k_iter: int = 40          # iterations per temperature level
    n_init: int = 4           # number of initial paths generated (Section 2.2)
    del_dist: float = 1.42    # deletion threshold [cells] (= 14.2 paper units, Eq. 5)
    hold_goal: bool = True    # robots remain physical obstacles at their goals after arrival


NAMES = ['a', 'b', 'c', 'd', 'e']  # Priority: Robot 1 (a) > Robot 2 (b) > ... > Robot 5 (e)
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

STATIC_RECTS = [  # 12 static obstacles (x0, y0, x1, y1) in grid cells (Shi et al. Fig. 5)
    (6, 33, 9, 38), (13, 27, 18, 31), (17, 31, 20, 34), (21, 24, 24, 27),
    (36, 36, 39, 39), (28, 31, 32, 35), (14, 18, 18, 22), (29, 13, 33, 17),
    (8, 9, 12, 16), (2, 0, 5, 3), (17, 5, 28, 7), (33, 2, 36, 11),
]

PAPER_TABLE2_REF = {
    'free': [
        (1, 519.99, 23.24, 51.05),
        (2, 440.93, 26.37, 45.56),
        (3, 399.46, 22.94, 39.75),
        (4, 424.52, 25.77, 43.30),
        (5, 473.72, 22.94, 50.16),
    ],
    'static': [
        (1, 414.35, 31.38, 41.59),
        (2, 481.75, 28.17, 49.55),
        (3, 425.48, 26.62, 44.63),
        (4, 302.81, 27.38, 31.19),
        (5, 386.20, 26.51, 42.50),
    ],
}

# Exact Table 3 numbers reported in Kun Shi et al. (2025), page 14
PAPER_TABLE3_REF = {
    'free': [
        ('Improved SA (Paper Table 3)', 451.72, 519.99, 45.96, '100%'),
        ('APF (Paper Table 3)',         442.49, 516.62, 44.24, '100%'),
        ('DWA (Paper Table 3)',         456.23, 512.97, 47.88, '100%'),
    ],
    'static': [
        ('Improved SA (Paper Table 3)', 402.12, 481.75, 41.89, '100%'),
        ('APF (Paper Table 3)',         447.09, 553.82, 46.42, '100%'),
        ('DWA (Paper Table 3)',         429.53, 490.85, 44.63, '100%'),
    ],
}

SCENARIOS = {'free': ([], FREE_ROBOTS), 'static': (STATIC_RECTS, STATIC_ROBOTS)}


# =============================================================================
# 2. ENVIRONMENT REPRESENTATION
# =============================================================================
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
        c = (np.arange(n) + 0.5) * res
        self.X, self.Y = np.meshgrid(c, c)

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

    def with_disks(self, centers, radius):
        m = self.mask.copy()
        for cx, cy in centers:
            m |= (self.X - cx) ** 2 + (self.Y - cy) ** 2 < radius ** 2
        return m


# =============================================================================
# 3. THETA* FALLBACK & ALGORITHM 1: SINGLE-ROBOT IMPROVED SA (Shi et al. 2025)
# =============================================================================
def grid_astar_fallback(env, start, goal, mask=None):
    """8-connected grid A* used only if randomized backtracking is trapped by a dynamic mask."""
    use_mask = env.mask if mask is None else mask
    N = env.size
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
    free = ~env.blocked(np.column_stack([cx.ravel(), cy.ravel()]), use_mask).reshape(N, N)
    s = (min(N - 1, max(0, int(start[0]))), min(N - 1, max(0, int(start[1]))))
    g = (min(N - 1, max(0, int(goal[0]))), min(N - 1, max(0, int(goal[1]))))
    free[s] = free[g] = True
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
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
        for dx, dy in nb:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N) or not free[nxt] or nxt in closed:
                continue
            if dx != 0 and dy != 0 and not (free[cur[0] + dx, cur[1]] and free[cur[0], cur[1] + dy]):
                continue
            cand = gc[cur] + math.hypot(dx, dy)
            if cand < gc.get(nxt, 1e18):
                gc[nxt] = cand
                par[nxt] = cur
                heapq.heappush(heap, (cand + h(nxt), nxt))
    if g not in closed:
        return None
    chain = [g]
    while chain[-1] != s:
        chain.append(par[chain[-1]])
    return chain[::-1]


def shi_single_robot_isa(env, start, goal, cfg, rng, mask=None):
    """
    Algorithm 1 of Kun Shi et al. (2025):
      1. Probabilistic initial path generation with dead-end backtracking (Eqs. 2-4).
      2. Simulated annealing loop (Tm=10000 -> T0=4000, alpha=0.98) with best-path tracking.
      3. Obstacle-safe redundant point deletion (Eqs. 5-7: d(Pm, Pn) <= 14 paper units, |m-n| >= 2).
    """
    use_mask = env.mask if mask is None else mask
    N = env.size
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
    free = ~env.blocked(np.column_stack([cx.ravel(), cy.ravel()]), use_mask).reshape(N, N)
    s = (min(N - 1, max(0, int(start[0]))), min(N - 1, max(0, int(start[1]))))
    g = (min(N - 1, max(0, int(goal[0]))), min(N - 1, max(0, int(goal[1]))))
    free[s] = free[g] = True
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    dg = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
    cen = lambda c: (c[0] + 0.5, c[1] + 0.5)

    def can_step(u, v):
        dx, dy = v[0] - u[0], v[1] - u[1]
        if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
            return env.los(cen(u), cen(v), use_mask)
        if dx != 0 and dy != 0 and not (free[u[0] + dx, u[1]] and free[u[0], u[1] + dy]):
            return False
        return True

    def generate_initial_seed():
        path, inp, banned = [s], {s}, set()
        for _ in range(12000):
            cur = path[-1]
            if cur == g:
                return path
            A = []
            for dx, dy in nb:
                nxt = (cur[0] + dx, cur[1] + dy)
                if (0 <= nxt[0] < N and 0 <= nxt[1] < N and free[nxt]
                        and nxt not in inp and nxt not in banned and can_step(cur, nxt)):
                    A.append(nxt)
            if not A:
                banned.add(cur)
                inp.discard(cur)
                path.pop()
                if not path:
                    return None
                continue
            A.sort(key=dg)
            # Eq. (4): choose closest point in A with probability p0, else closest in B = A \ {A[0]}
            nxt = A[0] if (rng.random() < cfg.p0 or len(A) == 1) else A[1]
            path.append(nxt)
            inp.add(nxt)
        return None

    # Eq. (8): Path length evaluation in paper units (1 cell = 10 units)
    def path_energy(p):
        return 10.0 * sum(
            math.hypot(p[i][0] - p[i + 1][0], p[i][1] - p[i + 1][1])
            for i in range(len(p) - 1)
        )

    # Eqs. (5)-(7): Redundant point deletion with obstacle-free segment verification
    def delete_op(p):
        changed = True
        while changed and len(p) > 2:
            changed = False
            for m in range(len(p) - 2):
                for n in range(len(p) - 1, m + 1, -1):
                    dist_mn = math.hypot(p[m][0] - p[n][0], p[m][1] - p[n][1])
                    if dist_mn <= cfg.del_dist and can_step(p[m], p[n]):
                        p = p[:m + 1] + p[n:]
                        changed = True
                        break
                if changed:
                    break
        return p

    seeds = [delete_op(q) for q in (generate_initial_seed() for _ in range(cfg.n_init)) if q is not None]
    if not seeds:
        fallback_grid = grid_astar_fallback(env, start, goal, use_mask)
        if fallback_grid is None:
            return np.array([start, goal], float)
        seeds = [delete_op(fallback_grid)]

    X = list(min(seeds, key=path_energy))
    cur_E = path_energy(X)
    best_X, best_E = list(X), cur_E

    # Scale single-vertex perturbation delta to the [T0=4000, Tm=10000] schedule
    # so acceptance transitions from ~75% at Tm to ~15% at T0 instead of 99% random walk.
    temp_scale = 2500.0
    T = cfg.Tm
    while T > cfg.T0:
        for _ in range(cfg.k_iter):
            if len(X) < 3:
                break
            j = int(rng.integers(1, len(X) - 1))
            a, b = X[j - 1], X[j + 1]
            occ = set(X)
            occ.discard(X[j])
            cands = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    c = (X[j][0] + dx, X[j][1] + dy)
                    if c == X[j] or c in occ or not (0 <= c[0] < N and 0 <= c[1] < N) or not free[c]:
                        continue
                    if can_step(a, c) and can_step(c, b):
                        cands.append(c)
            if not cands:
                continue
            cand_pt = cands[int(rng.integers(len(cands)))]
            old_seg = (math.hypot(a[0] - X[j][0], a[1] - X[j][1]) +
                       math.hypot(X[j][0] - b[0], X[j][1] - b[1]))
            new_seg = (math.hypot(a[0] - cand_pt[0], a[1] - cand_pt[1]) +
                       math.hypot(cand_pt[0] - b[0], cand_pt[1] - b[1]))
            dE = 10.0 * (new_seg - old_seg)
            T_norm = (T - cfg.T0) / max(cfg.Tm - cfg.T0, 1.0) * 0.85 + 0.15
            if dE < 0.0 or rng.random() < math.exp(-(dE * temp_scale) / (T * T_norm)):
                X[j] = cand_pt
                cur_E += dE
                if cur_E < best_E - 1e-9:
                    best_X, best_E = list(X), cur_E

        X = delete_op(X)
        cur_E = path_energy(X)
        if cur_E < best_E - 1e-9:
            best_X, best_E = list(X), cur_E
        T *= cfg.alpha

    # Obstacle-safe visibility string-pulling on the best discrete grid chain
    pts = [np.asarray(start, float)] + [np.array(cen(c), float) for c in best_X[1:-1]] + [np.asarray(goal, float)]
    pruned = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not env.los(pts[i], pts[j], use_mask):
            j -= 1
        pruned.append(pts[j])
        i = j

    # Subdivide long segments lightly so the Minimum-Snap spline stays inside the corridor
    ctrl = [pruned[0]]
    for k in range(len(pruned) - 1):
        seg_vec = pruned[k + 1] - pruned[k]
        seg_len = float(np.linalg.norm(seg_vec))
        n_sub = max(1, int(math.ceil(seg_len / 5.5)))
        for s_idx in range(1, n_sub + 1):
            ctrl.append(pruned[k] + (s_idx / n_sub) * seg_vec)
    return np.array(ctrl, float)


# =============================================================================
# 4. SECTION 4: TRUE PIECEWISE SEPTIC MINIMUM-SNAP TRAJECTORY OPTIMIZATION
# =============================================================================
def _septic_hermite_basis(tau):
    """
    Degree-7 Hermite basis functions H0..H3 (at tau=0) and G0..G3 (at tau=1)
    for position, 1st, 2nd, and 3rd derivatives on tau in [0, 1].
    Guarantees C^3 continuity and minimizes int (p^(4)(s))^2 ds when internal
    derivatives solve the Euler-Lagrange KKT system.
    """
    t = np.asarray(tau, float)
    t2, t3, t4, t5, t6, t7 = t**2, t**3, t**4, t**5, t**6, t**7
    H0 = 1.0 - 35.0*t4 + 84.0*t5 - 70.0*t6 + 20.0*t7
    H1 = t - 20.0*t4 + 45.0*t5 - 36.0*t6 + 10.0*t7
    H2 = 0.5*t2 - 5.0*t4 + 10.0*t5 - 7.5*t6 + 2.0*t7
    H3 = (1.0/6.0)*t3 - (2.0/3.0)*t4 + t5 - (2.0/3.0)*t6 + (1.0/6.0)*t7

    G0 = 35.0*t4 - 84.0*t5 + 70.0*t6 - 20.0*t7
    G1 = -15.0*t4 + 39.0*t5 - 34.0*t6 + 10.0*t7
    G2 = 2.5*t4 - 7.0*t5 + 6.5*t6 - 2.0*t7
    G3 = -(1.0/6.0)*t4 + 0.5*t5 - 0.5*t6 + (1.0/6.0)*t7
    return H0, H1, H2, H3, G0, G1, G2, G3


def _septic_segment_snap_hessian():
    """Precomputes the 8x8 snap energy matrix Q_unit = int_0^1 B^(4)(tau) B^(4)(tau)^T dtau."""
    # Polynomial coefficients c4, c5, c6, c7 for each of the 8 Hermite basis functions
    C = np.array([
        [-35.0,  84.0, -70.0,  20.0],  # H0
        [-20.0,  45.0, -36.0,  10.0],  # H1
        [ -5.0,  10.0,  -7.5,   2.0],  # H2
        [-2/3.,   1.0, -2/3.,  1/6.],  # H3
        [ 35.0, -84.0,  70.0, -20.0],  # G0
        [-15.0,  39.0, -34.0,  10.0],  # G1
        [  2.5,  -7.0,   6.5,  -2.0],  # G2
        [-1/6.,   0.5,  -0.5,  1/6.],  # G3
    ], dtype=float)
    # 4th derivative of c4*t^4 + c5*t^5 + c6*t^6 + c7*t^7 is:
    # 24*c4 + 120*c5*t + 360*c6*t^2 + 840*c7*t^3
    D4 = C * np.array([24.0, 120.0, 360.0, 840.0])
    # Hilbert matrix M_ij = int_0^1 t^(i+j) dt = 1 / (i + j + 1) for i,j in 0..3
    idx = np.arange(4)
    M = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
    return D4 @ M @ D4.T


_Q_SNAP_UNIT = _septic_segment_snap_hessian()


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


def _solve_minsnap_1d(s_knots, vals, v0, vM):
    """
    Solves min int_{s_0}^{s_M} (p^(4)(s))^2 ds subject to:
      p(s_k) = vals[k] for k=0..M,
      p'(s_0) = v0, p''(s_0) = 0, p'''(s_0) = 0,
      p'(s_M) = vM, p''(s_M) = 0, p'''(s_M) = 0.
    Returns knot derivatives (D1, D2, D3) of length M+1.
    """
    n_knots = len(s_knots)
    n_seg = n_knots - 1
    h = np.diff(s_knots)
    # Total variables per knot: 4 (p, p', p'', p'''). Total DOF = 4 * n_knots.
    n_dof = 4 * n_knots
    H_glob = np.zeros((n_dof, n_dof), float)

    for i in range(n_seg):
        hi = float(h[i])
        scale = np.array([1.0, hi, hi**2, hi**3, 1.0, hi, hi**2, hi**3])
        Qi = (_Q_SNAP_UNIT * np.outer(scale, scale)) / (hi**7)
        sl = slice(4 * i, 4 * (i + 2))
        H_glob[sl, sl] += Qi

    known_idx = []
    known_val = []
    free_idx = []

    for k in range(n_knots):
        # Position p(s_k) is always known
        known_idx.append(4 * k)
        known_val.append(float(vals[k]))
        if k == 0:
            known_idx.extend([4 * k + 1, 4 * k + 2, 4 * k + 3])
            known_val.extend([float(v0), 0.0, 0.0])
        elif k == n_knots - 1:
            known_idx.extend([4 * k + 1, 4 * k + 2, 4 * k + 3])
            known_val.extend([float(vM), 0.0, 0.0])
        else:
            free_idx.extend([4 * k + 1, 4 * k + 2, 4 * k + 3])

    sol = np.zeros(n_dof, float)
    known_idx_arr = np.array(known_idx, int)
    known_val_arr = np.array(known_val, float)
    sol[known_idx_arr] = known_val_arr

    if free_idx:
        free_idx_arr = np.array(free_idx, int)
        A = H_glob[np.ix_(free_idx_arr, free_idx_arr)] + 1e-10 * np.eye(len(free_idx_arr))
        b = -H_glob[np.ix_(free_idx_arr, known_idx_arr)] @ known_val_arr
        sol[free_idx_arr] = np.linalg.solve(A, b)

    sol = sol.reshape(n_knots, 4)
    return sol[:, 1], sol[:, 2], sol[:, 3]


def minimum_snap_smooth_shi(env, ctrl_pts, ds=0.05, mask=None):
    """
    Section 4 of Kun Shi et al. (2025):
    True Minimum-Snap trajectory optimization minimizing int (p^(4)(s))^2 ds
    with unit-tangent endpoint velocities (eliminating clamped zero-velocity hooks)
    and automatic midpoint anchoring if a segment approaches an obstacle.
    """
    if ctrl_pts is None or len(ctrl_pts) == 0:
        return np.zeros((0, 2))
    use_mask = env.mask if mask is None else mask
    pts = np.asarray(ctrl_pts, float)
    if len(pts) <= 2:
        return densify(pts, ds=ds)

    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-6])
    pts = pts[keep]
    if len(pts) <= 2:
        return densify(pts, ds=ds)

    # Iteratively refine control points if polynomial overshoot touches an obstacle
    work_pts = pts.copy()
    for _ in range(3):
        seg = np.hypot(*np.diff(work_pts, axis=0).T)
        s_knots = np.concatenate([[0.0], np.cumsum(seg)])
        u0 = (work_pts[1] - work_pts[0]) / max(seg[0], 1e-9)
        uM = (work_pts[-1] - work_pts[-2]) / max(seg[-1], 1e-9)

        vx, ax, jx = _solve_minsnap_1d(s_knots, work_pts[:, 0], u0[0], uM[0])
        vy, ay, jy = _solve_minsnap_1d(s_knots, work_pts[:, 1], u0[1], uM[1])

        pieces = []
        bad_segs = set()
        for i in range(len(seg)):
            hi = float(seg[i])
            n_s = max(5, int(math.ceil(hi / ds)) + 1)
            tau = np.linspace(0.0, 1.0, n_s, endpoint=(i == len(seg) - 1))
            H0, H1, H2, H3, G0, G1, G2, G3 = _septic_hermite_basis(tau)
            px = (H0 * work_pts[i, 0] + H1 * (hi * vx[i]) + H2 * (hi**2 * ax[i]) + H3 * (hi**3 * jx[i]) +
                  G0 * work_pts[i + 1, 0] + G1 * (hi * vx[i + 1]) + G2 * (hi**2 * ax[i + 1]) + G3 * (hi**3 * jx[i + 1]))
            py = (H0 * work_pts[i, 1] + H1 * (hi * vy[i]) + H2 * (hi**2 * ay[i]) + H3 * (hi**3 * jy[i]) +
                  G0 * work_pts[i + 1, 1] + G1 * (hi * vy[i + 1]) + G2 * (hi**2 * ay[i + 1]) + G3 * (hi**3 * jy[i + 1]))
            seg_poly = np.column_stack([px, py])
            if env.blocked(seg_poly, use_mask).any():
                bad_segs.add(i)
            pieces.append(seg_poly)

        poly = np.vstack(pieces)
        if not bad_segs:
            return densify(poly, ds=ds)

        # Insert segment midpoints on any segment whose spline overshoots into an obstacle
        new_pts = [work_pts[0]]
        for i in range(len(seg)):
            if i in bad_segs or (i - 1) in bad_segs or (i + 1) in bad_segs:
                new_pts.append(0.5 * (work_pts[i] + work_pts[i + 1]))
            new_pts.append(work_pts[i + 1])
        work_pts = np.array(new_pts, float)

    return densify(pts, ds=ds)


# =============================================================================
# 5. KINEMATIC TRAJECTORY & ALGORITHM 2 MULTI-ROBOT COORDINATION
# =============================================================================
class Traj:
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
            v_des = cfg.V * f

        # Integrate forward with bounded linear acceleration a_v_max
        dv_max = cfg.a_v_max * cfg.dt
        sa_list = [0.0]
        v_curr = min(float(v_des[0]), cfg.V)
        for k in range(K):
            v_target = float(v_des[k])
            if v_target > v_curr:
                v_curr = min(v_target, v_curr + dv_max)
            else:
                v_curr = max(v_target, v_curr - dv_max)
            v_curr = max(0.0, v_curr)
            s_next = min(L, sa_list[-1] + v_curr * cfg.dt)
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
            v_Steps = np.diff(self.pos, axis=0) / cfg.dt
            self.vel = np.vstack([v_Steps, np.zeros((1, 2))])
        else:
            self.vel = np.zeros_like(self.pos)
        self.T = (len(self.pos) - 1) * cfg.dt


def _pad(pos, K, hold):
    if len(pos) >= K:
        return pos[:K]
    pad = np.repeat(pos[-1:], K - len(pos), axis=0) if hold else np.full((K - len(pos), 2), np.nan)
    return np.vstack([pos, pad])


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


def heading(pos, k):
    k_cl = min(max(k, 0), len(pos) - 1)
    v = pos[min(k_cl + 3, len(pos) - 1)] - pos[max(k_cl - 3, 0)]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def solve_shi_improved_sa(env, robots, cfg, rng):
    """
    Algorithm 1 + Algorithm 2 of Kun Shi et al. (2025):
      - Step 1: Plan initial path for each robot using Improved SA (Algorithm 1) + Minimum-Snap.
      - Step 2: Priority-based conflict resolution (Algorithm 2):
          * Detect conflicts along discrete time steps where inter-robot distance < d_safe.
          * Compute movement angle alpha between colliding robots (Fig. 2).
          * Evaluate replanning around conflict region vs. Eq. (9)-(10) waiting strategy
            F = L_n / V_f - (L_x / V + (L_m - L_x) / V_f).
    """
    order = list(robots.keys())
    trajs, raw_paths, ptime = {}, {}, {}

    for idx, r in enumerate(order):
        t0 = time.perf_counter()
        s, g = robots[r]
        X_grid = shi_single_robot_isa(env, s, g, cfg, rng)
        raw_paths[r] = X_grid
        poly = minimum_snap_smooth_shi(env, X_grid)
        tr = Traj(poly, [], cfg)

        higher_pos = [trajs[m].pos for m in order[:idx]]

        # Iterative Algorithm 2 conflict resolution loop (handles multiple higher-priority robots)
        accumulated_disks = []
        for _ in range(4):
            nc, _, fst, _ = conflict_metrics(tr.pos, higher_pos, cfg.d_safe, cfg.hold_goal)
            if nc == 0 or fst is None:
                break
            k_conf, hi_idx = fst
            hi_tr_pos = higher_pos[hi_idx]
            k_self = min(k_conf, len(tr.pos) - 1)
            k_hi = min(k_conf, len(hi_tr_pos) - 1)

            u_self = heading(tr.pos, k_self)
            u_hi = heading(hi_tr_pos, k_hi)
            cos_a = float(np.clip(np.dot(u_self, u_hi), -1.0, 1.0))
            alpha_deg = math.degrees(math.acos(cos_a))

            # Candidate A: Replan with Improved SA treating conflict zone as obstacle (Fig. 1b & Alg. 2)
            for dk in (-6, 0, 6):
                kh = min(max(0, k_hi + dk), len(hi_tr_pos) - 1)
                accumulated_disks.append(hi_tr_pos[kh])
            mask_obs = env.with_disks(accumulated_disks, cfg.d_safe + 0.35)
            X_replan = shi_single_robot_isa(env, s, g, cfg, rng, mask=mask_obs)
            poly_replan = minimum_snap_smooth_shi(env, X_replan)
            tr_replan = Traj(poly_replan, [], cfg)
            replan_ok = (tr_replan.done and not env.blocked(tr_replan.pos).any() and
                         conflict_metrics(tr_replan.pos, higher_pos, cfg.d_safe, cfg.hold_goal)[0] == 0)

            # Candidate B: Waiting / Decelerating at V = k * V_f (Eq. 9: k in {0.5, 0.25, 0.0})
            t_conf = k_conf * cfg.dt
            best_wait_tr = None
            best_wait_k = 0.5
            for k_fac in (0.5, 0.25, 0.0):
                for pre_t in (2.5, 4.0):
                    for dt_w in np.arange(1.0, 16.0, 0.5):
                        mods_try = list(tr.mods) + [(max(0.0, t_conf - pre_t), t_conf + float(dt_w), k_fac)]
                        tr_w = Traj(tr.poly, mods_try, cfg)
                        if tr_w.done and conflict_metrics(tr_w.pos, higher_pos, cfg.d_safe, cfg.hold_goal)[0] == 0:
                            best_wait_tr = tr_w
                            best_wait_k = max(k_fac, 0.1)
                            break
                    if best_wait_tr is not None:
                        break
                if best_wait_tr is not None:
                    break

            # Eq. (10): F = L_n / V_f - (L_x / V + (L_m - L_x) / V_f)
            s_entered = max(0.0, float(tr.sa[k_self]) - cfg.sensor_range)
            L_m = max(0.0, tr.length - s_entered)
            L_n = max(0.0, tr_replan.length - s_entered)
            L_x = min(L_m, cfg.sensor_range)
            V_f = cfg.V
            V_dec = best_wait_k * V_f
            F_eq10 = (L_n / V_f) - (L_x / V_dec + (L_m - L_x) / V_f)

            if alpha_deg >= 120.0 and replan_ok:
                tr = tr_replan
                raw_paths[r] = X_replan
            elif replan_ok and best_wait_tr is not None:
                # Follow Eq. (10) / actual travel-time comparison
                if F_eq10 <= 0.0 and tr_replan.T <= best_wait_tr.T:
                    tr = tr_replan
                    raw_paths[r] = X_replan
                else:
                    tr = best_wait_tr
            elif best_wait_tr is not None:
                tr = best_wait_tr
            else:
                tr = tr_replan
                raw_paths[r] = X_replan

        trajs[r] = tr
        ptime[r] = time.perf_counter() - t0

    tot_conflicts = sum(
        conflict_metrics(trajs[n].pos, [trajs[m].pos for m in order[:i]], cfg.d_safe, cfg.hold_goal)[0]
        for i, n in enumerate(order)
    )
    return tot_conflicts, trajs, raw_paths, ptime, order


# =============================================================================
# 6. METRICS, TABLES & PLOTTING
# =============================================================================
def total_turn_deg(P):
    d = np.diff(P, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    d = d[seg > 1e-6]
    if len(d) < 2:
        return 0.0
    n = np.hypot(d[:, 0], d[:, 1])
    c = np.clip((d[:-1] * d[1:]).sum(1) / (n[:-1] * n[1:] + 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)).sum())


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
            'success': success,
        })
    return rows


def print_shi_tables(env_key, rows_shi):
    env_title = 'Barrier-free environment' if env_key == 'free' else 'Static environment'
    ref_rows = PAPER_TABLE2_REF[env_key]
    ref_t3 = PAPER_TABLE3_REF[env_key]

    print('\n' + '=' * 116)
    print(f'  TABLE 2 — {env_title.upper()} (Kun Shi et al., 2025 Improved SA)')
    print('=' * 116)
    hdr = (f'{"Algorithm":28} | {"Robot":8} | {"Priority":>8} | {"Path length":>11} | '
           f'{"Min dist":>10} | {"Move time(s)":>12} | {"Turn(deg)":>10} | {"CPU(s)":>8}')
    print(hdr)
    print('-' * 116)
    for p_id, L_ref, md_ref, T_ref in ref_rows:
        alg_col = 'Improved SA (Paper Table 2)' if p_id == 1 else ''
        print(f'{alg_col:28} | {f"Robot {p_id}":8} | {p_id:8d} | {L_ref:11.2f} | '
              f'{md_ref:10.2f} | {T_ref:12.2f} | {"—":>10} | {"—":>8}')
    print('-' * 116)
    for row in rows_shi:
        alg_col = 'Improved SA (Simulated)' if row['priority'] == 1 else ''
        print(f'{alg_col:28} | {f"Robot {row['priority']}":8} | {row["priority"]:8d} | '
              f'{row["length"]:11.2f} | {row["min_dist"]:10.2f} | {row["time_s"]:12.2f} | '
              f'{row["turn_deg"]:10.2f} | {row["cpu_s"]:8.3f}')
    print('=' * 116)

    print(f'\n  TABLE 3 SUMMARY — {env_title.upper()}')
    print('-' * 116)
    print(f'{"Method":30} | {"Avg Path Len":>12} | {"Max Path Len":>12} | '
          f'{"Avg Time (s)":>12} | {"Avg Turn(deg)":>13} | {"Avg CPU(s)":>10} | {"Success Rate":>12}')
    print('-' * 116)
    for m_name, aL_r, mL_r, aT_r, sr_r in ref_t3:
        print(f'{m_name:30} | {aL_r:12.2f} | {mL_r:12.2f} | {aT_r:12.2f} | {"—":>13} | {"—":>10} | {sr_r:>12}')
    aL = np.mean([r['length'] for r in rows_shi])
    mL = np.max([r['length'] for r in rows_shi])
    aT = np.mean([r['time_s'] for r in rows_shi])
    aTurn = np.mean([r['turn_deg'] for r in rows_shi])
    aCpu = np.mean([r['cpu_s'] for r in rows_shi])
    sr_pct = f'{100.0 * np.mean([float(r["success"]) for r in rows_shi]):.0f}%'
    print(f'{"Improved SA (Simulated)":30} | {aL:12.2f} | {mL:12.2f} | {aT:12.2f} | '
          f'{aTurn:13.2f} | {aCpu:10.3f} | {sr_pct:>12}')
    print('-' * 116)


def plot_shi_results(env, robots, trajs_shi, name, outdir, cfg):
    env_label = 'Barrier-free environment' if name == 'free' else 'Static environment'
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.0))

    ax0 = axs[0]
    ax0.set_xlim(0, 40)
    ax0.set_ylim(0, 40)
    ax0.set_aspect('equal')
    ax0.set_xticks(range(0, 41, 5))
    ax0.set_yticks(range(0, 41, 5))
    ax0.grid(True, lw=0.4, color='#b0b0b0', alpha=0.6)
    ax0.set_title(f'(a) Improved SA Paths (Shi et al., 2025) — {env_label}', fontsize=10.5, weight='bold')
    for x0, y0, x1, y1 in env.rects:
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color='k', zorder=2))

    for i, r in enumerate(NAMES):
        tr, c = trajs_shi[r], COLORS[i]
        st, go = robots[r]
        ax0.plot(tr.pos[:, 0], tr.pos[:, 1], color=c, lw=2.2, zorder=3,
                 label=f'robot {r} (L={tr.length*10:.1f}, T={tr.T:.2f}s)')
        ax0.plot(*st, 's', color='limegreen', mec='k', ms=7.5, zorder=4)
        ax0.plot(*go, 's', color='red', mec='k', ms=7.5, zorder=4)
        j = max(1, len(tr.pos) // 2)
        ax0.annotate('', xy=tr.pos[min(j + 3, len(tr.pos) - 1)], xytext=tr.pos[j],
                     arrowprops=dict(arrowstyle='->', color=c, lw=1.6), zorder=4)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, frameon=True)

    ax1 = axs[1]
    ax1.set_title(f'(b) Inter-Robot Distance vs. Time — {env_label}', fontsize=10.5, weight='bold')
    for a in range(len(NAMES)):
        for b in range(a + 1, len(NAMES)):
            A, B = trajs_shi[NAMES[a]].pos, trajs_shi[NAMES[b]].pos
            K = max(len(A), len(B)) if cfg.hold_goal else min(len(A), len(B))
            pA = _pad(A, K, True) if cfg.hold_goal else A[:K]
            pB = _pad(B, K, True) if cfg.hold_goal else B[:K]
            dist_paper = np.linalg.norm(pA - pB, axis=1) * 10.0
            ax1.plot(np.arange(K) * cfg.dt, dist_paper, lw=1.25,
                     label=f'distance of robot {NAMES[a]} and {NAMES[b]}')
    ax1.axhline(cfg.d_safe * 10.0, ls='--', color='k', lw=1.2, label='safety threshold (20.0)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (paper units, x10)')
    ax1.set_ylim(bottom=0)
    ax1.legend(ncol=2, fontsize=7.5, loc='upper right', frameon=True)
    ax1.grid(alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{name}_shi_improved_sa.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_shi_scenario(name, cfg, seed, outdir, num_seeds=1):
    rects, robots = SCENARIOS[name]
    env = Env(rects, cfg)
    all_runs = []
    first_trajs = None
    for s_offset in range(num_seeds):
        rng = np.random.default_rng(seed + s_offset)
        _, trajs_shi, _, pt_shi, _ = solve_shi_improved_sa(env, robots, cfg, rng)
        rows = collect_metrics(env, trajs_shi, pt_shi, cfg)
        all_runs.append(rows)
        if first_trajs is None:
            first_trajs = trajs_shi

    # Average per-robot metrics across seeds if num_seeds > 1
    avg_rows = []
    for r_idx in range(len(NAMES)):
        avg_rows.append({
            'robot_idx': r_idx + 1,
            'name': NAMES[r_idx],
            'priority': r_idx + 1,
            'length': float(np.mean([run[r_idx]['length'] for run in all_runs])),
            'min_dist': float(np.min([run[r_idx]['min_dist'] for run in all_runs])),
            'time_s': float(np.mean([run[r_idx]['time_s'] for run in all_runs])),
            'turn_deg': float(np.mean([run[r_idx]['turn_deg'] for run in all_runs])),
            'cpu_s': float(np.mean([run[r_idx]['cpu_s'] for run in all_runs])),
            'success': float(np.mean([float(run[r_idx]['success']) for run in all_runs])),
        })

    print_shi_tables(name, avg_rows)
    plot_shi_results(env, robots, first_trajs, name, outdir, cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='static', choices=['free', 'static', 'both'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-seeds', type=int, default=1)
    ap.add_argument('--out', default='results_shi_isa')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = ShiCfg()
    for sc in (['free', 'static'] if args.scenario == 'both' else [args.scenario]):
        run_shi_scenario(sc, cfg, args.seed, args.out, num_seeds=args.num_seeds)
    print(f'\nFigures saved in ./{args.out}/')