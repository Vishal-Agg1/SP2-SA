#!/usr/bin/env python3
"""
SP2-SA : Seeded, Pruned and Prioritised Simulated Annealing
Multi-robot path planning on the 40x40 grid used in Shi et al. (J. Franklin Inst. 2025)

Pipeline per robot (in priority order a > b > c > d > e):
  1. Theta* seed (any-angle, few waypoints)
  2. SA over waypoints: shift / insert / delete moves, calibrated temperature,
     energy = length + turning^2 + clearance penalty, early stopping
  3. Bezier corner rounding (C1 smooth, collision-checked)
  4. Space-time coordination against higher-priority robots:
        candidates = local SA repair | wait | slow-down  (wait/slow skipped if angle >= 120 deg)
        choose the candidate with fewest conflicts, then earliest arrival
        if still conflicting -> swap priorities and re-plan
Also included: a compact re-implementation of Shi's "improved SA" (grid greedy seed + deletion)
for single-robot comparison.   (APF and DWA are NOT implemented here.)

Usage:
    python sp2_sa.py                       # free + static scenarios, show & save figures
    python sp2_sa.py --scenario static --gif
    python sp2_sa.py --noshow              # only save PNGs to ./results
Requires: numpy, scipy, matplotlib   (pillow for --gif)

NOTE: obstacle rectangles in STATIC_RECTS are APPROXIMATIONS of Fig. 5 of the paper.
Replace them (x0, y0, x1, y1 in grid cells) with your exact map if you have it.
Units: 1 unit = 1 grid cell (paper uses cell side 10, so paper lengths = 10 x ours).
"""
import argparse, heapq, math, os, time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    V: float = 1.0            # robot speed [cells/s]  (paper: ~10 units/s with cell=10)
    dt: float = 0.1           # time step [s]
    tmax: float = 250.0       # time horizon [s]
    d_safe: float = 2.0       # min inter-robot distance (paper observed ~2.0-3.1 cells)
    margin: float = 0.3       # obstacle inflation (robot radius + safety)
    d_c: float = 0.8          # desired clearance in SA energy
    lam_turn: float = 1.0     # weight of turning-angle^2 term
    lam_clr: float = 2.0      # weight of clearance penalty
    lam_wp: float = 0.05      # cost per waypoint (keeps path compact)
    k_iter: int = 60          # SA moves per temperature level
    alpha: float = 0.9        # cooling rate
    r0: float = 2.0           # initial shift radius
    p_acc0: float = 0.8       # target initial acceptance probability
    stall_max: int = 3        # early stop after this many levels without improvement
    round_r: float = 1.5      # max corner rounding radius
    tau: float = 3.0          # half window for local repair / slow-down [s]
    n_max: int = 12           # max repair iterations per robot
    swap_tries: int = 3       # priority swaps allowed
    hold_goal: bool = False   # True: robots stay (as obstacles) after reaching goal


# ----------------------------------------------------------------------------- scenarios
NAMES = ['a', 'b', 'c', 'd', 'e']          # priority a (highest) ... e (lowest)
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
STATIC_RECTS = [  # approximate 12 static obstacles, (x0, y0, x1, y1)
    (6, 33, 9, 38), (13, 27, 18, 31), (17, 31, 20, 34), (21, 24, 24, 27),
    (36, 36, 39, 39), (28, 31, 32, 35), (14, 18, 18, 22), (29, 13, 33, 17),
    (8, 9, 12, 16), (2, 0, 5, 3), (17, 5, 28, 7), (33, 2, 36, 11)]
SCENARIOS = {'free': ([], FREE_ROBOTS), 'static': (STATIC_RECTS, STATIC_ROBOTS)}


# ----------------------------------------------------------------------------- environment
class Env:
    def __init__(self, rects, cfg, size=40, res=0.1):
        self.size, self.res, self.n, self.rects = size, res, int(round(size / res)), rects
        n = self.n
        raw = np.zeros((n, n), bool)                       # [iy, ix]
        for x0, y0, x1, y1 in rects:
            raw[int(round(y0 / res)):int(round(y1 / res)), int(round(x0 / res)):int(round(x1 / res))] = True
        self.raw = raw
        self.dist = distance_transform_edt(~raw) * res if raw.any() else np.full((n, n), 99.0)
        self.mask = self.dist < cfg.margin                  # inflated obstacles
        c = (np.arange(n) + 0.5) * res
        self.X, self.Y = np.meshgrid(c, c)

    def _idx(self, P):
        ix = np.floor(P[:, 0] / self.res).astype(int)
        iy = np.floor(P[:, 1] / self.res).astype(int)
        out = (ix < 0) | (ix >= self.n) | (iy < 0) | (iy >= self.n)
        return np.clip(ix, 0, self.n - 1), np.clip(iy, 0, self.n - 1), out

    def blocked(self, P, mask=None):
        m = self.mask if mask is None else mask
        P = np.atleast_2d(P)
        ix, iy, out = self._idx(P)
        return out | m[iy, ix]

    def los(self, a, b, mask=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / 0.1) + 1)
        P = a + (b - a) * np.linspace(0, 1, n)[:, None]
        return not self.blocked(P, mask).any()

    def clearance(self, P):
        ix, iy, _ = self._idx(np.atleast_2d(P))
        return self.dist[iy, ix]

    def with_disks(self, centers, radius):
        m = self.mask.copy()
        for cx, cy in centers:
            m |= (self.X - cx) ** 2 + (self.Y - cy) ** 2 < radius ** 2
        return m


# ----------------------------------------------------------------------------- Theta* seed
def theta_star(env, start, goal, mask=None):
    mask = env.mask if mask is None else mask
    N = env.size
    cen = lambda n: np.array([n[0] + 0.5, n[1] + 0.5])
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
    B = env.blocked(np.column_stack([cx.ravel(), cy.ravel()]), mask).reshape(N, N)
    s = (int(start[0]), int(start[1])); g = (int(goal[0]), int(goal[1]))
    B[s] = False; B[g] = False
    h = lambda n: math.hypot(n[0] - g[0], n[1] - g[1])
    gc = {s: 0.0}; par = {s: s}; closed = set(); heap = [(h(s), s)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in closed: continue
        closed.add(cur)
        if cur == g: break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < N and 0 <= nb[1] < N) or B[nb] or nb in closed: continue
                p = par[cur]
                if p == s or env.los(cen(p), cen(nb), mask) or p == cur:
                    use = p if env.los(cen(p), cen(nb), mask) else cur
                else:
                    use = cur
                if use == cur and not env.los(cen(cur), cen(nb), mask) and cur != s:
                    continue
                cand = gc[use] + math.hypot(use[0] - nb[0], use[1] - nb[1])
                if cand < gc.get(nb, 1e18):
                    gc[nb] = cand; par[nb] = use
                    heapq.heappush(heap, (cand + h(nb), nb))
    if g not in closed: return None
    chain = [g]
    while chain[-1] != s: chain.append(par[chain[-1]])
    chain = chain[::-1]
    pts = [np.array(start, float)] + [cen(n) for n in chain[1:-1]] + [np.array(goal, float)]
    out = [pts[0]]                                       # string-pulling
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not env.los(pts[i], pts[j], mask): j -= 1
        out.append(pts[j]); i = j
    return np.array(out)


# ----------------------------------------------------------------------------- SA on waypoints
def energy(env, X, cfg):
    d = np.diff(X, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    E = n.sum() + cfg.lam_wp * len(X)
    if len(X) > 2:
        c = np.clip((d[:-1] * d[1:]).sum(1) / (n[:-1] * n[1:] + 1e-12), -1, 1)
        E += cfg.lam_turn * (np.arccos(c) ** 2).sum()
        E += cfg.lam_clr * (np.maximum(0, cfg.d_c - env.clearance(X[1:-1])) ** 2).sum()
    return E


def propose(env, X, r, mask, rng):
    m, u = len(X), rng.random()
    if u < 0.6 and m > 2:                                           # shift
        j = rng.integers(1, m - 1); P = X[j] + rng.uniform(-r, r, 2)
        if env.blocked(P, mask)[0] or not (env.los(X[j-1], P, mask) and env.los(P, X[j+1], mask)): return None
        Xn = X.copy(); Xn[j] = P; return Xn
    if u < 0.8:                                                      # insert
        j = rng.integers(0, m - 1); P = (X[j] + X[j+1]) / 2 + rng.uniform(-r, r, 2)
        if env.blocked(P, mask)[0] or not (env.los(X[j], P, mask) and env.los(P, X[j+1], mask)): return None
        return np.insert(X, j + 1, P, axis=0)
    if m > 2:                                                        # delete (global shortcut)
        j = rng.integers(1, m - 1)
        if env.los(X[j-1], X[j+1], mask): return np.delete(X, j, axis=0)
    return None


def sp2_sa(env, start, goal, cfg, rng, mask=None):
    mask = env.mask if mask is None else mask
    X = theta_star(env, start, goal, mask)
    if X is None or len(X) <= 2: return X
    E = energy(env, X, cfg); bestX, bestE = X.copy(), E
    deltas = [abs(energy(env, Xn, cfg) - E) for Xn in
              (propose(env, X, cfg.r0, mask, rng) for _ in range(60)) if Xn is not None]
    Tm = max(-np.mean(deltas) / math.log(cfg.p_acc0), 1e-3) if deltas else 1.0
    T0, T, stall = 0.01 * Tm, Tm, 0
    while T > T0 and stall < cfg.stall_max:
        improved = False
        r = max(0.1, cfg.r0 * T / Tm)
        for _ in range(cfg.k_iter):
            Xn = propose(env, X, r, mask, rng)
            if Xn is None: continue
            En = energy(env, Xn, cfg); dE = En - E
            if dE < 0 or rng.random() < math.exp(-dE / T):
                X, E = Xn, En
                if E < bestE - 1e-9: bestX, bestE, improved = X.copy(), E, True
        T *= cfg.alpha
        stall = 0 if improved else stall + 1
    return bestX


# ----------------------------------------------------------------------------- smoothing
def densify(poly, ds=0.05):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9]); poly, s = poly[keep], s[keep]
    q = np.unique(np.append(np.arange(0, s[-1], ds), s[-1]))
    return np.column_stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])])


def smooth(env, X, cfg):
    """Quadratic-Bezier corner rounding; radius shrinks until the curve is collision-free."""
    if len(X) <= 2: return densify(X)
    pieces = [X[0:1]]
    t = np.linspace(0, 1, 15)[:, None]
    for j in range(1, len(X) - 1):
        P, a, b = X[j], X[j-1], X[j+1]
        la, lb = np.linalg.norm(a - P), np.linalg.norm(b - P)
        u1, u2 = (a - P) / la, (b - P) / lb
        d, curve = min(cfg.round_r, 0.5 * la, 0.5 * lb), None
        while d > 0.05:
            A, B = P + u1 * d, P + u2 * d
            c = (1 - t) ** 2 * A + 2 * (1 - t) * t * P + t ** 2 * B
            if not env.blocked(c).any(): curve = c; break
            d *= 0.5
        pieces.append(curve if curve is not None else P[None])
    pieces.append(X[-1:])
    return densify(np.vstack(pieces))


# ----------------------------------------------------------------------------- trajectories / coordination
class Traj:
    def __init__(self, poly, mods, cfg):
        self.poly, self.mods = poly, mods
        self.s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
        K = int(cfg.tmax / cfg.dt); t = np.arange(K) * cfg.dt
        f = np.ones(K)
        for t0, t1, fac in mods:
            w = (t >= t0) & (t < t1); f[w] = np.minimum(f[w], fac)
        sa = np.concatenate([[0], np.cumsum(cfg.V * f * cfg.dt)])
        L = self.s[-1]
        kend = int(np.argmax(sa >= L)) if (sa >= L).any() else len(sa) - 1
        self.sa = np.minimum(sa[:kend + 1], L)
        self.pos = np.column_stack([np.interp(self.sa, self.s, poly[:, 0]), np.interp(self.sa, self.s, poly[:, 1])])
        self.T = (len(self.pos) - 1) * cfg.dt
        self.done = self.sa[-1] >= L - 1e-9
        self.length = L


def _pad(pos, K, hold):
    if len(pos) >= K: return pos[:K]
    pad = np.repeat(pos[-1:], K - len(pos), 0) if hold else np.full((K - len(pos), 2), np.nan)
    return np.vstack([pos, pad])


def conflicts(pos, higher, cfg):
    """total conflicting samples and earliest (k, h)"""
    tot, first = 0, None
    for hi, ph in enumerate(higher):
        d = np.linalg.norm(pos - _pad(ph, len(pos), cfg.hold_goal), axis=1)
        bad = d < cfg.d_safe
        if bad.any():
            tot += int(bad.sum()); k = int(np.argmax(bad))
            if first is None or k < first[0]: first = (k, hi)
    return tot, first


def heading(pos, k):
    v = pos[min(k + 3, len(pos) - 1)] - pos[max(k - 3, 0)]
    return v / (np.linalg.norm(v) + 1e-9)


def local_repair(env, traj, higher, k, hi, goal, cfg, rng):
    t_r = max(0.0, k * cfg.dt - cfg.tau); ir = min(int(t_r / cfg.dt), len(traj.pos) - 1)
    p_r, s_r = traj.pos[ir], traj.sa[ir]
    k0, k1 = max(0, k - int(cfg.tau / cfg.dt)), k + int(cfg.tau / cfg.dt)
    ph = higher[hi]; ctr = [c for c in ph[k0:k1:10] if not np.isnan(c).any()]
    rad = 0.75 * cfg.d_safe
    ctr = [c for c in ctr if np.linalg.norm(c - p_r) > rad + 0.1 and np.linalg.norm(c - goal) > rad + 0.1]
    mask = env.with_disks(ctr, rad)
    X = sp2_sa(env, p_r, goal, cfg, rng, mask)
    if X is None: return None
    new = smooth(env, X, cfg)
    pre = traj.poly[:np.searchsorted(traj.s, s_r)]
    poly = np.vstack([pre, new]) if len(pre) else new
    return Traj(poly, [m for m in traj.mods if m[1] <= t_r], cfg)


def plan_robot(env, start, goal, higher, cfg, rng):
    X = sp2_sa(env, start, goal, cfg, rng)
    if X is None: return None, None
    traj = Traj(smooth(env, X, cfg), [], cfg)
    best = traj; bc, _ = conflicts(traj.pos, higher, cfg)
    cur = traj
    for _ in range(cfg.n_max):
        nc, first = conflicts(cur.pos, higher, cfg)
        if first is None: return cur, X
        k, hi = first; ts = k * cfg.dt
        ang = math.degrees(math.acos(np.clip(np.dot(heading(cur.pos, k), heading(higher[hi], min(k, len(higher[hi]) - 1))), -1, 1)))
        cands = []
        rep = local_repair(env, cur, higher, k, hi, goal, cfg, rng)
        if rep is not None: cands.append(rep)
        if ang < 120:                                                   # wait / slow only if not head-on
            th = max(0.0, ts - cfg.tau); bestw = None
            for dw in np.arange(0.5, 25.01, 0.5):
                c = Traj(cur.poly, cur.mods + [(th, th + dw, 0.0)], cfg)
                n_c, _ = conflicts(c.pos, higher, cfg)
                if bestw is None or n_c < bestw[0]: bestw = (n_c, c)
                if n_c == 0: break
            cands.append(bestw[1])
            for fac in (0.25, 0.5):
                for w in (cfg.tau, 2 * cfg.tau):
                    cands.append(Traj(cur.poly, cur.mods + [(max(0, ts - w), ts + w, fac)], cfg))
        cands = [c for c in cands if c.done]
        if not cands: break
        scored = [(conflicts(c.pos, higher, cfg)[0], c.T, i) for i, c in enumerate(cands)]
        _, _, bi = min(scored); cur = cands[bi]
        n_new, _ = conflicts(cur.pos, higher, cfg)
        if (n_new, cur.T) < (bc, best.T): best, bc = cur, n_new
    return best, X


def plan_team(env, robots, order, cfg, rng):
    trajs, raw, ptime = {}, {}, {}
    higher = []
    for name in order:
        st, go = robots[name]
        t0 = time.perf_counter()
        tr, X = plan_robot(env, st, go, higher, cfg, rng)
        ptime[name] = time.perf_counter() - t0
        if tr is None: raise RuntimeError(f'No path for robot {name}')
        trajs[name], raw[name] = tr, X; higher.append(tr.pos)
    return trajs, raw, ptime


def total_conflicts(trajs, order, cfg):
    tot, bad_pair = 0, None
    for i, n in enumerate(order):
        c, f = conflicts(trajs[n].pos, [trajs[m].pos for m in order[:i]], cfg)
        tot += c
        if f is not None and bad_pair is None: bad_pair = (i, f[1])
    return tot, bad_pair


def solve(env, robots, cfg, rng):
    order, best = list(robots), None
    for attempt in range(cfg.swap_tries + 1):
        trajs, raw, pt = plan_team(env, robots, order, cfg, rng)
        tot, pair = total_conflicts(trajs, order, cfg)
        if best is None or tot < best[0]: best = (tot, trajs, raw, pt, list(order))
        if tot == 0 or pair is None: break
        i, h = pair; order[i], order[h] = order[h], order[i]           # priority swap
        print(f'   residual conflicts -> swapping priorities: {order}')
    return best


# ----------------------------------------------------------------------------- Shi et al. improved SA (compact re-implementation)
def shi_isa(env, start, goal, rng, p0=0.85, Tm=1e4, T0=4e3, alpha=0.98, k=60, n_init=3):
    N = env.size
    cx, cy = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing='ij')
    free = ~env.blocked(np.column_stack([cx.ravel(), cy.ravel()])).reshape(N, N)
    s, g = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    free[s] = free[g] = True
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    dg = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])

    def seed():
        path, inp, banned = [s], {s}, set()
        for _ in range(30000):
            cur = path[-1]
            if cur == g: return path
            A = [(cur[0] + a, cur[1] + b) for a, b in nb
                 if 0 <= cur[0] + a < N and 0 <= cur[1] + b < N and free[cur[0] + a, cur[1] + b]
                 and (cur[0] + a, cur[1] + b) not in inp and (cur[0] + a, cur[1] + b) not in banned]
            if not A:
                banned.add(cur); inp.discard(cur); path.pop()
                if not path: return None
                continue
            A.sort(key=dg)
            c = A[0] if (rng.random() < p0 or len(A) == 1) else A[1]
            path.append(c); inp.add(c)
        return None

    E = lambda p: 10 * sum(math.hypot(p[i][0] - p[i+1][0], p[i][1] - p[i+1][1]) for i in range(len(p) - 1))

    def delete_op(p):
        ch = True
        while ch:
            ch = False
            for m in range(len(p) - 2):
                for n in range(len(p) - 1, m + 1, -1):
                    if math.hypot(p[m][0] - p[n][0], p[m][1] - p[n][1]) <= 1.4 + 1e-9:
                        p = p[:m + 1] + p[n:]; ch = True; break
                if ch: break
        return p

    seeds = [q for q in (seed() for _ in range(n_init)) if q]
    if not seeds: return None
    X = min(seeds, key=E); cur = E(X); T = Tm
    while T > T0:
        for _ in range(k):
            if len(X) < 3: break
            j = int(rng.integers(1, len(X) - 1)); a, b = X[j-1], X[j+1]; occ = set(X)
            C = [(a[0] + dx, a[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            C = [c for c in C if max(abs(c[0] - b[0]), abs(c[1] - b[1])) <= 1 and 0 <= c[0] < N and 0 <= c[1] < N
                 and free[c] and c not in occ and c != a]
            if not C: continue
            Xn = X.copy(); Xn[j] = C[int(rng.integers(len(C)))]; En = E(Xn); dE = En - cur
            if dE < 0 or rng.random() < math.exp(-dE / T): X, cur = Xn, En
        X = delete_op(X); cur = E(X); T *= alpha
    return np.array(X, float) + 0.5


# ----------------------------------------------------------------------------- metrics
def path_len(P): return float(np.hypot(*np.diff(P, axis=0).T).sum())

def total_turn_deg(P):
    d = np.diff(P, axis=0); d = d[np.hypot(*d.T) > 1e-9]
    if len(d) < 2: return 0.0
    c = np.clip((d[:-1] * d[1:]).sum(1) / (np.hypot(*d[:-1].T) * np.hypot(*d[1:].T)), -1, 1)
    return float(np.degrees(np.arccos(c)).sum())


# ----------------------------------------------------------------------------- plotting
def draw_map(ax, env, title):
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_aspect('equal')
    ax.set_xticks(range(0, 41, 5)); ax.set_yticks(range(0, 41, 5))
    ax.grid(True, lw=0.3, alpha=0.5); ax.set_title(title, fontsize=10)
    for x0, y0, x1, y1 in env.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color='k', zorder=2))


def plot_result(env, robots, trajs, order, name, outdir, cfg, show_cfg=True):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [1.1, 1]})
    draw_map(ax, env, f'SP²-SA paths — {name} environment')
    for i, r in enumerate(NAMES):
        if r not in trajs: continue
        tr, c = trajs[r], COLORS[i]; st, go = robots[r]
        ax.plot(tr.pos[:, 0], tr.pos[:, 1], color=c, lw=2, zorder=3, label=f'robot {r}  (L={tr.length*10:.0f}, T={tr.T:.1f}s)')
        ax.plot(*st, 's', color='limegreen', mec='k', ms=8, zorder=4); ax.plot(*go, 's', color='red', mec='k', ms=8, zorder=4)
        ax.text(st[0] + .5, st[1] + .5, r, fontsize=9, color=c, weight='bold', zorder=5)
        for k in range(0, len(tr.pos), 100):                                 # time ticks every 10 s
            ax.plot(*tr.pos[k], 'o', color=c, ms=3.5, mec='w', mew=0.5, zorder=4)
        j = max(1, len(tr.pos) // 2)
        ax.annotate('', xy=tr.pos[j + 2], xytext=tr.pos[j], arrowprops=dict(arrowstyle='->', color=c, lw=1.8), zorder=4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
    ax2.set_title('Inter-robot distance (dashed = d_safe)', fontsize=10)
    for a in range(len(NAMES)):
        for b in range(a + 1, len(NAMES)):
            A, B = trajs[NAMES[a]].pos, trajs[NAMES[b]].pos; n = min(len(A), len(B))
            ax2.plot(np.arange(n) * cfg.dt, np.linalg.norm(A[:n] - B[:n], axis=1), lw=1.2, label=f'{NAMES[a]}-{NAMES[b]}')
    ax2.axhline(cfg.d_safe, ls='--', color='k'); ax2.set_xlabel('time [s]'); ax2.set_ylabel('distance [cells]')
    ax2.legend(ncol=2, fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_paths.png'), dpi=150)


def plot_compare(env, robots, raw_sp2, raw_shi, name, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, data, ttl in ((axs[0], raw_shi, "Shi et al. improved SA (grid path)"), (axs[1], raw_sp2, 'SP²-SA raw waypoints (before corner rounding)')):
        draw_map(ax, env, f'{ttl} — {name}')
        for i, r in enumerate(NAMES):
            P = data.get(r)
            if P is None: continue
            ax.plot(P[:, 0], P[:, 1], '-o', color=COLORS[i], lw=1.6, ms=2.5 if ttl.startswith('Shi') else 4, zorder=3, label=f'{r}')
            st, go = robots[r]
            ax.plot(*st, 's', color='limegreen', mec='k', ms=7, zorder=4); ax.plot(*go, 's', color='red', mec='k', ms=7, zorder=4)
        ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_compare_single_robot.png'), dpi=150)


def make_gif(env, robots, trajs, name, outdir, cfg):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); draw_map(ax, env, f'SP²-SA — {name}')
    dots, trail = {}, {}
    for i, r in enumerate(NAMES):
        ax.plot(*robots[r][1], 's', color='red', mec='k', ms=7, zorder=4)
        dots[r], = ax.plot([], [], 'o', color=COLORS[i], ms=9, mec='k', zorder=5)
        trail[r], = ax.plot([], [], '-', color=COLORS[i], lw=1.5, zorder=3)
    K = max(len(t.pos) for t in trajs.values()); step = 4
    def upd(f):
        k = f * step
        for r in NAMES:
            P = trajs[r].pos; kk = min(k, len(P) - 1)
            dots[r].set_data([P[kk, 0]], [P[kk, 1]]); trail[r].set_data(P[:kk + 1, 0], P[:kk + 1, 1])
        return list(dots.values()) + list(trail.values())
    ani = FuncAnimation(fig, upd, frames=K // step + 1, blit=True)
    ani.save(os.path.join(outdir, f'{name}.gif'), writer=PillowWriter(fps=20)); plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(name, cfg, seed, outdir, gif):
    rects, robots = SCENARIOS[name]
    env = Env(rects, cfg); rng = np.random.default_rng(seed)
    for r, (s, g) in robots.items():
        assert not env.blocked(s)[0] and not env.blocked(g)[0], f'robot {r}: start/goal inside an obstacle'
    print(f'\n=== {name} environment ===')
    t0 = time.perf_counter()
    tot, trajs, raw, pt, order = solve(env, robots, cfg, rng)
    wall = time.perf_counter() - t0
    print(f'priority order used: {order} | residual conflict samples: {tot} | total planning time {wall:.2f}s')
    print(f'{"robot":6}{"length x10":>11}{"time[s]":>9}{"min dist":>10}{"plan[s]":>9}')
    for r in NAMES:
        md = min(np.nanmin(np.linalg.norm(trajs[r].pos[:min(len(trajs[r].pos), len(trajs[o].pos))] -
                 trajs[o].pos[:min(len(trajs[r].pos), len(trajs[o].pos))], axis=1)) for o in NAMES if o != r)
        print(f'{r:6}{trajs[r].length*10:11.1f}{trajs[r].T:9.1f}{md:10.2f}{pt[r]:9.3f}')
    L = [trajs[r].length * 10 for r in NAMES]
    print(f'avg path length {np.mean(L):.1f} | max path length {max(L):.1f} | avg time {np.mean([trajs[r].T for r in NAMES]):.1f}s')
    # single-robot comparison with Shi's improved SA
    raw_sp2, raw_shi = {}, {}
    print('\nSingle-robot comparison (no inter-robot coordination)')
    print(f'{"robot":6}{"method":10}{"len x10":>9}{"turn[deg]":>11}{"#pts":>6}{"time[s]":>9}')
    for r in NAMES:
        s, g = robots[r]
        t = time.perf_counter(); A = sp2_sa(env, s, g, cfg, rng); ta = time.perf_counter() - t
        t = time.perf_counter(); B = shi_isa(env, s, g, rng); tb = time.perf_counter() - t
        raw_sp2[r], raw_shi[r] = A, B
        print(f'{r:6}{"Shi-ISA":10}{path_len(B)*10:9.1f}{total_turn_deg(B):11.0f}{len(B):6d}{tb:9.3f}')
        print(f'{"":6}{"SP2-SA":10}{path_len(A)*10:9.1f}{total_turn_deg(A):11.0f}{len(A):6d}{ta:9.3f}')
    plot_result(env, robots, trajs, order, name, outdir, cfg)
    plot_compare(env, robots, raw_sp2, raw_shi, name, outdir)
    if gif: make_gif(env, robots, trajs, name, outdir, cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='both', choices=['free', 'static', 'both'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results')
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--noshow', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sc in (['free', 'static'] if a.scenario == 'both' else [a.scenario]):
        run(sc, Cfg(), a.seed, a.out, a.gif)
    print(f'\nFigures saved in ./{a.out}/')
    if not a.noshow: plt.show()
