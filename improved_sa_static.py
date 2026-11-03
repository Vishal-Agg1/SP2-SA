"""#!/usr/bin/env python3"""
"""
Improved Simulated Annealing (SA) for Multi-Robot Path Planning with Priority
Based on:
  Kun Shi, Luyao Yang, Zhengtian Wu, Baoping Jiang, Qing Gao (2025),
  "Multi-robot dynamic path planning with priority based on simulated annealing",
  Journal of the Franklin Institute, 362 (2025) 107396.

Implemented components from the paper:
  1. Raster Grid Environment Modelling (Section 2.1, Eq. 1):
     - 40x40 grid where each grid cell has side length 10 in paper units (400x400).
     - Uses the exact static environment obstacles and 5 robot start/goal pairs
       from the provided static environment specification.
  2. Improved SA Initial Path Selection (Section 2.2, Eq. 2-4):
     - Probabilistic greedy 8-neighbor expansion using probability p0 in [0.7, 0.95]:
         P_{j+1} = d_min(A_j, P_end) with probability p0,
                   d_min(B_j, P_end) with probability 1 - p0,
       where B_j = A_j \ {d_min(A_j, P_end)}.
     - Automatic dead-end rounding/backtracking when A_j is empty.
     - Multi-start initial path selection keeping the best collision-free seed.
  3. Redundant Point Deletion Operation (Section 2.2, Eq. 5-7):
     - Deletes all intermediate points between P_m and P_n whenever
       d(P_m, P_n) <= 14 (paper units, i.e. <= sqrt(2) grid cells) and |m - n| >= 2,
       plus collision-free line-of-sight shortcut pruning.
  4. Single-Robot Improved SA Optimization (Section 2.3, Eq. 8 & Algorithm 1):
     - Temperature schedule: T_m = 10000, T_0 = 4000, cooling rate alpha = 0.98.
     - Evaluation function E(X) = sum of Euclidean segment lengths (Eq. 8).
     - Metropolis acceptance criterion exp(-(E(X_t2) - E(X_t1)) / T) with
       deletion operation applied at each temperature stage.
  5. Multi-Robot Priority & Waiting/Replanning Strategy (Section 3, Eq. 9-10 & Algorithm 2):
     - Robots 1..5 ('a'..'e') have priority 1..5 (1 = highest, 5 = lowest).
     - Sensor detection range check between lower-priority robot R_a and
       higher-priority robot R_b.
     - Speed reduction rule (Eq. 9) and time-difference decision function F (Eq. 10):
         F = L_n / V_f - (L_x / V + (L_m - L_x) / V_f)
       combined with movement direction angle check (alpha >= 120 deg forces
       replanning with Improved SA around the higher-priority robot).
  6. Minimum-Snap Trajectory Smoothing (Section 4):
     - Smooths discrete grid waypoints by minimizing fourth-derivative (snap)
       energy subject to waypoint/boundary interpolation and obstacle/safety clearance.
"""

import argparse
import heapq
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import distance_transform_edt


# =============================================================================
# 1. PARAMETERS & STATIC ENVIRONMENT DEFINITION (from provided code & paper)
# =============================================================================
@dataclass
class SAParams:
    # Grid & scaling parameters (Section 2.2 & Section 4)
    grid_size: int = 40
    cell_scale: float = 10.0      # 1 grid cell = 10 paper units (Eq. 6: side length 10)
    dt: float = 0.1               # simulation time step [s]
    V_f: float = 1.0              # initial robot speed [cells/s] = 10 paper units/s
    k_slow: float = 0.40          # speed reduction factor k in [1/4, 1/2] (Eq. 9)

    # Improved SA parameters (Section 2.2, Section 4, Algorithm 1)
    T_m: float = 10000.0          # initial temperature (Section 4: 10,000)
    T_0: float = 4000.0           # ending temperature (Section 4: 4,000)
    alpha: float = 0.98           # cooling rate (Section 4: 0.98)
    k_iter: int = 12              # inner loop iterations per temperature step (Algorithm 1)
    p0: float = 0.85              # greedy neighbor selection probability in [0.7, 0.95] (Eq. 4)
    n_init_trials: int = 15       # multi-try initial solution selection (Section 2.2)
    delete_dist_paper: float = 14.2  # Eq. 6 threshold: d(P_m, P_n) <= 14 (sqrt(2)*10 = 14.14)

    # Multi-robot priority & collision avoidance parameters (Section 3, Eq. 9-10, Algorithm 2)
    d_safe_paper: float = 22.0    # target minimum inter-robot spacing in paper units (>20 in Table 2)
    sensor_range_paper: float = 45.0  # sensor detection range L_x in paper units
    head_on_angle_deg: float = 120.0  # angle threshold alpha >= 120 deg (Section 3.2, Fig. 2)
    obs_margin_cells: float = 0.28    # obstacle clearance margin in grid cells


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

SCENARIOS = {
    'free': ([], FREE_ROBOTS),
    'static': (STATIC_RECTS, STATIC_ROBOTS),
}

# 8-connected grid offsets
MOVES_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


# =============================================================================
# 2. RASTER GRID ENVIRONMENT (Section 2.1)
# =============================================================================
class GridEnv:
    def __init__(self, rects: List[Tuple[int, int, int, int]], params: SAParams, res: float = 0.1):
        self.size = params.grid_size
        self.scale = params.cell_scale
        self.rects = rects
        self.res = res
        self.n_fine = int(round(self.size / res))

        # Integer 40x40 obstacle grid for discrete SA path planning
        self.occ_grid = np.zeros((self.size, self.size), dtype=bool)
        for x0, y0, x1, y1 in rects:
            self.occ_grid[y0:y1, x0:x1] = True

        # Fine resolution grid for continuous collision / clearance checks
        raw = np.zeros((self.n_fine, self.n_fine), dtype=bool)
        for x0, y0, x1, y1 in rects:
            raw[int(round(y0 / res)):int(round(y1 / res)),
                int(round(x0 / res)):int(round(x1 / res))] = True
        self.raw_fine = raw
        self.dist_fine = (
            distance_transform_edt(~raw) * res if raw.any()
            else np.full((self.n_fine, self.n_fine), 99.0)
        )
        self.mask_fine = self.dist_fine < params.obs_margin_cells

    @staticmethod
    def pt_to_cell(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(math.floor(pt[0])), int(math.floor(pt[1])))

    @staticmethod
    def cell_to_pt(cell: Tuple[int, int]) -> np.ndarray:
        return np.array([cell[0] + 0.5, cell[1] + 0.5], dtype=float)

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def cell_free(self, cell: Tuple[int, int], extra_blocked: Optional[set] = None) -> bool:
        if not self.in_bounds(cell):
            return False
        if self.occ_grid[cell[1], cell[0]]:
            return False
        if extra_blocked and cell in extra_blocked:
            return False
        return True

    def valid_step(
        self,
        c1: Tuple[int, int],
        c2: Tuple[int, int],
        extra_blocked: Optional[set] = None,
    ) -> bool:
        """Checks 8-neighbor validity including preventing diagonal corner cutting through obstacles."""
        if not self.cell_free(c2, extra_blocked):
            return False
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        if abs(dx) == 1 and abs(dy) == 1:
            # Avoid squeezing diagonally through a blocked corner
            if self.occ_grid[c1[1], c1[0] + dx] or self.occ_grid[c1[1] + dy, c1[0]]:
                return False
        return True

    def points_blocked(self, P: np.ndarray) -> bool:
        P = np.atleast_2d(P)
        ix = np.floor(P[:, 0] / self.res).astype(int)
        iy = np.floor(P[:, 1] / self.res).astype(int)
        if (ix < 0).any() or (ix >= self.n_fine).any() or (iy < 0).any() or (iy >= self.n_fine).any():
            return True
        return bool(self.mask_fine[iy, ix].any())

    def line_of_sight(self, a: np.ndarray, b: np.ndarray, extra_blocked: Optional[set] = None) -> bool:
        dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(2, int(dist / 0.08) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = np.column_stack([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
        if self.points_blocked(pts):
            return False
        if extra_blocked:
            cells = set(zip(np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)))
            if cells & extra_blocked:
                return False
        return True


# =============================================================================
# 3. IMPROVED SIMULATED ANNEALING FOR SINGLE ROBOT (Section 2.2 - 2.3, Alg. 1)
# =============================================================================
def path_energy_paper(cells: List[Tuple[int, int]], scale: float = 10.0) -> float:
    """
    Evaluation function E (Eq. 8):
        E = sum_{j=start}^{end-1} d(P_j, P_{j+1})
    """
    if len(cells) < 2:
        return 0.0
    pts = np.array(cells, dtype=float) * scale
    diffs = np.diff(pts, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def delete_operation_eq6_eq7(
    cells: List[Tuple[int, int]],
    env: GridEnv,
    params: SAParams,
    extra_blocked: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    Deletion operation from Section 2.2 (Eq. 5 - Eq. 7):
      Given trajectory X^i = {P_start, ..., P_m, ..., P_n, ..., P_end} on a grid
      with side length 10:
          d(P_m, P_n) <= 14  and  |m - n| >= 2
      If satisfied, all points between P_m and P_n are deleted:
          X^i = {P_start, ..., P_m, P_n, ..., P_end}.
    """
    if len(cells) <= 2:
        return list(cells)

    path = list(cells)
    threshold_cells = params.delete_dist_paper / params.cell_scale  # 1.42 cells (>= sqrt(2))
    changed = True
    while changed and len(path) > 2:
        changed = False
        m = 0
        while m < len(path) - 2:
            # Search from farthest n back to m+2 to remove the largest redundant loop first
            jumped = False
            for n in range(len(path) - 1, m + 1, -1):
                c_m, c_n = path[m], path[n]
                dist_cells = math.hypot(c_n[0] - c_m[0], c_n[1] - c_m[1])
                if dist_cells <= threshold_cells and env.valid_step(c_m, c_n, extra_blocked):
                    path = path[:m + 1] + path[n:]
                    changed = True
                    jumped = True
                    break
            if not jumped:
                m += 1
    return path


def generate_initial_path_eq4(
    env: GridEnv,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
    max_steps: int = 2000,
) -> Optional[List[Tuple[int, int]]]:
    """
    Improved SA initial solution selection (Section 2.2, Eq. 2 - Eq. 4):
      - Starts at P_start and iteratively selects P_{j+1} from the 8-neighborhood A_j.
      - Obstacle points and points already in the path are excluded from A_j.
      - When A_j is empty, P_j is marked as a dead-end and backtracked.
      - Next point is chosen by Eq. (4):
            P_{j+1} = d_min(A_j, P_end)  if rand < p_0
                      d_min(B_j, P_end)  otherwise
        where B_j is A_j after deleting the point closest to P_end.
    """
    path = [start_cell]
    visited = {start_cell}
    dead_ends = set()

    for _ in range(max_steps):
        curr = path[-1]
        if curr == goal_cell:
            return delete_operation_eq6_eq7(path, env, params, extra_blocked)

        # Find valid adjacent points A^i_j
        A_j = []
        for dx, dy in MOVES_8:
            nb = (curr[0] + dx, curr[1] + dy)
            if nb == goal_cell or (
                nb not in visited
                and nb not in dead_ends
                and env.valid_step(curr, nb, extra_blocked)
            ):
                dist_goal = math.hypot(nb[0] - goal_cell[0], nb[1] - goal_cell[1])
                A_j.append((dist_goal + rng.uniform(0.0, 1e-6), nb))

        if not A_j:
            # Dead end reached: round off P_j and backtrack (Section 2.2)
            dead_ends.add(curr)
            visited.remove(curr)
            path.pop()
            if not path:
                break
            continue

        # Sort A_j by linear distance to P_end
        A_j.sort(key=lambda x: x[0])
        best_pt = A_j[0][1]

        if len(A_j) == 1 or rng.random() < params.p0:
            next_pt = best_pt
        else:
            # B_j is A_j after removing the point closest to P_end (Eq. 4)
            next_pt = A_j[1][1]

        path.append(next_pt)
        visited.add(next_pt)

    return None


def astar_grid_path(
    env: GridEnv,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    extra_blocked: Optional[set] = None,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Optional[List[Tuple[int, int]]]:
    """8-connected grid A* used as fallback or local segment bridge in SA neighborhood moves."""
    if start_cell == goal_cell:
        return [start_cell]
    h = lambda c: math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])
    g_cost = {start_cell: 0.0}
    parent = {start_cell: start_cell}
    closed = set()
    heap = [(h(start_cell), start_cell)]

    while heap:
        _, curr = heapq.heappop(heap)
        if curr in closed:
            continue
        closed.add(curr)
        if curr == goal_cell:
            break
        for dx, dy in MOVES_8:
            nb = (curr[0] + dx, curr[1] + dy)
            if nb in closed:
                continue
            if nb != goal_cell and not env.valid_step(curr, nb, extra_blocked):
                continue
            if nb == goal_cell and not env.in_bounds(nb):
                continue
            step = math.hypot(dx, dy)
            if noise_scale > 0.0 and rng is not None:
                step += float(rng.uniform(0.0, noise_scale))
            cand = g_cost[curr] + step
            if cand < g_cost.get(nb, 1e18):
                g_cost[nb] = cand
                parent[nb] = curr
                heapq.heappush(heap, (cand + h(nb), nb))

    if goal_cell not in closed:
        return None
    chain = [goal_cell]
    while chain[-1] != start_cell:
        chain.append(parent[chain[-1]])
    chain.reverse()
    return chain


def select_initial_solution(
    env: GridEnv,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    Runs the probabilistic Eq. (4) initial path generator multiple times and
    keeps the best collision-free initial path (Section 2.2).
    """
    best_path = None
    best_E = 1e18
    for _ in range(params.n_init_trials):
        cand = generate_initial_path_eq4(env, start_cell, goal_cell, params, rng, extra_blocked)
        if cand is not None:
            E = path_energy_paper(cand, params.cell_scale)
            if E < best_E:
                best_E = E
                best_path = cand
    if best_path is None:
        best_path = astar_grid_path(env, start_cell, goal_cell, extra_blocked)
    return best_path


def propose_neighbor_solution(
    path: List[Tuple[int, int]],
    env: GridEnv,
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    Generates a random neighbor solution X_t2 in the neighborhood of X_t1 (Algorithm 1, line 6):
      - Selects a random sub-segment [i1, i2] along the discrete path and either:
        (a) Perturbs an intermediate pivot cell and reconnects via 8-connected paths, or
        (b) Reroutes the sub-segment using a randomized Eq. (4) / noisy bridge.
    """
    n = len(path)
    if n <= 3:
        return None

    u = rng.random()
    if u < 0.55:
        # Local single-vertex or short-window nudge on the 8-connected grid
        idx = int(rng.integers(1, n - 1))
        prev_c, curr_c, next_c = path[idx - 1], path[idx], path[idx + 1]
        candidates = []
        for dx, dy in MOVES_8:
            alt = (curr_c[0] + dx, curr_c[1] + dy)
            if alt != curr_c and env.valid_step(prev_c, alt, extra_blocked) and env.valid_step(alt, next_c, extra_blocked):
                candidates.append(alt)
        if candidates:
            chosen = candidates[int(rng.integers(0, len(candidates)))]
            new_path = list(path)
            new_path[idx] = chosen
            return new_path

    # Sub-segment rerouting between two indices i1 < i2
    span = int(rng.integers(3, max(4, min(16, n))))
    i1 = int(rng.integers(0, max(1, n - span)))
    i2 = min(n - 1, i1 + span)
    c1, c2 = path[i1], path[i2]

    if rng.random() < 0.65:
        sub = astar_grid_path(env, c1, c2, extra_blocked, noise_scale=0.35, rng=rng)
    else:
        sub = generate_initial_path_eq4(env, c1, c2, params, rng, extra_blocked, max_steps=150)

    if sub is None or len(sub) < 2:
        return None
    return path[:i1] + sub + path[i2 + 1:]


def improved_sa_single_robot(
    env: GridEnv,
    start_pt: Tuple[float, float],
    goal_pt: Tuple[float, float],
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    Algorithm 1: Single robot dynamic/static path planning via Improved SA.
      Input: Initial temperature T_m; ending temperature T_0; cooling rate alpha;
             maximum number of iterations k.
      Output: Global optimal discrete grid path X_f.
    """
    start_cell = env.pt_to_cell(start_pt)
    goal_cell = env.pt_to_cell(goal_pt)

    # Line 2: Randomly generated initial path X_0 using Eq. (2)-(4) + Eq. (6)-(7)
    X_t1 = select_initial_solution(env, start_cell, goal_cell, params, rng, extra_blocked)
    X_t1 = delete_operation_eq6_eq7(X_t1, env, params, extra_blocked)
    E_t1 = path_energy_paper(X_t1, params.cell_scale)

    X_best = list(X_t1)
    E_best = E_t1

    # Line 1 & 4: T = T_m; while T > T_0 do
    T = params.T_m
    while T > params.T_0:
        X_f = list(X_t1)
        E_f = E_t1
        # Line 5: for i = 1 to k do
        for _ in range(params.k_iter):
            # Line 6: Select new solution X_t2 in the neighborhood of X_t1
            X_t2 = propose_neighbor_solution(X_t1, env, params, rng, extra_blocked)
            if X_t2 is None:
                continue
            E_t2 = path_energy_paper(X_t2, params.cell_scale)
            # Line 7-14: Metropolis acceptance criterion
            if E_t2 < E_t1:
                X_f = X_t2
                E_f = E_t2
                X_t1 = X_t2
                E_t1 = E_t2
            else:
                prob = math.exp(-(E_t2 - E_t1) / max(T, 1e-6))
                if rng.random() < prob:
                    X_f = X_t2
                    E_f = E_t2
                    X_t1 = X_t2
                    E_t1 = E_t2

        # Line 17: A delete operation on X_f (Eq. 5-7)
        X_f = delete_operation_eq6_eq7(X_f, env, params, extra_blocked)
        E_f = path_energy_paper(X_f, params.cell_scale)
        X_t1, E_t1 = X_f, E_f

        if E_f < E_best - 1e-6:
            X_best = list(X_f)
            E_best = E_f

        # Line 18: T = T * alpha
        T *= params.alpha

    return X_best


# =============================================================================
# 4. MINIMUM SNAP TRAJECTORY SMOOTHING (Section 4)
# =============================================================================
def extract_key_waypoints(
    cells: List[Tuple[int, int]],
    env: GridEnv,
    extra_blocked: Optional[set] = None,
) -> np.ndarray:
    """
    Extracts key turning waypoints from the discrete grid path while preserving
    collision-free visibility and anchoring intermediate points for minimum-snap fitting.
    """
    pts = np.array([env.cell_to_pt(c) for c in cells], dtype=float)
    if len(pts) <= 2:
        return pts

    # Line-of-sight pruning to remove staircase artifacts while keeping obstacle clearance
    pruned = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not env.line_of_sight(pts[i], pts[j], extra_blocked):
            j -= 1
        pruned.append(pts[j])
        i = j
    pruned_arr = np.array(pruned, dtype=float)

    # Subdivide long segments so the polynomial trajectory stays tightly bound to the corridor
    refined = [pruned_arr[0]]
    for k in range(len(pruned_arr) - 1):
        a, b = pruned_arr[k], pruned_arr[k + 1]
        seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n_sub = max(1, int(math.ceil(seg_len / 4.5)))
        for s in range(1, n_sub + 1):
            refined.append(a + (b - a) * (s / n_sub))
    return np.array(refined, dtype=float)


def minimum_snap_smooth(
    cells: List[Tuple[int, int]],
    env: GridEnv,
    extra_blocked: Optional[set] = None,
    ds: float = 0.05,
) -> np.ndarray:
    """
    Minimum-snap trajectory optimization (Section 4):
      min f(p) = integral (p^(4)(t))^2 dt  subject to A_eq p = b_eq.
    Solved on the discretized trajectory via penalized fourth-difference energy
    minimization with pinned start/end points and collision-free corridor clamping.
    """
    waypoints = extract_key_waypoints(cells, env, extra_blocked)
    if len(waypoints) < 2:
        return waypoints

    # Densify linearly first
    seg = np.hypot(*np.diff(waypoints, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(s_cum[-1])
    if total_len < 1e-6:
        return waypoints

    n_pts = max(10, int(round(total_len / ds)) + 1)
    q = np.linspace(0.0, total_len, n_pts)
    P_ref = np.column_stack([
        np.interp(q, s_cum, waypoints[:, 0]),
        np.interp(q, s_cum, waypoints[:, 1]),
    ])

    if n_pts < 8:
        return P_ref

    # Construct 4th-derivative finite-difference operator D4 (minimizing snap (p^(4))^2)
    # and 2nd-derivative operator D2 for tension regularization:
    #   min ||D4 * P||^2 + w_acc * ||D2 * P||^2 + w_fit * ||P - P_ref||^2
    # with exact equality constraints on start/end positions and velocities.
    idx = np.arange(n_pts - 4)
    D4 = np.zeros((n_pts - 4, n_pts), dtype=float)
    D4[idx, idx] = 1.0
    D4[idx, idx + 1] = -4.0
    D4[idx, idx + 2] = 6.0
    D4[idx, idx + 3] = -4.0
    D4[idx, idx + 4] = 1.0

    idx2 = np.arange(n_pts - 2)
    D2 = np.zeros((n_pts - 2, n_pts), dtype=float)
    D2[idx2, idx2] = 1.0
    D2[idx2, idx2 + 1] = -2.0
    D2[idx2, idx2 + 2] = 1.0

    # Try progressively tighter tracking weights until the minimum-snap curve is 100% obstacle-free
    H_snap = D4.T @ D4
    H_acc = D2.T @ D2
    I_mat = np.eye(n_pts)

    for w_fit in (0.002, 0.008, 0.03, 0.12, 0.5, 2.0):
        W = w_fit * I_mat.copy()
        # Pin endpoints (A_eq p = b_eq at boundary positions/tangents)
        W[0, 0] = W[-1, -1] = 1e6
        W[1, 1] = W[-2, -2] = max(w_fit * 10.0, 0.5)
        A_sys = H_snap + 0.02 * H_acc + W
        rhs = W @ P_ref
        P_opt = np.linalg.solve(A_sys, rhs)
        P_opt[0] = P_ref[0]
        P_opt[-1] = P_ref[-1]
        if not env.points_blocked(P_opt):
            return P_opt

    return P_ref


# =============================================================================
# 5. MULTI-ROBOT PRIORITY & WAITING / REPLANNING STRATEGY (Section 3, Alg. 2)
# =============================================================================
@dataclass
class RobotTrajectory:
    name: str
    priority: int
    cells: List[Tuple[int, int]]
    poly: np.ndarray          # smoothed spatial path [M, 2] in grid coordinates
    pos: np.ndarray           # time-sampled positions [K, 2] at dt = 0.1s
    speeds: np.ndarray        # speed at each time step [K] in cells/s
    length_paper: float       # path length E in paper units (x10)
    time_s: float             # total movement time in seconds
    replan_count: int = 0
    wait_steps: int = 0


def rollout_trajectory_with_speed_profile(
    poly: np.ndarray,
    dt: float,
    V_f: float,
    slow_windows: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simulates robot motion along `poly` at initial speed V_f, applying any
    deceleration/waiting windows (t_start, t_end, V_reduced) from Eq. (9).
    Returns (pos_array, speed_array, length_paper, total_time_s).
    """
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len_cells = float(s_cum[-1])

    s_curr = 0.0
    t_curr = 0.0
    s_trace = [0.0]
    v_trace = [V_f]

    max_steps = 4000
    for _ in range(max_steps):
        if s_curr >= total_len_cells - 1e-6:
            break
        v_step = V_f
        if slow_windows:
            for t0, t1, v_red in slow_windows:
                if t0 <= t_curr <= t1:
                    v_step = min(v_step, v_red)
        s_next = min(total_len_cells, s_curr + max(v_step, 0.05) * dt)
        s_trace.append(s_next)
        v_trace.append(v_step)
        s_curr = s_next
        t_curr += dt

    s_arr = np.array(s_trace, dtype=float)
    pos = np.column_stack([
        np.interp(s_arr, s_cum, poly[:, 0]),
        np.interp(s_arr, s_cum, poly[:, 1]),
    ])
    speeds = np.array(v_trace, dtype=float)
    total_time_s = (len(pos) - 1) * dt
    return pos, speeds, total_len_cells * 10.0, total_time_s


def movement_direction_angle_deg(pos_a: np.ndarray, k_a: int, pos_b: np.ndarray, k_b: int) -> float:
    """
    Computes the angle alpha (in degrees, [0, 180]) between the movement directions
    of two robots at discrete step k (Section 3.2, Fig. 2).
    """
    ka0 = max(0, min(k_a - 2, len(pos_a) - 2))
    ka1 = min(len(pos_a) - 1, ka0 + 4)
    kb0 = max(0, min(k_b - 2, len(pos_b) - 2))
    kb1 = min(len(pos_b) - 1, kb0 + 4)

    va = pos_a[ka1] - pos_a[ka0]
    vb = pos_b[kb1] - pos_b[kb0]
    na = float(np.hypot(va[0], va[1]))
    nb = float(np.hypot(vb[0], vb[1]))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def find_first_conflict(
    pos_low: np.ndarray,
    higher_trajs: List[RobotTrajectory],
    d_safe_cells: float,
) -> Optional[Tuple[int, RobotTrajectory, float]]:
    """
    Finds the earliest discrete step k where `pos_low` comes within `d_safe_cells`
    of any higher-priority robot trajectory during their movement.
    """
    best_k = None
    best_hi = None
    best_dist = 1e18

    for hi_tr in higher_trajs:
        K = min(len(pos_low), len(hi_tr.pos))
        if K == 0:
            continue
        dists = np.hypot(
            pos_low[:K, 0] - hi_tr.pos[:K, 0],
            pos_low[:K, 1] - hi_tr.pos[:K, 1],
        )
        bad = np.where(dists < d_safe_cells)[0]
        if len(bad) > 0:
            k0 = int(bad[0])
            d_min = float(np.min(dists[bad]))
            if best_k is None or k0 < best_k:
                best_k = k0
                best_hi = hi_tr
                best_dist = d_min

    if best_k is None:
        return None
    return best_k, best_hi, best_dist


def min_pairwise_distance_paper(
    pos_a: np.ndarray,
    higher_trajs: List[RobotTrajectory],
    scale: float = 10.0,
) -> float:
    """Computes minimum Euclidean distance (in paper units) between pos_a and higher-priority robots."""
    if not higher_trajs:
        return 999.0
    min_d = 999.0
    for hi_tr in higher_trajs:
        K = min(len(pos_a), len(hi_tr.pos))
        if K == 0:
            continue
        d = np.hypot(pos_a[:K, 0] - hi_tr.pos[:K, 0], pos_a[:K, 1] - hi_tr.pos[:K, 1]) * scale
        min_d = min(min_d, float(np.min(d)))
    return min_d


def build_temporary_obstacle_cells(
    env: GridEnv,
    hi_pos: np.ndarray,
    k_conflict: int,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    radius_cells: int = 2,
) -> set:
    """
    Marks the spatial region occupied by the higher-priority robot around the
    conflict time step as a temporary obstacle for Improved SA replanning
    (Section 2.3 & Section 3.1).
    """
    k_cl = min(max(0, k_conflict), len(hi_pos) - 1)
    blocked = set()
    for k in range(max(0, k_cl - 6), min(len(hi_pos), k_cl + 7)):
        cx, cy = env.pt_to_cell((float(hi_pos[k, 0]), float(hi_pos[k, 1])))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells + 1:
                    c = (cx + dx, cy + dy)
                    if c != start_cell and c != goal_cell:
                        blocked.add(c)
    return blocked


def multi_robot_improved_sa_planner(
    env: GridEnv,
    robots: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    params: SAParams,
    seed: int = 0,
) -> Tuple[Dict[str, RobotTrajectory], Dict[str, float]]:
    """
    Algorithm 2: Multi-robot path planning algorithm based on Improved SA,
    priority rules, angle check (alpha > 120 deg), time-difference evaluation F (Eq. 10),
    and deceleration/waiting strategy (Eq. 9).
    """
    rng = np.random.default_rng(seed)
    order = list(robots.keys())
    d_safe_cells = params.d_safe_paper / params.cell_scale

    trajs: Dict[str, RobotTrajectory] = {}
    cpu_times: Dict[str, float] = {}

    # Lines 1-3 of Algorithm 2: Obtain initial Improved SA path x^i for each robot
    initial_cells: Dict[str, List[Tuple[int, int]]] = {}
    for idx, r_name in enumerate(order):
        t0 = time.perf_counter()
        s_pt, g_pt = robots[r_name]
        cells = improved_sa_single_robot(env, s_pt, g_pt, params, rng)
        poly = minimum_snap_smooth(cells, env)
        pos, speeds, len_paper, time_s = rollout_trajectory_with_speed_profile(
            poly, params.dt, params.V_f
        )
        initial_cells[r_name] = cells
        trajs[r_name] = RobotTrajectory(
            name=r_name,
            priority=idx + 1,
            cells=cells,
            poly=poly,
            pos=pos,
            speeds=speeds,
            length_paper=len_paper,
            time_s=time_s,
        )
        cpu_times[r_name] = time.perf_counter() - t0

    # Lines 4-18 of Algorithm 2: Priority-based collision checking, Eq. (9)-(10) & replanning/waiting
    for idx, r_name in enumerate(order):
        if idx == 0:
            continue  # Robot 1 ('a') has highest priority (Priority 1) and keeps its optimal path

        t0 = time.perf_counter()
        s_pt, g_pt = robots[r_name]
        s_cell, g_cell = env.pt_to_cell(s_pt), env.pt_to_cell(g_pt)
        higher_trajs = [trajs[m] for m in order[:idx]]

        curr_tr = trajs[r_name]
        conflict = find_first_conflict(curr_tr.pos, higher_trajs, d_safe_cells)
        if conflict is None:
            cpu_times[r_name] += time.perf_counter() - t0
            continue

        k_conf, hi_tr, min_d_cells = conflict
        t_conf = k_conf * params.dt

        # Compute movement direction angle alpha between the two meeting robots (Fig. 2)
        alpha_deg = movement_direction_angle_deg(curr_tr.pos, k_conf, hi_tr.pos, k_conf)

        # Build candidate 1: Replanned spatial detour using Improved SA around higher-priority robot
        best_replan_tr: Optional[RobotTrajectory] = None
        acc_blocked = set()
        for radius in (2, 3):
            temp_obs = build_temporary_obstacle_cells(
                env, hi_tr.pos, k_conf, s_cell, g_cell, radius_cells=radius
            )
            acc_blocked |= temp_obs
            cells_new = improved_sa_single_robot(env, s_pt, g_pt, params, rng, extra_blocked=acc_blocked)
            if cells_new is not None:
                poly_new = minimum_snap_smooth(cells_new, env, extra_blocked=acc_blocked)
                pos_new, sp_new, len_new, time_new = rollout_trajectory_with_speed_profile(
                    poly_new, params.dt, params.V_f
                )
                if find_first_conflict(pos_new, higher_trajs, d_safe_cells) is None:
                    best_replan_tr = RobotTrajectory(
                        name=r_name,
                        priority=idx + 1,
                        cells=cells_new,
                        poly=poly_new,
                        pos=pos_new,
                        speeds=sp_new,
                        length_paper=len_new,
                        time_s=time_new,
                        replan_count=1,
                    )
                    break

        # Build candidate 2: Waiting / deceleration strategy (Section 3.2, Eq. 9)
        # Speed drops to V = k * V_f prior to the conflict window so higher-priority robot passes first
        best_wait_tr: Optional[RobotTrajectory] = None
        for wait_dur in (3.0, 4.5, 6.0, 7.5, 9.0, 11.0):
            for k_fac in (params.k_slow, 0.25, 0.15):
                V_red = k_fac * params.V_f
                t_start = max(0.0, t_conf - 0.65 * wait_dur)
                t_end = t_start + wait_dur
                pos_w, sp_w, len_w, time_w = rollout_trajectory_with_speed_profile(
                    curr_tr.poly, params.dt, params.V_f, slow_windows=[(t_start, t_end, V_red)]
                )
                if find_first_conflict(pos_w, higher_trajs, d_safe_cells) is None:
                    best_wait_tr = RobotTrajectory(
                        name=r_name,
                        priority=idx + 1,
                        cells=curr_tr.cells,
                        poly=curr_tr.poly,
                        pos=pos_w,
                        speeds=sp_w,
                        length_paper=len_w,
                        time_s=time_w,
                        wait_steps=int(round(wait_dur / params.dt)),
                    )
                    break
            if best_wait_tr is not None:
                break

        # Evaluate Eq. (10) time-difference function F and angle rule (alpha >= 120 deg):
        #   F = L_n / V_f - (L_x / V + (L_m - L_x) / V_f)
        # Note from Section 3.2 & Algorithm 2:
        #   - If alpha >= 120 deg (robots moving nearly head-on), deceleration is NOT performed;
        #     replanning with Improved SA is mandatory (Algorithm 2 lines 8-11).
        #   - Otherwise compare travel time of replanned path vs deceleration/waiting (Eq. 10).
        frac_done = min(0.95, (k_conf * params.dt) / max(curr_tr.time_s, 1e-6))
        L_m = curr_tr.length_paper * (1.0 - frac_done)
        L_x = min(L_m, params.sensor_range_paper)
        V_f_paper = params.V_f * params.cell_scale
        V_slow_paper = params.k_slow * V_f_paper
        L_n = (
            best_replan_tr.length_paper - curr_tr.length_paper * frac_done
            if best_replan_tr is not None
            else L_m * 1.25
        )
        F_val = (L_n / V_f_paper) - ((L_x / V_slow_paper) + (L_m - L_x) / V_f_paper)

        if alpha_deg >= params.head_on_angle_deg and best_replan_tr is not None:
            trajs[r_name] = best_replan_tr
        elif best_replan_tr is not None and best_wait_tr is not None:
            # Choose the collision-free strategy that minimizes travel time (Section 3.2)
            if best_replan_tr.time_s <= best_wait_tr.time_s or F_val < 0.0:
                trajs[r_name] = best_replan_tr
            else:
                trajs[r_name] = best_wait_tr
        elif best_replan_tr is not None:
            trajs[r_name] = best_replan_tr
        elif best_wait_tr is not None:
            trajs[r_name] = best_wait_tr

        cpu_times[r_name] += time.perf_counter() - t0

    return trajs, cpu_times


# =============================================================================
# 6. EVALUATION METRICS & VISUALIZATION (Table 2, Table 3, Fig. 5(c) & Fig. 6)
# =============================================================================
def evaluate_solution(
    env: GridEnv,
    trajs: Dict[str, RobotTrajectory],
    cpu_times: Dict[str, float],
    params: SAParams,
) -> List[Dict[str, float]]:
    rows = []
    for idx, r_name in enumerate(NAMES):
        tr = trajs[r_name]
        min_d_paper = 999.0
        for o_name in NAMES:
            if o_name == r_name:
                continue
            other = trajs[o_name]
            K = min(len(tr.pos), len(other.pos))
            dists = np.hypot(
                tr.pos[:K, 0] - other.pos[:K, 0],
                tr.pos[:K, 1] - other.pos[:K, 1],
            ) * params.cell_scale
            min_d_paper = min(min_d_paper, float(np.min(dists)))

        obs_collision = env.points_blocked(tr.poly)
        success = (not obs_collision) and (min_d_paper >= 20.0)
        rows.append({
            'robot_idx': idx + 1,
            'name': r_name,
            'priority': tr.priority,
            'path_length': tr.length_paper,
            'min_distance': min_d_paper,
            'movement_time': tr.time_s,
            'cpu_time': cpu_times[r_name],
            'success': 100.0 if success else 0.0,
        })
    return rows


def print_paper_tables(scenario_name: str, rows: List[Dict[str, float]]) -> None:
    env_label = "Static environment" if scenario_name == "static" else "Barrier-free environment"
    print("\n" + "=" * 95)
    print(f"  Table 2 & Table 3 Reproduction — {env_label} (Improved SA with Priority)")
    print("=" * 95)
    print(
        f"{'Environment':24} | {'Robot':8} | {'Priority':>8} | "
        f"{'Path length':>12} | {'Minimum distance':>16} | {'Movement time (s)':>17}"
    )
    print("-" * 95)
    for r in rows:
        env_col = env_label if r['robot_idx'] == 1 else ""
        print(
            f"{env_col:24} | {'Robot ' + str(r['robot_idx']):8} | {r['priority']:8d} | "
            f"{r['path_length']:12.2f} | {r['min_distance']:16.2f} | {r['movement_time']:17.2f}"
        )
    print("-" * 95)
    avg_len = float(np.mean([r['path_length'] for r in rows]))
    max_len = float(np.max([r['path_length'] for r in rows]))
    avg_time = float(np.mean([r['movement_time'] for r in rows]))
    min_spacing = float(np.min([r['min_distance'] for r in rows]))
    succ_rate = float(np.mean([r['success'] for r in rows]))
    avg_cpu = float(np.mean([r['cpu_time'] for r in rows]))
    print(
        f"  Table 3 Summary ({env_label}):\n"
        f"    Average path length   : {avg_len:.2f} (Paper static Improved SA: 402.12)\n"
        f"    Maximum path length   : {max_len:.2f} (Paper static Improved SA: 481.75)\n"
        f"    Average movement time : {avg_time:.2f} s (Paper static Improved SA: 41.89 s)\n"
        f"    Minimum robot spacing : {min_spacing:.2f} (> 20.0 safety threshold)\n"
        f"    Success rate          : {succ_rate:.0f}%\n"
        f"    Average CPU time/robot: {avg_cpu:.3f} s"
    )
    print("=" * 95)


def plot_paper_figures(
    env: GridEnv,
    robots: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    trajs: Dict[str, RobotTrajectory],
    scenario_name: str,
    out_path: str,
    params: SAParams,
) -> None:
    """
    Generates a two-panel figure matching Fig. 5(c) (Paths of Proposed Algorithm
    in a static environment) and Fig. 6 (Distance between the robots in a static
    environment) from Shi et al. (2025).
    """
    env_title = "static environment" if scenario_name == "static" else "barrier-free environment"
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # Left panel: Fig. 5(c) / Fig. 3(c) Path plot on 40x40 grid
    ax0 = axs[0]
    ax0.set_xlim(0, 40)
    ax0.set_ylim(0, 40)
    ax0.set_aspect("equal")
    ax0.set_xticks(range(0, 41, 5))
    ax0.set_yticks(range(0, 41, 5))
    # Draw 1x1 minor grid and 5x5 major grid like the paper
    ax0.set_xticks(range(0, 41, 1), minor=True)
    ax0.set_yticks(range(0, 41, 1), minor=True)
    ax0.grid(which="minor", color="#dcdcdc", linestyle="-", linewidth=0.35)
    ax0.grid(which="major", color="#b0b0b0", linestyle="-", linewidth=0.6)
    ax0.set_title(f"(a) Proposed Improved SA paths ({env_title})", fontsize=11, weight="bold", pad=8)

    for x0, y0, x1, y1 in env.rects:
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="k", zorder=2))

    for idx, r_name in enumerate(NAMES):
        tr = trajs[r_name]
        c = COLORS[idx]
        s_pt, g_pt = robots[r_name]
        ax0.plot(
            tr.pos[:, 0],
            tr.pos[:, 1],
            color=c,
            lw=2.2,
            zorder=3,
            label=f"robot {r_name} (L={tr.length_paper:.1f}, t={tr.time_s:.1f}s)",
        )
        # Green start square, red target square (Section 4.1)
        ax0.plot(s_pt[0], s_pt[1], "s", color="limegreen", mec="k", ms=7.0, zorder=4)
        ax0.plot(g_pt[0], g_pt[1], "s", color="red", mec="k", ms=7.0, zorder=4)

        # Direction arrow near the beginning of each robot trajectory
        k_arr = min(15, max(2, len(tr.pos) // 8))
        ax0.annotate(
            "",
            xy=tr.pos[min(k_arr + 5, len(tr.pos) - 1)],
            xytext=tr.pos[k_arr],
            arrowprops=dict(arrowstyle="->", color="k", lw=1.3),
            zorder=5,
        )

    ax0.legend(loc="upper right", fontsize=8.5, frameon=True, edgecolor="black")

    # Right panel: Fig. 6 / Fig. 4 Pairwise inter-robot distances over time
    ax1 = axs[1]
    ax1.set_title(
        f"(b) Distance between the robots in a {env_title}",
        fontsize=11,
        weight="bold",
        pad=8,
    )
    pair_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    p_idx = 0
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            ra, rb = NAMES[i], NAMES[j]
            pos_a, pos_b = trajs[ra].pos, trajs[rb].pos
            K = min(len(pos_a), len(pos_b))
            t_vec = np.arange(K) * params.dt
            d_vec = np.hypot(pos_a[:K, 0] - pos_b[:K, 0], pos_a[:K, 1] - pos_b[:K, 1]) * params.cell_scale
            ax1.plot(
                t_vec,
                d_vec,
                lw=1.3,
                color=pair_colors[p_idx],
                label=f"distance of robot {ra} and {rb}",
            )
            p_idx += 1

    ax1.axhline(20.0, ls="--", color="k", lw=1.0, alpha=0.7, label="safety threshold (20)")
    ax1.set_xlabel("Time(s)", fontsize=10)
    ax1.set_ylabel("Distance", fontsize=10)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 500)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Improved Simulated Annealing Multi-Robot Path Planning (Shi et al., 2025)"
    )
    parser.add_argument("--scenario", default="static", choices=["static", "free", "both"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="results_improved_sa")
    args = parser.parse_args()

    params = SAParams()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = ["static", "free"] if args.scenario == "both" else [args.scenario]
    for sc in scenarios:
        rects, robots = SCENARIOS[sc]
        env = GridEnv(rects, params)
        trajs, cpu_times = multi_robot_improved_sa_planner(env, robots, params, seed=args.seed)
        rows = evaluate_solution(env, trajs, cpu_times, params)
        print_paper_tables(sc, rows)
        fig_path = os.path.join(args.out_dir, f"improved_sa_{sc}_results.png")
        plot_paper_figures(env, robots, trajs, sc, fig_path, params)
        print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()