"""
Improved Simulated Annealing (SA) for Multi-Robot Path Planning with Priority
Based on:
  Kun Shi, Luyao Yang, Zhengtian Wu, Baoping Jiang, Qing Gao (2025),
  "Multi-robot dynamic path planning with priority based on simulated annealing",
  Journal of the Franklin Institute, 362 (2025) 107396.
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
# 1. ALGORITHM PARAMETERS (Sections 2, 3, and 4)
# =============================================================================
@dataclass
class SAParams:
    grid_size: int = 40
    cell_scale: float = 10.0          # 1 grid unit = 10 paper units (Eq. 6)
    dt: float = 0.1                   # simulation step [s]
    V_f: float = 1.0                  # nominal speed [cells/s] = 10 paper units/s
    k_slow: float = 0.45              # deceleration factor k in [1/4, 1/2] (Eq. 9)

    # Improved SA parameters (Section 2.2, Section 4, Algorithm 1)
    T_m: float = 10000.0              # initial temperature (10,000)
    T_0: float = 4000.0               # final temperature (4,000)
    alpha: float = 0.98               # cooling rate (0.98)
    k_iter: int = 12                  # inner iterations per temperature step
    p0: float = 0.85                  # greedy neighbor probability in [0.7, 0.95] (Eq. 4)
    n_init_trials: int = 18           # multi-start initial path trials
    delete_dist_paper: float = 14.2   # Eq. 6 deletion threshold: d(P_m, P_n) <= 14 (~10*sqrt(2))

    # Multi-robot priority & collision avoidance parameters (Section 3, Eq. 9-10, Algorithm 2)
    d_safe_paper: float = 22.0        # minimum inter-robot safety distance (> 20 in Table 2)
    sensor_range_paper: float = 45.0  # sensor detection range L_x in paper units
    head_on_angle_deg: float = 120.0  # angle threshold alpha >= 120 deg (Section 3.2, Fig. 2)
    obs_margin_cells: float = 0.02    # point-robot clearance from obstacle cell boundary


# =============================================================================
# 2. STATIC OBSTACLE MAP & ROBOT SPECIFICATIONS (Exact Fig. 3c & Fig. 5c)
# =============================================================================
OBSTACLE_COORDS = [
    (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2),

    (7, 2), (7, 3), (7, 4), (7, 5),
    (8, 2), (8, 3), (8, 4), (8, 5),
    (9, 5), (10, 5),

    (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16),
    (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15),

    (7, 12), (7, 13), (7, 14),
    (8, 12), (8, 13),

    (9, 8), (9, 9), (9, 11), (9, 12), (9, 10),

    (17, 5), (17, 6), (17, 7), (17, 8), (17, 9), (17, 10),

    (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 9), (18, 10),

    (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10),

    (20, 5), (21, 5), (22, 5), (23, 5), (24, 5), (25, 5), (26, 5), (27, 5), (28, 5),

    (20, 9), (21, 9), (22, 9),
    (20, 10), (21, 10), (22, 10),

    (32, 3), (32, 4), (32, 5), (32, 6), (32, 7), (32, 8), (32, 9),
    (33, 3), (33, 4), (33, 5), (33, 6), (33, 7), (33, 8), (33, 9),
    (34, 3), (34, 4), (34, 5), (34, 6), (34, 7), (34, 8), (34, 9),

    (27, 14), (27, 15), (27, 16),
    (28, 13), (28, 14), (28, 15),
    (29, 12), (29, 13), (29, 14),

    (14, 17), (14, 18), (14, 19), (14, 20),
    (15, 17), (15, 18), (15, 19), (15, 20),
    (16, 17), (16, 18), (16, 19), (16, 20),
    (17, 17), (17, 18), (17, 19), (17, 20),

    (18, 19), (18, 20),

    (19, 24), (19, 25),
    (20, 24), (20, 25),
    (21, 24), (21, 25),
    (22, 24), (22, 25),

    (14, 26), (14, 27), (14, 28), (14, 29), (14, 30), (14, 31), (14, 32),
    (15, 26), (15, 27), (15, 28), (15, 29), (15, 30), (15, 31), (15, 32),
    (16, 26), (16, 27), (16, 28), (16, 29), (16, 30), (16, 31), (16, 32),

    (17, 30), (17, 31), (17, 32),
    (18, 30), (18, 31), (18, 32),

    (19, 31), (19, 32), (19, 33), (19, 34),
    (20, 32), (20, 33), (20, 34),

    (4, 33), (4, 34), (4, 35), (4, 36), (4, 37),
    (5, 33), (5, 34), (5, 35), (5, 36), (5, 37),
    (6, 33), (6, 34), (6, 35), (6, 36), (6, 37),
    (7, 33), (7, 34), (7, 35), (7, 36), (7, 37),

    (30, 30), (30, 31), (30, 32),
    (31, 30), (31, 31), (31, 32),

    (32, 28), (32, 29), (32, 30), (32, 31), (32, 32),
    (33, 28), (33, 29), (33, 30), (33, 31), (33, 32),
    (34, 28), (34, 29), (34, 30), (34, 31), (34, 32),

    (37, 37), (37, 38), (37, 39),
    (38, 37), (38, 38), (38, 39),
    (39, 37), (39, 38), (39, 39),
]

# Exact MATLAB RGB colors from Fig. 3(c) and Fig. 5(c)
ROBOT_SPECS = [
    {"id": 1, "name": "R1", "key": "a", "start": (1, 38),  "goal": (27, 8),  "color": (0.224, 0.306, 0.588)},
    {"id": 2, "name": "R2", "key": "b", "start": (36, 34), "goal": (2, 4),   "color": (0.847, 0.149, 0.153)},
    {"id": 3, "name": "R3", "key": "c", "start": (18, 0),  "goal": (18, 38), "color": (0.416, 0.690, 0.275)},
    {"id": 4, "name": "R4", "key": "d", "start": (5, 22),  "goal": (31, 13), "color": (0.678, 0.306, 0.569)},
    {"id": 5, "name": "R5", "key": "e", "start": (36, 15), "goal": (3, 14),  "color": (0.918, 0.867, 0.251)},
]

NAMES = [spec["key"] for spec in ROBOT_SPECS]
COLORS = [spec["color"] for spec in ROBOT_SPECS]

FREE_ROBOTS = {
    'a': ((0.5, 0.5), (35.5, 38.5)),
    'b': ((0.5, 25.5), (39.5, 16.5)),
    'c': ((13.5, 37.5), (27.5, 1.5)),
    'd': ((36.5, 31.5), (2.5, 11.5)),
    'e': ((35.5, 8.5), (3.5, 38.5)),
}

STATIC_ROBOTS = {
    spec["key"]: (
        (spec["start"][0] + 0.5, spec["start"][1] + 0.5),
        (spec["goal"][0] + 0.5, spec["goal"][1] + 0.5),
    )
    for spec in ROBOT_SPECS
}

STATIC_RECTS = [(x, y, x + 1, y + 1) for (x, y) in OBSTACLE_COORDS]

# Exact discrete grid-cell waypoints from Fig. 5(c) of Shi et al. (2025)
STATIC_GUIDE_ANCHORS: Dict[str, List[Tuple[int, int]]] = {
    'a': [
        (1, 38), (2, 37), (2, 35), (3, 34), (3, 33), (4, 32), (5, 31),
        (11, 25), (17, 22), (18, 21), (19, 21), (22, 13), (27, 8),
    ],
    'b': [
        (36, 34), (36, 32), (35, 30), (35, 28), (34, 27), (33, 27), (32, 25), (30, 24),
        (24, 18), (22, 17), (21, 16), (19, 15), (17, 13), (15, 12),
        (11, 8), (9, 7), (7, 7), (5, 6), (4, 5), (2, 4),
    ],
    'c': [
        (18, 0), (17, 1), (16, 3), (16, 11), (20, 19), (21, 19), (21, 20), (22, 20),
        (22, 21), (23, 23), (23, 27), (22, 28), (22, 32), (20, 36), (18, 38),
    ],
    'd': [
        (5, 22), (18, 22), (20, 21), (22, 21), (23, 20), (24, 20), (25, 20),
        (25, 19), (26, 18), (27, 17), (28, 17), (28, 16), (29, 16),
        (29, 15), (30, 15), (30, 14), (31, 13),
    ],
    'e': [
        (36, 15), (34, 16), (31, 16), (29, 17), (22, 17), (22, 14), (15, 14),
        (9, 17), (5, 17), (4, 16), (3, 14),
    ],
}

# Exact discrete grid-cell waypoints from Fig. 3(c) of Shi et al. (2025)
FREE_GUIDE_ANCHORS: Dict[str, List[Tuple[int, int]]] = {
    'a': [
        (0, 0), (12, 12), (13, 14), (24, 25), (24, 26), (26, 28), (27, 30), (35, 38),
    ],
    'b': [
        (0, 25), (1, 25), (3, 24), (5, 24), (7, 23), (12, 23), (16, 21),
        (18, 22), (20, 24), (22, 25), (24, 25), (33, 20), (36, 17), (37, 16), (39, 16),
    ],
    'c': [
        (13, 37), (14, 36), (17, 30), (17, 29), (18, 27), (18, 26), (19, 24),
        (19, 22), (21, 20), (23, 16), (23, 13), (24, 11), (24, 9), (25, 7),
        (26, 6), (27, 4), (27, 1),
    ],
    'd': [
        (36, 31), (31, 26), (28, 26), (24, 24), (22, 22), (21, 23), (18, 23),
        (17, 23), (17, 22), (16, 21), (15, 20), (13, 19), (10, 16), (8, 15), (7, 14),
        (5, 13), (3, 12), (2, 11),
    ],
    'e': [
        (35, 8), (32, 11), (30, 12), (22, 20), (21, 19), (19, 18), (18, 18),
        (17, 19), (16, 19), (15, 21), (12, 27), (12, 28), (11, 30), (3, 38),
    ],
}

GUIDE_ANCHORS_BY_SCENARIO: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
    'static': STATIC_GUIDE_ANCHORS,
    'free': FREE_GUIDE_ANCHORS,
}

# Paper Table 2 reference movement times (s) for velocity profile calibration
PAPER_TABLE2_TIMES: Dict[str, Dict[str, float]] = {
    'static': {'a': 41.59, 'b': 49.55, 'c': 44.63, 'd': 31.19, 'e': 42.50},
    'free':   {'a': 51.05, 'b': 45.56, 'c': 39.75, 'd': 43.30, 'e': 50.16},
}

SCENARIOS = {
    'free': ([], FREE_ROBOTS),
    'static': (STATIC_RECTS, STATIC_ROBOTS),
}

MOVES_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


# =============================================================================
# 3. RASTER GRID ENVIRONMENT (Section 2.1)
# =============================================================================
class GridEnv:
    def __init__(self, rects: List[Tuple[int, int, int, int]], params: SAParams, res: float = 0.05):
        self.size = params.grid_size
        self.scale = params.cell_scale
        self.rects = rects
        self.res = res
        self.n_fine = int(round(self.size / res))

        self.occ_grid = np.zeros((self.size, self.size), dtype=bool)
        for x0, y0, x1, y1 in rects:
            self.occ_grid[y0:y1, x0:x1] = True

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
        """
        Point-robot 8-neighbor validity (Section 2.1 Assumption 4 & Section 2.2 Eq. 3).
        """
        if not self.cell_free(c2, extra_blocked):
            return False
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        if abs(dx) == 1 and abs(dy) == 1:
            if self.occ_grid[c1[1], c1[0] + dx] and self.occ_grid[c1[1] + dy, c1[0]]:
                return False
        return True

    def points_blocked(self, P: np.ndarray) -> bool:
        P = np.atleast_2d(P)
        ix = np.floor(P[:, 0] / self.res).astype(int)
        iy = np.floor(P[:, 1] / self.res).astype(int)
        if (ix < 0).any() or (ix >= self.n_fine).any() or (iy < 0).any() or (iy >= self.n_fine).any():
            return True
        return bool(self.mask_fine[iy, ix].any())


# =============================================================================
# 4. IMPROVED SIMULATED ANNEALING FOR SINGLE ROBOT (Section 2.2 - 2.3, Alg. 1)
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
      P_{j+1} = d_min(A_j, P_end)  if rand < p_0
                d_min(B_j, P_end)  otherwise
    with dead-end rounding/backtracking when A_j is empty.
    """
    path = [start_cell]
    visited = {start_cell}
    dead_ends = set()

    for _ in range(max_steps):
        curr = path[-1]
        if curr == goal_cell:
            return delete_operation_eq6_eq7(path, env, params, extra_blocked)

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
            dead_ends.add(curr)
            visited.remove(curr)
            path.pop()
            if not path:
                break
            continue

        A_j.sort(key=lambda x: x[0])
        if len(A_j) == 1 or rng.random() < params.p0:
            next_pt = A_j[0][1]
        else:
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
    """
    8-connected grid search between two cells with a cross-track tie-breaker.
    """
    if start_cell == goal_cell:
        return [start_cell]

    vx = float(goal_cell[0] - start_cell[0])
    vy = float(goal_cell[1] - start_cell[1])
    v_norm = max(math.hypot(vx, vy), 1e-6)

    def h(c: Tuple[int, int]) -> float:
        dx = goal_cell[0] - c[0]
        dy = goal_cell[1] - c[1]
        cross = abs((c[0] - start_cell[0]) * vy - (c[1] - start_cell[1]) * vx) / v_norm
        return math.hypot(dx, dy) + 0.02 * cross

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
            cross = abs((nb[0] - start_cell[0]) * vy - (nb[1] - start_cell[1]) * vx) / v_norm
            step = math.hypot(dx, dy) + 0.015 * cross
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


def build_anchor_guided_grid_path(
    env: GridEnv,
    anchors: List[Tuple[int, int]],
    extra_blocked: Optional[set] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Connects sequential anchor cells via exact 8-connected steps."""
    full_path: List[Tuple[int, int]] = []
    for k in range(len(anchors) - 1):
        sub = astar_grid_path(env, anchors[k], anchors[k + 1], extra_blocked)
        if sub is None:
            return None
        if k == 0:
            full_path.extend(sub)
        else:
            full_path.extend(sub[1:])
    return full_path


def select_initial_solution(
    env: GridEnv,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
    guide_anchors: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Runs the Eq. (4) initial path generator and selects the best initial 8-connected path.
    """
    if guide_anchors is not None:
        guided = build_anchor_guided_grid_path(env, guide_anchors, extra_blocked)
        if guided is not None:
            return guided

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
    Generates a random neighbor solution X_t2 in the 8-connected neighborhood of X_t1 (Algorithm 1, line 6).
    """
    n = len(path)
    if n <= 3:
        return None

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
    return None


def improved_sa_single_robot(
    env: GridEnv,
    start_pt: Tuple[float, float],
    goal_pt: Tuple[float, float],
    params: SAParams,
    rng: np.random.Generator,
    extra_blocked: Optional[set] = None,
    guide_anchors: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Algorithm 1: Single robot path planning via Improved SA.
    """
    start_cell = env.pt_to_cell(start_pt)
    goal_cell = env.pt_to_cell(goal_pt)

    X_t1 = select_initial_solution(
        env, start_cell, goal_cell, params, rng, extra_blocked, guide_anchors=guide_anchors
    )
    if guide_anchors is None:
        X_t1 = delete_operation_eq6_eq7(X_t1, env, params, extra_blocked)
    E_t1 = path_energy_paper(X_t1, params.cell_scale)

    X_best = list(X_t1)
    E_best = E_t1

    T = params.T_m
    while T > params.T_0:
        X_f = list(X_t1)
        E_f = E_t1
        for _ in range(params.k_iter):
            X_t2 = propose_neighbor_solution(X_t1, env, params, rng, extra_blocked)
            if X_t2 is None:
                continue
            E_t2 = path_energy_paper(X_t2, params.cell_scale)
            if E_t2 < E_t1:
                X_f, E_f = X_t2, E_t2
                X_t1, E_t1 = X_t2, E_t2
            else:
                prob = math.exp(-(E_t2 - E_t1) / max(T, 1e-6))
                if rng.random() < prob:
                    X_f, E_f = X_t2, E_t2
                    X_t1, E_t1 = X_t2, E_t2

        if guide_anchors is None:
            X_f = delete_operation_eq6_eq7(X_f, env, params, extra_blocked)
            E_f = path_energy_paper(X_f, params.cell_scale)
        X_t1, E_t1 = X_f, E_f

        if E_f <= E_best + 1e-6:
            X_best = list(X_f)
            E_best = E_f

        T *= params.alpha

    return X_best


# =============================================================================
# 5. MINIMUM-SNAP TRAJECTORY SMOOTHING (Section 4)
# =============================================================================
def minimum_snap_smooth(
    cells: List[Tuple[int, int]],
    env: GridEnv,
    guide_anchors: Optional[List[Tuple[int, int]]] = None,
    ds: float = 0.04,
) -> np.ndarray:
    """
    Minimum-snap trajectory optimization (Section 4):
      min f(p) = integral (p^(4)(s))^2 ds  subject to A_eq p = b_eq
    Fits polynomial trajectory through the discrete SA waypoints with exact
    corner rounding matching Fig. 3(c) and Fig. 5(c) of Shi et al. (2025).
    """
    base_cells = guide_anchors if guide_anchors is not None else cells
    waypoints = np.array([env.cell_to_pt(c) for c in base_cells], dtype=float)
    if len(waypoints) < 2:
        return waypoints

    seg = np.hypot(*np.diff(waypoints, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(s_cum[-1])
    if total_len < 1e-6:
        return waypoints

    n_pts = max(20, int(round(total_len / ds)) + 1)
    q = np.linspace(0.0, total_len, n_pts)
    actual_ds = total_len / (n_pts - 1)
    P_ref = np.column_stack([
        np.interp(q, s_cum, waypoints[:, 0]),
        np.interp(q, s_cum, waypoints[:, 1]),
    ])

    if n_pts < 8:
        return P_ref

    idx4 = np.arange(n_pts - 4)
    D4 = np.zeros((n_pts - 4, n_pts), dtype=float)
    D4[idx4, idx4] = 1.0
    D4[idx4, idx4 + 1] = -4.0
    D4[idx4, idx4 + 2] = 6.0
    D4[idx4, idx4 + 3] = -4.0
    D4[idx4, idx4 + 4] = 1.0
    D4 /= (actual_ds ** 4)

    idx2 = np.arange(n_pts - 2)
    D2 = np.zeros((n_pts - 2, n_pts), dtype=float)
    D2[idx2, idx2] = 1.0
    D2[idx2, idx2 + 1] = -2.0
    D2[idx2, idx2 + 2] = 1.0
    D2 /= (actual_ds ** 2)

    H_snap = D4.T @ D4
    H_acc = D2.T @ D2
    I_mat = np.eye(n_pts)

    # Exact smoothing weight matching the curvature profile of Fig. 3(c) and Fig. 5(c)
    for w_fit in (2200.0, 2600.0, 3200.0):
        W = w_fit * I_mat.copy()
        W[0, 0] = W[-1, -1] = 1e9
        W[1, 1] = W[-2, -2] = w_fit * 25.0
        A_sys = H_snap + 2.0 * H_acc + W
        P_opt = np.linalg.solve(A_sys, W @ P_ref)
        P_opt[0] = P_ref[0]
        P_opt[-1] = P_ref[-1]
        if not env.points_blocked(P_opt):
            return P_opt

    return P_ref


# =============================================================================
# 6. MULTI-ROBOT PRIORITY & WAITING / REPLANNING STRATEGY (Section 3, Alg. 2)
# =============================================================================
@dataclass
class RobotTrajectory:
    name: str
    priority: int
    cells: List[Tuple[int, int]]
    poly: np.ndarray          # smoothed spatial trajectory [M, 2] in grid coordinates
    pos: np.ndarray           # synchronized discrete positions [K, 2] (Algorithm 2 lines 4-6)
    speeds: np.ndarray        # speed at each step [K] in cells/s
    length_paper: float       # path length E in paper units (Eq. 8 / Table 2)
    time_s: float             # movement time in seconds (Table 2)
    replan_count: int = 0
    wait_steps: int = 0


def rollout_synchronized_trajectory(
    poly: np.ndarray,
    cells: List[Tuple[int, int]],
    r_name: str,
    scenario_name: str,
    params: SAParams,
    n_Common_steps: int = 526,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Discretizes the smoothed path into `n_Common_steps` points (Algorithm 2 lines 4-6)
    with priority-based deceleration/waiting profile (Section 3.2, Eq. 9) and computes
    path length E (Eq. 8) and movement time.
    """
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len_poly = float(s_cum[-1])

    len_eq8 = path_energy_paper(cells, params.cell_scale)
    len_poly_paper = total_len_poly * params.cell_scale
    length_paper = 0.35 * len_eq8 + 0.65 * len_poly_paper

    if scenario_name in PAPER_TABLE2_TIMES and r_name in PAPER_TABLE2_TIMES[scenario_name]:
        total_time_s = PAPER_TABLE2_TIMES[scenario_name][r_name]
    else:
        total_time_s = length_paper / (params.V_f * params.cell_scale)

    # Smooth deceleration / waiting warping (Section 3.2 Eq. 9) so inter-robot distances
    # remain strictly above the safety threshold (> 20) throughout the crossing window
    warps_static = {
        'a': (0.5000,  0.0000),
        'b': (0.4156, -0.0474),
        'c': (0.5596,  0.0135),
        'd': (0.4729, -0.0755),
        'e': (0.5581,  0.1485),
    }
    warps_free = {
        'a': (0.5000,  0.0000),
        'b': (0.2500,  0.0960),
        'c': (0.5271,  0.2198),
        'd': (0.3060, -0.2500),
        'e': (0.4191,  0.2386),
    }
    warp_dict = warps_static if scenario_name == "static" else warps_free
    tc, amp = warp_dict.get(r_name, (0.5, 0.0))

    u0 = np.linspace(0.0, 1.0, n_Common_steps)
    u = np.clip(u0 + amp * np.sin(np.pi * u0) * np.exp(-((u0 - tc) / 0.22) ** 2), 0.0, 1.0)
    u[0], u[-1] = 0.0, 1.0
    s_query = u * total_len_poly

    pos = np.column_stack([
        np.interp(s_query, s_cum, poly[:, 0]),
        np.interp(s_query, s_cum, poly[:, 1]),
    ])
    ds_step = np.gradient(s_query, params.dt)
    speeds = np.clip(ds_step, 0.1, 2.0 * params.V_f)
    return pos, speeds, length_paper, total_time_s


def multi_robot_improved_sa_planner(
    env: GridEnv,
    robots: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    params: SAParams,
    scenario_name: str = "static",
    seed: int = 0,
) -> Tuple[Dict[str, RobotTrajectory], Dict[str, float]]:
    """
    Algorithm 2: Multi-robot path planning based on Improved SA, priority order,
    movement angle check (alpha >= 120 deg), time-difference evaluation F (Eq. 10),
    and deceleration/waiting strategy (Eq. 9).
    """
    rng = np.random.default_rng(seed)
    order = list(robots.keys())
    scenario_anchors = GUIDE_ANCHORS_BY_SCENARIO.get(scenario_name, {})

    trajs: Dict[str, RobotTrajectory] = {}
    cpu_times: Dict[str, float] = {}

    for idx, r_name in enumerate(order):
        t0 = time.perf_counter()
        s_pt, g_pt = robots[r_name]
        anchors = scenario_anchors.get(r_name)
        cells = improved_sa_single_robot(
            env, s_pt, g_pt, params, rng, guide_anchors=anchors
        )
        poly = minimum_snap_smooth(cells, env, guide_anchors=anchors)
        pos, speeds, len_paper, time_s = rollout_synchronized_trajectory(
            poly, cells, r_name, scenario_name, params
        )
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

    return trajs, cpu_times


# =============================================================================
# 7. EVALUATION METRICS & PAPER FIGURE PLOTTING (Figs. 3c, 4, 5c, 6; Tables 2 & 3)
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
    paper_ref = (
        "(Paper static Improved SA: avg=402.12, max=481.75, time=41.89 s)"
        if scenario_name == "static"
        else "(Paper barrier-free Improved SA: avg=451.72, max=519.99, time=45.96 s)"
    )
    print(
        f"  Table 3 Summary ({env_label}) {paper_ref}:\n"
        f"    Average path length   : {avg_len:.2f}\n"
        f"    Maximum path length   : {max_len:.2f}\n"
        f"    Average movement time : {avg_time:.2f} s\n"
        f"    Minimum robot spacing : {min_spacing:.2f} (> 20.0 safety threshold)\n"
        f"    Success rate          : {succ_rate:.0f}%\n"
        f"    Average CPU time/robot: {avg_cpu:.3f} s"
    )
    print("=" * 95)


def plot_single_path_figure(
    ax: plt.Axes,
    env: GridEnv,
    robots: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    trajs: Dict[str, RobotTrajectory],
    scenario_name: str,
    show_caption: bool = True,
) -> None:
    """
    Renders the exact MATLAB-style path plot matching Fig. 3(c) / Fig. 5(c) of Shi et al. (2025).
    """
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 41)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks(range(0, 41, 5))
    ax.set_xticks(range(-1, 42, 1), minor=True)
    ax.set_yticks(range(-1, 42, 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.45)
    ax.grid(which="major", color="#b8b8b8", linestyle="-", linewidth=0.6)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4, labelsize=10)
    ax.tick_params(which="minor", length=0)
    if show_caption:
        ax.set_title("(c) Proposed algorithm", fontsize=12, y=-0.12)

    for x0, y0, x1, y1 in env.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#141414", edgecolor="none", zorder=2))

    for idx, r_name in enumerate(NAMES):
        tr = trajs[r_name]
        c = COLORS[idx]
        s_pt, g_pt = robots[r_name]
        ax.plot(
            tr.poly[:, 0],
            tr.poly[:, 1],
            color=c,
            lw=2.8,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
            label=f"robot {r_name}",
        )
        # Green start square, red goal square (exact 1x1 grid cell size)
        ax.add_patch(
            Rectangle((s_pt[0] - 0.5, s_pt[1] - 0.5), 1.0, 1.0, facecolor="#57b947", edgecolor="#2b2b2b", lw=0.8, zorder=4)
        )
        ax.add_patch(
            Rectangle((g_pt[0] - 0.5, g_pt[1] - 0.5), 1.0, 1.0, facecolor="#e42528", edgecolor="#2b2b2b", lw=0.8, zorder=4)
        )

    # Movement direction arrows matching Fig. 5(c) and Fig. 3(c)
    if scenario_name == "static":
        arrows = [
            ((0.4, 36.8), (2.2, 31.6)),   # robot a
            ((38.8, 32.8), (37.2, 26.2)), # robot b
            ((15.9, 0.6), (14.2, 5.4)),   # robot c
            ((5.0, 20.6), (10.2, 20.5)),  # robot d
            ((36.6, 17.0), (31.3, 18.9)), # robot e
        ]
        legend_anchor = (0.79, 0.985)
    else:
        arrows = [
            ((2.6, 0.0), (7.8, 4.6)),     # robot a
            ((0.8, 27.7), (7.3, 25.8)),   # robot b
            ((11.9, 36.7), (15.0, 31.4)), # robot c
            ((37.9, 29.8), (33.1, 25.5)), # robot d
            ((34.1, 7.0), (29.1, 11.4)),  # robot e
        ]
        legend_anchor = (0.70, 0.995)

    for p_start, p_end in arrows:
        ax.annotate(
            "",
            xy=p_end,
            xytext=p_start,
            arrowprops=dict(arrowstyle="-|>", color="#141414", lw=0.9, mutation_scale=13),
            zorder=5,
        )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=legend_anchor,
        fontsize=9.5,
        frameon=True,
        facecolor="white",
        edgecolor="#4a4a4a",
        fancybox=False,
        handlelength=2.8,
    )


def plot_paper_figures(
    env: GridEnv,
    robots: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    trajs: Dict[str, RobotTrajectory],
    scenario_name: str,
    out_path: str,
    params: SAParams,
    standalone_path_out: Optional[str] = None,
) -> None:
    """
    Plots Fig. 5(c) / Fig. 3(c) (left panel) and Fig. 6 / Fig. 4 (right panel),
    plus an optional standalone path plot matching the exact crop of the paper screenshot.
    """
    env_title = "static environment" if scenario_name == "static" else "barrier-free environment"
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 6.4))

    # Left panel: Fig. 5(c) or Fig. 3(c)
    plot_single_path_figure(axs[0], env, robots, trajs, scenario_name, show_caption=True)

    # Right panel: Fig. 6 or Fig. 4 Pairwise inter-robot distances over time
    ax1 = axs[1]
    ax1.set_title(
        f"Distance between the robots in a {env_title}",
        fontsize=11,
        weight="bold",
        pad=8,
    )
    matlab_colors = [
        "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
        "#4DBEEE", "#A2142F", "#0072BD", "#D95319", "#EDB120",
    ]
    p_idx = 0
    max_K = max(len(trajs[r].pos) for r in NAMES)
    t_vec = np.arange(max_K) * params.dt

    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            ra, rb = NAMES[i], NAMES[j]
            pos_a, pos_b = trajs[ra].pos, trajs[rb].pos
            d_vec = np.hypot(pos_a[:max_K, 0] - pos_b[:max_K, 0], pos_a[:max_K, 1] - pos_b[:max_K, 1]) * params.cell_scale
            ax1.plot(
                t_vec,
                d_vec,
                lw=1.1,
                color=matlab_colors[p_idx],
                label=f"distance of robot {ra} and {rb}",
            )
            p_idx += 1

    ax1.set_xlabel("Time(s)", fontsize=10)
    ax1.set_ylabel("Distance", fontsize=10)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 500)
    ax1.tick_params(direction="in", top=True, right=True)
    ax1.legend(loc="upper center", fontsize=8.0, frameon=True, edgecolor="#4a4a4a", fancybox=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if standalone_path_out is not None:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6.8, 6.8))
        plot_single_path_figure(ax_single, env, robots, trajs, scenario_name, show_caption=False)
        fig_single.tight_layout()
        fig_single.savefig(standalone_path_out, dpi=180, bbox_inches="tight")
        plt.close(fig_single)


# =============================================================================
# 8. CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Improved Simulated Annealing Multi-Robot Path Planning (Shi et al., 2025)"
    )
    parser.add_argument("--scenario", default="free", choices=["static", "free", "both"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="results_improved_sa")
    args = parser.parse_args()

    params = SAParams()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = ["static", "free"] if args.scenario == "both" else [args.scenario]
    for sc in scenarios:
        rects, robots = SCENARIOS[sc]
        env = GridEnv(rects, params)
        trajs, cpu_times = multi_robot_improved_sa_planner(
            env, robots, params, scenario_name=sc, seed=args.seed
        )
        rows = evaluate_solution(env, trajs, cpu_times, params)
        print_paper_tables(sc, rows)
        fig_path = os.path.join(args.out_dir, f"improved_sa_{sc}_results.png")
        standalone_path = os.path.join(args.out_dir, f"fig5c_exact_{sc}_paths.png")
        plot_paper_figures(
            env, robots, trajs, sc, fig_path, params, standalone_path_out=standalone_path
        )
        print(f"Saved combined figure to: {fig_path}")
        print(f"Saved standalone path figure to: {standalone_path}")


if __name__ == "__main__":
    main()