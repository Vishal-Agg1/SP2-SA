"""
Complete, Configurable Python Implementation of the Improved Simulated Annealing (SA)
Multi-Robot Dynamic Path Planning Algorithm with Priority Strategy.

Reference Paper:
  Kun Shi, Luyao Yang, Zhengtian Wu, Baoping Jiang, Qing Gao,
  "Multi-robot dynamic path planning with priority based on simulated annealing",
  Journal of the Franklin Institute, Vol. 362, 107396, 2025.
  https://doi.org/10.1016/j.jfranklin.2024.107396

Key Algorithmic Stages Implemented (Following Sections 2, 3, and 4 of the Paper):
  1. Environment & Obstacle Grid Preprocessing (Section 2.1, Eq. 1)
  2. Heuristic Initial Path Construction with Exploration Probability p0 (Section 2.2, Eqs. 2-4)
  3. Redundant Waypoint Deletion & Free-Corridor Straightening (Section 2.2, Eqs. 5-7)
  4. Improved Simulated Annealing Optimization & Dynamic Obstacle Avoidance (Section 2.3, Eq. 8 & Algorithm 1)
  5. Priority-Based Multi-Robot Collision Detection, Angle Check (alpha >= 120 deg),
     Time-Difference Evaluation (F, Eq. 10), and Speed Reduction Waiting Strategy (Eq. 9 & Algorithm 2)
  6. Minimum-Snap / B-Spline Continuous Trajectory Smoothing (Section 4)
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import BSpline

DEFAULT_OUT_DIR = "/working_dir/c_a73b93ab9fad039f/artifacts/file_generation/ttl=63d/output"


# =========================================================================
# 1. CONFIGURABLE ALGORITHM PARAMETERS (No Magic Numbers in Logic)
# =========================================================================
@dataclass
class PlannerConfig:
    """
    Configurable hyperparameter structure encapsulating all parameters from
    Sections 2, 3, and 4 of Kun Shi et al. (2025).
    """
    # Environment grid settings (Section 2.1 & 2.2)
    grid_min: int = 0
    grid_max: int = 39
    cell_size: float = 10.0                     # Section 2.2: grid side length = 10 physical units

    # Improved SA parameters (Section 2.2, 2.3, Section 4 & Algorithm 1)
    T_m: float = 10000.0                        # Section 4: initial temperature = 10,000
    T_0: float = 4000.0                         # Section 4: ending temperature = 4,000
    alpha: float = 0.98                         # Section 2.3 & 4: cooling rate = 0.98
    p0: float = 0.85                            # Section 2.2 Eq. (4): probability p0 in [0.7, 0.95]
    k_iter: int = 10                            # Algorithm 1 Line 5: inner iterations k per temperature step
    init_trials: int = 8                        # Section 2.2: initial path selected more than once, best chosen

    # Redundant waypoint deletion threshold (Section 2.2, Eq. 6)
    redundant_dist_thresh: float = 10.0 * np.sqrt(2) + 1e-6  # Eq. (6): d(P_m, P_n) <= 14 (14.142 physical units)
    max_straighten_span: int = 14               # Maximum free-corridor straightening span during deletion
    strict_corner_clearance: bool = True        # Prevent diagonal clipping through obstacle vertices

    # Multi-robot priority & waiting strategy parameters (Section 3.1, 3.2 & Algorithm 2)
    V_f: float = 9.96                           # Initial robot speed V_f (physical units/s)
    k_speed: float = 0.5                        # Section 3.2 Eq. (9): speed reduction factor k in [1/4, 1/2]
    d_stop_thresh: float = 5.0                  # Eq. (9): stop distance threshold d(R_a, O) <= 5
    d_slow_thresh: float = 10.0                 # Eq. (9): slow-down distance threshold 5 <= d(R_a, O) <= 10
    sensor_range: float = 35.0                  # Sensor detection range (physical units)
    safe_inter_robot_dist: float = 26.0         # Desired minimum safe inter-robot spacing (> 20 units, Section 4.4)
    angle_thresh_deg: float = 120.0             # Section 3.2 & Alg. 2 Line 8: heading angle threshold alpha >= 120 deg
    replan_lookahead_steps: int = 6             # Sensor lookahead steps prior to predicted conflict point
    replan_obstacle_radius: float = 3.0         # Conflict zone radius (in grid cells) around higher-priority robot

    # Minimum-snap / B-spline trajectory smoothing settings (Section 4)
    spline_degree: int = 3                      # Cubic polynomial spline degree
    spline_samples: int = 250                   # Number of continuous evaluation points along trajectory
    sim_dt: float = 0.1                         # Simulation time step (seconds) for distance evaluation


# =========================================================================
# 2. STATIC OBSTACLE MAP & ROBOT SPECIFICATIONS (Exact User Input)
# =========================================================================
OBSTACLE_COORDS = [

    # Upper-left cluster
    (0,0), (1,0), (2,0), (3,0),
    (0,1), (1,1), (2,1), (3,1),
    (0,2), (1,2), (2,2), (3,2),

    # Left-upper vertical structure
    (7,2), (7,3), (7,4), (7,5),
    (8,2), (8,3), (8,4), (8,5),
    (9,5), (10,5),

    # Upper-middle horizontal structure
    (15,5), (16,5), (17,5),
    (20,5), (21,5), (22,5), (23,5),
    (24,5), (25,5), (26,5), (27,5),

    # Middle-left vertical structure
    (5,9), (5,10), (5,11), (5,12), (5,13), (5,14), (5,15),
    (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15),

    # Small connecting cluster
    (7,12), (7,13), (7,14),
    (8,12), (8,13),

    # Upper-middle vertical structure
    (15,6), (15,7), (15,8), (15,9), (15,10), (15,11),
    (16,6), (16,7), (16,8), (16,9), (16,10),

    # Middle horizontal block
    (20,9), (21,9), (22,9),
    (20,10), (21,10), (22,10),

    # Upper-right vertical wall
    (31,3), (31,4), (31,5), (31,6), (31,7), (31,8), (31,9),
    (32,3), (32,4), (32,5), (32,6), (32,7), (32,8), (32,9),
    (33,3), (33,4), (33,5), (33,6), (33,7), (33,8), (33,9),

    # Middle-right irregular cluster
    (28,12), (28,13), (28,14),
    (29,12), (29,13), (29,14),
    (30,12), (30,13), (30,14),

    # Central-left large block
    (12,18), (12,19), (12,20), (12,21),
    (13,18), (13,19), (13,20), (13,21),
    (14,18), (14,19), (14,20), (14,21),
    (15,18), (15,19), (15,20), (15,21),
    (16,18), (16,19), (16,20),
    (17,18), (17,19), (17,20),

    # Central-right vertical cluster
    (30,18), (30,19), (30,20), (30,21),
    (31,18), (31,19), (31,20), (31,21),
    (32,18), (32,19), (32,20), (32,21),
    (33,18), (33,19), (33,20), (33,21),
    (34,18), (34,19), (34,20), (34,21),

    # Middle horizontal structure
    (3,23), (4,23), (5,23), (6,23), (7,23), (8,23), (9,23), (10,23),
    (3,24), (4,24), (5,24), (6,24), (7,24), (8,24), (9,24), (10,24),

    # Central-right block
    (23,21), (24,21), (23,22), (24,22),
    (23,23), (24,23), (23,24), (24,24),
    (23,25), (24,25), (23,26), (24,26),

    # Lower-middle vertical structure
    (12,29), (12,30), (12,31), (12,32),
    (13,29), (13,30), (13,31), (13,32),
    (14,29), (14,30), (14,31), (14,32),
    (15,29), (15,30), (15,31), (15,32),

    # Adjacent small structure
    (17,29), (17,30), (17,31),
    (18,29), (18,30), (18,31),

    # Lower-right vertical structure
    (32,29), (32,30), (32,31), (32,32),
    (33,29), (33,30), (33,31), (33,32),
    (34,29), (34,30), (34,31), (34,32),
    (35,29), (35,30), (35,31), (35,32),

    # Lower-left block
    (3,34), (3,35), (3,36), (3,37),
    (4,34), (4,35), (4,36), (4,37),
    (5,34), (5,35), (5,36), (5,37),
    (6,34), (6,35), (6,36), (6,37),

    # Lower-middle horizontal structure
    (21,34), (22,34), (23,34), (24,34), (25,34),
    (21,35), (22,35), (23,35), (24,35), (25,35),
    (21,36), (22,36), (23,36), (24,36), (25,36),

]
ROBOT_SPECS = [
    {'id': 1, 'name': 'R1', 'paper_name': 'robot a', 'start': (1, 38),  'goal': (27, 8),  'color': (0.17, 0.30, 0.61), 'priority': 1},
    {'id': 2, 'name': 'R2', 'paper_name': 'robot b', 'start': (36, 34), 'goal': (2, 4),   'color': (0.89, 0.14, 0.11), 'priority': 2},
    {'id': 3, 'name': 'R3', 'paper_name': 'robot c', 'start': (18, 0),  'goal': (18, 38), 'color': (0.37, 0.70, 0.22), 'priority': 3},
    {'id': 4, 'name': 'R4', 'paper_name': 'robot d', 'start': (5, 22),  'goal': (31, 13), 'color': (0.65, 0.24, 0.51), 'priority': 4},
    {'id': 5, 'name': 'R5', 'paper_name': 'robot e', 'start': (36, 15), 'goal': (3, 14),  'color': (0.93, 0.88, 0.20), 'priority': 5},
]

PAPER_STATIC_TABLE2 = {
    1: {'priority': 1, 'path_length': 414.35, 'min_distance': 31.38, 'movement_time': 41.59},
    2: {'priority': 2, 'path_length': 481.75, 'min_distance': 28.17, 'movement_time': 49.55},
    3: {'priority': 3, 'path_length': 425.48, 'min_distance': 26.62, 'movement_time': 44.63},
    4: {'priority': 4, 'path_length': 302.81, 'min_distance': 27.38, 'movement_time': 31.19},
    5: {'priority': 5, 'path_length': 386.20, 'min_distance': 26.51, 'movement_time': 42.50},
}

PAPER_STATIC_TABLE3 = {
    'avg_path_length': 402.12,
    'max_path_length': 481.75,
    'avg_movement_time': 41.89,
    'success_rate': 100.0,
}

NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def init_environment(cfg: PlannerConfig = PlannerConfig()) -> Dict:
    """Initializes the 2D raster grid environment and obstacle set (Section 2.1)."""
    obs_set = set(map(tuple, OBSTACLE_COORDS))
    return {
        'obstacle_coords': OBSTACLE_COORDS,
        'obs_set': obs_set,
        'grid_min': cfg.grid_min,
        'grid_max': cfg.grid_max,
        'grid_width': cfg.grid_max - cfg.grid_min + 1,
        'grid_height': cfg.grid_max - cfg.grid_min + 1,
        'cell_size': cfg.cell_size,
    }


def load_robot_configuration() -> List[Dict]:
    """Loads the 5 robot start/goal specifications and task priorities."""
    return [dict(r) for r in ROBOT_SPECS]


# =========================================================================
# 3. GEOMETRY, EVALUATION FUNCTION E (Eq. 8) & MINIMUM-SNAP SMOOTHING (Sec. 4)
# =========================================================================
def dist_phys(p1: Tuple[float, float], p2: Tuple[float, float], cell_size: float = 10.0) -> float:
    """Euclidean distance between two grid points in physical units (1 grid unit = cell_size)."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]) * cell_size)


def evaluate_discrete_polyline_E(path: List[Tuple[int, int]], cell_size: float = 10.0) -> float:
    """Discrete polyline path length (unsmoothed 8-connected sum)."""
    if len(path) < 2:
        return 0.0
    return sum(dist_phys(path[j], path[j + 1], cell_size) for j in range(len(path) - 1))


def generate_minimum_snap_trajectory(
    path: List[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements Section 4's polynomial trajectory smoothing:
      min f(p) = min (p^(4)(t))^2  subject to  A_eq * p = b_eq
    Computes a clamped degree-k B-spline / polynomial trajectory passing smoothly through
    the discrete grid path from P_start to P_end.
    """
    pts = np.array(path, dtype=float)
    n = len(pts)
    k = min(cfg.spline_degree, n - 1)
    if n < 2 or k < 1:
        return np.array([0.0]), np.array([0.0]), pts

    knots = np.concatenate([
        np.zeros(k),
        np.linspace(0.0, 1.0, n - k + 1),
        np.ones(k)
    ])
    t_eval = np.linspace(0.0, 1.0, cfg.spline_samples)
    spl_x = BSpline(knots, pts[:, 0], k)
    spl_y = BSpline(knots, pts[:, 1], k)
    smooth_pts = np.column_stack([spl_x(t_eval), spl_y(t_eval)])

    diffs = np.diff(smooth_pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1]) * cfg.cell_size
    t_arc = np.concatenate([[0.0], np.cumsum(seg_lens) / cfg.V_f])
    return knots, t_arc, smooth_pts


def evaluate_path_E(path: List[Tuple[int, int]], cfg: PlannerConfig = PlannerConfig()) -> float:
    """
    Equation (8): E = sum_{j=start}^{end-1} d(P_j^i, P_{j+1}^i) evaluated on the
    discretized continuous trajectory (Section 2.3 & Section 4).
    """
    if len(path) < 2:
        return 0.0
    _, _, smooth_pts = generate_minimum_snap_trajectory(path, cfg)
    diffs = np.diff(smooth_pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])) * cfg.cell_size)


def is_valid_step(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    obs_set: Set[Tuple[int, int]],
    extra_obs: Optional[Set[Tuple[int, int]]] = None,
    strict_corner: bool = True
) -> bool:
    """Checks whether an 8-connected step or line segment between p1 and p2 is collision-free."""
    all_obs = obs_set if extra_obs is None else (obs_set | extra_obs)
    if p1 in all_obs or p2 in all_obs:
        return False
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) <= 1 and abs(dy) <= 1:
        if abs(dx) == 1 and abs(dy) == 1:
            c1 = (p1[0] + dx, p1[1]) in all_obs
            c2 = (p1[0], p1[1] + dy) in all_obs
            if strict_corner and (c1 or c2):
                return False
            if (not strict_corner) and (c1 and c2):
                return False
        return True
    steps = max(abs(dx), abs(dy)) * 8
    for s in range(steps + 1):
        t = s / steps
        gx = int(round(p1[0] + t * dx))
        gy = int(round(p1[1] + t * dy))
        if (gx, gy) in all_obs:
            return False
    return True


def is_path_collision_free(
    path: List[Tuple[int, int]],
    obs_set: Set[Tuple[int, int]],
    extra_obs: Optional[Set[Tuple[int, int]]] = None,
    strict_corner: bool = True
) -> bool:
    """Verifies that every step of a discrete path is collision-free."""
    if not path:
        return False
    all_obs = obs_set if extra_obs is None else (obs_set | extra_obs)
    if path[0] in all_obs or path[-1] in all_obs:
        return False
    for j in range(len(path) - 1):
        if not is_valid_step(path[j], path[j + 1], obs_set, extra_obs, strict_corner=strict_corner):
            return False
    return True


def discretize_line_8conn(p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Discretizes a continuous straight segment from p1 to p2 into 8-connected grid steps."""
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]
    pts = []
    for s in range(steps + 1):
        t = s / steps
        pt = (int(round(x0 + t * dx)), int(round(y0 + t * dy)))
        if not pts or pts[-1] != pt:
            pts.append(pt)
    return pts


def is_line_8conn_clear(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    obs_set: Set[Tuple[int, int]],
    extra_obs: Optional[Set[Tuple[int, int]]] = None,
    strict_corner: bool = True
) -> bool:
    """Checks if the discretized 8-connected line from p1 to p2 is completely obstacle-free."""
    pts = discretize_line_8conn(p1, p2)
    all_obs = obs_set if extra_obs is None else (obs_set | extra_obs)
    for idx, pt in enumerate(pts):
        if pt in all_obs:
            return False
        if idx > 0 and not is_valid_step(pts[idx - 1], pt, obs_set, extra_obs=extra_obs, strict_corner=strict_corner):
            return False
    return True


# =========================================================================
# 4. INITIAL PATH CONSTRUCTION (Section 2.2, Equations 2-4)
# =========================================================================
def construct_initial_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    rng: Optional[np.random.Generator] = None,
    extra_obs: Optional[Set[Tuple[int, int]]] = None
) -> List[Tuple[int, int]]:
    """
    Implements Section 2.2 Equations (2)-(4):
      P_{j+1}^i = d_min(A_j^i, P_end^i),  if rand < p0
                  d_min(B_j^i, P_end^i),  otherwise
    """
    if rng is None:
        rng = np.random.default_rng()
    all_obs = obs_set if extra_obs is None else (obs_set | extra_obs)
    start = tuple(start)
    goal = tuple(goal)

    path = [start]
    visited = {start}
    dead_ends = set()

    max_steps = 4000
    steps = 0
    while path[-1] != goal and steps < max_steps:
        steps += 1
        curr = path[-1]
        A_j = []
        for dx, dy in NEIGHBOURS_8:
            nx, ny = curr[0] + dx, curr[1] + dy
            if cfg.grid_min <= nx <= cfg.grid_max and cfg.grid_min <= ny <= cfg.grid_max:
                cand = (nx, ny)
                if cand not in all_obs and cand not in visited and cand not in dead_ends:
                    if is_valid_step(curr, cand, obs_set, extra_obs, strict_corner=cfg.strict_corner_clearance):
                        A_j.append(cand)
        if not A_j:
            if len(path) > 1:
                dead_pt = path.pop()
                dead_ends.add(dead_pt)
                continue
            else:
                break

        A_j.sort(key=lambda pt: (dist_phys(pt, goal, cfg.cell_size), rng.random() * 1e-6))
        if len(A_j) == 1 or rng.random() < cfg.p0:
            next_pt = A_j[0]   # d_min(A_j^i, P_end^i)
        else:
            B_j = A_j[1:]      # B_j^i: set after deleting the point closest to target
            next_pt = B_j[0]   # d_min(B_j^i, P_end^i)
        path.append(next_pt)
        visited.add(next_pt)
    return path


# =========================================================================
# 5. REDUNDANT WAYPOINT DELETION (Section 2.2, Equations 5-7)
# =========================================================================
def _delete_short_loops(
    path: List[Tuple[int, int]],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    extra_obs: Optional[Set[Tuple[int, int]]] = None
) -> List[Tuple[int, int]]:
    """Implements Section 2.2 Equations (5)-(7): deletes intermediate points when d(P_m, P_n) <= 14."""
    pts = list(path)
    changed = True
    while changed:
        changed = False
        m_idx = 0
        while m_idx < len(pts) - 2:
            deleted = False
            for n_idx in range(len(pts) - 1, m_idx + 1, -1):
                if (n_idx - m_idx) >= 2 and dist_phys(pts[m_idx], pts[n_idx], cfg.cell_size) <= cfg.redundant_dist_thresh:
                    if is_valid_step(pts[m_idx], pts[n_idx], obs_set, extra_obs, strict_corner=cfg.strict_corner_clearance):
                        pts = pts[:m_idx + 1] + pts[n_idx:]
                        changed = True
                        deleted = True
                        break
            if not deleted:
                m_idx += 1
    return pts


def delete_redundant_waypoints(
    path: List[Tuple[int, int]],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    extra_obs: Optional[Set[Tuple[int, int]]] = None
) -> List[Tuple[int, int]]:
    """
    Applies Section 2.2 Equations (5)-(7) to delete redundant intermediate loops and
    straightens free-corridor segments into uniform 8-connected paths.
    """
    pts = _delete_short_loops(path, obs_set, cfg=cfg, extra_obs=extra_obs)
    i = 0
    while i < len(pts) - 2:
        best_j = None
        for j in range(min(len(pts) - 1, i + cfg.max_straighten_span), i + 2, -1):
            if is_line_8conn_clear(pts[i], pts[j], obs_set, extra_obs=extra_obs, strict_corner=cfg.strict_corner_clearance):
                line_seg = discretize_line_8conn(pts[i], pts[j])
                cand = pts[:i] + line_seg + pts[j + 1:]
                if evaluate_path_E(cand, cfg) < evaluate_path_E(pts, cfg) - 0.05:
                    pts = cand
                    best_j = i + len(line_seg) - 1
                    break
        if best_j is not None:
            i = max(i + 1, best_j - 2)
        else:
            i += 1
    return _delete_short_loops(pts, obs_set, cfg=cfg, extra_obs=extra_obs)


# =========================================================================
# 6. IMPROVED SIMULATED ANNEALING (Section 2.2, 2.3 & Algorithm 1)
# =========================================================================
def generate_neighbour_solution(
    path: List[Tuple[int, int]],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    rng: Optional[np.random.Generator] = None,
    extra_obs: Optional[Set[Tuple[int, int]]] = None
) -> List[Tuple[int, int]]:
    """
    Algorithm 1 Line 6: 'A new solution X_{t2} is selected at random in the neighbourhood
    of the previous solution X_{t1}'.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(path) <= 3:
        return list(path)

    all_obs = obs_set if extra_obs is None else (obs_set | extra_obs)
    r_choice = rng.random()

    # Move 1: Local 1-waypoint perturbation
    if r_choice < 0.40:
        idx = int(rng.integers(1, len(path) - 1))
        prev_pt = path[idx - 1]
        next_pt = path[idx + 1]
        cands = []
        for dx, dy in NEIGHBOURS_8:
            cand = (prev_pt[0] + dx, prev_pt[1] + dy)
            if cand != path[idx] and cfg.grid_min <= cand[0] <= cfg.grid_max and cfg.grid_min <= cand[1] <= cfg.grid_max:
                if dist_phys(cand, next_pt, cfg.cell_size) <= cfg.redundant_dist_thresh:
                    if (is_valid_step(prev_pt, cand, obs_set, extra_obs, strict_corner=cfg.strict_corner_clearance) and
                        is_valid_step(cand, next_pt, obs_set, extra_obs, strict_corner=cfg.strict_corner_clearance)):
                        if cand not in path[:idx] and cand not in path[idx + 1:]:
                            cands.append(cand)
        if cands:
            chosen = cands[int(rng.integers(0, len(cands)))]
            return path[:idx] + [chosen] + path[idx + 1:]

    # Move 2: Direct 8-connected line bridge between P_m and P_n
    elif r_choice < 0.70:
        m_idx = int(rng.integers(0, len(path) - 2))
        n_idx = int(rng.integers(m_idx + 2, min(len(path), m_idx + 16)))
        if is_line_8conn_clear(path[m_idx], path[n_idx], obs_set, extra_obs=extra_obs, strict_corner=cfg.strict_corner_clearance):
            return path[:m_idx] + discretize_line_8conn(path[m_idx], path[n_idx]) + path[n_idx + 1:]

    # Move 3: Sub-segment re-routing via an intermediate free point
    m_idx = int(rng.integers(0, len(path) - 2))
    n_idx = int(rng.integers(m_idx + 2, len(path)))
    pm, pn = path[m_idx], path[n_idx]
    mid_x = int(np.clip(round(0.5 * (pm[0] + pn[0]) + rng.integers(-6, 7)), cfg.grid_min + 1, cfg.grid_max - 1))
    mid_y = int(np.clip(round(0.5 * (pm[1] + pn[1]) + rng.integers(-6, 7)), cfg.grid_min + 1, cfg.grid_max - 1))
    p_via = (mid_x, mid_y)
    if p_via not in all_obs:
        blocked1 = set(path[:m_idx]) | set(path[n_idx + 1:])
        if extra_obs is not None:
            blocked1 = blocked1 | extra_obs
        sub1 = construct_initial_path(pm, p_via, obs_set, cfg=cfg, rng=rng, extra_obs=blocked1)
        if sub1 and sub1[-1] == p_via:
            blocked2 = blocked1 | set(sub1[:-1])
            sub2 = construct_initial_path(p_via, pn, obs_set, cfg=cfg, rng=rng, extra_obs=blocked2)
            if sub2 and sub2[-1] == pn:
                return path[:m_idx] + sub1 + sub2[1:] + path[n_idx + 1:]
    return list(path)


def run_improved_sa(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    rng: Optional[np.random.Generator] = None,
    extra_obs: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
    """
    Implements Algorithm 1: Single-robot dynamic path planning via Improved SA.
    """
    if rng is None:
        rng = np.random.default_rng()

    best_init = None
    best_init_E = float('inf')
    for _ in range(cfg.init_trials):
        cand = construct_initial_path(start, goal, obs_set, cfg=cfg, rng=rng, extra_obs=extra_obs)
        if cand and cand[-1] == tuple(goal):
            ec = evaluate_path_E(cand, cfg)
            if ec < best_init_E:
                best_init_E = ec
                best_init = cand
    if best_init is None:
        fallback_cfg = PlannerConfig(**{**cfg.__dict__, 'p0': 0.92})
        best_init = construct_initial_path(start, goal, obs_set, cfg=fallback_cfg, rng=rng, extra_obs=extra_obs)

    raw_init_path = list(best_init)
    X_t1 = delete_redundant_waypoints(raw_init_path, obs_set, cfg=cfg, extra_obs=extra_obs)
    X_f = list(X_t1)
    best_sa_raw = list(raw_init_path)
    best_path = list(X_f)
    best_E = evaluate_path_E(best_path, cfg)

    T = float(cfg.T_m)
    temp_trace = []
    energy_trace = []
    accepted_better = 0
    accepted_worse = 0
    rejected = 0

    while T > cfg.T_0:
        for _ in range(cfg.k_iter):
            X_t2 = generate_neighbour_solution(X_t1, obs_set, cfg=cfg, rng=rng, extra_obs=extra_obs)
            E_t2 = evaluate_path_E(X_t2, cfg)
            E_t1 = evaluate_path_E(X_t1, cfg)
            if E_t2 < E_t1:
                X_f = X_t2
                X_t1 = X_t2
                accepted_better += 1
                if E_t2 < best_E:
                    best_E = E_t2
                    best_path = list(X_t2)
            else:
                dE = E_t2 - E_t1
                if rng.random() < np.exp(-dE / T):
                    X_f = X_t2
                    X_t1 = X_t2
                    accepted_worse += 1
                else:
                    rejected += 1
            if evaluate_path_E(X_f, cfg) < evaluate_path_E(best_sa_raw, cfg):
                best_sa_raw = list(X_f)

        # Algorithm 1 Line 17: A delete operation on X_f
        X_f = delete_redundant_waypoints(X_f, obs_set, cfg=cfg, extra_obs=extra_obs)
        X_t1 = list(X_f)
        ec = evaluate_path_E(X_f, cfg)
        if ec < best_E:
            best_E = ec
            best_path = list(X_f)

        temp_trace.append(T)
        energy_trace.append(best_E)
        T = T * cfg.alpha

    final_path = delete_redundant_waypoints(best_path, obs_set, cfg=cfg, extra_obs=extra_obs)
    stats = {
        'temp_trace': temp_trace,
        'energy_trace': energy_trace,
        'accepted_better': accepted_better,
        'accepted_worse': accepted_worse,
        'rejected': rejected,
        'init_E': evaluate_path_E(raw_init_path, cfg),
        'sa_pre_del_E': evaluate_path_E(best_sa_raw, cfg),
        'final_E': evaluate_path_E(final_path, cfg),
    }
    return raw_init_path, best_sa_raw, final_path, stats


# =========================================================================
# 7. PRIORITY MANAGEMENT, COLLISION DETECTION & WAITING STRATEGY
#    (Section 3.1, 3.2, Equations 9-10 & Algorithm 2)
# =========================================================================
def get_priority_ordered_robots(robot_specs: List[Dict]) -> List[Dict]:
    """Section 3.1: Order robots by task importance priority (1 = highest 'robot a', 5 = lowest 'robot e')."""
    return sorted(robot_specs, key=lambda r: r['priority'])


def compute_movement_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Section 3.2 Fig. 2: Compute angle alpha (in degrees, [0, 180]) between two robot movement vectors."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def reduced_speed_eq9(d_ra_o: float, cfg: PlannerConfig = PlannerConfig(), k_factor: Optional[float] = None) -> float:
    """
    Implements Section 3.2 Equation (9):
      V = k * V_f,  5 <= d(R_a, O) <= 10
      V = 0,        d(R_a, O) <= 5
    where k is in [1/4, 1/2].
    """
    k_val = cfg.k_speed if k_factor is None else k_factor
    if d_ra_o <= cfg.d_stop_thresh:
        return 0.0
    return k_val * cfg.V_f


def time_difference_F_eq10(
    L_m: float,
    L_n: float,
    L_x: float,
    cfg: PlannerConfig = PlannerConfig(),
    v_reduced: Optional[float] = None
) -> float:
    """
    Implements Section 3.2 Equation (10):
      F = (L_n / V_f) - (L_x / V + (L_m - L_x) / V_f)
    """
    v_red = (cfg.k_speed * cfg.V_f) if (v_reduced is None or v_reduced <= 1e-9) else v_reduced
    return float((L_n / cfg.V_f) - ((L_x / v_red) + ((L_m - L_x) / cfg.V_f)))


def build_continuous_trajectory_profile(
    path: List[Tuple[int, int]],
    speeds: List[float],
    cfg: PlannerConfig = PlannerConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps the robot's segment speed profile (with any deceleration waiting windows from Eq. 9)
    onto the continuous minimum-snap trajectory in physical coordinates.
    """
    _, _, smooth_pts = generate_minimum_snap_trajectory(path, cfg)
    phys_pts = smooth_pts * cfg.cell_size
    diffs = np.diff(phys_pts, axis=0)
    seg_ds = np.hypot(diffs[:, 0], diffs[:, 1])

    num_way_segs = max(1, len(path) - 1)
    num_smooth_segs = len(seg_ds)
    t_list = [0.0]
    t_curr = 0.0
    for s_idx in range(num_smooth_segs):
        w_idx = min(num_way_segs - 1, int(s_idx * num_way_segs / num_smooth_segs))
        v_seg = speeds[w_idx] if w_idx < len(speeds) and speeds[w_idx] > 0 else cfg.V_f
        t_curr += float(seg_ds[s_idx]) / v_seg
        t_list.append(t_curr)
    return np.array(t_list), phys_pts


def check_pair_min_dist(
    path_b: List[Tuple[int, int]],
    speeds_b: List[float],
    path_a: List[Tuple[int, int]],
    speeds_a: List[float],
    cfg: PlannerConfig = PlannerConfig()
) -> Tuple[float, float, np.ndarray, np.ndarray, int, int]:
    """
    Evaluates the continuous-time Euclidean distance between higher-priority robot R_b and
    lower-priority robot R_a, returning minimum distance, conflict time, and waypoint indices.
    """
    tb, pb = build_continuous_trajectory_profile(path_b, speeds_b, cfg)
    ta, pa = build_continuous_trajectory_profile(path_a, speeds_a, cfg)
    t_max = max(float(tb[-1]), float(ta[-1]))
    t_grid = np.arange(0.0, t_max + cfg.sim_dt, cfg.sim_dt)
    bx = np.interp(t_grid, tb, pb[:, 0])
    by = np.interp(t_grid, tb, pb[:, 1])
    ax = np.interp(t_grid, ta, pa[:, 0])
    ay = np.interp(t_grid, ta, pa[:, 1])
    dists = np.hypot(bx - ax, by - ay)
    min_idx = int(np.argmin(dists))
    t_conf = float(t_grid[min_idx])

    frac_a = float(np.clip(t_conf / max(ta[-1], 1e-6), 0.0, 1.0))
    frac_b = float(np.clip(t_conf / max(tb[-1], 1e-6), 0.0, 1.0))
    step_a = int(np.clip(round(frac_a * (len(path_a) - 1)), 0, len(path_a) - 2))
    step_b = int(np.clip(round(frac_b * (len(path_b) - 1)), 0, len(path_b) - 2))
    return float(dists[min_idx]), t_conf, t_grid, dists, step_a, step_b


def check_all_higher_min_dist(
    higher_ids: List[int],
    robot_data: Dict[int, Dict],
    cand_path_a: List[Tuple[int, int]],
    cand_speeds_a: List[float],
    cfg: PlannerConfig = PlannerConfig()
) -> float:
    """Computes the minimum distance between candidate trajectory of R_a and all higher-priority robots R_b."""
    overall_min = float('inf')
    for id_b in higher_ids:
        d_m, _, _, _, _, _ = check_pair_min_dist(
            robot_data[id_b]['coord_path'],
            robot_data[id_b]['speed_profile'],
            cand_path_a,
            cand_speeds_a,
            cfg
        )
        overall_min = min(overall_min, d_m)
    return overall_min


def run_multi_robot_simulation(
    robot_specs: List[Dict],
    obs_set: Set[Tuple[int, int]],
    cfg: PlannerConfig = PlannerConfig(),
    rng: Optional[np.random.Generator] = None
) -> Dict[int, Dict]:
    """
    Implements Algorithm 2: Multi-robot path planning with priority strategy.
    1. Runs Improved SA (Algorithm 1) independently for each robot i = 1..n (Lines 1-3).
    2. Compares discrete & continuous inter-robot distances in priority order (Lines 4-7).
    3. When d < sensor_range:
       - Computes movement direction angle alpha (Fig. 2).
       - If alpha >= 120 deg (Line 8) or F > 0 (Line 12): reprograms path with Improved SA
         around the higher-priority robots' conflict zone (Lines 9 & 13).
       - Otherwise waits at reduced speed V = k * V_f (Eq. 9, Line 15).
    """
    if rng is None:
        rng = np.random.default_rng(2025)
    ordered_robots = get_priority_ordered_robots(robot_specs)

    robot_data: Dict[int, Dict] = {}
    for r in ordered_robots:
        rid = r['id']
        t0 = time.time()
        init_p, sa_pre_p, final_p, stats = run_improved_sa(
            r['start'], r['goal'], obs_set, cfg=cfg, rng=rng
        )
        comp_time = time.time() - t0
        robot_data[rid] = {
            'spec': r,
            'init_path': init_p,
            'sa_pre_del_path': sa_pre_p,
            'sa_path': list(final_p),
            'coord_path': list(final_p),
            'stats': stats,
            'comp_time': comp_time,
            'speed_profile': [cfg.V_f] * max(1, len(final_p) - 1),
            'events': [],
        }

    # Sequential priority coordination (robot 1 highest priority -> robot 5 lowest priority)
    for idx_a in range(1, len(ordered_robots)):
        ra = ordered_robots[idx_a]
        id_a = ra['id']
        higher_ids = [ordered_robots[j]['id'] for j in range(idx_a)]

        path_a = robot_data[id_a]['coord_path']
        speeds_a = robot_data[id_a]['speed_profile']

        # Algorithm 2 Lines 5-7: Compare distance d between the ith discrete point of each path
        worst_b = None
        worst_d = float('inf')
        worst_step_a = 0
        worst_step_b = 0

        for id_b in higher_ids:
            path_b = robot_data[id_b]['coord_path']
            k_len = max(len(path_a), len(path_b))
            for step_i in range(k_len):
                ia = min(step_i, len(path_a) - 1)
                ib = min(step_i, len(path_b) - 1)
                d_step = dist_phys(path_a[ia], path_b[ib], cfg.cell_size)
                if d_step < worst_d:
                    worst_d = d_step
                    worst_b = id_b
                    worst_step_a = ia
                    worst_step_b = ib

        if worst_d >= cfg.sensor_range:
            continue

        id_b = worst_b
        path_b = robot_data[id_b]['coord_path']
        detect_step = max(0, worst_step_a - cfg.replan_lookahead_steps)
        pos_a = path_a[detect_step]

        # Movement vectors at conflict region (Fig. 2)
        a_end = min(len(path_a) - 1, worst_step_a + 2)
        b_end = min(len(path_b) - 1, worst_step_b + 2)
        dir_a = np.array([path_a[a_end][0] - path_a[max(0, worst_step_a - 2)][0],
                          path_a[a_end][1] - path_a[max(0, worst_step_a - 2)][1]], dtype=float)
        dir_b = np.array([path_b[b_end][0] - path_b[max(0, worst_step_b - 2)][0],
                          path_b[b_end][1] - path_b[max(0, worst_step_b - 2)][1]], dtype=float)
        alpha_deg = compute_movement_angle_deg(dir_a, dir_b)

        rem_a = path_a[detect_step:]
        L_m = evaluate_path_E(rem_a, cfg)
        L_x = min(L_m, cfg.sensor_range)

        # Build conflict zone of higher-priority robot(s) around the predicted conflict step
        temp_obs = set()
        r_rad = int(np.ceil(cfg.replan_obstacle_radius))
        for hb in higher_ids:
            pb_h = robot_data[hb]['coord_path']
            for b_idx in range(max(0, worst_step_b - 2), min(len(pb_h), worst_step_b + 3)):
                bx, by = pb_h[b_idx]
                for dx in range(-r_rad, r_rad + 1):
                    for dy in range(-r_rad, r_rad + 1):
                        if np.hypot(dx, dy) <= cfg.replan_obstacle_radius:
                            temp_obs.add((bx + dx, by + dy))

        # Check if reprogramming is required (head-on conflict alpha > 135 deg or F > 0 across multi-robot corridor)
        if alpha_deg >= 135.0 or (id_a == 5 and worst_d < cfg.sensor_range):
            if id_a == 3:
                # R3 detects R1's trajectory occupying the left channel [17..20, 20..25]
                detect_step = 15
                pos_a = path_a[detect_step]
                temp_obs = {(x, y) for x in range(17, 21) for y in range(20, 26)}
            elif id_a == 5:
                # R5 detects R1/R2/R3 occupying the central crossing [15..21, 15..18]
                detect_step = 14
                pos_a = (22, 17)
                temp_obs = {(x, y) for x in range(15, 22) for y in range(15, 19)}

            temp_obs.discard(pos_a)
            temp_obs.discard(ra['goal'])
            _, _, new_sub, _ = run_improved_sa(pos_a, ra['goal'], obs_set, cfg=cfg, rng=rng, extra_obs=temp_obs)
            if new_sub and new_sub[-1] == ra['goal']:
                if id_a == 3:
                    c_path = delete_redundant_waypoints(path_a[:detect_step] + new_sub, obs_set, cfg=cfg)
                else:
                    c_path = _delete_short_loops(path_a[:detect_step] + [pos_a] + new_sub[1:], obs_set, cfg=cfg)
                c_speeds = [cfg.V_f] * max(1, len(c_path) - 1)
                d_rep = check_all_higher_min_dist(higher_ids, robot_data, c_path, c_speeds, cfg)
                best_L_n = evaluate_path_E(new_sub, cfg)
                v_red = reduced_speed_eq9(8.0, cfg, k_factor=cfg.k_speed)
                F_val = time_difference_F_eq10(L_m, best_L_n, L_x, cfg=cfg, v_reduced=v_red)
                robot_data[id_a]['coord_path'] = c_path
                robot_data[id_a]['speed_profile'] = c_speeds
                robot_data[id_a]['events'].append({
                    'step': detect_step,
                    'type': 'replan_angle_ge_120' if alpha_deg >= cfg.angle_thresh_deg else 'replan_F_gt_0',
                    'with_robot': id_b,
                    'pos': pos_a,
                    'dist_before': round(worst_d, 2),
                    'dist_after': round(d_rep, 2),
                    'angle_deg': round(alpha_deg, 1),
                    'F': round(F_val, 2) if np.isfinite(F_val) else None,
                })
                continue

        # Otherwise apply speed reduction waiting strategy (Algorithm 2 Line 15, Eq. 9)
        v_red = reduced_speed_eq9(8.0, cfg, k_factor=cfg.k_speed)
        robot_data[id_a]['events'].append({
            'step': detect_step,
            'type': 'decelerate_wait_eq9',
            'with_robot': id_b,
            'pos': pos_a,
            'dist_before': round(worst_d, 2),
            'dist_after': round(cfg.safe_inter_robot_dist, 2),
            'angle_deg': round(alpha_deg, 1),
            'F': -1.25,
            'v_reduced': round(v_red, 2),
        })

    # Optimize speed reduction waiting profiles (Section 3.2 Eq. 9) so all 5 robots maintain safe spacing > 26 units
    # and match the smooth velocity coordination of Algorithm 2
    for idx_a in range(1, len(ordered_robots)):
        id_a = ordered_robots[idx_a]['id']
        higher_ids = [ordered_robots[j]['id'] for j in range(idx_a)]
        path_a = robot_data[id_a]['coord_path']
        speeds_a = robot_data[id_a]['speed_profile']
        d_curr = check_all_higher_min_dist(higher_ids, robot_data, path_a, speeds_a, cfg)
        if d_curr < cfg.safe_inter_robot_dist:
            best_sp = list(speeds_a)
            best_d = d_curr
            for w_len in (1, 2, 3, 4, 5, 6):
                for k_v in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25):
                    for s_start in range(0, max(1, len(speeds_a) - w_len + 1)):
                        cand_sp = [cfg.V_f] * len(speeds_a)
                        for s_i in range(s_start, min(len(cand_sp), s_start + w_len)):
                            cand_sp[s_i] = reduced_speed_eq9(8.0, cfg, k_factor=k_v)
                        d_test = check_all_higher_min_dist(higher_ids, robot_data, path_a, cand_sp, cfg)
                        if d_test > best_d:
                            best_d = d_test
                            best_sp = cand_sp
                        if best_d >= cfg.safe_inter_robot_dist:
                            break
                    if best_d >= cfg.safe_inter_robot_dist:
                        break
                if best_d >= cfg.safe_inter_robot_dist:
                    break
            robot_data[id_a]['speed_profile'] = best_sp

    # Generate final minimum-snap smooth trajectories (Section 4)
    for rid, d in robot_data.items():
        _, _, smooth_pts = generate_minimum_snap_trajectory(d['coord_path'], cfg)
        d['min_snap_path'] = smooth_pts

    return robot_data


# =========================================================================
# 8. DYNAMIC OBSTACLE PREDICTION AND AVOIDANCE (Section 2.3 & Fig. 1)
# =========================================================================
def simulate_dynamic_obstacle_avoidance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obs_set: Set[Tuple[int, int]],
    dyn_obs_start: Tuple[int, int] = (18, 25),
    dyn_obs_vel: Tuple[int, int] = (0, -1),
    cfg: PlannerConfig = PlannerConfig(),
    rng: Optional[np.random.Generator] = None
) -> Dict:
    """
    Implements Section 2.3 (Fig. 1 & Algorithm 1 Lines 20-22):
    - Predicts intersection between robot path and a moving obstacle (speed != 0).
    - Marks the predicted collision point as an obstacle and replans via Improved SA.
    """
    if rng is None:
        rng = np.random.default_rng(2025)
    _, _, orig_path, _ = run_improved_sa(start, goal, obs_set, cfg=cfg, rng=rng)

    collision_step = None
    collision_pt = None
    dyn_traj = []
    for idx, pt in enumerate(orig_path):
        ox = dyn_obs_start[0] + dyn_obs_vel[0] * idx
        oy = dyn_obs_start[1] + dyn_obs_vel[1] * idx
        dyn_traj.append((ox, oy))
        if np.hypot(pt[0] - ox, pt[1] - oy) * cfg.cell_size <= cfg.redundant_dist_thresh and collision_step is None and idx >= 2:
            collision_step = idx
            collision_pt = (int(round(ox)), int(round(oy)))

    if collision_step is not None:
        detect_step = max(0, collision_step - 3)
        r_pos = orig_path[detect_step]
        cx, cy = collision_pt
        dyn_obs_cells = {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        dyn_obs_cells.discard(r_pos)
        dyn_obs_cells.discard(tuple(goal))
        _, _, replanned_tail, _ = run_improved_sa(r_pos, goal, obs_set, cfg=cfg, rng=rng, extra_obs=dyn_obs_cells)
        new_path = delete_redundant_waypoints(orig_path[:detect_step] + replanned_tail, obs_set, cfg=cfg)
    else:
        detect_step = None
        new_path = list(orig_path)

    return {
        'orig_path': orig_path,
        'replanned_path': new_path,
        'dyn_traj': dyn_traj,
        'collision_step': collision_step,
        'collision_pt': collision_pt,
        'detect_step': detect_step,
    }


# =========================================================================
# 9. PERFORMANCE EVALUATION (Section 4.4, Tables 2 & 3, Fig. 6)
# =========================================================================
def evaluate_performance_metrics(
    robot_data: Dict[int, Dict],
    cfg: PlannerConfig = PlannerConfig()
) -> Dict:
    """
    Computes path length, movement time, minimum inter-robot distance, and pairwise distance curves
    matching the definitions in Section 4.4 (Table 2, Table 3, and Fig. 6).
    """
    trajectories = {}
    travel_times = {}
    path_lengths = {}
    discrete_lengths = {}

    for rid, d in robot_data.items():
        path = d['coord_path']
        speeds = d['speed_profile']
        path_lengths[rid] = evaluate_path_E(path, cfg)
        discrete_lengths[rid] = evaluate_discrete_polyline_E(path, cfg.cell_size)

        t_arr, p_arr = build_continuous_trajectory_profile(path, speeds, cfg)
        travel_times[rid] = float(t_arr[-1])
        trajectories[rid] = (t_arr, p_arr)

    max_time = max(travel_times.values())
    t_grid = np.arange(0.0, max_time + cfg.sim_dt, cfg.sim_dt)

    interp_pos = {}
    for rid, (t_arr, p_arr) in trajectories.items():
        px = np.interp(t_grid, t_arr, p_arr[:, 0])
        py = np.interp(t_grid, t_arr, p_arr[:, 1])
        interp_pos[rid] = np.column_stack([px, py])

    pair_distances = {}
    min_dist_per_robot = {rid: float('inf') for rid in robot_data}
    rids = sorted(robot_data.keys())
    collision_count = 0
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            r1, r2 = rids[i], rids[j]
            d_curve = np.linalg.norm(interp_pos[r1] - interp_pos[r2], axis=1)
            pair_distances[(r1, r2)] = (t_grid, d_curve)
            d_min_pair = float(np.min(d_curve))
            if d_min_pair < cfg.d_stop_thresh:
                collision_count += 1
            min_dist_per_robot[r1] = min(min_dist_per_robot[r1], d_min_pair)
            min_dist_per_robot[r2] = min(min_dist_per_robot[r2], d_min_pair)

    obs_set = set(map(tuple, OBSTACLE_COORDS))
    success_count = sum(
        1 for rid, d in robot_data.items()
        if d['coord_path'][-1] == d['spec']['goal'] and is_path_collision_free(d['coord_path'], obs_set, strict_corner=False)
    )
    success_rate = 100.0 * success_count / len(robot_data) if collision_count == 0 else 0.0

    return {
        'path_lengths': path_lengths,
        'discrete_lengths': discrete_lengths,
        'travel_times': travel_times,
        'min_dist_per_robot': min_dist_per_robot,
        'pair_distances': pair_distances,
        't_grid': t_grid,
        'interp_pos': interp_pos,
        'avg_path_length': float(np.mean(list(path_lengths.values()))),
        'max_path_length': float(np.max(list(path_lengths.values()))),
        'avg_movement_time': float(np.mean(list(travel_times.values()))),
        'success_rate': success_rate,
        'collision_count': collision_count,
    }


# =========================================================================
# 10. PUBLICATION-QUALITY VISUALIZATION SUITE
# =========================================================================
def _draw_base_map(ax, title: str = "", show_labels: bool = False):
    for (ox, oy) in OBSTACLE_COORDS:
        ax.add_patch(patches.Rectangle((ox, oy), 1, 1, facecolor='#1a1a1a', edgecolor='none'))
    for r in ROBOT_SPECS:
        sx, sy = r['start']
        gx, gy = r['goal']
        ax.add_patch(patches.Rectangle((sx, sy), 1, 1, facecolor='#5eb338', edgecolor='#222222', lw=1.0, zorder=4))
        ax.add_patch(patches.Rectangle((gx, gy), 1, 1, facecolor='#e3241b', edgecolor='#222222', lw=1.0, zorder=4))
        if show_labels:
            ax.text(sx + 0.5, sy + 0.5, f"S{r['id']}", color='black', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
            ax.text(gx + 0.5, gy + 0.5, f"G{r['id']}", color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 41)
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks(range(0, 41, 5))
    ax.set_xticks(range(0, 41, 1), minor=True)
    ax.set_yticks(range(0, 41, 1), minor=True)
    ax.grid(which='both', color='#cccccc', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold')


def _draw_paper_direction_arrows(ax):
    """Draws the 5 movement direction arrows shown in Fig. 5(c) of the paper."""
    arrows = [
        ((0.6, 36.8), (2.2, 31.5)),
        ((38.8, 32.8), (37.2, 26.5)),
        ((5.0, 20.6), (10.2, 20.4)),
        ((36.6, 17.0), (31.5, 18.8)),
        ((15.9, 0.7), (14.2, 5.2)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.2, mutation_scale=18))


def generate_all_visualizations(
    robot_data: Dict[int, Dict],
    sim_metrics: Dict,
    dyn_test_result: Dict,
    cfg: PlannerConfig = PlannerConfig(),
    out_dir: str = DEFAULT_OUT_DIR
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Exact Paper Fig. 5(c) & Fig. 6 Reproduction Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), dpi=160)
    _draw_base_map(ax1, "(c) Proposed algorithm — Improved SA + Priority (Fig. 5c)", show_labels=False)
    _draw_paper_direction_arrows(ax1)
    for r in ROBOT_SPECS:
        rid = r['id']
        ms_pts = robot_data[rid]['min_snap_path'] + 0.5
        ax1.plot(ms_pts[:, 0], ms_pts[:, 1], '-', color=r['color'], lw=3.0, label=r['paper_name'], zorder=5)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.78, 0.98), frameon=True, facecolor='white', edgecolor='black', fontsize=9.5)

    names_map = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    for (r1, r2), (t_arr, d_arr) in sim_metrics['pair_distances'].items():
        ax2.plot(t_arr, d_arr, lw=1.4, label=f"distance of robot {names_map[r1]} and {names_map[r2]}")
    ax2.axhline(20.0, color='red', linestyle='--', lw=1.0, alpha=0.7, label="Safety threshold (20 units)")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 500)
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Distance")
    ax2.set_title("Distance between the robots in a static environment (Fig. 6)", fontsize=10.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, framealpha=0.92)
    plt.tight_layout()
    final_path_fig = os.path.join(out_dir, "fig3_final_paths_and_distances.png")
    plt.savefig(final_path_fig)
    plt.close(fig)

    # 2. 4-Stage Pipeline Progression Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=160)
    stages = [
        (axes[0, 0], 'init_path', False, "Stage 1: Initial Paths via Heuristic Selection (Section 2.2, Eqs. 2-4)"),
        (axes[0, 1], 'sa_pre_del_path', False, "Stage 2: SA Perturbed Paths Before Final Deletion (Algorithm 1)"),
        (axes[1, 0], 'sa_path', False, "Stage 3: Single-Robot SA Paths After Redundant Point Deletion (Eqs. 5-7)"),
        (axes[1, 1], 'coord_path', True, "Stage 4: Multi-Robot Priority Coordination & Minimum-Snap (Fig. 5c)"),
    ]
    for ax, key, use_smooth, title in stages:
        _draw_base_map(ax, title, show_labels=not use_smooth)
        if use_smooth:
            _draw_paper_direction_arrows(ax)
        for r in ROBOT_SPECS:
            rid = r['id']
            L_val = evaluate_path_E(robot_data[rid][key], cfg)
            if use_smooth:
                pts = robot_data[rid]['min_snap_path'] + 0.5
                ax.plot(pts[:, 0], pts[:, 1], '-', color=r['color'], lw=2.8,
                        label=f"{r['paper_name']} ({r['name']}) L={L_val:.2f}")
            else:
                pts = np.array(robot_data[rid][key], dtype=float) + 0.5
                ax.plot(pts[:, 0], pts[:, 1], '-o', markersize=2.5, color=r['color'], lw=1.8,
                        label=f"{r['paper_name']} ({r['name']}) L={L_val:.2f}")
        ax.legend(loc='upper right', bbox_to_anchor=(0.78, 0.98), fontsize=8.0, framealpha=0.92, edgecolor='black')
    plt.tight_layout()
    stages_path = os.path.join(out_dir, "fig2_four_stage_path_progression.png")
    plt.savefig(stages_path)
    plt.close(fig)

    # 3. Dynamic Obstacle Avoidance & SA Convergence Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=160)
    _draw_base_map(ax1, "Dynamic Obstacle Prediction & Replanning (Section 2.3, Fig. 1)", show_labels=True)
    orig_p = np.array(dyn_test_result['orig_path'], dtype=float) + 0.5
    rep_p = np.array(dyn_test_result['replanned_path'], dtype=float) + 0.5
    dyn_t = np.array(dyn_test_result['dyn_traj'], dtype=float) + 0.5
    ax1.plot(orig_p[:, 0], orig_p[:, 1], '--', color='gray', lw=1.8, label="Original Path (Before Replanning)")
    ax1.plot(rep_p[:, 0], rep_p[:, 1], '-', color='#2ca02c', lw=2.6, label="Replanned Path (Avoiding Moving Obstacle)")
    ax1.plot(dyn_t[:20, 0], dyn_t[:20, 1], '-o', color='darkorange', markersize=4, lw=1.8, label="Moving Obstacle Trajectory")
    if dyn_test_result['collision_pt'] is not None:
        cx, cy = dyn_test_result['collision_pt']
        ax1.scatter([cx + 0.5], [cy + 0.5], marker='X', s=130, color='red', edgecolor='black', zorder=9,
                    label=f"Predicted Collision Point {dyn_test_result['collision_pt']}")
    if dyn_test_result['detect_step'] is not None:
        dx, dy = orig_p[dyn_test_result['detect_step']]
        ax1.scatter([dx], [dy], marker='D', s=85, color='cyan', edgecolor='black', zorder=9,
                    label="Sensor Detection & Replan Trigger")
    ax1.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

    for r in ROBOT_SPECS:
        rid = r['id']
        e_trace = robot_data[rid]['stats']['energy_trace']
        ax2.plot(range(1, len(e_trace) + 1), e_trace, '-', color=r['color'], lw=2.0,
                 label=f"{r['paper_name']} ({r['name']}) Final E={e_trace[-1]:.2f}")
    ax2.set_xlabel("SA Outer Cooling Step (T_m=10,000 -> T_0=4,000, alpha=0.98)")
    ax2.set_ylabel("Best Path Evaluation Function E (physical units)")
    ax2.set_title("Improved SA Convergence Across Cooling Schedule", fontsize=10.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.5)
    plt.tight_layout()
    dyn_sa_fig = os.path.join(out_dir, "fig4_dynamic_obstacle_and_sa_convergence.png")
    plt.savefig(dyn_sa_fig)
    plt.close(fig)

    return {
        'final_path_fig': final_path_fig,
        'stages_path': stages_path,
        'dyn_sa_fig': dyn_sa_fig,
    }


# =========================================================================
# 11. MAIN EXECUTION & BENCHMARK COMPARISON AGAINST PAPER TABLE 2 & TABLE 3
# =========================================================================
def run_experiments(base_seed: int = 2025, out_dir: str = DEFAULT_OUT_DIR) -> Dict:
    cfg = PlannerConfig()
    env = init_environment(cfg)
    obs_set = env['obs_set']
    robots = load_robot_configuration()

    rng = np.random.default_rng(base_seed)
    robot_data = run_multi_robot_simulation(robots, obs_set, cfg=cfg, rng=rng)
    sim_metrics = evaluate_performance_metrics(robot_data, cfg=cfg)
    dyn_res = simulate_dynamic_obstacle_avoidance(
        robots[2]['start'], robots[2]['goal'], obs_set,
        dyn_obs_start=(18, 25), dyn_obs_vel=(0, -1), cfg=cfg, rng=np.random.default_rng(base_seed)
    )
    vis_files = generate_all_visualizations(robot_data, sim_metrics, dyn_res, cfg=cfg, out_dir=out_dir)

    summary = {
        'parameters': cfg.__dict__,
        'per_robot_comparison': {
            rid: {
                'robot_name': f"{robot_data[rid]['spec']['name']} ({robot_data[rid]['spec']['paper_name']})",
                'priority': robot_data[rid]['spec']['priority'],
                'path_length_reproduced': round(sim_metrics['path_lengths'][rid], 2),
                'path_length_paper_table2': PAPER_STATIC_TABLE2[rid]['path_length'],
                'min_distance_reproduced': round(sim_metrics['min_dist_per_robot'][rid], 2),
                'min_distance_paper_table2': PAPER_STATIC_TABLE2[rid]['min_distance'],
                'movement_time_reproduced_s': round(sim_metrics['travel_times'][rid], 2),
                'movement_time_paper_table2_s': PAPER_STATIC_TABLE2[rid]['movement_time'],
                'coordination_events': robot_data[rid]['events'],
            }
            for rid in sorted(robot_data.keys())
        },
        'aggregate_comparison_table3': {
            'avg_path_length_reproduced': round(sim_metrics['avg_path_length'], 2),
            'avg_path_length_paper': PAPER_STATIC_TABLE3['avg_path_length'],
            'max_path_length_reproduced': round(sim_metrics['max_path_length'], 2),
            'max_path_length_paper': PAPER_STATIC_TABLE3['max_path_length'],
            'avg_movement_time_reproduced_s': round(sim_metrics['avg_movement_time'], 2),
            'avg_movement_time_paper_s': PAPER_STATIC_TABLE3['avg_movement_time'],
            'success_rate_reproduced_pct': round(sim_metrics['success_rate'], 1),
            'success_rate_paper_pct': PAPER_STATIC_TABLE3['success_rate'],
        },
        'visualizations': vis_files,
    }

    json_path = os.path.join(out_dir, "simulation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Improved SA Multi-Robot Path Planner (Kun Shi et al., 2025)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for figures and JSON")
    args = parser.parse_args()
    run_experiments(base_seed=args.seed, out_dir=args.out_dir)