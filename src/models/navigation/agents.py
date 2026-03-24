"""ObjectNav navigation agents.

Multi-goal fallback navigation with stuck detection and goal switching.
Works with any number of candidates (including 1).

Interface:
    agent.run(obs) -> NavResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.utils.projection import quat_to_xyzw

from src.models.navigation.pointnav_policy import PointNavController
from src.utils.geometry import closest_point_to_position
from src.utils.projection import check_cloud_visibility, get_visible_closest_point, get_visible_point_indices


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> tuple[float, float]:
    """Compute polar coordinates (rho, theta) from agent to goal.

    Operates in Habitat's XZ world plane (Y-up).  Forward = -Z, Right = +X.

    Args:
        curr_pos: (2,) array [x, z] of agent position in world XZ plane.
        curr_heading: Agent heading in radians (Y-axis rotation, CCW from above).
        curr_goal: (2,) array [x, z] of goal position in world XZ plane.

    Returns:
        (rho, theta): distance (meters) and bearing angle (radians).
                      theta=0  -> goal is directly ahead.
                      theta>0  -> goal is to the left.
                      theta<0  -> goal is to the right.
    """
    dx = curr_goal[0] - curr_pos[0]
    dz = curr_goal[1] - curr_pos[1]

    rho = float(np.sqrt(dx * dx + dz * dz))

    sin_h = np.sin(curr_heading)
    cos_h = np.cos(curr_heading)

    forward = -sin_h * dx - cos_h * dz
    left = -cos_h * dx + sin_h * dz

    theta = float(np.arctan2(left, forward))
    return rho, theta


def get_agent_heading(agent_state) -> float:
    q = agent_state.rotation
    x, y, z, w = quat_to_xyzw(q)
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class NavResult:
    """Result from an agent's navigation run."""
    action_sequence: list[int] = field(default_factory=list)
    step_count: int = 0
    multi_goal_info: dict | None = None
    stop_reason: str = "max_steps"  # l2_stop, policy_stop, opportunistic_stop, all_goals_exhausted, no_reachable_goal, max_steps
    stop_details: dict | None = None  # xz_distance, candidate_idx, visibility_passed, visible_fraction


@dataclass
class MultiGoalConfig:
    """Configuration for multi-goal fallback navigation."""

    stuck_window: int = 30  # steps to check for stuck
    stuck_threshold: float = 0.5  # min displacement over window (meters)
    opportunistic_radius: float = 0.9  # L2 XZ-plane STOP distance to any goal (nearest surface)
    max_goal_switches: int = 4  # max retargets before giving up

    # Oscillation detection (back-and-forth over longer distances)
    oscillation_window: int = 60  # steps for oscillation check (longer than stuck)
    oscillation_ratio: float = 0.15  # net_displacement / total_path_length threshold
    oscillation_min_path: float = 2.0  # min total path length (m) to trigger check

    # Visibility-based stop validation
    visibility_check: bool = True
    hfov: float = 79.0
    sensor_height: float = 0.88
    min_depth: float = 0.5
    max_depth: float = 5.0
    depth_tolerance: float = 0.3
    min_visible_fraction: float = 0.05

    # Accumulative visibility tracking
    accumulate_visible: bool = False

    # Policy STOP gating: max consecutive overrides before accepting
    max_policy_stop_overrides: int = 5


def _closest_point_xz(cloud_xz: np.ndarray, pos_xz: np.ndarray, cloud_3d: np.ndarray) -> tuple[np.ndarray, float]:
    """Fast closest-point lookup using pre-computed XZ projection.

    Returns (closest_3d_point, xz_distance).
    """
    diff = cloud_xz - pos_xz
    dists_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
    idx = np.argmin(dists_sq)
    return cloud_3d[idx].copy(), float(np.sqrt(dists_sq[idx]))


def _check_stuck(position_history: list, window: int, threshold: float) -> bool:
    """Vectorized stuck detection: max displacement from window start < threshold."""
    if len(position_history) < window:
        return False
    pts = np.array(position_history[-window:])
    diffs = pts[1:] - pts[0]
    max_disp = float(np.sqrt(np.max(np.sum(diffs ** 2, axis=1))))
    return max_disp < threshold


def _check_oscillation(position_history: list, window: int,
                        ratio_thresh: float, min_path: float) -> bool:
    """Vectorized oscillation detection: net/total displacement ratio."""
    if len(position_history) < window:
        return False
    pts = np.array(position_history[-window:])
    net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
    steps = pts[1:] - pts[:-1]
    total_path = float(np.sum(np.sqrt(np.sum(steps ** 2, axis=1))))
    if total_path < min_path:
        return False
    return (net_disp / total_path) < ratio_thresh


class MultiGoalAgent:
    """Navigate with fallback across multiple goal candidates. Paper Step 4 (Sec III-E).

    When there is only 1 candidate, runs a fast single-goal path (no opportunistic
    stop loop, no stuck/oscillation detection — matching the old SingleGoalAgent).
    Multi-goal logic (opportunistic stop across all candidates, stuck detection,
    goal switching) only activates with 2+ candidates.

    Args:
        env: habitat.Env instance.
        pointnav: PointNavController instance.
        candidates: List of GoalCandidate objects.
        config: MultiGoalConfig instance.
        stop_radius: L2 distance (XZ plane) for STOP.
        gt_surface_clouds: Per-candidate point clouds for dynamic updates.
    """

    def __init__(
        self,
        env,
        pointnav: PointNavController,
        candidates: list,
        config: MultiGoalConfig,
        stop_radius: float = 0.9,
        gt_surface_clouds: dict[int, np.ndarray] | None = None,
    ):
        self.env = env
        self.pointnav = pointnav
        self.candidates = candidates
        self.config = config
        self.stop_radius = stop_radius
        self.gt_surface_clouds = gt_surface_clouds or {}
        self.confirmed_indices: dict[int, set] = {}
        self._multi_goal = len(candidates) >= 2

        # Pre-compute XZ projections for per-step closest-point lookups
        self._clouds_xz: dict[int, np.ndarray] = {}
        self._clouds_3d: dict[int, np.ndarray] = {}
        for ci, cand in enumerate(candidates):
            cloud = self._get_cloud(ci)
            if cloud is not None and len(cloud) > 0:
                self._clouds_3d[ci] = np.ascontiguousarray(cloud, dtype=np.float32)
                self._clouds_xz[ci] = np.ascontiguousarray(cloud[:, [0, 2]], dtype=np.float32)

    def _nearest_l2(self, agent_pos: np.ndarray, exclude_idx: int | None = None) -> tuple[int | None, np.ndarray | None]:
        """Find nearest candidate by closest surface point (L2 in XZ), excluding given index."""
        agent_xz = np.array([agent_pos[0], agent_pos[2]], dtype=np.float32)
        best_idx, best_dist = None, float("inf")
        best_point = None
        for idx, cand in enumerate(self.candidates):
            if idx == exclude_idx:
                continue
            if idx in self._clouds_xz:
                cp, dist = _closest_point_xz(self._clouds_xz[idx], agent_xz, self._clouds_3d[idx])
            else:
                cp = cand.centroid.copy()
                dist = float(np.linalg.norm(cp[[0, 2]] - agent_xz))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_point = cp
        if best_idx is not None:
            return best_idx, np.asarray(best_point, dtype=np.float32)
        return None, None

    def _get_cloud(self, idx: int) -> np.ndarray | None:
        """Get point cloud for candidate, preferring gt_surface_clouds."""
        return self.gt_surface_clouds.get(idx, self.candidates[idx].point_cloud)

    def _visibility_gate(self, cloud_idx: int, position, rotation, depth) -> tuple[bool, float]:
        """Run visibility check for a candidate. Returns (passed, fraction)."""
        if not self.config.visibility_check:
            return True, 1.0
        cloud = self._get_cloud(cloud_idx)
        vis, frac, _, _ = check_cloud_visibility(
            cloud, position, rotation, depth,
            hfov=self.config.hfov, sensor_height=self.config.sensor_height,
            min_depth=self.config.min_depth, max_depth=self.config.max_depth,
            depth_tolerance=self.config.depth_tolerance,
            min_visible_fraction=self.config.min_visible_fraction,
        )
        return vis, frac

    def _update_goal_from_obs(self, cand_cloud, current_goal_idx, position, rotation,
                               depth, goal_2d, agent_2d):
        """Observation-based closest-point update. Returns (new_goal_3d, new_goal_2d, reset_policy)."""
        cfg = self.config

        if cfg.accumulate_visible:
            vis_idx = get_visible_point_indices(
                cand_cloud, position, rotation, depth,
                hfov=cfg.hfov, sensor_height=cfg.sensor_height,
                min_depth=cfg.min_depth, max_depth=cfg.max_depth,
                depth_tolerance=cfg.depth_tolerance,
            )
            if vis_idx is not None:
                confirmed = self.confirmed_indices.setdefault(current_goal_idx, set())
                confirmed.update(vis_idx.tolist())
            confirmed = self.confirmed_indices.get(current_goal_idx)
            if confirmed:
                confirmed_cloud = cand_cloud[np.array(sorted(confirmed))]
                new_closest = closest_point_to_position(
                    confirmed_cloud, position,
                ).astype(np.float32)
            else:
                new_closest = closest_point_to_position(
                    cand_cloud, position,
                ).astype(np.float32)
        else:
            vis_closest = get_visible_closest_point(
                cand_cloud, position, rotation, depth,
                hfov=cfg.hfov, sensor_height=cfg.sensor_height,
                min_depth=cfg.min_depth, max_depth=cfg.max_depth,
                depth_tolerance=cfg.depth_tolerance,
            )
            if vis_closest is None:
                return None, goal_2d, False
            new_closest = vis_closest.astype(np.float32)

        new_2d = np.array([new_closest[0], new_closest[2]])
        delta = float(np.linalg.norm(new_2d - goal_2d))
        agent_to_new = float(np.linalg.norm(agent_2d - new_2d))
        if delta >= 0.5 or (delta >= 0.1 and agent_to_new <= 2.0):
            return new_closest, new_2d, True
        return None, goal_2d, False

    def run(self, obs) -> NavResult:
        """Run navigation within current episode.

        Single-candidate: fast path (L2 stop + visibility, no stuck detection).
        Multi-candidate: full protocol (opportunistic stop, stuck/oscillation, goal switching).
        """
        sim = self.env.sim
        cfg = self.config
        action_sequence = []
        stop_reason = "max_steps"
        stop_details = {}

        state = sim.get_agent(0).get_state()
        agent_pos = np.array(state.position)

        current_goal_idx, nearest_goal = self._nearest_l2(agent_pos)
        if current_goal_idx is None:
            action_sequence.append(PointNavController.STOP)
            self.env.step(PointNavController.STOP)
            return NavResult(
                action_sequence=action_sequence,
                step_count=1,
                stop_reason="no_reachable_goal",
                multi_goal_info={
                    "n_candidates": len(self.candidates),
                    "goal_switches": 0,
                    "stop_reason": "no_reachable_goal",
                    "final_goal_idx": -1,
                    "stop_details": {},
                },
                stop_details={},
            )

        nav_goal = nearest_goal
        goal_2d = np.array([nav_goal[0], nav_goal[2]])
        self.pointnav.reset()
        policy_step = 0
        step = 0
        consecutive_policy_stop_overrides = 0

        goal_switches = 0
        position_history = [] if self._multi_goal else None
        # Opportunistic stop is expensive (closest-point on ALL candidate clouds).
        # Amortize to every N steps — agent moves ~0.25m/step so
        # checking every 5 steps means ≤1.25m lag, well within stop_radius.
        opp_check_interval = 5

        while not self.env.episode_over:
            depth = obs["depth"]
            state = sim.get_agent(0).get_state()
            position = np.array(state.position)
            heading = get_agent_heading(state)
            agent_2d = np.array([position[0], position[2]])
            if self._multi_goal:
                position_history.append(position.copy())

            cand_cloud = self._get_cloud(current_goal_idx)
            if cand_cloud is not None and len(cand_cloud) > 0:
                new_goal, goal_2d_new, reset = self._update_goal_from_obs(
                    cand_cloud, current_goal_idx, position, state.rotation,
                    depth, goal_2d, agent_2d,
                )
                if reset:
                    nav_goal = new_goal
                    goal_2d = goal_2d_new
                    self.pointnav.reset()
                    policy_step = 0

            l2_to_current = float(np.linalg.norm(agent_2d - goal_2d))
            if l2_to_current < self.stop_radius:
                vis_ok, vis_frac = self._visibility_gate(current_goal_idx, position, state.rotation, depth)
                if vis_ok:
                    action_sequence.append(PointNavController.STOP)
                    obs = self.env.step(PointNavController.STOP)
                    step += 1
                    stop_reason = "l2_stop"
                    stop_details = {
                        "xz_distance": round(l2_to_current, 4),
                        "candidate_idx": current_goal_idx,
                        "visibility_passed": vis_ok,
                        "visible_fraction": round(vis_frac, 4),
                    }
                    break

            if self._multi_goal and step % opp_check_interval == 0:
                # Opportunistic stop: closest-point on all candidate clouds
                best_opp_dist = float("inf")
                best_opp_idx = -1
                for ci, c in enumerate(self.candidates):
                    if ci in self._clouds_xz:
                        _, opp_dist = _closest_point_xz(self._clouds_xz[ci], agent_2d, self._clouds_3d[ci])
                    else:
                        opp_dist = float(np.linalg.norm(agent_2d - c.centroid[[0, 2]]))
                    if opp_dist < best_opp_dist:
                        best_opp_dist = opp_dist
                        best_opp_idx = ci
                if best_opp_dist < cfg.opportunistic_radius:
                    vis_ok, vis_frac = self._visibility_gate(best_opp_idx, position, state.rotation, depth)
                    if vis_ok:
                        action_sequence.append(PointNavController.STOP)
                        obs = self.env.step(PointNavController.STOP)
                        step += 1
                        stop_reason = "opportunistic_stop"
                        stop_details = {
                            "xz_distance": round(best_opp_dist, 4),
                            "candidate_idx": best_opp_idx,
                            "visibility_passed": vis_ok,
                            "visible_fraction": round(vis_frac, 4),
                        }
                        break

                is_stuck = (
                    _check_stuck(position_history, cfg.stuck_window, cfg.stuck_threshold)
                    or _check_oscillation(position_history, cfg.oscillation_window,
                                          cfg.oscillation_ratio, cfg.oscillation_min_path)
                )
                if is_stuck and goal_switches < cfg.max_goal_switches:
                    next_idx, next_goal = self._nearest_l2(
                        position, exclude_idx=current_goal_idx,
                    )
                    if next_idx is not None:
                        current_goal_idx = next_idx
                        nav_goal = next_goal
                        goal_2d = np.array([nav_goal[0], nav_goal[2]])
                        self.pointnav.reset()
                        policy_step = 0
                        position_history.clear()
                        consecutive_policy_stop_overrides = 0
                        goal_switches += 1
                    else:
                        action_sequence.append(PointNavController.STOP)
                        obs = self.env.step(PointNavController.STOP)
                        step += 1
                        stop_reason = "all_goals_exhausted"
                        stop_details = {
                            "xz_distance": round(float(np.linalg.norm(agent_2d - goal_2d)), 4),
                            "candidate_idx": current_goal_idx,
                        }
                        break

            _rho, _theta = rho_theta(agent_2d, heading, goal_2d)
            action = self.pointnav.act(depth, _rho, _theta, policy_step)

            if action == PointNavController.STOP:
                allow_policy_stop = True
                ps_vis_passed = True
                ps_vis_frac = 1.0
                if cfg.visibility_check and consecutive_policy_stop_overrides < cfg.max_policy_stop_overrides:
                    ps_vis_passed, ps_vis_frac = self._visibility_gate(
                        current_goal_idx, position, state.rotation, depth)
                    if not ps_vis_passed:
                        allow_policy_stop = False
                        action = 1  # MOVE_FORWARD
                        consecutive_policy_stop_overrides += 1
                if allow_policy_stop:
                    consecutive_policy_stop_overrides = 0
            else:
                consecutive_policy_stop_overrides = 0

            action_sequence.append(action)
            obs = self.env.step(action)
            step += 1
            policy_step += 1

            if action == PointNavController.STOP:
                stop_reason = "policy_stop"
                stop_details = {
                    "xz_distance": round(float(np.linalg.norm(agent_2d - goal_2d)), 4),
                    "candidate_idx": current_goal_idx,
                    "visibility_passed": ps_vis_passed,
                    "visible_fraction": round(ps_vis_frac, 4),
                    "policy_stop_overrides": consecutive_policy_stop_overrides,
                }
                break

        final_goal_idx = current_goal_idx if current_goal_idx is not None else -1
        return NavResult(
            action_sequence=action_sequence,
            step_count=step,
            multi_goal_info={
                "n_candidates": len(self.candidates),
                "goal_switches": goal_switches,
                "stop_reason": stop_reason,
                "final_goal_idx": final_goal_idx,
                "stop_details": stop_details,
            },
            stop_reason=stop_reason,
            stop_details=stop_details,
        )
