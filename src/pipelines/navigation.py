"""Step 4: Navigation pipeline (goal determination + episode processing).

Paper: Sec III-E (Goal-Directed Navigation).

Orchestrates retrieval + localization per episode, then navigates to the
predicted 3D goal using DD-PPO PointNav. Multi-goal fallback enabled by default.

Key functions:
- ``determine_nav_goal``: Runs Steps 1-3 for a single episode (retrieval → localization).
- ``collect_outcome``: Collects habitat metrics into a serializable EpisodeOutcome.
- ``build_episode_result``: Assembles a JSON-serializable dict for one episode.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any

import numpy as np

from src.pipelines.localization import localize
CAMERA_FLOOR_THRESHOLD = 1.0
from src.pipelines.retrieval import search_scene
from src.utils.geometry import closest_point_to_position
from src.utils.multi_view_fusion import group_predictions_by_overlap, l2_sort_candidates


@dataclass
class GoalDeterminationResult:
    """Output of determine_nav_goal()."""

    nav_goal: np.ndarray | None = None
    goal_candidates: list = field(default_factory=list)
    preds: list | None = None
    valid_indices: list | None = None
    pred_metadata: list | None = None
    top1_image: np.ndarray | None = None
    status_tags: list = field(default_factory=list)
    skip_reason: str | None = None

    def release_heavy_data(self):
        """Free point clouds and images that are no longer needed after navigation.

        Call after build_episode_result() / navigation to release large arrays
        (point clouds in pred_metadata and goal_candidates, top1_image) while
        keeping lightweight scalar fields for metrics.
        """
        if self.pred_metadata:
            for m in self.pred_metadata:
                m.pop("point_cloud", None)
        self.pred_metadata = None
        for cand in self.goal_candidates:
            cand.point_cloud = None
        self.top1_image = None
        self.preds = None


@dataclass
class EpisodeOutcome:
    """Post-navigation metrics from habitat.Env."""

    success: float = 0.0
    spl: float = 0.0
    soft_spl: float = 0.0
    dtg: float = 0.0
    num_steps: int = 0
    query_time_s: float = 0.0
    multi_goal_info: dict | None = None
    action_sequence: list = field(default_factory=list)
    stop_reason: str = "max_steps"
    stop_details: dict | None = None


def record_skip(
    env: Any,
    nav_acc: Any,
    all_results: dict[str, dict],
    scene_name: str,
    category: str,
    ep_id: int,
    skip_reason: str,
    *,
    send_stop: bool = True,
    q_start: float | None = None,
) -> None:
    """Record an episode as skipped in metrics.

    Episodes are counted in the denominator (no silent inflation).

    Args:
        send_stop: If True, send STOP action so habitat computes metrics.
            Set False when the episode already terminated (e.g. after reset).
        q_start: If provided, compute and record query_time_s.
    """
    if send_stop:
        env.step(0)  # STOP → habitat computes metrics (success=0)
    metrics = env.get_metrics()
    success = metrics.get("success", 0.0)
    spl = metrics.get("spl", 0.0)
    soft_spl = metrics.get("soft_spl", 0.0)
    dtg = metrics.get("distance_to_goal", 0.0)
    nav_acc.update(success, spl, soft_spl, dtg, 0,
                   scene=scene_name, category=category)
    result = {
        "scene": scene_name, "episode_id": str(ep_id),
        "category": category, "success": success, "spl": spl,
        "soft_spl": soft_spl, "distance_to_goal": dtg,
        "num_steps": 0, "nav_goal": None,
        "skip_reason": skip_reason,
    }
    if q_start is not None:
        q_time = time.perf_counter() - q_start
        result["query_time_s"] = round(q_time, 3)
        print(f" -> SKIP ({q_time:.1f}s)")
    else:
        print(f" -> SKIP ({skip_reason})")
    all_results[f"{scene_name}/ep{ep_id}/{category}"] = result


def snap_viewpoints_to_navmesh(env) -> int:
    """Snap episode viewpoint Y values to navmesh elevation in-place.

    Habitat's DistanceToGoal caches references to vp.agent_state.position,
    so in-place mutation of position[1] propagates to the measurement cache.
    The agent walks on the navmesh, so its Y follows navmesh elevation.
    Without this fix, the Y gap between raw viewpoint Y and navmesh Y inflates
    geodesic distance, causing false failures even when XZ distance is tiny.

    Returns number of viewpoints snapped.
    """
    sim = env.sim
    episode = env.current_episode
    count = 0
    for goal in episode.goals:
        if not hasattr(goal, "view_points") or not goal.view_points:
            continue
        for vp in goal.view_points:
            pos = vp.agent_state.position
            snapped = sim.pathfinder.snap_point(pos)
            if not np.isfinite(snapped).all():
                continue
            if abs(pos[1] - snapped[1]) > 1e-3:
                pos[1] = snapped[1]
                count += 1
    dtg_uuid = "distance_to_goal"
    if hasattr(env, "task") and hasattr(env.task, "measurements"):
        measures = env.task.measurements.measures
        if dtg_uuid in measures:
            dtg = measures[dtg_uuid]
            # Re-cache viewpoint positions and recompute initial DTG
            dtg.reset_metric(episode=episode)
    return count


def reset_metric_caches(env) -> None:
    """Clear stale shortest-path and DTG caches after viewpoint snapping.

    Without this, geodesic_distance reuses un-snapped Y coordinates,
    inflating DTG. Must be called after snap_viewpoints_to_navmesh().
    """
    env.current_episode._shortest_path_cache = None
    dtg = env.task.measurements.measures.get("distance_to_goal")
    if dtg is not None:
        dtg._previous_position = None
        dtg.update_metric(episode=env.current_episode)
        spl = env.task.measurements.measures.get("spl")
        if spl is not None:
            spl._start_end_episode_distance = dtg.get_metric()


def handle_episode_over(env, nav_acc, all_results, scene_name, scene_eps, i):
    """Handle malformed episodes that terminate during init. Returns True if skipped."""
    if not env.episode_over:
        return False
    episode = env.current_episode
    category = episode.object_category
    ep_id = episode.episode_id
    print(f"  [{i+1}/{len(scene_eps)}] ep={ep_id} cat={category}", end="")
    record_skip(
        env, nav_acc, all_results, scene_name, category, ep_id,
        "episode_over_after_reset", send_stop=False,
    )
    return True


def fuse_goal_candidates(
    goal_candidates: list,
    category: str,
    scene_loader: Any,
    sam3_segmenter: Any,
    start_pos: np.ndarray,
    args: Any,
    pred_metadata: list[dict],
    all_poses_raw: tuple | None = None,
    filtered_preds: list[np.ndarray] | None = None,
    filtered_meta: list[dict] | None = None,
) -> None:
    """Run spatial fusion for nav candidates with floor filtering.

    Thin wrapper around ``fuse_candidates()``: filters cameras to the same
    floor as the agent, then delegates fusion.

    Args:
        goal_candidates: List of GoalCandidate objects to update in-place.
        category: Object category string (e.g. "chair").
        scene_loader: Scene loader providing depth, poses, and intrinsics.
        sam3_segmenter: SAM3Segmenter instance for neighbor segmentation.
        start_pos: Agent start position (3,).
        args: Namespace with spatial_max_views, spatial_max_distance,
            frustum_margin, fusion_threshold, overlap_threshold,
            hdbscan, hdbscan_min_cluster_size, hdbscan_min_samples,
            max_mask_depth, and sensor_height attributes.
        pred_metadata: Per-rank metadata dicts.
        all_poses_raw: Optional (all_poses, all_fids) tuple.
        filtered_preds: Per-rank 3D predictions (parallel to filtered_meta).
        filtered_meta: Per-rank metadata dicts with point_cloud, scores.
    """
    from src.utils.multi_view_fusion import fuse_candidates

    if all_poses_raw is not None:
        all_poses, all_fids = all_poses_raw
    else:
        all_poses, all_fids = scene_loader.get_all_poses()
        all_fids = np.asarray(all_fids)

    reference_cam_y = start_pos[1] + args.sensor_height
    all_cam_ys = all_poses[:, 1, 3]
    same_floor_mask = np.abs(all_cam_ys - reference_cam_y) <= CAMERA_FLOOR_THRESHOLD
    all_poses = all_poses[same_floor_mask]
    all_fids = all_fids[same_floor_mask]
    all_poses_cache = (all_poses, all_fids)

    query = category.replace("_", " ")

    fuse_candidates(
        goal_candidates,
        query=query,
        scene_loader=scene_loader,
        sam3_segmenter=sam3_segmenter,
        pred_metadata=pred_metadata,
        preds=filtered_preds,
        pred_meta=filtered_meta,
        all_poses_cache=all_poses_cache,
        spatial_max_views=args.spatial_max_views,
        spatial_max_distance=args.spatial_max_distance,
        frustum_margin=args.frustum_margin,
        fusion_threshold=args.fusion_threshold,
        overlap_threshold=args.overlap_threshold,
        hdbscan_clean=args.hdbscan,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        max_mask_depth=args.max_mask_depth,
        use_centroid=False,
        robot_position=start_pos,
    )


def determine_nav_goal(
    args: Any,
    episode: Any,
    start_pos: np.ndarray,
    scene_loader: Any,
    images: list[np.ndarray],
    all_frame_ids: list[int],
    retriever: Any,
    sam3_segmenter: Any,
    all_poses_raw: tuple | None = None,
) -> GoalDeterminationResult:
    """Goal selection via retrieval pipeline (Steps 1-3).

    Returns GoalDeterminationResult with nav_goal set (or skip_reason).
    """
    result = GoalDeterminationResult()
    category = episode.object_category

    if scene_loader is None:
        result.status_tags.append(" [no scene loader]")
        result.skip_reason = "no_scene_loader"
        return result

    query_text = category.replace("_", " ")
    search_res = search_scene(
        query=query_text,
        images=images,
        frame_ids=all_frame_ids,
        retriever=retriever,
        top_k=args.top_k,
        scene_loader=scene_loader,
        robot_position=start_pos,
        sensor_height=args.sensor_height,
        min_retrieval_score=args.min_retrieval_score,
    )

    preds, top1_img_idx, ep_pred_metadata = localize(
        query=query_text,
        images=images,
        frame_ids=all_frame_ids,
        search_result=search_res,
        scene_loader=scene_loader,
        sam3_segmenter=sam3_segmenter,
        top_k=args.top_k,
        robot_position=start_pos,
        max_mask_depth=args.max_mask_depth,
    )

    result.preds = preds
    result.pred_metadata = ep_pred_metadata
    result.top1_image = images[top1_img_idx] if top1_img_idx is not None else None

    if preds:
        has_vlm = args.vlm

        result.valid_indices = [j for j, m in enumerate(ep_pred_metadata)
                                 if not has_vlm or m.get("detected", True)]
        if result.valid_indices:
            filtered_preds = [preds[j] for j in result.valid_indices]
            filtered_meta = [ep_pred_metadata[j] for j in result.valid_indices]
        else:
            # All predictions invalid — keep top-1 only as fallback
            # (VLM mode: VLM-reranked top-1; feature-only: retrieval top-1)
            filtered_preds = [preds[0]]
            filtered_meta = [ep_pred_metadata[0]]

        n_kept = len(result.valid_indices) or 1
        n_filtered_out = len(preds) - n_kept
        if n_filtered_out > 0:
            result.status_tags.append(f" [filtered {n_filtered_out}/{len(preds)} invalid]")

        goal_candidates = group_predictions_by_overlap(
            filtered_preds, metadata=filtered_meta,
            overlap_threshold=args.grouping_overlap_threshold,
            point_threshold=args.fusion_threshold,
            centroid_fallback_distance=args.grouping_centroid_fallback,
            proximity_threshold=args.grouping_proximity_threshold,
        )

        if args.spatial_fusion and sam3_segmenter is not None and goal_candidates:
            fuse_goal_candidates(
                goal_candidates, category, scene_loader,
                sam3_segmenter, start_pos, args, ep_pred_metadata,
                all_poses_raw=all_poses_raw,
                filtered_preds=filtered_preds,
                filtered_meta=filtered_meta,
            )

        l2_sort_candidates(goal_candidates, start_pos)

        if args.multi_goal:
            result.goal_candidates = goal_candidates
        # Initial nav goal: closest point on object PCD (not fused centroid)
        cand0 = goal_candidates[0]
        if cand0.point_cloud is not None and len(cand0.point_cloud) > 0:
            result.nav_goal = closest_point_to_position(
                cand0.point_cloud, start_pos,
            ).astype(np.float32)
        else:
            result.nav_goal = cand0.centroid

        n_cand = len(goal_candidates)
        fused_tag = "+fused" if args.spatial_fusion else ""
        if n_cand > 1:
            result.status_tags.append(
                f" [{len(filtered_preds)} preds -> {n_cand} candidates{fused_tag}]")
        elif args.spatial_fusion:
            result.status_tags.append(f" [1 candidate{fused_tag}]")
    else:
        # No 3D predictions (all depth/projection failed) — navigate to
        # top-1 retrieval image's camera position as fallback.
        if top1_img_idx is not None:
            top1_pose = scene_loader.load_pose(all_frame_ids[top1_img_idx])
            result.nav_goal = np.array(top1_pose[:3, 3], dtype=np.float32)
            result.status_tags.append(" [no pred, cam-pos fallback]")
        else:
            result.status_tags.append(" [no retrieval candidates]")
            result.skip_reason = "no_retrieval_candidates"

    return result


def collect_outcome(env, nav_result, q_start) -> EpisodeOutcome:
    """Read habitat metrics after navigation → EpisodeOutcome."""
    metrics = env.get_metrics()

    return EpisodeOutcome(
        success=metrics.get("success", 0.0),
        spl=metrics.get("spl", 0.0),
        soft_spl=metrics.get("soft_spl", 0.0),
        dtg=metrics.get("distance_to_goal", 0.0),
        num_steps=nav_result.step_count,
        query_time_s=time.perf_counter() - q_start,
        multi_goal_info=nav_result.multi_goal_info,
        action_sequence=nav_result.action_sequence,
        stop_reason=nav_result.stop_reason,
        stop_details=nav_result.stop_details,
    )


def build_episode_result(
    scene_name, ep_id, category, outcome, nav_goal,
    preds, valid_indices, goal_candidates,
) -> dict:
    """Assemble JSON dict for one episode."""
    ep_result = {
        "scene": scene_name,
        "episode_id": str(ep_id),
        "category": category,
        "success": outcome.success,
        "spl": outcome.spl,
        "soft_spl": outcome.soft_spl,
        "distance_to_goal": outcome.dtg,
        "num_steps": outcome.num_steps,
        "nav_goal": nav_goal.tolist() if nav_goal is not None else None,
        "query_time_s": round(outcome.query_time_s, 3),
        "stop_reason": outcome.stop_reason,
    }
    if outcome.multi_goal_info is not None:
        ep_result["multi_goal"] = outcome.multi_goal_info
    if preds:
        ep_result["pred_filtering"] = {
            "n_total": len(preds),
            "n_valid": len(valid_indices) if valid_indices else len(preds),
            "n_filtered_out": len(preds) - (len(valid_indices) if valid_indices else len(preds)),
            "n_candidates": len(goal_candidates) if goal_candidates else 0,
        }
    return ep_result


RESULTS_CONFIG_KEYS = [
    "data_root", "split", "scene", "agent_height", "agent_radius",
    "sensor_height", "forward_step_size", "turn_angle",
    "policy_input_size", "min_depth", "max_depth",
    "nav_max_steps", "stop_radius", "hfov",
    "retrieval_model", "vlm_model", "spatial_fusion", "hdbscan",
    "spatial_max_views", "spatial_max_distance", "frustum_margin",
    "fusion_threshold", "multi_goal", "stuck_window", "stuck_threshold",
    "oscillation_window", "oscillation_ratio", "oscillation_min_path",
    "opportunistic_radius", "cluster_distance", "max_goal_switches",
    "device", "success_distance", "distance_to",
]


def save_results_json(args, run_dir, config_path, nav_acc, all_results,
                      *, partial=False):
    """Save results.json. Called after each scene (partial) and at the end."""
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "run_dir": str(run_dir),
        "partial": partial,
        "config": {k: getattr(args, k, None) for k in RESULTS_CONFIG_KEYS},
        "results": nav_acc.to_json(),
        "episode_results": all_results,
    }
    output_data["config"]["config_file"] = str(config_path) if config_path.exists() else None
    output_data["config"]["metrics_source"] = "habitat.Env (ObjectNav-v1)"
    output_data["config"]["sam3"] = True

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path


def run_nav_episodes(
    env, scene_eps, scene_name, args,
    scene_loader, images, all_frame_ids,
    retriever, sam3_segmenter, pointnav,
    nav_acc, all_results,
    all_poses_raw=None,
):
    """Run the per-episode nav evaluation loop for a single scene.

    Shared by eval_hm3d, eval_mp3d, and eval_ovon.
    """
    from src.evaluation.helpers import cleanup, quiet_reset
    from src.models.navigation.agents import MultiGoalAgent, MultiGoalConfig

    for i, _ in enumerate(scene_eps):
        obs = quiet_reset(env)

        snap_viewpoints_to_navmesh(env)
        reset_metric_caches(env)

        if handle_episode_over(env, nav_acc, all_results, scene_name, scene_eps, i):
            continue

        episode = env.current_episode
        category = episode.object_category
        ep_id = episode.episode_id
        start_pos = np.array(episode.start_position)
        q_start = time.perf_counter()

        print(f"  [{i+1}/{len(scene_eps)}] ep={ep_id} cat={category}", end="")

        goal_det = determine_nav_goal(
            args, episode, start_pos,
            scene_loader, images, all_frame_ids,
            retriever, sam3_segmenter,
            all_poses_raw=all_poses_raw,
        )

        if goal_det.skip_reason:
            record_skip(
                env, nav_acc, all_results, scene_name,
                category, ep_id, goal_det.skip_reason, q_start=q_start)
            goal_det.release_heavy_data()
            continue

        nav_goal = goal_det.nav_goal

        gt_clouds = {ci: c.point_cloud for ci, c in enumerate(goal_det.goal_candidates)
                     if c.point_cloud is not None}
        nav_config = MultiGoalConfig(
            stuck_window=args.stuck_window,
            stuck_threshold=args.stuck_threshold,
            opportunistic_radius=args.opportunistic_radius,
            max_goal_switches=args.max_goal_switches,
            oscillation_window=args.oscillation_window,
            oscillation_ratio=args.oscillation_ratio,
            oscillation_min_path=args.oscillation_min_path,
            visibility_check=args.visibility_check,
            hfov=args.hfov,
            sensor_height=args.sensor_height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            depth_tolerance=args.depth_tolerance,
            min_visible_fraction=args.min_visible_fraction,
            accumulate_visible=args.accumulate_visible,
            max_policy_stop_overrides=args.max_policy_stop_overrides,
        )
        agent = MultiGoalAgent(
            env=env,
            pointnav=pointnav,
            candidates=goal_det.goal_candidates,
            config=nav_config,
            stop_radius=args.stop_radius,
            gt_surface_clouds=gt_clouds,
        )

        nav_result = agent.run(obs)

        outcome = collect_outcome(env, nav_result, q_start)
        nav_acc.update(outcome.success, outcome.spl, outcome.soft_spl, outcome.dtg,
                       outcome.num_steps, scene=scene_name, category=category)
        tag = "SUCCESS" if outcome.success else "FAIL"
        print(f" -> {tag} dtg={outcome.dtg:.2f}m spl={outcome.spl:.3f} "
              f"sspl={outcome.soft_spl:.3f} steps={outcome.num_steps} ({outcome.query_time_s:.1f}s)")

        ep_result = build_episode_result(
            scene_name, ep_id, category, outcome, nav_goal,
            goal_det.preds, goal_det.valid_indices,
            goal_det.goal_candidates,
        )
        all_results[f"{scene_name}/ep{ep_id}/{category}"] = ep_result

        goal_det.release_heavy_data()
        cleanup(scene_loader=scene_loader, sam3_segmenter=sam3_segmenter,
                images=images, episode_idx=i, interval=5)
