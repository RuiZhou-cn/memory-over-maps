"""Configuration utilities: YAML loading, config+CLI merging for all datasets.

All pipeline parameters come from YAML. CLI only provides data selection
(--scene, --episode) and hardware (--device, --output).
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


def get_scene_name(scene_id: str) -> str:
    """Extract scene name from habitat scene_id path.

    Works for HM3D (``TEEsavR23oF.basis.glb`` → ``TEEsavR23oF``)
    and MP3D (``8194nk5LbLH.glb`` → ``8194nk5LbLH``) paths.
    """
    return Path(scene_id).stem.replace(".basis", "")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config file with ``_base`` inheritance, return nested dict.

    If the YAML contains a ``_base`` key, the referenced file is loaded first
    (resolved relative to the current config's directory) and the current
    config's values are deep-merged on top.  Inheritance can be chained
    (e.g. ``hm3d.yaml`` → ``base_nav.yaml`` → ``base.yaml``).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("_base", None)
    if base_rel:
        base_path = str(Path(config_path).parent / base_rel)
        base_cfg = load_config(base_path)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def cfg_get(cfg: dict, *keys, default=None):
    """Get nested value from config dict. e.g. cfg_get(cfg, 'agent', 'height')."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _merge_pipeline_args(args: argparse.Namespace, cfg: dict[str, Any], cli_args) -> None:
    """Set retrieval, VLM, SAM3, and multi-view fusion args shared by all evals."""
    args.retrieval_model = cfg_get(cfg, "retrieval", "model")
    args.stage1_top_k = cfg_get(cfg, "retrieval", "stage1_top_k")
    args.stage2_top_k = cfg_get(cfg, "retrieval", "stage2_top_k")
    args.min_retrieval_score = cfg_get(cfg, "retrieval", "min_score")
    args.extractor_kwargs = cfg_get(cfg, "retrieval", "extractor_kwargs") or {}

    args.vlm_model = cfg_get(cfg, "vlm", "model")
    args.vlm_batch_size = cfg_get(cfg, "vlm", "batch_size")
    args.vlm = not getattr(cli_args, "no_vlm", False)

    args.sam3_confidence = cfg_get(cfg, "sam3", "confidence")
    args.sam3_batch_size = cfg_get(cfg, "sam3", "batch_size")

    args.spatial_fusion = cfg_get(cfg, "multi_view", "spatial_fusion")
    args.fusion_threshold = cfg_get(cfg, "multi_view", "fusion_threshold")
    args.overlap_threshold = cfg_get(cfg, "multi_view", "overlap_threshold")
    args.spatial_max_views = cfg_get(cfg, "multi_view", "spatial_max_views")
    args.spatial_max_distance = cfg_get(cfg, "multi_view", "spatial_max_distance")
    args.frustum_margin = cfg_get(cfg, "multi_view", "frustum_margin")
    args.hdbscan = cfg_get(cfg, "multi_view", "hdbscan")
    args.hdbscan_min_cluster_size = cfg_get(cfg, "multi_view", "hdbscan_min_cluster_size")
    args.hdbscan_min_samples = cfg_get(cfg, "multi_view", "hdbscan_min_samples")
    args.grouping_overlap_threshold = cfg_get(cfg, "multi_view", "grouping_overlap_threshold")
    args.grouping_centroid_fallback = cfg_get(cfg, "multi_view", "grouping_centroid_fallback")
    args.grouping_proximity_threshold = cfg_get(cfg, "multi_view", "grouping_proximity_threshold")
    args.max_mask_depth = cfg_get(cfg, "multi_view", "max_mask_depth")

    args.top_k = cfg_get(cfg, "eval", "top_k")


def _merge_common_nav_args(cfg: dict[str, Any], cli_args) -> argparse.Namespace:
    """Merge YAML config with CLI args for fields shared across HM3D/MP3D/OVON.

    All pipeline parameters come from YAML config. CLI only provides:
    - Data selection: scene, split, episode
    - Hardware: device, output
    """
    args = argparse.Namespace()

    args.scene = cli_args.scene
    args.split = cli_args.split or cfg_get(cfg, "data", "split", default="val")

    args.agent_height = cfg_get(cfg, "agent", "height")
    args.agent_radius = cfg_get(cfg, "agent", "radius")
    args.sensor_height = cfg_get(cfg, "agent", "sensor_height")

    args.forward_step_size = cfg_get(cfg, "actions", "forward_step_size")
    args.turn_angle = cfg_get(cfg, "actions", "turn_angle")

    args.pointnav_weights = cfg_get(cfg, "policy", "weights")
    args.policy_input_size = cfg_get(cfg, "policy", "input_size")

    args.min_depth = cfg_get(cfg, "sensor", "min_depth")
    args.max_depth = cfg_get(cfg, "sensor", "max_depth")

    args.nav_max_steps = cfg_get(cfg, "navigation", "max_steps")
    args.stop_radius = cfg_get(cfg, "navigation", "stop_radius")
    args.success_distance = cfg_get(cfg, "navigation", "success_distance")
    args.distance_to = cfg_get(cfg, "navigation", "distance_to")

    args.multi_goal = cfg_get(cfg, "navigation", "multi_goal")
    args.stuck_window = cfg_get(cfg, "navigation", "stuck_window")
    args.stuck_threshold = cfg_get(cfg, "navigation", "stuck_threshold")
    args.opportunistic_radius = cfg_get(cfg, "navigation", "opportunistic_radius")
    args.cluster_distance = cfg_get(cfg, "navigation", "cluster_distance")
    args.max_goal_switches = cfg_get(cfg, "navigation", "max_goal_switches")
    args.oscillation_window = cfg_get(cfg, "navigation", "oscillation_window")
    args.oscillation_ratio = cfg_get(cfg, "navigation", "oscillation_ratio")
    args.oscillation_min_path = cfg_get(cfg, "navigation", "oscillation_min_path")

    args.visibility_check = cfg_get(cfg, "navigation", "visibility_check")
    args.depth_tolerance = cfg_get(cfg, "navigation", "depth_tolerance")
    args.min_visible_fraction = cfg_get(cfg, "navigation", "min_visible_fraction")
    args.accumulate_visible = cfg_get(cfg, "navigation", "accumulate_visible")
    args.max_policy_stop_overrides = cfg_get(cfg, "navigation", "max_policy_stop_overrides")

    args.hfov = cfg_get(cfg, "camera", "hfov")

    args.keyframing = cfg_get(cfg, "keyframing", "enabled")
    args.keyframe_rotation = cfg_get(cfg, "keyframing", "rotation_threshold")
    args.keyframe_translation = cfg_get(cfg, "keyframing", "translation_threshold")

    _merge_pipeline_args(args, cfg, cli_args)

    args.episode = cli_args.episode

    args.device = cli_args.device or "cuda"
    args.output = cli_args.output

    return args


def merge_nav_config_and_args(
    cfg: dict[str, Any], cli_args, dataset_name: str
) -> argparse.Namespace:
    """Merge config and CLI args for navigation datasets (HM3D, MP3D).

    Wraps :func:`_merge_common_nav_args` and appends the dataset-specific
    ``data_root`` field, defaulting to ``data/{dataset_name}``.

    Args:
        cfg: Nested config dict loaded from a YAML file.
        cli_args: Parsed CLI arguments (``argparse.Namespace``).
        dataset_name: Dataset identifier used to build the default data root
            (e.g. ``"hm3d"``, ``"mp3d"``).

    Returns:
        Merged ``argparse.Namespace`` with all pipeline and data parameters.
    """
    args = _merge_common_nav_args(cfg, cli_args)
    default_root = f"data/{dataset_name}"
    args.data_root = cli_args.data_root or cfg_get(cfg, "data", "root", default=default_root)
    return args


# ---------------------------------------------------------------------------
# Dataset-specific merge functions
# ---------------------------------------------------------------------------

OVON_SPLIT_FILENAMES = {
    "val_seen": "val_seen.json.gz",
    "val_seen_synonyms": "val_unseen_easy.json.gz",
    "val_unseen": "val_unseen_hard.json.gz",
}


def merge_hm3d_config_and_args(cfg, cli_args):
    return merge_nav_config_and_args(cfg, cli_args, dataset_name="hm3d")


def merge_mp3d_config_and_args(cfg, cli_args):
    return merge_nav_config_and_args(cfg, cli_args, dataset_name="mp3d")


def merge_goatcore_config_and_args(cfg, cli_args):
    args = argparse.Namespace()
    args.goatcore_root = cli_args.goatcore_root or cfg_get(cfg, "data", "root")
    args.scene = cli_args.scene
    args.task_type = getattr(cli_args, "task_type", "all")
    args.threshold = cfg_get(cfg, "eval", "threshold")
    _merge_pipeline_args(args, cfg, cli_args)
    args.device = cli_args.device or "cuda"
    args.output = cli_args.output
    return args


def merge_ovon_config_and_args(cfg, cli_args):
    args = merge_hm3d_config_and_args(cfg, cli_args)
    args.split = (
        cli_args.split
        if cli_args.split is not None
        else cfg_get(cfg, "data", "split", default=None)
    )
    args.ovon_root = (
        cli_args.ovon_root
        if getattr(cli_args, "ovon_root", None) is not None
        else cfg_get(cfg, "data", "ovon_root")
    )
    return args
