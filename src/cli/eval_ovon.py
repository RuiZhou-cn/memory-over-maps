"""HM3D-OVON ObjectNav evaluation using habitat.Env with built-in metrics.

Paper: Sec IV-C, Table II(b) (HM3D-OVON ObjectNav results).

Same pipeline and navigation as eval_hm3d.py, but uses OVON-v1 episodes
(379 open-vocabulary categories, 3 val splits: val_seen, val_seen_synonyms,
val_unseen). Same 36 scenes and meshes as standard HM3D ObjectNav.

Requires setup: python scripts/prepare_data.py --dataset ovon
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

from src.evaluation.helpers import suppress_stderr

suppress_stderr(lambda: __import__("src.dataloaders.ovon"))  # registers OVON-v1 dataset

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.models.navigation.pointnav_policy import PointNavController
from src.pipelines.navigation import (
    record_skip,
    run_nav_episodes,
)
from src.pipelines.retrieval import build_retriever
from src.utils.config import (
    OVON_SPLIT_FILENAMES,
    get_scene_name,
    load_config,
    merge_ovon_config_and_args,
)

ALL_SPLITS = list(OVON_SPLIT_FILENAMES.keys())


def parse_args():
    from src.cli.arg_groups import (
        add_habitat_data_args,
        add_hardware_args,
        add_pipeline_args,
    )

    parser = argparse.ArgumentParser(
        description="HM3D-OVON ObjectNav evaluation (379 open-vocabulary categories)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pipeline params come from YAML config. CLI provides mode toggles only.",
    )

    add_habitat_data_args(parser, "hm3d_ovon", include_episode=True)
    add_hardware_args(parser)
    add_pipeline_args(parser)

    return parser.parse_args()


def _create_ovon_env(args, split, get_config, Env, suppress_stderr):
    """Create a habitat.Env for a specific OVON split."""
    data_root = Path(args.data_root)
    ovon_root = Path(args.ovon_root)

    episode_filename = OVON_SPLIT_FILENAMES.get(split, f"{split}.json.gz")
    episode_path = str(ovon_root / "hm3d" / split / episode_filename)

    if not Path(episode_path).exists():
        print(f"ERROR: Episode file not found: {episode_path}")
        print(f"  Expected: {ovon_root}/hm3d/{split}/{episode_filename}")
        print("  Run: python scripts/prepare_data.py --dataset ovon")
        return None

    scenes_dir = str(data_root / "meshes")

    cfg = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml", overrides=[
        "habitat.dataset.type=OVON-v1",
        f"habitat.dataset.split={split}",
        f"habitat.dataset.data_path={episode_path}",
        f"habitat.dataset.scenes_dir={scenes_dir}",
        f"habitat.environment.max_episode_steps={args.nav_max_steps}",
        f"habitat.task.measurements.success.success_distance={args.success_distance}",
        f"habitat.task.measurements.distance_to_goal.distance_to={args.distance_to}",
    ])

    env = suppress_stderr(Env, cfg)

    return env


def _evaluate_split(
    args, split, env, pointnav, sam3_segmenter, run_dir,
):
    """Evaluate all episodes for a single OVON split. Returns (nav_acc, all_results, categories)."""
    from src.evaluation import (
        cleanup,
        filter_episodes,
        group_by_scene,
        quiet_reset,
    )
    from src.evaluation.metrics import NavMetricsAccumulator

    nav_acc = NavMetricsAccumulator()
    all_results = {}

    all_episodes = list(env.episodes)
    print(f"  Loaded {len(all_episodes)} episodes")

    all_episodes, skipped_cross_floor = filter_episodes(
        all_episodes,
        scene=args.scene,
        episode_ids=args.episode,
        floor_filter=True,
    )
    if args.scene:
        print(f"  Filtered to {len(all_episodes)} episodes for scene '{args.scene}'")
    if args.episode is not None:
        print(f"  Filtered to {len(all_episodes)} episodes for IDs {sorted(args.episode)}")
    if skipped_cross_floor > 0:
        print(f"  Floor filter: skipped {skipped_cross_floor} cross-floor episodes")

    if not all_episodes:
        print("  WARNING: No episodes to evaluate for this split")
        return nav_acc, all_results, set()

    episodes_by_scene = group_by_scene(all_episodes, get_scene_name)
    scene_names = sorted(episodes_by_scene.keys())
    print(f"  Scenes: {len(scene_names)} ({', '.join(scene_names[:5])}{'...' if len(scene_names) > 5 else ''})")

    categories = set()
    for ep in all_episodes:
        categories.add(ep.object_category)
    print(f"  Categories: {len(categories)} unique"
          f" (e.g., {', '.join(sorted(categories)[:5])}...)")

    data_root = Path(args.data_root)
    retriever = None

    for scene_idx, scene_name in enumerate(scene_names, 1):
        scene_eps = episodes_by_scene[scene_name]
        scene_start = time.time()
        print(f"\n  --- [{scene_idx}/{len(scene_names)}] Scene: {scene_name} ({len(scene_eps)} episodes) ---")

        env.episode_iterator = iter(scene_eps)

        scenes_dir_path = data_root / "scenes"
        scene_path = scenes_dir_path / scene_name
        scene_loader = None
        images = None
        all_frame_ids = None

        if not scene_path.exists():
            print(f"  WARNING: Scene images not found at {scene_path}, skipping scene")
            for _ in scene_eps:
                quiet_reset(env)
                episode = env.current_episode
                q_start = time.perf_counter()
                record_skip(
                    env, nav_acc, all_results, scene_name,
                    episode.object_category, episode.episode_id,
                    "no_scene_images", q_start=q_start)
            continue

        from src.dataloaders.hm3d import HM3DSceneDatasetLoader

        scene_loader = HM3DSceneDatasetLoader(
            str(scene_path), max_image_size=640, hfov=args.hfov,
        )
        from src.evaluation import apply_keyframing
        all_frame_ids = apply_keyframing(scene_loader, args)
        from src.dataloaders import LazyImageList
        images = LazyImageList(scene_loader, all_frame_ids)
        retriever = build_retriever(
            images, scene_name,
            retrieval_model=args.retrieval_model,
            vlm_model=args.vlm_model,
            device=args.device,
            stage1_top_k=args.stage1_top_k,
            stage2_top_k=args.stage2_top_k,
            vlm_batch_size=args.vlm_batch_size,
            use_vlm=args.vlm,
            existing_retriever=retriever,
            keyframe_ids=all_frame_ids if args.keyframing else None,
            extractor_kwargs=args.extractor_kwargs,
        )

        raw_poses, raw_fids = scene_loader.get_all_poses()
        all_poses_raw = (raw_poses, np.asarray(raw_fids))

        run_nav_episodes(
            env, scene_eps, scene_name, args,
            scene_loader, images, all_frame_ids,
            retriever, sam3_segmenter, pointnav,
            nav_acc, all_results,
            all_poses_raw=all_poses_raw,
        )

        cleanup(sam3_segmenter=sam3_segmenter)
        scene_time = time.time() - scene_start
        print(f"  Scene {scene_name} done in {scene_time:.1f}s")

    del retriever
    cleanup()

    return nav_acc, all_results, categories


def main():
    cli_args = parse_args()

    from src.evaluation import load_and_merge_config
    args, config_path = load_and_merge_config(cli_args, load_config, merge_ovon_config_and_args)

    from src.evaluation import validate_device

    validate_device(args)

    splits = [args.split] if args.split else ALL_SPLITS

    splits_str = ", ".join(splits)
    print("=" * 60)
    print("HM3D-OVON ObjectNav Evaluation")
    print("=" * 60)
    print("  Dataset: HM3D-OVON (379 open-vocabulary categories)")
    print(f"  Splits: {splits_str}")

    print("\n[1/4] Loading models...")

    pointnav = PointNavController(
        args.pointnav_weights, device=args.device,
        policy_input_size=args.policy_input_size,
    )
    print("  PointNav policy loaded")

    from src.evaluation import load_sam3

    sam3_segmenter = load_sam3(args)

    from src.evaluation import create_run_dir, get_scene_tag
    scene_tag = get_scene_tag(args)
    run_dir = create_run_dir("hm3d_ovon", scene_tag)

    from src.evaluation import suppress_stderr

    def _import_habitat():
        from habitat import Env
        from habitat.config import get_config
        return get_config, Env
    get_config, Env = suppress_stderr(_import_habitat)

    per_split_results = {}

    for split_idx, split in enumerate(splits, 1):
        print(f"\n{'=' * 60}")
        print(f"[2/4] Split {split_idx}/{len(splits)}: {split}")
        print(f"{'=' * 60}")

        env = _create_ovon_env(args, split, get_config, Env, suppress_stderr)
        if env is None:
            continue

        nav_acc, all_results, categories = _evaluate_split(
            args, split, env, pointnav, sam3_segmenter, run_dir,
        )

        env.close()

        per_split_results[split] = (nav_acc, all_results, categories)

        print(f"\n  --- Results for {split} ---")
        nav_acc.print_table(success_distance=args.success_distance,
                            distance_to=args.distance_to)

    from src.evaluation import cleanup
    del sam3_segmenter, pointnav
    cleanup()

    if not per_split_results:
        print("ERROR: No splits evaluated successfully")
        return

    if len(per_split_results) > 1:
        from src.evaluation.metrics import NavMetricsAccumulator

        print(f"\n{'=' * 60}")
        print("[3/4] Combined results (all splits)")
        print(f"{'=' * 60}")

        combined_acc = NavMetricsAccumulator()
        for split, (nav_acc, _, _) in per_split_results.items():
            combined_acc.merge(nav_acc)
        combined_acc.print_table(success_distance=args.success_distance,
                                 distance_to=args.distance_to)
    else:
        combined_acc = list(per_split_results.values())[0][0]

    print("\n[4/4] Saving results...")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_episode_results = {}
    all_categories = set()
    per_split_json = {}
    for split, (nav_acc, all_results, categories) in per_split_results.items():
        for k, v in all_results.items():
            v["split"] = split
            merged_episode_results[f"{split}/{k}"] = v
        all_categories |= categories
        per_split_json[split] = nav_acc.to_json()

    output_data = {
        "run_dir": str(run_dir),
        "dataset": "hm3d_ovon",
        "splits": list(per_split_results.keys()),
        "config": {k: getattr(args, k, None) for k in [
            "data_root", "ovon_root", "split", "scene", "agent_height", "agent_radius",
            "sensor_height", "forward_step_size", "turn_angle",
            "policy_input_size", "min_depth", "max_depth",
            "nav_max_steps", "stop_radius", "hfov",
            "retrieval_model", "vlm_model", "spatial_fusion", "hdbscan",
            "spatial_max_views", "spatial_max_distance", "frustum_margin",
            "fusion_threshold", "multi_goal", "stuck_window", "stuck_threshold",
            "oscillation_window", "oscillation_ratio", "oscillation_min_path",
            "opportunistic_radius", "cluster_distance", "max_goal_switches",
            "device", "success_distance", "distance_to",
        ]},
        "results": combined_acc.to_json(),
        "per_split_results": per_split_json,
        "episode_results": merged_episode_results,
    }
    output_data["config"]["config_file"] = str(config_path) if config_path.exists() else None
    output_data["config"]["metrics_source"] = "habitat.Env (OVON-v1)"
    output_data["config"]["sam3"] = True
    output_data["config"]["n_categories"] = len(all_categories)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
