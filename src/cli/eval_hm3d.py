"""HM3D ObjectNav evaluation using habitat.Env with built-in metrics.

Paper: Sec IV-C, Table II (HM3D ObjectNav results).

Uses habitat-lab's ObjectNav task with official measurements (Success, SPL,
SoftSPL, DTG). Pipeline per episode: retrieval -> SAM3 + depth -> 3D goal ->
DD-PPO PointNav navigation. Metrics computed by habitat.Env.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

import argparse
import time
from pathlib import Path

import numpy as np

from src.utils.config import get_scene_name, load_config, merge_hm3d_config_and_args
from src.models.navigation.pointnav_policy import PointNavController
from src.pipelines.navigation import (
    run_nav_episodes,
    save_results_json,
)
from src.pipelines.retrieval import build_retriever


def parse_args():
    from src.cli.arg_groups import (
        add_habitat_data_args,
        add_hardware_args,
        add_pipeline_args,
    )

    parser = argparse.ArgumentParser(
        description="HM3D ObjectNav evaluation (habitat.Env built-in metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pipeline params set in YAML config; use --config to override.",
    )

    add_habitat_data_args(parser, "hm3d", include_episode=True)
    add_hardware_args(parser)
    add_pipeline_args(parser)

    return parser.parse_args()


def main():
    cli_args = parse_args()

    from src.evaluation import load_and_merge_config
    args, config_path = load_and_merge_config(cli_args, load_config, merge_hm3d_config_and_args)

    from src.evaluation import validate_device

    validate_device(args)

    print("=" * 60)
    print("HM3D ObjectNav Evaluation")
    print("=" * 60)
    print(f"  Dataset: HM3D ObjectNav v2 (6 categories, split={args.split})")

    print(f"\n[1/5] Creating habitat.Env (split={args.split})...")

    from src.evaluation import (
        cleanup,
        filter_episodes,
        group_by_scene,
        import_habitat,
        suppress_stderr,
    )

    get_config, Env = import_habitat()

    data_root = Path(args.data_root)
    episodes_dir = data_root / "episodes"
    split = args.split
    episode_path = str(episodes_dir / split / f"{split}.json.gz")

    cfg = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml", overrides=[
        f"habitat.dataset.split={split}",
        f"habitat.dataset.data_path={episode_path}",
        f"habitat.dataset.scenes_dir={data_root / 'meshes'}",
        f"habitat.environment.max_episode_steps={args.nav_max_steps}",
        f"habitat.task.measurements.success.success_distance={args.success_distance}",
        f"habitat.task.measurements.distance_to_goal.distance_to={args.distance_to}",
    ])

    env = suppress_stderr(Env, cfg)

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
        print("ERROR: No episodes to evaluate")
        env.close()
        return

    episodes_by_scene = group_by_scene(all_episodes, get_scene_name)

    scene_names = sorted(episodes_by_scene.keys())
    print(f"  Scenes: {len(scene_names)} ({', '.join(scene_names[:5])}{'...' if len(scene_names) > 5 else ''})")

    from src.evaluation import print_episode_summary
    print_episode_summary(all_episodes)

    print("\n[2/5] Loading models...")

    pointnav = PointNavController(
        args.pointnav_weights, device=args.device,
        policy_input_size=args.policy_input_size,
    )
    print("  PointNav policy loaded")

    from src.evaluation import load_sam3

    sam3_segmenter = load_sam3(args)

    from src.evaluation import create_run_dir, get_scene_tag
    scene_tag = get_scene_tag(args)
    run_dir = create_run_dir("hm3d_objectnav", scene_tag)

    print("\n[3/5] Running evaluation...")

    from src.evaluation.metrics import NavMetricsAccumulator
    nav_acc = NavMetricsAccumulator()
    all_results = {}

    retriever = None

    for scene_idx, scene_name in enumerate(scene_names, 1):
        scene_eps = episodes_by_scene[scene_name]
        scene_start = time.time()
        print(f"\n  --- [{scene_idx}/{len(scene_names)}] Scene: {scene_name} ({len(scene_eps)} episodes) ---")

        env.episode_iterator = iter(scene_eps)

        scenes_dir = data_root / "scenes"
        scene_path = scenes_dir / scene_name
        scene_loader = None
        images = None
        all_frame_ids = None

        if not scene_path.exists():
            print(f"  WARNING: Scene images not found at {scene_path}, skipping scene")
        else:
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
            )

        all_poses_raw = None
        if scene_loader is not None:
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

        save_results_json(args, run_dir, config_path, nav_acc, all_results,
                           partial=True)

    del retriever, sam3_segmenter, pointnav
    cleanup()

    print("\n[4/5] Results...")
    nav_acc.print_table(success_distance=args.success_distance,
                        distance_to=args.distance_to)

    print("\n[5/5] Saving results...")

    output_path = save_results_json(
        args, run_dir, config_path, nav_acc, all_results,
        partial=False,
    )
    print(f"\nResults saved to: {output_path}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
