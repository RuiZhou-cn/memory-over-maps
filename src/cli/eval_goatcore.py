"""Goat-Core benchmark evaluation: retrieval + SAM3 segmentation + 3D projection.

Paper: Sec IV-B, Table I (GOAT-Core 3D localization results).

End-to-end pipeline:
1. Load scene data (GoatCoreSceneDatasetLoader)
2. Build/load feature index (HybridRetriever)
3. For each query: hybrid retrieval -> SAM3 segmentation -> depth backprojection
4. Evaluate Success Rate @ threshold distance
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.dataloaders import LazyImageList
from src.dataloaders.goatcore import GoatCoreGroundTruthLoader, GoatCoreSceneDatasetLoader
from src.evaluation import cleanup
from src.pipelines import localize, search_scene
from src.pipelines.retrieval import build_retriever


def parse_args():
    from src.cli.arg_groups import (
        add_goatcore_data_args,
        add_hardware_args,
        add_pipeline_args,
    )

    parser = argparse.ArgumentParser(
        description="Goat-Core benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_goatcore_data_args(parser)
    add_hardware_args(parser)
    add_pipeline_args(parser)

    parser.add_argument("--task-type", type=str, default="all",
                        choices=["all", "language", "object", "image"],
                        help="Filter by query mode (default: all)")

    return parser.parse_args()


def main():
    import logging
    logging.getLogger("src").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    from functools import partialmethod

    from tqdm import tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    cli_args = parse_args()
    from src.evaluation import load_and_merge_config
    from src.utils.config import load_config, merge_goatcore_config_and_args
    args, config_path = load_and_merge_config(cli_args, load_config, merge_goatcore_config_and_args)

    from src.evaluation import validate_device

    validate_device(args)

    print("=" * 60)
    print("Goat-Core 3D Localization Evaluation")
    print("=" * 60)
    print(f"  Success threshold: {args.threshold}m")
    print(f"  Task type: {args.task_type}")

    print("\n[1/5] Loading ground truth...")
    gt_dataset = GoatCoreGroundTruthLoader(args.goatcore_root)
    print(f"  Samples: {len(gt_dataset)}")
    all_scenes = gt_dataset.get_scenes()
    print(f"  Scenes: {len(all_scenes)} ({', '.join(all_scenes[:5])}{'...' if len(all_scenes) > 5 else ''})")

    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = gt_dataset.get_scenes()

    print("\n[2/5] Loading models...")
    from src.evaluation import load_sam3

    sam3_segmenter = load_sam3(args)

    print("\n[3/5] Running evaluation...")
    from src.evaluation.metrics import LocMetricsAccumulator
    loc_acc = LocMetricsAccumulator(k_values=[args.top_k], threshold=args.threshold)

    from src.evaluation import create_run_dir
    scene_tag = args.scene if args.scene else "all"
    run_dir = create_run_dir("goatcore_loc", scene_tag)

    retriever = None
    prev_scene_name = None

    query_times = []
    query_times_by_type = defaultdict(list)
    scene_times = {}
    eval_start = time.time()

    for scene_idx, scene_id in enumerate(scene_ids, 1):
        scene_path = Path(args.goatcore_root) / "dataset" / scene_id

        if not scene_path.exists():
            print(f"\n  WARNING: Scene {scene_id} not found at {scene_path}, skipping")
            continue

        scene_start = time.time()
        scene_samples = gt_dataset.filter_by_scene(scene_id)
        if args.task_type != "all":
            scene_samples = [s for s in scene_samples if s["task_type"] == args.task_type]

        if not scene_samples:
            print(f"\n  --- [{scene_idx}/{len(scene_ids)}] Scene: {scene_id} (0 queries) ---")
            print(f"  No matching samples for scene {scene_id}")
            continue

        task_counts = {}
        for s in scene_samples:
            task_counts[s["task_type"]] = task_counts.get(s["task_type"], 0) + 1
        task_summary = ", ".join(f"{k}={v}" for k, v in sorted(task_counts.items()))
        print(f"\n  --- [{scene_idx}/{len(scene_ids)}] Scene: {scene_id} ({len(scene_samples)} queries) ---")
        print(f"  Query types: {task_summary}")

        scene_loader = GoatCoreSceneDatasetLoader(scene_path, max_image_size=640)
        frame_ids = scene_loader.frame_ids
        all_poses, all_fids = scene_loader.get_all_poses()
        all_poses_cache = (all_poses, np.asarray(all_fids))

        images = LazyImageList(scene_loader, frame_ids)

        if prev_scene_name != scene_id:
            retriever = build_retriever(
                images, scene_id,
                retrieval_model=args.retrieval_model,
                vlm_model=args.vlm_model,
                device=args.device,
                stage1_top_k=args.stage1_top_k,
                stage2_top_k=args.stage2_top_k,
                vlm_batch_size=args.vlm_batch_size,
                use_vlm=args.vlm,
                existing_retriever=retriever,
                cache_prefix="goatcore",
                extractor_kwargs=args.extractor_kwargs,
            )
            prev_scene_name = scene_id

        for i, sample in enumerate(scene_samples):
            text_query = sample["text_query"]
            task_type = sample["task_type"]
            key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"

            q_image = None
            if task_type == "image":
                q_image = cv2.imread(sample["query"])
                if q_image is not None:
                    q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)

            q_start = time.time()
            display_query = sample["query"] if task_type == "image" else text_query
            print(f"  [{i+1}/{len(scene_samples)}] [{task_type}] {key}: \"{display_query[:60]}\"", end="")

            search_res = search_scene(
                query=text_query,
                images=images,
                frame_ids=frame_ids,
                retriever=retriever,
                top_k=args.top_k,
                min_retrieval_score=args.min_retrieval_score,
                query_image=q_image,
            )

            preds, _, pred_metadata = localize(
                query=text_query,
                images=images,
                frame_ids=frame_ids,
                search_result=search_res,
                scene_loader=scene_loader,
                sam3_segmenter=sam3_segmenter,
                top_k=args.top_k,
                use_centroid=True,
                max_mask_depth=args.max_mask_depth,
            )

            if args.spatial_fusion and sam3_segmenter is not None and preds:
                from src.utils.multi_view_fusion import GoalCandidate, fuse_candidates

                candidates = [
                    GoalCandidate(
                        centroid=np.array(p, dtype=np.float32),
                        point_cloud=m.get("point_cloud"),
                        source_ranks=[i],
                        confidence=m.get("sam3_score", 0.0),
                        instance_id=i,
                    )
                    for i, (p, m) in enumerate(zip(preds, pred_metadata))
                    if m.get("sam3_score", 0.0) > 0
                ]
                unfused = [
                    (p, m) for p, m in zip(preds, pred_metadata)
                    if m.get("sam3_score", 0.0) <= 0
                ]

                if candidates:
                    fuse_candidates(
                        candidates,
                        query=text_query,
                        scene_loader=scene_loader,
                        sam3_segmenter=sam3_segmenter,
                        pred_metadata=pred_metadata,
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
                        use_centroid=True,
                    )

                preds = (
                    [np.array(c.centroid) for c in candidates]
                    + [p for p, _ in unfused]
                )

            q_time = time.time() - q_start

            gt_goals = sample["goals"]
            loc_acc.update(preds, gt_goals, scene=sample["scene"], category=task_type)
            min_dist = float("inf")
            for p in preds:
                for g in gt_goals:
                    d_xz = math.sqrt((p[0] - g[0]) ** 2 + (p[2] - g[2]) ** 2)
                    if d_xz < min_dist:
                        min_dist = d_xz
            success = min_dist <= args.threshold

            query_times.append(q_time)
            query_times_by_type[task_type].append(q_time)

            status = "SUCCESS" if success else "MISS"
            dist_str = f"{min_dist:.2f}m" if min_dist < float("inf") else "no preds"
            print(f" -> {status}, dist={dist_str}, {q_time:.1f}s")

            if sam3_segmenter is not None:
                sam3_segmenter.clear_cache()

        cleanup(sam3_segmenter=sam3_segmenter)

        scene_time = time.time() - scene_start
        scene_times[scene_id] = scene_time
        print(f"  Scene {scene_id} done in {scene_time:.1f}s")

    had_sam3 = sam3_segmenter is not None
    del retriever, sam3_segmenter
    cleanup()

    total_eval_time = time.time() - eval_start

    print("\n[4/5] Results...")
    loc_acc.print_scene_category_table(
        title="Goat-Core Evaluation Results",
        category_order=["object", "image", "language"],
    )

    if query_times:
        avg_query = sum(query_times) / len(query_times)
        print(f"\n  Timing: {len(query_times)} queries, "
              f"avg {avg_query:.2f}s/query, total {total_eval_time:.1f}s")
        for tt in ["object", "image", "language"]:
            times = query_times_by_type.get(tt, [])
            if times:
                print(f"    {tt:>8}: {len(times)} queries, avg {sum(times)/len(times):.2f}s/query")

    print("\n[5/5] Saving results...")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    timing_data = {
        "total_time": round(total_eval_time, 2),
        "num_queries": len(query_times),
        "avg_query_time": round(sum(query_times) / len(query_times), 2) if query_times else 0.0,
        "avg_by_type": {
            tt: round(sum(times) / len(times), 2)
            for tt, times in query_times_by_type.items()
        },
        "per_scene": {s: round(t, 2) for s, t in scene_times.items()},
    }

    output_data = {
        "config": {
            "scene": args.scene,
            "task_type": args.task_type,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "retrieval_model": args.retrieval_model,
            "vlm_model": args.vlm_model,
            "sam3": had_sam3,
            "spatial_fusion": args.spatial_fusion and had_sam3,
            "fusion_threshold": args.fusion_threshold,
        },
        "results": loc_acc.to_json(),
        "timing": timing_data,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
