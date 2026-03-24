"""SUN RGB-D text-to-image retrieval evaluation.

Paper: Sec IV-D, Table III (SUN RGB-D retrieval results).

Evaluates vision-language retrieval models on the SUN RGB-D dataset using
AR@K metric. Builds a **separate FAISS index per sensor type** (kv1, kv2,
xtion, realsense) and retrieves within each sensor group independently.

Default evaluates our full method (siglip2+vlm). Use --model all to add
feature-only baselines (clip-base, clip-large, align, flava, siglip2).

GT matching is **exact string match** on lowercased annotation labels — no synonym
merging. "table" and "desk" are separate categories; a scene annotated with "desk"
will NOT count as GT for a "table" query.

Results are reported per-sensor (kv1, kv2, xtion, realsense),
each with its own retrieval index.

Pipeline:
1. Discover scenes (walk SUNRGBD/ tree, find dirs with image/ + annotation)
2. Load annotations -> build category_to_scene_indices mapping
3. Load all images once (shared across models)
4. For each model, evaluate per-sensor:
   Build FAISS index per sensor group, search per category
5. Report per-instance AR@K per sensor
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.dataloaders.sunrgbd import (
    SENSOR_TYPES,
    build_category_index,
    build_sensor_groups,
    discover_scenes,
    load_scene_image,
)
from src.models.retrieval import (
    MODEL_REGISTRY,
    HybridRetriever,
    create_feature_extractor,
)

OURS_KEY = "ours"  # virtual model key: siglip2 features + VLM re-ranking


def _ours_config(model_key: str) -> dict:
    """Resolve model config, mapping 'ours' to the siglip2 registry entry."""
    if model_key == OURS_KEY:
        return MODEL_REGISTRY["siglip2"]
    return MODEL_REGISTRY[model_key]


def parse_args():
    from src.cli.arg_groups import add_hardware_args, add_pipeline_args

    parser = argparse.ArgumentParser(
        description="SUN RGB-D text-to-image retrieval evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sunrgbd-root", type=str, default="data/SUNRGBD",
                        help="Path to SUN RGB-D dataset root")
    parser.add_argument("--model", type=str, nargs="+", default=["ours"],
                        help="Model key(s): 'ours' (siglip2+vlm, default), "
                             "feature-only baselines (clip-base, clip-large, align, flava, siglip2), "
                             "or 'all'")
    parser.add_argument("--top-k", type=str, default="1,5",
                        help="Comma-separated K values for AR@K")
    parser.add_argument("--sensor", type=str, nargs="+", default=["all"],
                        choices=["all"] + SENSOR_TYPES,
                        help="Sensor type(s) to evaluate, or 'all' (default: all)")

    add_hardware_args(parser)
    add_pipeline_args(parser)

    return parser.parse_args()


def build_features_cache_dir(model_key: str, max_size: int) -> Path:
    """Build canonical cache directory path for global features."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "results" / "features" / f"{model_key}_sunrgbd_global_{max_size}"


def save_features(
    cache_dir: Path,
    features: np.ndarray,
    scene_paths: list[str],
    model_key: str,
    model_name: str,
    max_size: int,
):
    """Save extracted features and scene list to cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", features)
    np.save(cache_dir / "frame_ids.npy", np.arange(len(features)))

    metadata = {
        "model_key": model_key,
        "feature_extractor_model": model_name,
        "dataset": "sunrgbd",
        "max_size": max_size,
        "num_frames": len(features),
        "feature_dim": features.shape[1],
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(cache_dir / "scene_paths.json", "w") as f:
        json.dump(scene_paths, f, indent=2)


def validate_cache(cache_dir: Path, scene_paths: list[str]) -> bool:
    """Check if cached features match the current scene list."""
    scene_paths_file = cache_dir / "scene_paths.json"
    if not scene_paths_file.exists():
        return False
    try:
        with open(scene_paths_file, "r") as f:
            cached_paths = json.load(f)
        return cached_paths == scene_paths
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def _eval_embedding_group(
    retriever: HybridRetriever,
    local_to_global: list[int],
    category_to_scenes: dict[str, set[int]],
    k_values: list[int],
    group_images: list[np.ndarray] | None = None,
) -> dict:
    """Evaluate one retrieval group (global or per-sensor) with a pre-built FAISS index.

    Args:
        retriever: HybridRetriever with FAISS index already built for this group.
        local_to_global: Maps retriever-local index -> global scene index.
        category_to_scenes: GT mapping (global scene indices).
        k_values: K values for AR@K.
        group_images: Images for this group. VLM re-ranking is skipped when None.

    Returns dict with per_instance_ar, query_results, num_instances.
    """
    max_k = max(k_values)
    global_set = set(local_to_global)
    use_vlm = retriever.has_vlm and group_images is not None

    valid_categories = []
    for category in sorted(category_to_scenes.keys()):
        gt = category_to_scenes[category] & global_set
        if gt:
            valid_categories.append(category)
    num_queries = len(valid_categories)

    instance_hits = {k: 0 for k in k_values}
    instance_total = 0
    query_results = []

    for qi, category in enumerate(valid_categories):
        gt_global = category_to_scenes[category] & global_set

        if use_vlm:
            results = retriever.search(category, group_images, top_k=max_k)
            retrieved_local = [r["image_index"] for r in results]
            scores = [r["confidence"] for r in results]
        else:
            retrieved_local, scores = retriever.search_features(
                category, top_k=max_k, return_scores=True,
                similarity_threshold=1.0,
            )

        retrieved_global = [local_to_global[i] for i in retrieved_local]

        hit_at_first_k = False
        for k in k_values:
            hit = any(idx in gt_global for idx in retrieved_global[:k])
            if hit:
                instance_hits[k] += len(gt_global)
            if k == k_values[0]:
                hit_at_first_k = hit
        instance_total += len(gt_global)

        status = "hit" if hit_at_first_k else "miss"
        print(f"    [{qi+1}/{num_queries}] {category}: {status}", flush=True)

        query_results.append({
            "category": category,
            "num_gt_scenes": len(gt_global),
            "retrieved_indices": retrieved_global[:max_k],
            "scores": [float(s) for s in scores[:max_k]],
        })

    per_instance_ar = {
        k: instance_hits[k] / instance_total * 100 if instance_total > 0 else 0.0
        for k in k_values
    }

    return {
        "per_instance_ar": per_instance_ar,
        "query_results": query_results,
        "num_instances": instance_total,
    }


def evaluate_embedding_model(
    model_key: str,
    config: dict,
    images: list[np.ndarray],
    scene_paths: list[str],
    category_to_scenes: dict[str, set[int]],
    sensor_groups: dict[str, list[int]],
    k_values: list[int],
    args,
) -> dict:
    """Evaluate an embedding-based model per-sensor."""
    import torch
    model_type = config["model_type"]
    model_name = config["model_name"]

    print(f"\n  Loading {model_key}...")
    extractor = create_feature_extractor(
        model_type=model_type,
        model_name=model_name,
        device=args.device,
        normalize=True,
        batch_size=args.batch_size,
    )

    # Create VLM for stage-2 re-ranking (only for "ours" = full pipeline, unless --no-vlm)
    vlm = None
    if model_key == OURS_KEY and not getattr(args, "no_vlm", False):
        from src.models.vlm import Qwen2_5VL
        print(f"  Loading VLM: {args.vlm_model}...")
        vlm = Qwen2_5VL(model_name=args.vlm_model, device=args.device)
    else:
        print("  Stage 1 only (no VLM re-ranking)")

    # "ours" shares siglip2 features — cache under "siglip2" to avoid duplication
    cache_key = "siglip2" if model_key == OURS_KEY else model_key
    cache_dir = build_features_cache_dir(cache_key, args.max_size)
    use_cache = (
        not args.no_cache
        and cache_dir.exists()
        and (cache_dir / "features.npy").exists()
        and validate_cache(cache_dir, scene_paths)
    )

    if use_cache:
        print(f"  Loading cached features from {cache_dir}")
        all_features = np.load(cache_dir / "features.npy")
        print(f"  Loaded {len(all_features)} image features")
    else:
        print(f"  Extracting features for {len(images)} images...")
        t0 = time.time()
        all_features = extractor.extract_image_features(images)
        print(f"  Feature extraction: {time.time() - t0:.1f}s")

        if not args.no_cache:
            save_features(
                cache_dir, all_features,
                scene_paths, cache_key, model_name, args.max_size,
            )
            print(f"  Cached features to {cache_dir}")

    vlm_tag = " + VLM" if vlm else ""

    eval_sensors = args.eval_sensors
    per_sensor = {}
    total_eval_time = 0.0
    for sensor in eval_sensors:
        indices = sensor_groups.get(sensor, [])
        if len(indices) < 2:
            continue

        print(f"\n  Evaluating {sensor} ({len(indices)} images{vlm_tag})...")
        t0 = time.time()

        sensor_features = all_features[indices]
        sensor_retriever = HybridRetriever(
            feature_extractor=extractor,
            vlm=vlm,
            stage1_top_k=args.stage1_top_k,
            stage2_top_k=args.stage2_top_k,
            batch_size=args.vlm_batch_size,
        )
        sensor_retriever.build_index_from_features(sensor_features)

        sensor_images = [images[i] for i in indices]
        sensor_result = _eval_embedding_group(
            sensor_retriever, indices,
            category_to_scenes, k_values,
            group_images=sensor_images,
        )
        sensor_result["eval_time"] = time.time() - t0
        sensor_result["num_scenes"] = len(indices)
        per_sensor[sensor] = sensor_result
        total_eval_time += sensor_result["eval_time"]

        inst1 = sensor_result["per_instance_ar"][k_values[0]]
        print(f"  {sensor}: AR@{k_values[0]}={inst1:.1f}%")

    del extractor
    if vlm is not None:
        del vlm
    torch.cuda.empty_cache()

    return {"per_sensor": per_sensor, "eval_time": total_eval_time}


def print_single_model_results(
    model_key: str,
    result: dict,
    k_values: list[int],
):
    """Print results for a single model (per-sensor breakdown)."""
    display_ks = k_values[:4]
    per_sensor = result.get("per_sensor", {})

    print(f"\n{'=' * 80}")
    print(f"  {model_key}")
    print("  GT matching: exact label string (no synonym merging)")
    print(f"{'=' * 80}")

    k_headers = "".join(f"  {'AR@' + str(k):<8}" for k in display_ks)
    print(f"\n  {'Sensor':<14} {'Images':>6} {'Insts':>6}  {k_headers}")
    print(f"  {'-' * 60}")

    for sensor in SENSOR_TYPES:
        if sensor in per_sensor:
            _print_group_row(sensor, per_sensor[sensor], display_ks)

    print(f"{'=' * 80}")


def _print_group_row(label: str, result: dict, k_values: list[int]):
    """Print one row of the results table."""
    ar = result["per_instance_ar"]
    n_scenes = result.get("num_scenes", "?")
    n_inst = result.get("num_instances", "?")
    vals = "".join(f"  {ar[k]:>6.1f}% " for k in k_values)
    print(f"  {label:<14} {n_scenes:>6} {n_inst:>6}  {vals}")


def print_comparison_table(
    all_results: dict[str, dict],
    k_values: list[int],
    num_scenes: int,
):
    """Print comparison table across all evaluated models (per-sensor)."""
    if len(all_results) <= 1:
        return

    display_ks = k_values[:3]

    print(f"\n{'=' * 90}")
    print(f"  SUN RGB-D Retrieval Baselines Comparison ({num_scenes} images)")
    print(f"{'=' * 90}")

    k_headers = "".join(f" | {'AR@' + str(k):<8}" for k in display_ks)

    for sensor in SENSOR_TYPES:
        if not any(sensor in r.get("per_sensor", {}) for r in all_results.values()):
            continue

        print(f"\n  AR@K ({sensor})")
        print(f"  {'Model':<22} | {'Mode':<9}{k_headers} | {'Time':<6}")
        print(f"  {'-' * 75}")

        for model_key, result in all_results.items():
            sensor_result = result.get("per_sensor", {}).get(sensor)
            if not sensor_result:
                continue
            config = _ours_config(model_key)
            mode = "siglip2+vlm" if model_key == OURS_KEY else config["model_type"]
            per_inst = sensor_result["per_instance_ar"]
            t = sensor_result["eval_time"]
            time_str = f"{t / 60:.0f}m" if t >= 60 else f"{t:.0f}s"

            k_vals = "".join(f" | {per_inst[k]:6.2f}%  " for k in display_ks)
            print(f"  {model_key:<22} | {mode:<9}{k_vals} | {time_str:<6}")

    print(f"{'=' * 90}")


def main():
    args = parse_args()

    args.max_size = 640
    args.batch_size = 32
    args.no_cache = False
    args.stage1_top_k = 10
    args.stage2_top_k = 5
    args.vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    args.vlm_batch_size = 1

    k_values = [int(k.strip()) for k in args.top_k.split(",")]

    if "all" in args.sensor:
        args.eval_sensors = SENSOR_TYPES
    else:
        args.eval_sensors = args.sensor

    from src.evaluation import validate_device

    validate_device(args)

    resolved = []
    for m in args.model:
        if m == "all":
            resolved = [OURS_KEY] + list(MODEL_REGISTRY.keys())
            break
        if m == OURS_KEY:
            resolved.append(OURS_KEY)
        elif m in MODEL_REGISTRY:
            resolved.append(m)
        else:
            available = ", ".join([OURS_KEY] + list(MODEL_REGISTRY.keys()))
            raise ValueError(f"Unknown model key '{m}'. Available: {available}")
    model_keys = resolved

    print("=" * 60)
    print("SUN RGB-D Retrieval Evaluation")
    print("=" * 60)
    print(f"  Models: {', '.join(model_keys)}")
    has_ours = OURS_KEY in model_keys
    print(f"  VLM re-ranking: {'ON for ours' if has_ours else 'OFF'} ({args.vlm_model})")
    print(f"  K values: {k_values}")
    print(f"  Sensors: {', '.join(args.eval_sensors)}")
    print("  GT matching: exact label string (no synonym merging)")

    print(f"\n[1/4] Discovering scenes in {args.sunrgbd_root}...")
    scenes = discover_scenes(args.sunrgbd_root)
    print(f"  Found {len(scenes)} valid scenes")
    if not scenes:
        print("ERROR: No valid scenes found. Check --sunrgbd-root path.")
        sys.exit(1)

    sensor_groups = build_sensor_groups(scenes, args.sunrgbd_root)
    for sensor in SENSOR_TYPES:
        if sensor in sensor_groups:
            print(f"    {sensor}: {len(sensor_groups[sensor])} scenes")

    print("\n[2/4] Loading annotations and building category index...")
    category_to_scenes, _ = build_category_index(scenes)

    num_instances = sum(len(s) for s in category_to_scenes.values())
    print(f"  {len(category_to_scenes)} categories, {num_instances} total instances")

    if len(category_to_scenes) == 0:
        print("ERROR: No categories found after filtering.")
        sys.exit(1)

    print(f"\n[3/4] Loading {len(scenes)} images...")
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor
    max_size = args.max_size
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda sd: load_scene_image(sd, max_size=max_size), scenes
        ))

    images = []
    valid_indices = []
    for idx, img in enumerate(results):
        if img is not None:
            images.append(img)
            valid_indices.append(idx)

    load_time = time.time() - t0
    print(f"  Loaded {len(images)} images in {load_time:.1f}s")

    if len(valid_indices) != len(scenes):
        scenes = [scenes[i] for i in valid_indices]
        category_to_scenes, _ = build_category_index(scenes)
        sensor_groups = build_sensor_groups(scenes, args.sunrgbd_root)
        num_instances = sum(len(s) for s in category_to_scenes.values())
        print(f"  Rebuilt category index: {len(category_to_scenes)} categories, "
              f"{num_instances} instances")

    scene_paths = [str(s) for s in scenes]

    print(f"\n[4/4] Running evaluation (K={k_values})...")
    all_results = {}

    for model_idx, model_key in enumerate(model_keys):
        config = _ours_config(model_key)
        if model_key == OURS_KEY:
            display_name = "ours (siglip2 + vlm)"
        else:
            display_name = f"{model_key} (feature-only)"
        print(f"\n{'=' * 60}")
        print(f"  Model {model_idx + 1}/{len(model_keys)}: {display_name}")
        print(f"  Mode: {config['model_type']} | Model: {config['model_name']}")
        print(f"{'=' * 60}")

        result = evaluate_embedding_model(
            model_key, config, images, scene_paths,
            category_to_scenes, sensor_groups, k_values, args,
        )

        all_results[model_key] = result

        print_single_model_results(model_key, result, k_values)

    print_comparison_table(all_results, k_values, len(scenes))

    from src.evaluation import create_run_dir

    run_dir = create_run_dir("sunrgbd_eval", "all")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_group(r: dict) -> dict:
        return {
            "per_instance": {f"AR@{k}": r["per_instance_ar"][k] for k in k_values},
            "num_scenes": r.get("num_scenes"),
            "num_instances": r.get("num_instances"),
        }

    output_data = {
        "config": {
            "models": model_keys,
            "vlm": True,
            "vlm_model": args.vlm_model,
            "top_k": k_values,
            "num_scenes": len(scenes),
            "num_instances": num_instances,
        },
        "comparison": {
            model_key: {
                "per_sensor": {
                    s: _serialize_group(sr)
                    for s, sr in r.get("per_sensor", {}).items()
                },
            }
            for model_key, r in all_results.items()
        },
        "models": {
            model_key: {
                "config": _ours_config(model_key),
                "results": {
                    "per_sensor": {
                        s: _serialize_group(sr)
                        for s, sr in r.get("per_sensor", {}).items()
                    },
                },
            }
            for model_key, r in all_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
