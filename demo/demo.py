#!/usr/bin/env python3
"""Build TSDF mesh from posed RGBD and visualize with viser.

Supports interactive text queries with the full retrieval pipeline:
feature search -> VLM re-ranking -> SAM3 segmentation -> 3D localization.

Usage:
    python demo/demo.py --dataset hm3d --scene 4ok3usBNeis
    python demo/demo.py --dataset mp3d --scene 8194nk5LbLH
    python demo/demo.py --dataset goatcore --scene nfv
    python demo/demo.py --dataset custom --scene /path/to/my_scene --pose-convention opencv
    python demo/demo.py --dataset hm3d --list-scenes

Edit configs/demo.yaml to change TSDF, retrieval, and display parameters.
"""

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataloaders import get_dataloader
from src.evaluation.helpers import cleanup
from src.models.segmentation import SAM3Segmenter
from src.pipelines.localization import localize
from src.pipelines.retrieval import build_retriever, search_scene
from src.utils.config import cfg_get, load_config
from src.utils.image import resize_images_batch
from src.utils.keyframe import KeyframeManager
from src.utils.multi_view_fusion import (
    fuse_candidates,
    group_predictions_by_overlap,
    hdbscan_filter_cloud,
)

logger = logging.getLogger(__name__)


def list_scenes(dataset: str):
    if dataset == "custom":
        print("Custom dataset: pass --scene /path/to/your/scene directly.")
        return
    if dataset == "goatcore":
        scenes_dir = ROOT / "data" / "Goat-core" / "dataset"
    else:
        scenes_dir = ROOT / "data" / dataset / "scenes"
    if not scenes_dir.exists():
        print(f"No scenes directory found: {scenes_dir}")
        return
    scenes = sorted(p.name for p in scenes_dir.iterdir() if p.is_dir())
    print(f"\n{dataset.upper()} scenes ({len(scenes)}):")
    for s in scenes:
        print(f"  {s}")


def habitat_c2w_to_o3d_extrinsic(c2w: np.ndarray) -> np.ndarray:
    """Convert Habitat camera-to-world (Y-up, -Z forward) to Open3D extrinsic.

    Open3D TSDF expects world-to-camera in OpenCV convention (Y-down, Z-forward).
    Habitat convention has Y-up and Z-backward, so we flip columns 1 and 2
    of the C2W rotation to get OpenCV C2W, then invert.
    """
    c2w_cv = c2w.copy().astype(np.float64)
    c2w_cv[:3, 1] *= -1
    c2w_cv[:3, 2] *= -1
    return np.linalg.inv(c2w_cv)


def _load_frame(loader, fid):
    """Load all data for a single frame (runs in thread pool)."""
    depth = loader.load_depth(fid)
    if depth is None:
        return None
    rgb = loader.load_rgb(fid)
    pose = loader.load_pose(fid)
    intr = loader.get_intrinsics_for_frame(fid)
    return depth, rgb, pose, intr


def build_tsdf_mesh(
    loader,
    frame_ids: list,
    voxel_size: float = 0.015,
    sdf_trunc_factor: float = 5.0,
    max_depth: float = 6.0,
    prefetch_workers: int = 8,
):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * sdf_trunc_factor,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n = len(frame_ids)
    print(f"Integrating {n}/{len(loader)} frames")

    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        futures = {pool.submit(_load_frame, loader, fid): (i, fid)
                   for i, fid in enumerate(frame_ids)}

        # Collect results keyed by index to iterate in order
        results = [None] * n
        for fut in futures:
            idx, _ = futures[fut]
            results[idx] = fut.result()

    for i, result in enumerate(results):
        if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
            print(f"  [{i + 1}/{n}] frame {frame_ids[i]}")

        if result is None:
            continue

        depth, rgb, pose, intr = result
        h, w = depth.shape[:2]

        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        o3d_intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, intr["fx"], intr["fy"], intr["cx"], intr["cy"],
        )

        color = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_scale=1.0,  # already in meters
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        extrinsic = habitat_c2w_to_o3d_extrinsic(pose)
        volume.integrate(rgbd, o3d_intr, extrinsic)

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    return mesh


def o3d_mesh_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if mesh.has_vertex_colors():
        colors_float = np.asarray(mesh.vertex_colors)  # (N, 3) in [0, 1]
        colors_uint8 = (colors_float * 255).clip(0, 255).astype(np.uint8)
        rgba = np.column_stack([colors_uint8, np.full(len(colors_uint8), 255, dtype=np.uint8)])
    else:
        rgba = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=rgba,
        process=False,
    )


def _select_keyframes_raw(loader, rot_deg, trans_m, min_frames_between=5):
    """Select keyframes from poses using explicit thresholds."""
    manager = KeyframeManager(
        rotation_threshold_deg=rot_deg,
        translation_threshold_m=trans_m,
        min_frames_between=min_frames_between,
    )
    poses, fids = loader.get_all_poses()
    return manager.select_keyframes(poses, fids)


def _select_keyframes(loader, cfg):
    """Compute keyframe IDs from poses using config thresholds."""
    return _select_keyframes_raw(
        loader,
        cfg_get(cfg, "keyframing", "rotation_threshold"),
        cfg_get(cfg, "keyframing", "translation_threshold"),
        cfg_get(cfg, "keyframing", "min_frames_between"),
    )


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion for viser.

    Applies 180-deg X rotation so Habitat camera poses (Y-up, -Z forward)
    produce correctly oriented frustums in viser.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    xyzw = Rotation.from_matrix(R @ flip).as_quat()  # scipy returns (x, y, z, w)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


_RANK_COLORS = [
    (0, 200, 50),
    (50, 150, 255),
    (255, 200, 0),
    (255, 100, 50),
    (200, 50, 200),
]


def setup_gui(server, retriever, loader, images, frame_ids, sam3_segmenter, cfg, query_res=0):
    intr = loader.intrinsics
    vfov = float(2.0 * np.arctan(intr.height / (2.0 * intr.fy)))
    aspect = intr.width / intr.height

    search_top_k = cfg_get(cfg, "retrieval", "stage1_top_k")

    # Pre-compute poses once (constant across queries)
    all_poses, all_fids = loader.get_all_poses()
    all_poses_cache = (all_poses, np.asarray(all_fids))

    with server.gui.add_folder("Query"):
        query_input = server.gui.add_text("Text Query", initial_value="")
        top_k_slider = server.gui.add_slider(
            "Top-K", min=1, max=50, step=1, initial_value=search_top_k,
        )
        vlm_cb = server.gui.add_checkbox("VLM Re-ranking", initial_value=retriever.has_vlm)
        search_btn = server.gui.add_button("Search")

    with server.gui.add_folder("Results"):
        status_md = server.gui.add_markdown("*Enter a query and click Search*")

    with server.gui.add_folder("Visualization"):
        show_frustums = server.gui.add_checkbox("Show Frustums", initial_value=True)
        show_cloud = server.gui.add_checkbox("Show Object Cloud", initial_value=True)
        cloud_size_slider = server.gui.add_slider(
            "Point Size", min=0.005, max=0.05, step=0.005, initial_value=0.015,
        )

    spatial_default = cfg_get(cfg, "multi_view", "spatial_fusion")
    with server.gui.add_folder("Multi-View Fusion"):
        spatial_cb = server.gui.add_checkbox("Spatial Fusion", initial_value=spatial_default)
        spatial_neighbors_slider = server.gui.add_slider(
            "Spatial Neighbors", min=1, max=20, step=1,
            initial_value=cfg_get(cfg, "multi_view", "spatial_max_views"),
            hint="More neighbors = denser object cloud but slower queries",
        )
        hdbscan_cb = server.gui.add_checkbox(
            "HDBSCAN", initial_value=cfg_get(cfg, "multi_view", "hdbscan"),
        )
        hdbscan_min_cluster = server.gui.add_slider(
            "Min Cluster Size", min=5, max=50, step=5,
            initial_value=cfg_get(cfg, "multi_view", "hdbscan_min_cluster_size"),
        )
        hdbscan_min_samples_slider = server.gui.add_slider(
            "Min Samples", min=1, max=15, step=1,
            initial_value=cfg_get(cfg, "multi_view", "hdbscan_min_samples"),
        )

    state = {
        "frustum_handles": [],
        "neighbor_handles": [],
        "cloud_handles_base": [],
        "cloud_handles_fused": [],
        "raw_fused_clouds": [],
        "busy": False,
    }

    def clear_results():
        all_handles = (state["frustum_handles"] + state["neighbor_handles"]
                       + state["cloud_handles_base"] + state["cloud_handles_fused"])
        for h in all_handles:
            h.remove()
        state["frustum_handles"].clear()
        state["neighbor_handles"].clear()
        state["cloud_handles_base"].clear()
        state["cloud_handles_fused"].clear()
        state["raw_fused_clouds"].clear()

        cleanup(scene_loader=loader, sam3_segmenter=sam3_segmenter)

    def _do_search(query, current_top_k, use_vlm):
        """Run pipeline and update visualization (called in background thread).

        Follows the nav pipeline's approach: localize once (no spatial fusion),
        group predictions into instances, then fuse each instance using
        grouped retrieval views as precomputed views — instances with enough
        views skip neighbor search entirely.
        """
        t0 = time.time()
        try:
            t_step = time.time()
            search_result = search_scene(
                query=query, images=images, frame_ids=frame_ids,
                retriever=retriever, top_k=current_top_k,
                use_vlm=use_vlm,
            )
            print(f"  Retrieval: {time.time() - t_step:.3f}s")

            t_step = time.time()
            preds, _, meta = localize(
                query=query, images=images, frame_ids=frame_ids,
                search_result=search_result, scene_loader=loader,
                sam3_segmenter=sam3_segmenter, top_k=current_top_k,
                use_centroid=True,
                max_mask_depth=cfg_get(cfg, "multi_view", "max_mask_depth"),
            )
            print(f"  Localization (SAM3 + backproject): {time.time() - t_step:.3f}s")
        except Exception as e:
            status_md.content = f"**Error:** {e}"
            logger.exception("Query failed")
            state["busy"] = False
            return

        if not preds:
            status_md.content = f"**No results found** ({time.time() - t0:.1f}s)"
            state["busy"] = False
            return

        t_step = time.time()
        cands_base = group_predictions_by_overlap(preds, meta)
        print(f"  Grouping: {time.time() - t_step:.3f}s ({len(cands_base)} instances)")

        cands_fused = [replace(c) for c in cands_base]

        t_step = time.time()
        per_seed_nearby = fuse_candidates(
            cands_fused,
            query=query,
            scene_loader=loader,
            sam3_segmenter=sam3_segmenter,
            pred_metadata=meta,
            preds=preds,
            all_poses_cache=all_poses_cache,
            use_centroid=True,
            query_res=query_res,
            spatial_max_views=int(spatial_neighbors_slider.value),
            spatial_max_distance=cfg_get(cfg, "multi_view", "spatial_max_distance"),
            frustum_margin=cfg_get(cfg, "multi_view", "frustum_margin"),
            fusion_threshold=cfg_get(cfg, "multi_view", "fusion_threshold"),
            overlap_threshold=cfg_get(cfg, "multi_view", "overlap_threshold"),
            max_mask_depth=cfg_get(cfg, "multi_view", "max_mask_depth"),
            hdbscan_clean=False,
        )
        print(f"  Multi-view fusion: {time.time() - t_step:.3f}s ({len(cands_fused)} fused)")

        state["raw_fused_clouds"] = [
            cand.point_cloud.copy() if cand.point_cloud is not None else None
            for cand in cands_fused
        ]
        if hdbscan_cb.value:
            for cand in cands_fused:
                if cand.point_cloud is not None and len(cand.point_cloud) >= 10:
                    cand.point_cloud = hdbscan_filter_cloud(
                        cand.point_cloud,
                        min_cluster_size=int(hdbscan_min_cluster.value),
                        min_samples=int(hdbscan_min_samples_slider.value),
                    )

        elapsed = time.time() - t0
        print(f"  TOTAL: {elapsed:.3f}s")
        status_md.content = (
            f"**Done** ({elapsed:.1f}s) — "
            f"{len(cands_base)} instances, {len(cands_fused)} fused"
        )

        # Cache retrieval images by frame ID to avoid redundant disk loads
        vis_image_cache = dict(zip(frame_ids, images))

        def _get_vis_image(fid):
            if fid not in vis_image_cache:
                vis_image_cache[fid] = loader.load_rgb(fid)
            return vis_image_cache[fid]

        for inst_idx, cand in enumerate(cands_base):
            color = _RANK_COLORS[inst_idx % len(_RANK_COLORS)]
            display_ranks = cand.source_ranks[:int(spatial_neighbors_slider.value)]
            for j, rank in enumerate(display_ranks):
                fid = meta[rank]["frame_id"]
                c2w = loader.load_pose(fid)
                position = c2w[:3, 3].astype(np.float64)
                wxyz = rotmat_to_wxyz(c2w[:3, :3])

                is_primary = (j == 0)
                scale = 0.25 if is_primary else 0.15
                img_size = (320, 240) if is_primary else (160, 120)
                img = _get_vis_image(fid)
                img_small = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

                h = server.scene.add_camera_frustum(
                    f"/results/frustums/inst_{inst_idx}_r{j}",
                    fov=vfov, aspect=aspect, scale=scale,
                    color=color, image=img_small,
                    wxyz=wxyz, position=position,
                    visible=show_frustums.value,
                )
                state["frustum_handles"].append(h)

        for i, nearby in enumerate(per_seed_nearby):
            if not nearby:
                continue
            color = _RANK_COLORS[i % len(_RANK_COLORS)]
            for j, (nb_fid, _nb_dist, _) in enumerate(nearby):
                c2w = loader.load_pose(nb_fid)
                position = c2w[:3, 3].astype(np.float64)
                wxyz = rotmat_to_wxyz(c2w[:3, :3])
                nb_img = _get_vis_image(nb_fid)
                nb_img_small = cv2.resize(nb_img, (160, 120), interpolation=cv2.INTER_AREA)

                h = server.scene.add_camera_frustum(
                    f"/results/neighbors/inst_{i}_nb{j}",
                    fov=vfov, aspect=aspect, scale=0.15,
                    color=color, image=nb_img_small,
                    wxyz=wxyz, position=position,
                    visible=spatial_cb.value and show_frustums.value,
                )
                state["neighbor_handles"].append(h)

        use_fused = spatial_cb.value
        for inst_idx, cand in enumerate(cands_base):
            if cand.point_cloud is None or len(cand.point_cloud) == 0:
                continue
            color = _RANK_COLORS[inst_idx % len(_RANK_COLORS)]
            cloud_colors = np.tile(
                np.array(color, dtype=np.uint8), (len(cand.point_cloud), 1),
            )
            h = server.scene.add_point_cloud(
                f"/results/object_base/inst_{inst_idx}",
                points=cand.point_cloud.astype(np.float32),
                colors=cloud_colors,
                point_size=cloud_size_slider.value,
                visible=show_cloud.value and not use_fused,
            )
            state["cloud_handles_base"].append(h)

        for inst_idx, cand in enumerate(cands_fused):
            if cand.point_cloud is None or len(cand.point_cloud) == 0:
                continue
            color = _RANK_COLORS[inst_idx % len(_RANK_COLORS)]
            cloud_colors = np.tile(
                np.array(color, dtype=np.uint8), (len(cand.point_cloud), 1),
            )
            h = server.scene.add_point_cloud(
                f"/results/object_fused/inst_{inst_idx}",
                points=cand.point_cloud.astype(np.float32),
                colors=cloud_colors,
                point_size=cloud_size_slider.value,
                visible=show_cloud.value and use_fused,
            )
            state["cloud_handles_fused"].append(h)

        state["busy"] = False

    @search_btn.on_click
    def _on_search(_):
        query = query_input.value.strip()
        if not query:
            status_md.content = "*Please enter a query*"
            return
        if state["busy"]:
            status_md.content = "*Search in progress...*"
            return

        state["busy"] = True
        current_top_k = int(top_k_slider.value)
        use_vlm = vlm_cb.value
        clear_results()
        status_md.content = f"**Searching:** {query}..."

        threading.Thread(
            target=_do_search,
            args=(query, current_top_k, use_vlm),
            daemon=True,
        ).start()

    def _sync_visibility():
        use_fused = spatial_cb.value
        for h in state["frustum_handles"]:
            h.visible = show_frustums.value
        for h in state["neighbor_handles"]:
            h.visible = show_frustums.value and use_fused
        for h in state["cloud_handles_base"]:
            h.visible = show_cloud.value and not use_fused
        for h in state["cloud_handles_fused"]:
            h.visible = show_cloud.value and use_fused

    show_frustums.on_update(lambda _: _sync_visibility())
    show_cloud.on_update(lambda _: _sync_visibility())
    spatial_cb.on_update(lambda _: _sync_visibility())

    def _reapply_hdbscan(_):
        if not state["raw_fused_clouds"] or state["busy"]:
            return

        use_fused = spatial_cb.value
        for i, raw_cloud in enumerate(state["raw_fused_clouds"]):
            if raw_cloud is None or len(raw_cloud) == 0:
                continue
            if i >= len(state["cloud_handles_fused"]):
                continue

            if hdbscan_cb.value:
                cleaned = hdbscan_filter_cloud(
                    raw_cloud,
                    min_cluster_size=int(hdbscan_min_cluster.value),
                    min_samples=int(hdbscan_min_samples_slider.value),
                )
            else:
                cleaned = raw_cloud

            if len(cleaned) == 0:
                state["cloud_handles_fused"][i].visible = False
                continue

            color = _RANK_COLORS[i % len(_RANK_COLORS)]
            cloud_colors = np.tile(
                np.array(color, dtype=np.uint8), (len(cleaned), 1),
            )
            state["cloud_handles_fused"][i].remove()
            state["cloud_handles_fused"][i] = server.scene.add_point_cloud(
                f"/results/object_fused/inst_{i}",
                points=cleaned.astype(np.float32),
                colors=cloud_colors,
                point_size=cloud_size_slider.value,
                visible=show_cloud.value and use_fused,
            )

    hdbscan_cb.on_update(_reapply_hdbscan)
    hdbscan_min_cluster.on_update(_reapply_hdbscan)
    hdbscan_min_samples_slider.on_update(_reapply_hdbscan)


def main():
    parser = argparse.ArgumentParser(description="TSDF mesh viewer with scene search")
    parser.add_argument("--dataset", type=str, default="hm3d", choices=["hm3d", "mp3d", "goatcore", "custom"])
    parser.add_argument("--scene", type=str, help="Scene ID")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Force rebuild mesh")
    parser.add_argument("--query-res", type=int, default=120,
                        help="Max image dimension (px) for VLM re-ranking + "
                             "SAM3 segmentation; feature retrieval uses original "
                             "resolution (0 = original everywhere)")
    parser.add_argument("--pose-convention", type=str, default="habitat",
                        choices=["habitat", "opencv"],
                        help="Pose coordinate convention for custom datasets "
                             "(default: habitat). habitat=Y-up/-Z-forward, "
                             "opencv=Y-down/Z-forward (common SLAM/VO output)")
    args = parser.parse_args()

    if args.list_scenes:
        list_scenes(args.dataset)
        return

    if not args.scene:
        parser.error("--scene is required (use --list-scenes to see available)")

    cfg = load_config(str(ROOT / "configs" / "demo.yaml"))

    if args.dataset == "mp3d":
        voxel_size = cfg_get(cfg, "tsdf", "voxel_size_mp3d")
    else:
        voxel_size = cfg_get(cfg, "tsdf", "voxel_size")
    max_depth = cfg_get(cfg, "tsdf", "max_depth")
    rot_thresh = cfg_get(cfg, "tsdf", "rot_thresh")
    trans_thresh = cfg_get(cfg, "tsdf", "trans_thresh")
    max_faces = cfg_get(cfg, "tsdf", "max_faces")
    port = cfg_get(cfg, "display", "port")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cache_dir = ROOT / "results" / "meshes" / f"{args.dataset}_{args.scene}"
    cache_path = cache_dir / "mesh.ply"

    loader = get_dataloader(args.dataset, args.scene, args.pose_convention)
    print(f"Scene: {args.scene}")

    if cache_path.exists() and not args.no_cache:
        print(f"Loading cached mesh: {cache_path}")
        mesh = o3d.io.read_triangle_mesh(str(cache_path))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
    else:
        if args.dataset == "mp3d":
            tsdf_frame_ids = list(range(len(loader)))
        else:
            tsdf_frame_ids = _select_keyframes_raw(
                loader, rot_thresh, trans_thresh, min_frames_between=2,
            )
        t0 = time.time()
        mesh = build_tsdf_mesh(
            loader, tsdf_frame_ids,
            voxel_size=voxel_size,
            max_depth=max_depth,
        )
        elapsed = time.time() - t0
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.triangles)
        print(f"Mesh: {n_verts} vertices, {n_faces} faces ({elapsed:.1f}s)")

        if n_faces > max_faces:
            print(f"Decimating mesh: {n_faces} -> {max_faces} faces...")
            t_dec = time.time()
            if n_faces > max_faces * 4:
                surface_area = mesh.get_surface_area()
                target_intermediate = max_faces * 3
                voxel_est = np.sqrt(2.0 * surface_area / target_intermediate)
                mesh = mesh.simplify_vertex_clustering(
                    voxel_size=voxel_est,
                    contraction=o3d.geometry.SimplificationContraction.Average,
                )
                print(f"  Vertex clustering: {n_faces} -> {len(mesh.triangles)} faces ({time.time() - t_dec:.1f}s)")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
            mesh.compute_vertex_normals()
            print(f"  Done: {len(mesh.triangles)} faces ({time.time() - t_dec:.1f}s)")

        cache_dir.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(cache_path), mesh)
        print(f"Cached to: {cache_path}")

    keyframe_ids = _select_keyframes(loader, cfg)
    retrieval_images = loader.load_all_rgb_parallel(keyframe_ids)

    retriever = build_retriever(
        retrieval_images, args.scene,
        retrieval_model=cfg_get(cfg, "retrieval", "model"),
        vlm_model=cfg_get(cfg, "vlm", "model"),
        device="cuda",
        stage1_top_k=cfg_get(cfg, "retrieval", "stage1_top_k"),
        stage2_top_k=cfg_get(cfg, "retrieval", "stage2_top_k"),
        vlm_batch_size=cfg_get(cfg, "vlm", "batch_size"),
        keyframe_ids=keyframe_ids, cache_prefix=args.dataset,
    )

    if args.query_res > 0:
        orig_shape = retrieval_images[0].shape[:2] if retrieval_images else None
        retrieval_images = resize_images_batch(retrieval_images, args.query_res)
        if orig_shape:
            new_shape = retrieval_images[0].shape[:2]
            print(f"Query images resized: {orig_shape[1]}x{orig_shape[0]} -> {new_shape[1]}x{new_shape[0]}")

    sam3_segmenter = SAM3Segmenter(
        device="cuda",
        confidence_threshold=cfg_get(cfg, "sam3", "confidence"),
        batch_size=cfg_get(cfg, "sam3", "batch_size"),
    )

    tm = o3d_mesh_to_trimesh(mesh)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+y")
    server.gui.set_panel_label("Scene Viewer")
    server.gui.configure_theme(control_width="large", dark_mode=False)
    server.gui.add_html(
        "<style>"
        ":root { font-size: 18px !important; }"
        "[class*='mantine-'] { font-size: 16px !important; }"
        "[class*='Accordion-label'] { font-size: 18px !important; font-weight: 600 !important; }"
        "[class*='Button-root'] { height: 40px !important; min-height: 40px !important; }"
        "[class*='Slider-thumb'] { width: 20px !important; height: 20px !important; }"
        "</style>",
    )
    server.scene.add_mesh_trimesh("/mesh", tm)

    # Set initial camera so the user sees the full mesh from an isometric angle.
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    radius = np.linalg.norm(extent) / 2.0
    fov_rad = server.initial_camera.fov
    cam_dist = radius / np.tan(fov_rad / 2.0) * 1.3
    elev = np.radians(40)
    azim = np.radians(25)
    direction = np.array([
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
        np.cos(elev) * np.cos(azim),
    ])
    # Shift look_at left to compensate for the right-side GUI panel.
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    look_at = center - right * radius * 0.08
    server.initial_camera.look_at = look_at
    server.initial_camera.position = look_at + cam_dist * direction

    sam3_segmenter.segment(retrieval_images[0], "warmup", cache_key=None)
    sam3_segmenter.clear_cache()
    if retriever.vlm:
        retriever.vlm.batch_query([retrieval_images[0]], "warmup")

    setup_gui(
        server, retriever, loader,
        retrieval_images, keyframe_ids,
        sam3_segmenter, cfg, query_res=args.query_res,
    )

    print(f"\nOpen http://localhost:{port}")
    print("Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
