#!/usr/bin/env python3
"""Render RGB + depth for HM3D scenes using habitat-sim.

Loads each scene's GLB mesh and camera poses, renders at 1600x1200 / HFOV 79°.
  - RGB:   saved as JPG  -> data/hm3d/scenes/{scene}/images/{frame}.jpg
  - Depth: saved as uint16 PNG (mm) -> data/hm3d/scenes/{scene}/depth/{frame}.png

Only poses.json + mesh are needed — both modalities are rendered from scratch.

Usage:
    python scripts/render_rgbd.py                          # All 36 scenes, RGB+depth
    python scripts/render_rgbd.py --depth-only             # Depth only (keep existing RGB)
    python scripts/render_rgbd.py --scenes Dd4bFSTQ8gi     # Single scene
    python scripts/render_rgbd.py --workers 4              # Parallel (4 GPU processes)
"""

import argparse
import json
import logging
import os
from multiprocessing import Pool, set_start_method

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["MAGNUM_GPU_VALIDATION"] = "OFF"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"
os.environ["HABITAT_SIM_SEMANTIC_LOG"] = "quiet"

import cv2  # noqa: E402
import habitat_sim  # noqa: E402
import numpy as np  # noqa: E402
import quaternion  # noqa: E402, F401 — enables np.quaternion
from tqdm import tqdm  # noqa: E402

from src.evaluation.helpers import suppress_stderr

logging.getLogger("habitat_sim").setLevel(logging.WARNING)

WIDTH = 1600
HEIGHT = 1200
HFOV = 79

SCENES_DIR = os.path.join("data", "hm3d", "scenes")
MESH_ROOT = os.path.join("data", "hm3d", "meshes")


def find_mesh_path(scene_name, mesh_root):
    if not os.path.isdir(mesh_root):
        return None
    for d in os.listdir(mesh_root):
        if d.endswith(f"-{scene_name}"):
            for ext in [".basis.glb", ".glb"]:
                path = os.path.join(mesh_root, d, scene_name + ext)
                if os.path.exists(path):
                    return path
    return None


def make_sim_cfg(scene_mesh, width, height, hfov, render_rgb):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.gpu_device_id = 0
    backend_cfg.scene_id = scene_mesh
    backend_cfg.load_semantic_mesh = False

    sensors = []

    if render_rgb:
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [height, width]
        rgb_spec.position = [0.0, 0.0, 0.0]
        rgb_spec.hfov = hfov
        rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors.append(rgb_spec)

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, 0.0, 0.0]
    depth_spec.hfov = hfov
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensors.append(depth_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def process_scene(args_tuple):
    scene_name, scenes_dir, mesh_root, width, height, hfov, depth_only, show_pbar = args_tuple
    render_rgb = not depth_only

    scene_dir = os.path.join(scenes_dir, scene_name)
    poses_file = os.path.join(scene_dir, "poses.json")
    depth_dir = os.path.join(scene_dir, "depth")
    images_dir = os.path.join(scene_dir, "images")

    if not os.path.isfile(poses_file):
        print(f"[{scene_name}] SKIP: no poses.json")
        return scene_name, False

    mesh_path = find_mesh_path(scene_name, mesh_root)
    if mesh_path is None:
        print(f"[{scene_name}] ERROR: mesh not found under {mesh_root}")
        return scene_name, False

    with open(poses_file) as f:
        poses = json.load(f)

    os.makedirs(depth_dir, exist_ok=True)
    if render_rgb:
        os.makedirs(images_dir, exist_ok=True)

    # Per-frame resume: skip frames whose output files already exist
    done_depth = set(os.listdir(depth_dir))
    done_rgb = set(os.listdir(images_dir)) if render_rgb else set()

    def _needs_render(pose_entry):
        fname = pose_entry["filename"]
        if os.path.splitext(fname)[0] + ".png" not in done_depth:
            return True
        if render_rgb and fname not in done_rgb:
            return True
        return False

    pending = [p for p in poses if _needs_render(p)]
    n_done = len(poses) - len(pending)

    if not pending:
        print(f"[{scene_name}] SKIP: already complete ({len(poses)} frames)")
        return scene_name, True
    if n_done:
        print(f"[{scene_name}] Resuming: {n_done}/{len(poses)} frames done, {len(pending)} remaining")

    cfg = make_sim_cfg(mesh_path, width, height, hfov, render_rgb)
    sim = suppress_stderr(habitat_sim.Simulator, cfg)
    agent = sim.initialize_agent(0)

    sensor_uuids = ["depth_sensor"]
    if render_rgb:
        sensor_uuids.append("color_sensor")

    iterator = (
        tqdm(pending, desc=f"[{scene_name}]", initial=n_done, total=len(poses), leave=True)
        if show_pbar else pending
    )
    for pose_entry in iterator:
        filename = pose_entry["filename"]
        c2w = pose_entry["camera_to_world"]
        t = c2w["translation"]
        q = c2w["quaternion"]

        position = np.array([t["x"], t["y"], t["z"]])
        rotation = np.quaternion(q["w"], q["x"], q["y"], q["z"])

        agent_state = agent.get_state()
        for uuid in sensor_uuids:
            agent_state.sensor_states[uuid].position = position
            agent_state.sensor_states[uuid].rotation = rotation
        agent.set_state(agent_state, reset_sensors=True, infer_sensor_states=False)

        obs = sim.get_sensor_observations(0)

        depth = obs["depth_sensor"]
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_filename = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(depth_dir, depth_filename), depth_mm)

        if render_rgb:
            bgr = obs["color_sensor"][:, :, :3][:, :, ::-1]
            cv2.imwrite(os.path.join(images_dir, filename), bgr)

    sim.close()
    mode = "depth" if depth_only else "RGB+depth"
    print(f"[{scene_name}] Done. {len(pending)} new {mode} frames ({len(poses)} total)")
    return scene_name, True


def main():
    parser = argparse.ArgumentParser(description="Render RGB + depth for HM3D scenes")
    parser.add_argument(
        "--scenes-dir", default=SCENES_DIR,
        help="Directory containing scene subdirs (default: %(default)s)",
    )
    parser.add_argument(
        "--mesh-root", default=MESH_ROOT,
        help="Directory with downloaded HM3D meshes (default: %(default)s)",
    )
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Scene names to process (default: all scenes in scenes-dir)",
    )
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--hfov", type=int, default=HFOV)
    parser.add_argument(
        "--depth-only", action="store_true",
        help="Render depth only (skip RGB)",
    )
    parser.add_argument(
        "--workers", type=int, default=18,
        help="Parallel workers (each loads a habitat-sim GPU instance)",
    )
    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted(
            d for d in os.listdir(args.scenes_dir)
            if os.path.isdir(os.path.join(args.scenes_dir, d))
        )

    mode = "depth" if args.depth_only else "RGB+depth"
    print(f"Rendering {mode} for {len(scenes)} scenes ({args.width}x{args.height}, hfov={args.hfov}°)")

    multi = args.workers > 1
    task_args = [
        (scene, args.scenes_dir, args.mesh_root, args.width, args.height, args.hfov, args.depth_only, not multi)
        for scene in scenes
    ]

    if not multi:
        for ta in task_args:
            process_scene(ta)
    else:
        succeeded = 0
        with Pool(processes=args.workers) as pool:
            with tqdm(total=len(task_args), desc="Scenes", unit="scene") as pbar:
                for scene_name, ok in pool.imap_unordered(process_scene, task_args):
                    succeeded += ok
                    pbar.set_postfix(last=scene_name)
                    pbar.update(1)
        print(f"Completed: {succeeded}/{len(task_args)} scenes")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
