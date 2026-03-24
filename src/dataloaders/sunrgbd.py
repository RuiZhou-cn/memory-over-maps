"""SUN RGB-D dataset utilities (scene discovery, image loading, label parsing)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.utils.image import resize_image

SENSOR_TYPES = ["kv1", "kv2", "xtion", "realsense"]


def discover_scenes(root: str) -> list[Path]:
    """Recursively find valid SUN RGB-D scenes.

    A valid scene has an image/ directory and at least one annotation file.
    Returns sorted list of scene directory paths for deterministic order.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SUN RGB-D root not found: {root}")

    scenes = []
    for image_dir in sorted(root.rglob("image")):
        scene_dir = image_dir.parent
        has_ann = (
            (scene_dir / "annotation2D3D" / "index.json").exists()
            or (scene_dir / "annotation" / "index.json").exists()
        )
        if not has_ann:
            continue
        jpgs = list(image_dir.glob("*.jpg"))
        if not jpgs:
            continue
        scenes.append(scene_dir)

    return sorted(scenes)


def detect_sensor(scene_path: Path, sunrgbd_root: Path) -> str:
    """Determine sensor type from scene path (kv1, kv2, xtion, realsense)."""
    try:
        rel = scene_path.relative_to(sunrgbd_root)
    except ValueError:
        return "unknown"
    top = rel.parts[0]
    if top in SENSOR_TYPES:
        return top
    return "unknown"


def build_sensor_groups(
    scenes: list[Path], sunrgbd_root: str,
) -> dict[str, list[int]]:
    """Map sensor type -> list of global scene indices."""
    root = Path(sunrgbd_root)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, scene_dir in enumerate(scenes):
        sensor = detect_sensor(scene_dir, root)
        groups[sensor].append(idx)
    return dict(groups)


def load_scene_image(
    scene_dir: Path,
    max_size: int = 640,
) -> np.ndarray | None:
    """Load the single RGB image from a SUN RGB-D scene directory."""
    image_dir = scene_dir / "image"
    jpgs = sorted(image_dir.glob("*.jpg"))
    if not jpgs:
        return None

    img = cv2.imread(str(jpgs[0]))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_image(img, max_size)


def load_scene_objects(scene_dir: Path) -> list[str]:
    """Load object category names from a scene's annotations.

    Prefers annotation2D3D (cleaner labels), falls back to annotation/.
    Returns lowercased, deduplicated object names (exact strings, no synonym merging).
    """
    ann_path = scene_dir / "annotation2D3D" / "index.json"
    if not ann_path.exists():
        ann_path = scene_dir / "annotation" / "index.json"
    if not ann_path.exists():
        return []

    try:
        with open(ann_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    objects = data.get("objects", [])
    names = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name", "").strip().lower()
        # Strip annotation suffixes like :truncated, :occluded
        if ":" in name:
            name = name.split(":")[0].strip()
        if name:
            names.add(name)

    return sorted(names)


def build_category_index(
    scenes: list[Path],
) -> tuple[dict[str, set[int]], dict[int, list[str]]]:
    """Build mappings between categories and scene indices.

    Uses exact string matching -- no synonym merging. "table" and "desk"
    are separate categories.
    """
    category_to_scenes: dict[str, set[int]] = defaultdict(set)
    scene_to_categories: dict[int, list[str]] = {}

    for idx, scene_dir in enumerate(scenes):
        objects = load_scene_objects(scene_dir)
        scene_to_categories[idx] = objects
        for name in objects:
            category_to_scenes[name].add(idx)

    return dict(category_to_scenes), scene_to_categories
