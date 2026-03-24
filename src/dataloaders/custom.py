"""Custom dataset loader for user-provided posed RGBD sequences.

Expected directory structure:
    my_scene/
        rgb/                  # RGB images (sorted alphabetically)
            000000.jpg        # .jpg or .png
            000001.jpg
            ...
        depth/                # uint16 PNG depth in millimeters
            000000.png
            000001.png
            ...
        intrinsics.json       # {"fx", "fy", "cx", "cy", "width", "height"}
        poses.txt             # One line per frame: tx ty tz qw qx qy qz
                              # Camera-to-world. Convention set via pose_convention:
                              #   "habitat" (default): Y-up, -Z forward
                              #   "opencv": Y-down, Z-forward (SLAM/VO output)

Notes:
    - RGB and depth filenames must sort in the same order.
    - poses.txt line i corresponds to the i-th sorted RGB image.
    - depth_scale in intrinsics.json (default 1000.0) converts raw uint16 to meters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .base import (
    BaseSceneDatasetLoader,
    CameraIntrinsics,
    _quat_translation_to_matrix,
)

logger = logging.getLogger(__name__)

_RGB_EXTS = {".jpg", ".jpeg", ".png"}


class CustomSceneDatasetLoader(BaseSceneDatasetLoader):
    """Loader for user-provided posed RGBD sequences."""

    def __init__(self, dataset_path: str, max_image_size: int = 640,
                 pose_convention: str = "habitat"):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.max_image_size = max_image_size
        self.pose_convention = pose_convention

        self.image_dir = self.dataset_path / "rgb"
        if not self.image_dir.exists():
            raise ValueError(f"RGB directory not found: {self.image_dir}")

        self.depth_dir = self.dataset_path / "depth"
        if not self.depth_dir.exists():
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        self._rgb_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in _RGB_EXTS
        )
        if not self._rgb_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        self.frame_ids = list(range(len(self._rgb_paths)))

        # Discover depth images (matched by sorted order)
        self._depth_paths = sorted(self.depth_dir.glob("*.png"))
        if len(self._depth_paths) != len(self._rgb_paths):
            raise ValueError(
                f"RGB/depth count mismatch: {len(self._rgb_paths)} RGB vs "
                f"{len(self._depth_paths)} depth images"
            )

        self.intrinsics, self._depth_scale = self._load_intrinsics()
        self._poses = self._load_poses()
        if self.pose_convention == "opencv":
            self._poses = [self._opencv_to_habitat(p) for p in self._poses]
        logger.info(
            "Custom scene: %s (%d frames)",
            self.dataset_path.name, len(self.frame_ids),
        )

    @staticmethod
    def _opencv_to_habitat(c2w: np.ndarray) -> np.ndarray:
        """Convert OpenCV C2W (Y-down, Z-forward) to Habitat C2W (Y-up, -Z forward)."""
        c2w = c2w.copy()
        c2w[:3, 1] *= -1  # Y-down -> Y-up
        c2w[:3, 2] *= -1  # Z-forward -> Z-backward
        return c2w

    def _load_intrinsics(self):
        intr_path = self.dataset_path / "intrinsics.json"
        if not intr_path.exists():
            raise ValueError(
                f"intrinsics.json not found: {intr_path}\n"
                f"Expected format: {{\"fx\", \"fy\", \"cx\", \"cy\", \"width\", \"height\"}}"
            )
        with open(intr_path) as f:
            data = json.load(f)

        for key in ("fx", "fy", "cx", "cy", "width", "height"):
            if key not in data:
                raise ValueError(f"Missing key '{key}' in intrinsics.json")

        depth_scale = data.get("depth_scale", 1000.0)

        return CameraIntrinsics(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        ), depth_scale

    def _load_poses(self) -> list[np.ndarray]:
        poses_path = self.dataset_path / "poses.txt"
        if not poses_path.exists():
            raise ValueError(
                f"poses.txt not found: {poses_path}\n"
                f"Expected format: one line per frame with 'tx ty tz qw qx qy qz'"
            )

        poses = []
        with open(poses_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 7:
                    raise ValueError(
                        f"poses.txt line {line_num}: expected 7 values "
                        f"(tx ty tz qw qx qy qz), got {len(parts)}"
                    )
                tx, ty, tz, qw, qx, qy, qz = map(float, parts)
                poses.append(_quat_translation_to_matrix(qw, qx, qy, qz, tx, ty, tz))

        if len(poses) != len(self.frame_ids):
            raise ValueError(
                f"Pose count mismatch: {len(poses)} poses vs {len(self.frame_ids)} images"
            )
        return poses

    def _get_rgb_path(self, frame_id: int) -> Path:
        return self._rgb_paths[frame_id]

    def load_pose(self, frame_id: int) -> np.ndarray:
        return self._poses[frame_id]

    def _load_depth_from_disk(self, frame_id: int) -> np.ndarray | None:
        depth_uint16 = cv2.imread(str(self._depth_paths[frame_id]), cv2.IMREAD_UNCHANGED)
        if depth_uint16 is None:
            return None
        return depth_uint16.astype(np.float32) / self._depth_scale
