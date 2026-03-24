"""HM3D dataset loaders.

- ``HM3DSceneDatasetLoader`` -- HM3D ObjectNav challenge scene loader (poses.json)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .base import (
    BaseSceneDatasetLoader,
    CameraIntrinsics,
    _quat_translation_to_matrix,
)

logger = logging.getLogger(__name__)


class HM3DSceneDatasetLoader(BaseSceneDatasetLoader):
    """Dataloader for HM3D ObjectNav challenge scene format (RGB images + poses).

    Dataset structure:
    - {scene_id}/
        - images/00000.jpg, 00001.jpg, ...   # RGB (1600x1200)
        - depth/00000.png, 00001.png, ...    # Depth (uint16 PNG, millimeters)
        - poses.json                          # camera_to_world per frame

    Intrinsics are computed from assumed HFOV and image resolution.
    """

    def __init__(
        self,
        dataset_path: str,
        max_image_size: int = 640,
        hfov: float = 79.0,
    ):
        """Initialize HM3D scene dataloader.

        Args:
            dataset_path: Path to a single scene directory (e.g., data/hm3d/scenes/TEEsavR23oF)
            max_image_size: Maximum image dimension for downsampling
            hfov: Assumed horizontal field of view (degrees) for intrinsics computation
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.max_image_size = max_image_size
        self.hfov = hfov

        self.image_dir = self.dataset_path / "images"
        if not self.image_dir.exists():
            raise ValueError(f"Images directory not found: {self.image_dir}")

        self.depth_dir = self.dataset_path / "depth"
        self._has_depth_dir = self.depth_dir.exists()

        self._pose_map = self._load_poses()

        self.frame_ids = sorted(self._pose_map.keys())
        if not self.frame_ids:
            raise ValueError(f"No frames found in {self.dataset_path}")

        self.intrinsics = self._compute_intrinsics()

    def _get_rgb_path(self, frame_id: int) -> Path:
        return self.image_dir / f"{frame_id:05d}.jpg"

    def _load_poses(self) -> dict[int, np.ndarray]:
        """Load camera-to-world poses from poses.json.

        Format: [{frame_id, filename, camera_to_world: {translation: {x,y,z}, quaternion: {w,x,y,z}}}]
        Returns {frame_id: 4x4 matrix}.
        """
        poses_file = self.dataset_path / "poses.json"
        if not poses_file.exists():
            raise ValueError(f"poses.json not found: {poses_file}")

        with open(poses_file, 'r') as f:
            data = json.load(f)

        pose_map = {}
        for entry in data:
            frame_id = entry["frame_id"]
            c2w = entry["camera_to_world"]
            t = c2w["translation"]
            q = c2w["quaternion"]

            pose_map[frame_id] = _quat_translation_to_matrix(
                q["w"], q["x"], q["y"], q["z"],
                t["x"], t["y"], t["z"],
            )

        return pose_map

    def _compute_intrinsics(self) -> CameraIntrinsics:
        """Compute intrinsics from HFOV and image resolution."""
        first_frame_id = self.frame_ids[0]
        img_path = self.image_dir / f"{first_frame_id:05d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot load first image: {img_path}")
        height, width = img.shape[:2]

        hfov_rad = math.radians(self.hfov)
        fx = width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        return CameraIntrinsics(
            width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )

    def load_pose(self, frame_id: int) -> np.ndarray:
        if frame_id not in self._pose_map:
            raise ValueError(f"No pose for frame {frame_id}")
        return self._pose_map[frame_id]

    def _load_depth_from_disk(self, frame_id: int) -> np.ndarray | None:
        """Load uint16 PNG depth (millimeters -> float32 meters)."""
        if not self._has_depth_dir:
            return None
        depth_path = self.depth_dir / f"{frame_id:05d}.png"
        depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_uint16 is None:
            return None
        return depth_uint16.astype(np.float32) / 1000.0
