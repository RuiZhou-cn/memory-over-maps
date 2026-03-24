"""MP3D (Matterport3D) dataset loader.

- ``MP3DSceneDatasetLoader`` -- undistorted images + per-file poses, Z-up -> Y-up
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import (
    BaseSceneDatasetLoader,
    CameraIntrinsics,
)

logger = logging.getLogger(__name__)


class MP3DSceneDatasetLoader(BaseSceneDatasetLoader):
    """Dataloader for Matterport3D scene format (undistorted images + per-file poses).

    Dataset structure (per scene dir):
    - undistorted_color_images/{uuid}_i{tripod}_{dir}.jpg   (1280x1024)
    - matterport_camera_poses/{uuid}_pose_{tripod}_{dir}.txt  (4x4 matrix, Z-up)
    - matterport_camera_intrinsics/{uuid}_intrinsics_{tripod}.txt
    - undistorted_depth_images/{uuid}_d{tripod}_{dir}.png   (uint16 millimeters)

    Matterport uses Z-up coordinates. All poses are converted to Y-up (Habitat):
        (x, y, z)_zup  ->  (x, z, -y)_yup
    """

    _ZUP_TO_YUP = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=np.float64)

    def __init__(self, dataset_path: str, max_image_size: int = 640):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.max_image_size = max_image_size

        self._img_dir = self.dataset_path / "undistorted_color_images"
        self._pose_dir = self.dataset_path / "matterport_camera_poses"
        self._depth_dir = self.dataset_path / "undistorted_depth_images"
        self._intrinsics_dir = self.dataset_path / "matterport_camera_intrinsics"

        if not self._img_dir.exists():
            raise ValueError(f"Image directory not found: {self._img_dir}")
        if not self._pose_dir.exists():
            raise ValueError(f"Pose directory not found: {self._pose_dir}")

        self._frame_map: dict[int, tuple[str, int, int]] = {}
        self._build_frame_map()

        self.frame_ids = sorted(self._frame_map.keys())
        if not self.frame_ids:
            raise ValueError(f"No images found in {self._img_dir}")

        self.intrinsics = self._load_intrinsics()
        # Per-camera-direction intrinsics (Matterport has 3 cameras: dirs 0,1,2)
        # Each direction captures different parts of the panorama with slightly
        # different fx/fy/cx/cy. Using the wrong set causes systematic 3D offset.
        self._intrinsics_by_dir: dict[int, CameraIntrinsics] = self._load_all_direction_intrinsics()

        self._pose_map = self._load_all_poses()

        logger.info("Loaded %d frames from %s", len(self.frame_ids), self.dataset_path.name)

    def _build_frame_map(self):
        """Enumerate undistorted color images and assign linear frame IDs."""
        entries = []
        for f in sorted(self._img_dir.glob("*_i*_*.jpg")):
            name = f.stem  # e.g. "2393bffb53fe4205bcc67796c6fb76e3_i0_0"
            idx = name.index("_i")
            uuid = name[:idx]
            rest = name[idx + 2:]  # "0_0"
            parts = rest.split("_")
            tripod = int(parts[0])
            direction = int(parts[1])
            entries.append((uuid, tripod, direction, f.name))

        for frame_id, (uuid, tripod, direction, _) in enumerate(entries):
            self._frame_map[frame_id] = (uuid, tripod, direction)

    def _get_rgb_path(self, frame_id: int) -> Path:
        uuid, tripod, direction = self._frame_map[frame_id]
        return self._img_dir / f"{uuid}_i{tripod}_{direction}.jpg"

    def _load_intrinsics(self) -> CameraIntrinsics:
        """Load intrinsics from first available intrinsics file.

        Format: w h fx fy cx cy k1 k2 p1 p2 k3
        Distortion coefficients are ignored (images are undistorted).
        """
        intrinsics_files = sorted(self._intrinsics_dir.glob("*_intrinsics_*.txt"))
        if not intrinsics_files:
            raise FileNotFoundError(
                f"No intrinsics files found in {self._intrinsics_dir}"
            )

        with open(intrinsics_files[0], 'r') as f:
            parts = f.read().strip().split()
        w, h = int(float(parts[0])), int(float(parts[1]))
        fx, fy = float(parts[2]), float(parts[3])
        cx, cy = float(parts[4]), float(parts[5])
        return CameraIntrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

    def _load_all_direction_intrinsics(self) -> dict[int, CameraIntrinsics]:
        """Load intrinsics for each camera direction (0, 1, 2).

        Matterport cameras within a tripod are distinct lenses with separate
        calibration files: ``{uuid}_intrinsics_{direction}.txt``.
        The fx/fy/cx/cy values differ across directions (e.g. cx varies ~10px
        at 1280px width), causing systematic backprojection offset when the
        wrong set is applied.

        Returns:
            Dict mapping camera direction index -> CameraIntrinsics.
            Keys present depend on which files exist; may be empty.
        """
        result: dict[int, CameraIntrinsics] = {}
        if not self._intrinsics_dir.exists():
            return result
        for direction in range(3):
            # Each UUID has one file per direction; any UUID's file is fine --
            # intrinsics are camera-hardware constants, not pose-dependent.
            files = sorted(self._intrinsics_dir.glob(f"*_intrinsics_{direction}.txt"))
            if not files:
                continue
            with open(files[0], 'r') as f:
                parts = f.read().strip().split()
            w, h = int(float(parts[0])), int(float(parts[1]))
            fx, fy = float(parts[2]), float(parts[3])
            cx, cy = float(parts[4]), float(parts[5])
            result[direction] = CameraIntrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        return result

    def get_intrinsics_for_frame(self, frame_id: int) -> dict[str, float]:
        """Return the correct intrinsics for this frame's camera direction.

        Overrides ``BaseSceneDatasetLoader.get_intrinsics_for_frame`` to use
        per-direction calibration instead of a single global intrinsics dict.
        Falls back to global intrinsics if direction-specific data is missing.
        """
        _, _, direction = self._frame_map[frame_id]
        intr = self._intrinsics_by_dir.get(direction)
        if intr is None:
            return self.get_intrinsics_dict()
        return {'fx': intr.fx, 'fy': intr.fy, 'cx': intr.cx, 'cy': intr.cy}


    def _load_all_poses(self) -> dict[int, np.ndarray]:
        """Preload all poses from disk and convert Z-up to Y-up.

        The Matterport camera uses OpenCV convention (X-right, Y-down, Z-forward).
        The pose file gives a camera-to-world matrix in Z-up coordinates.

        Conversion:
        1. Left-multiply R by _ZUP_TO_YUP to convert the world frame only.
           (Camera coordinates are defined by the physical lens, not the world.)
        2. Negate Y and Z columns to flip the camera from OpenCV convention
           (Y-down, Z-forward) to Habitat convention (Y-up, Z-backward).
        """
        pose_map = {}
        for frame_id, (uuid, tripod, direction) in self._frame_map.items():
            pose_file = self._pose_dir / f"{uuid}_pose_{tripod}_{direction}.txt"
            if not pose_file.exists():
                logger.warning("Pose file not found: %s", pose_file)
                continue

            pose_zup = np.loadtxt(str(pose_file))  # 4x4

            R_zup = pose_zup[:3, :3]
            t_zup = pose_zup[:3, 3]

            pose_yup = np.eye(4, dtype=np.float64)
            pose_yup[:3, :3] = self._ZUP_TO_YUP @ R_zup
            pose_yup[:3, 3] = self._ZUP_TO_YUP @ t_zup

            pose_yup[:3, 1] *= -1
            pose_yup[:3, 2] *= -1

            pose_map[frame_id] = pose_yup.astype(np.float32)
        return pose_map

    def load_pose(self, frame_id: int) -> np.ndarray:
        """Load 4x4 camera-to-world pose (preloaded, Y-up convention)."""
        if frame_id not in self._pose_map:
            raise ValueError(f"No pose for frame {frame_id}")
        return self._pose_map[frame_id]

    def _load_depth_from_disk(self, frame_id: int) -> np.ndarray | None:
        """Load uint16 depth (0.25mm per unit -> float32 meters)."""
        uuid, tripod, direction = self._frame_map[frame_id]
        depth_path = self._depth_dir / f"{uuid}_d{tripod}_{direction}.png"
        if not depth_path.exists():
            return None
        depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_uint16 is None:
            return None
        return depth_uint16.astype(np.float32) / 4000.0
