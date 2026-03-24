"""Dataset loaders for scene formats used in evaluation.

Provides ``BaseSceneDatasetLoader`` (ABC) and concrete loaders in submodules:
- ``goatcore`` -- GoatCoreSceneDatasetLoader, GoatCoreGroundTruthLoader
- ``hm3d`` -- HM3DSceneDatasetLoader
- ``mp3d`` -- MP3DSceneDatasetLoader
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.image import resize_image

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _quat_translation_to_matrix(
    qw: float, qx: float, qy: float, qz: float,
    tx: float, ty: float, tz: float,
) -> np.ndarray:
    """Convert quaternion (wxyz) + translation to 4x4 transformation matrix."""
    rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy expects xyzw
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


class BaseSceneDatasetLoader(ABC):
    """Abstract base for scene dataset loaders (GoatCore, HM3D, MP3D).

    Subclasses must implement:
      - ``_get_rgb_path(frame_id)`` -> Path
      - ``load_pose(frame_id)`` -> np.ndarray (4x4)

    And set in ``__init__``:
      - ``self.frame_ids``: list[int]
      - ``self.intrinsics``: CameraIntrinsics
      - ``self.max_image_size``: int | None
    """

    frame_ids: list[int]
    intrinsics: CameraIntrinsics
    max_image_size: int | None

    def __init__(self) -> None:
        self._depth_cache: dict[int, np.ndarray] = {}

    @abstractmethod
    def _get_rgb_path(self, frame_id: int) -> Path:
        """Return the filesystem path for the RGB image of *frame_id*."""

    @abstractmethod
    def load_pose(self, frame_id: int) -> np.ndarray:
        """Return the 4x4 camera-to-world pose matrix for *frame_id*."""

    def __len__(self) -> int:
        return len(self.frame_ids)

    def load_rgb(self, frame_id: int) -> np.ndarray:
        """Load RGB image, convert BGR->RGB, and optionally resize."""
        path = self._get_rgb_path(frame_id)
        rgb_bgr = cv2.imread(str(path))
        if rgb_bgr is None:
            raise ValueError(f"Could not load image: {path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        return resize_image(rgb, self.max_image_size)

    def load_depth(self, frame_id: int) -> np.ndarray | None:
        """Load depth map with caching. Subclasses override ``_load_depth_from_disk``."""
        if frame_id in self._depth_cache:
            return self._depth_cache[frame_id]
        depth = self._load_depth_from_disk(frame_id)
        if depth is not None:
            self._depth_cache[frame_id] = depth
        return depth

    def _load_depth_from_disk(self, frame_id: int) -> np.ndarray | None:
        """Load raw depth from disk. Override in subclasses with depth data."""
        return None

    def get_all_poses(self) -> tuple[np.ndarray, list[int]]:
        poses = [self.load_pose(fid) for fid in self.frame_ids]
        return np.array(poses, dtype=np.float32), list(self.frame_ids)

    def get_intrinsics_dict(self) -> dict[str, float]:
        return {
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'cx': self.intrinsics.cx,
            'cy': self.intrinsics.cy,
        }

    def get_intrinsics_for_frame(self, frame_id: int) -> dict[str, float]:
        """Return per-frame intrinsics; override for multi-camera rigs."""
        return self.get_intrinsics_dict()

    def get_pose_rt(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        pose = self.load_pose(frame_id)
        return pose[:3, :3], pose[:3, 3]

    def clear_depth_cache(self):
        self._depth_cache.clear()

    def load_all_rgb_parallel(
        self,
        frame_ids: list[int] | None = None,
        max_workers: int = 8,
    ) -> list[np.ndarray]:
        """Load all RGB images in parallel using ThreadPoolExecutor.

        cv2.imread releases the GIL, so threads give near-linear speedup.
        """
        from concurrent.futures import ThreadPoolExecutor

        if frame_ids is None:
            frame_ids = self.frame_ids

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            images = list(pool.map(self.load_rgb, frame_ids))
        return images


class LazyImageList:
    """List-like wrapper that loads images on demand instead of eagerly.

    Supports integer indexing, slicing, len(), and iteration -- enough to
    be a drop-in replacement for ``[loader.load_rgb(fid) for fid in ids]``
    in retrieval, VLM, and segmentation code paths.

    When features are cached (the common case), FAISS search never touches
    raw images, so most entries are never loaded.  Only the ~5-10 images
    accessed by VLM re-ranking or SAM3 segmentation are actually read from
    disk.
    """

    def __init__(self, loader, frame_ids: list[int]):
        self._loader = loader
        self._frame_ids = frame_ids
        self._cache = {}

    def __len__(self):
        return len(self._frame_ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if idx not in self._cache:
            self._cache[idx] = self._loader.load_rgb(self._frame_ids[idx])
        return self._cache[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def clear(self):
        self._cache.clear()


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_dataloader(
    dataset: str, scene: str, pose_convention: str = "habitat",
) -> "BaseSceneDatasetLoader":
    """Create a dataset-specific scene loader at native resolution (no downscaling).

    Uses lazy imports to avoid eagerly loading all submodules.

    Args:
        dataset: One of ``"hm3d"``, ``"mp3d"``, ``"goatcore"``, ``"custom"``.
        scene: Scene ID (or absolute path for custom datasets).
        pose_convention: Coordinate convention for custom datasets
            (``"habitat"`` or ``"opencv"``).

    Returns:
        A concrete ``BaseSceneDatasetLoader`` for the requested scene.
    """
    if dataset == "hm3d":
        from src.dataloaders.hm3d import HM3DSceneDatasetLoader

        scene_path = _PROJECT_ROOT / "data" / "hm3d" / "scenes" / scene
        return HM3DSceneDatasetLoader(str(scene_path), max_image_size=0)
    elif dataset == "mp3d":
        from src.dataloaders.mp3d import MP3DSceneDatasetLoader

        scene_path = _PROJECT_ROOT / "data" / "mp3d" / "scenes" / scene
        return MP3DSceneDatasetLoader(str(scene_path), max_image_size=0)
    elif dataset == "goatcore":
        from src.dataloaders.goatcore import GoatCoreSceneDatasetLoader

        scene_path = _PROJECT_ROOT / "data" / "Goat-core" / "dataset" / scene
        return GoatCoreSceneDatasetLoader(str(scene_path), max_image_size=0)
    elif dataset == "custom":
        from src.dataloaders.custom import CustomSceneDatasetLoader

        scene_path = Path(scene) if Path(scene).is_absolute() else _PROJECT_ROOT / scene
        return CustomSceneDatasetLoader(
            str(scene_path), max_image_size=0, pose_convention=pose_convention,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
