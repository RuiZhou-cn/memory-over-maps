"""Goat-Core dataset loaders.

- ``GoatCoreSceneDatasetLoader`` -- scene loader (COLMAP poses + depth)
- ``GoatCoreGroundTruthLoader`` -- ground truth reader (goals, queries, task types)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .base import (
    BaseSceneDatasetLoader,
    CameraIntrinsics,
    _quat_translation_to_matrix,
)

logger = logging.getLogger(__name__)


class GoatCoreSceneDatasetLoader(BaseSceneDatasetLoader):
    """Dataloader for Goat-Core benchmark scene format.

    Dataset structure:
    - dataset/{scene_id}/
        - images/img{XXXX}.png       # RGB images (1-indexed)
        - depth/img{XXXX}.npy        # Depth maps (float32, meters)
        - local_pos.txt              # frame_idx qw qx qy qz tx ty tz
        - sparse/0/cameras.txt       # COLMAP PINHOLE format
    """

    def __init__(self, dataset_path: str, max_image_size: int = 640):
        """Initialize GoatCore scene dataloader.

        Args:
            dataset_path: Path to a single scene directory (e.g., data/Goat-core/dataset/nfv)
            max_image_size: Maximum image dimension for downsampling
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.max_image_size = max_image_size

        self.image_dir = self.dataset_path / "images"
        self.depth_dir = self.dataset_path / "depth"

        if not self.image_dir.exists():
            raise ValueError(f"Images directory not found: {self.image_dir}")

        self.intrinsics = self._load_intrinsics()
        self._pose_map = self._load_poses()

        self._rgb_path_map: dict[int, Path] = {}
        for f in self.image_dir.glob("img*.*"):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                fid = int(f.stem.replace("img", ""))
                self._rgb_path_map[fid] = f

        self.frame_ids = sorted(self._rgb_path_map.keys())

        if not self.frame_ids:
            raise ValueError(f"No images found in {self.image_dir}")

        logger.info("Loaded %d frames from %s", len(self.frame_ids), self.dataset_path.name)

    def _get_rgb_path(self, frame_id: int) -> Path:
        return self._rgb_path_map[frame_id]

    def _load_intrinsics(self) -> CameraIntrinsics:
        """Load camera intrinsics from COLMAP cameras.txt."""
        cam_file = self.dataset_path / "sparse" / "0" / "cameras.txt"
        if not cam_file.exists():
            raise ValueError(f"cameras.txt not found: {cam_file}")

        with open(cam_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                model_type = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                if model_type == "PINHOLE":
                    return CameraIntrinsics(
                        width=width, height=height,
                        fx=float(parts[4]), fy=float(parts[5]),
                        cx=float(parts[6]), cy=float(parts[7])
                    )
                elif model_type == "SIMPLE_PINHOLE":
                    f_val = float(parts[4])
                    return CameraIntrinsics(
                        width=width, height=height,
                        fx=f_val, fy=f_val,
                        cx=float(parts[5]), cy=float(parts[6])
                    )

        raise ValueError(f"No supported camera model found in {cam_file}")

    def _load_poses(self) -> dict[int, np.ndarray]:
        """Load camera-to-world poses from local_pos.txt.

        Format: frame_idx qw qx qy qz tx ty tz
        Quaternion order in file is wxyz; scipy expects xyzw.
        Returns dict mapping frame_idx -> 4x4 camera-to-world matrix.
        """
        pos_file = self.dataset_path / "local_pos.txt"
        if not pos_file.exists():
            raise ValueError(f"local_pos.txt not found: {pos_file}")

        poses = {}
        with open(pos_file, 'r') as f:
            for line in f:
                parts = [float(x) for x in line.split()]
                idx = int(parts[0])
                qw, qx, qy, qz = parts[1:5]
                tx, ty, tz = parts[5:8]
                poses[idx] = _quat_translation_to_matrix(qw, qx, qy, qz, tx, ty, tz)

        return poses

    def load_pose(self, frame_id: int) -> np.ndarray:
        if frame_id not in self._pose_map:
            raise ValueError(f"No pose for frame {frame_id}")
        return self._pose_map[frame_id]

    def _load_depth_from_disk(self, frame_id: int) -> np.ndarray | None:
        """Load depth (.npy or .npz) as float32 meters."""
        npy_file = self.depth_dir / f"img{frame_id:04d}.npy"
        if npy_file.exists():
            return np.load(str(npy_file)).astype(np.float32)
        npz_file = self.depth_dir / f"img{frame_id:04d}.npz"
        if npz_file.exists():
            return np.load(str(npz_file))["depth"].astype(np.float32)
        return None


class GoatCoreGroundTruthLoader:
    """Goat-Core ground truth dataset loader.

    Reads the Goat-Core ground truth directory structure::

        Goat-core/
        +-- groundtruth/{scene}/{episode}/{object}/
        |       task_type.txt   # "language", "object", or "image"
        |       pos.txt         # [x, y, z] per line (one or more goals)
        |       language.txt    # text query (for language/object tasks)
        |       *.png           # image query (for image tasks)
        +-- dataset/{scene}/    # scene data (images, depth, poses)
    """

    def __init__(self, goatcore_root: str = "data/Goat-core"):
        self.root_dir = Path(goatcore_root)
        self.gt_root = self.root_dir / "groundtruth"
        self.scenes_root = self.root_dir / "dataset"
        self.samples: list[dict] = []
        self._scan_dataset()

    def _scan_dataset(self):
        if not self.gt_root.exists():
            raise ValueError(f"Ground truth directory not found: {self.gt_root}")

        for scene_gt_path in sorted(self.gt_root.iterdir()):
            if not scene_gt_path.is_dir():
                continue
            scene = scene_gt_path.name
            scene_path = self.scenes_root / scene

            for ep_id in range(6):
                ep_path = scene_gt_path / str(ep_id)
                if not ep_path.exists():
                    continue

                for obj_dir in sorted(ep_path.iterdir()):
                    if not obj_dir.is_dir():
                        continue
                    obj_name = obj_dir.name

                    task_type_file = obj_dir / "task_type.txt"
                    if not task_type_file.exists():
                        continue
                    task_type = task_type_file.read_text().strip()

                    pos_file = obj_dir / "pos.txt"
                    if not pos_file.exists():
                        continue
                    goals = []
                    for line in pos_file.read_text().splitlines():
                        if line.strip():
                            clean = line.replace('[', '').replace(']', '').replace(',', ' ')
                            goals.append([float(x) for x in clean.split()])

                    if not goals:
                        continue

                    query = None
                    text_query = None
                    if task_type == "image":
                        img_files = sorted(obj_dir.glob("*.png"))
                        if img_files:
                            query = str(img_files[0])
                        else:
                            continue
                        lang_file = obj_dir / "language.txt"
                        if lang_file.exists():
                            text_query = lang_file.read_text().strip()
                        else:
                            text_query = re.sub(r'\d+$', '', obj_name).strip('_').replace('_', ' ')
                    else:
                        lang_file = obj_dir / "language.txt"
                        if lang_file.exists():
                            query = lang_file.read_text().strip()
                        else:
                            continue
                        text_query = query

                    self.samples.append({
                        'scene': scene,
                        'episode': ep_id,
                        'target_name': obj_name,
                        'task_type': task_type,
                        'goals': goals,
                        'query': query,
                        'text_query': text_query,
                        'scene_path': str(scene_path),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_scenes(self) -> list[str]:
        return sorted({s['scene'] for s in self.samples})

    def filter_by_scene(self, scene_id: str) -> list[dict]:
        return [s for s in self.samples if s['scene'] == scene_id]
