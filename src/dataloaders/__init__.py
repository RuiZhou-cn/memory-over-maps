"""Dataset loaders for all supported scene formats.

Provides ``BaseSceneDatasetLoader`` (ABC) and concrete loaders:
- ``goatcore`` -- GoatCoreSceneDatasetLoader, GoatCoreGroundTruthLoader
- ``hm3d`` -- HM3DSceneDatasetLoader
- ``mp3d`` -- MP3DSceneDatasetLoader
- ``custom`` -- CustomSceneDatasetLoader
- ``sunrgbd`` -- SUN RGB-D utilities
"""

from .base import (
    BaseSceneDatasetLoader,
    CameraIntrinsics,
    LazyImageList,
    get_dataloader,
)
from .goatcore import (
    GoatCoreGroundTruthLoader,
    GoatCoreSceneDatasetLoader,
)
from .hm3d import HM3DSceneDatasetLoader
from .mp3d import MP3DSceneDatasetLoader

__all__ = [
    'CameraIntrinsics',
    'BaseSceneDatasetLoader',
    'LazyImageList',
    'get_dataloader',
    'GoatCoreSceneDatasetLoader',
    'GoatCoreGroundTruthLoader',
    'HM3DSceneDatasetLoader',
    'MP3DSceneDatasetLoader',
]
