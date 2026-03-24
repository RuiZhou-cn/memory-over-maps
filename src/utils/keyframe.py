"""Keyframe selection based on pose-based rotation/translation thresholds.

Paper: Sec IV-E (Table IV) — memory-accuracy trade-off and efficiency.
"""

from __future__ import annotations

import numpy as np


class KeyframeManager:
    """Pose-based keyframe selector.

    Selects diverse viewpoints by enforcing minimum rotation and translation
    changes between consecutive keyframes.

    Example:
        manager = KeyframeManager(
            rotation_threshold_deg=30.0,
            translation_threshold_m=1.0,
        )
        keyframe_ids = manager.select_keyframes(poses, frame_ids)
    """

    def __init__(
        self,
        rotation_threshold_deg: float = 30.0,
        translation_threshold_m: float = 1.0,
        min_frames_between: int = 5,
    ):
        self.rotation_threshold_deg = rotation_threshold_deg
        self.translation_threshold_m = translation_threshold_m
        self.min_frames_between = min_frames_between

    def select_keyframes(
        self,
        poses: np.ndarray,
        frame_ids: list[int],
        max_keyframes: int | None = None,
    ) -> list[int]:
        """Select keyframes from a sequence of poses.

        Uses rotation and translation thresholds to select diverse viewpoints.

        Args:
            poses: (N, 4, 4) array of camera poses
            frame_ids: Corresponding frame IDs
            max_keyframes: Maximum number of keyframes to select (None = no limit)

        Returns:
            List of selected keyframe frame IDs.
        """
        if len(poses) == 0:
            return []

        keyframe_indices = [0]

        last_R = poses[0, :3, :3].copy()
        last_t = poses[0, :3, 3].copy()
        last_keyframe_idx = 0

        trans_threshold_sq = self.translation_threshold_m ** 2
        cos_rot_threshold = np.cos(np.radians(self.rotation_threshold_deg))

        for i in range(1, len(poses)):
            if i - last_keyframe_idx < self.min_frames_between:
                continue

            curr_t = poses[i, :3, 3]
            trans_diff_sq = np.sum((curr_t - last_t) ** 2)

            if trans_diff_sq >= trans_threshold_sq:
                keyframe_indices.append(i)
                last_R = poses[i, :3, :3].copy()
                last_t = curr_t.copy()
                last_keyframe_idx = i

                if max_keyframes and len(keyframe_indices) >= max_keyframes:
                    break
                continue

            curr_R = poses[i, :3, :3]
            trace = np.sum(curr_R * last_R)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)

            if cos_angle <= cos_rot_threshold:
                keyframe_indices.append(i)
                last_R = curr_R.copy()
                last_t = curr_t.copy()
                last_keyframe_idx = i

                if max_keyframes and len(keyframe_indices) >= max_keyframes:
                    break

        return [frame_ids[i] for i in keyframe_indices]
