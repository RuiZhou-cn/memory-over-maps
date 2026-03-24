"""Evaluation helpers: episode filtering, grouping, stderr suppression, cleanup."""

from __future__ import annotations

import gc
import os
from collections import defaultdict

from collections.abc import Callable, Sequence


def print_episode_summary(episodes):
    """Print category summary for a list of episodes."""
    categories = sorted({ep.object_category for ep in episodes})
    if len(categories) <= 10:
        print(f"  Categories: {len(categories)} ({', '.join(categories)})")
    else:
        print(f"  Categories: {len(categories)} ({', '.join(categories[:5])}...)")


def suppress_stderr(fn, *args, **kwargs):
    """Call *fn* with C++ stderr suppressed (Magnum/PluginManager noise)."""
    fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        return fn(*args, **kwargs)
    finally:
        os.dup2(fd, 2)
        os.close(devnull)
        os.close(fd)


def quiet_reset(env):
    """``env.reset()`` with C++ stderr suppressed (scene load warnings)."""
    return suppress_stderr(env.reset)


def import_habitat():
    """Import habitat with C++ stderr suppressed."""
    def _do_import():
        from habitat import Env
        from habitat.config import get_config
        return get_config, Env
    return suppress_stderr(_do_import)


def filter_episodes(
    all_episodes: list,
    scene: str | None = None,
    episode_ids: Sequence[int] | None = None,
    floor_filter: bool = True,
    floor_threshold: float = 0.5,
) -> tuple[list, int]:
    """Filter episodes by scene, IDs, and floor.

    Args:
        all_episodes: Full episode list from env.episodes.
        scene: Scene name substring to filter by (e.g. "TEEsavR23oF").
        episode_ids: Specific episode IDs to keep.
        floor_filter: Whether to apply viewpoint floor filter.
        floor_threshold: Height difference threshold in meters.

    Returns:
        (filtered_episodes, skipped_cross_floor_count)
    """
    eps = list(all_episodes)

    if scene:
        eps = [ep for ep in eps if scene in ep.scene_id]

    if episode_ids is not None:
        ep_id_set = {str(e) for e in episode_ids}
        eps = [ep for ep in eps if ep.episode_id in ep_id_set]

    skipped = 0
    if floor_filter:
        filtered = []
        for ep in eps:
            start_y = ep.start_position[1]
            has_same_floor_vp = False
            for goal in ep.goals:
                if hasattr(goal, "view_points") and goal.view_points:
                    for vp in goal.view_points:
                        if abs(vp.agent_state.position[1] - start_y) <= floor_threshold:
                            has_same_floor_vp = True
                            break
                if has_same_floor_vp:
                    break
            if has_same_floor_vp:
                filtered.append(ep)
            else:
                skipped += 1
        eps = filtered

    return eps, skipped


def group_by_scene(
    episodes: list,
    get_scene_name_fn: Callable[[str], str],
) -> dict[str, list]:
    """Group episodes by scene name, sorted by (episode_id, category).

    Args:
        episodes: Filtered episode list.
        get_scene_name_fn: Callable that extracts scene name from scene_id
            string (e.g. ``get_scene_name``).

    Returns:
        Dict mapping scene_name -> sorted list of episodes.
    """
    by_scene: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_scene[get_scene_name_fn(ep.scene_id)].append(ep)
    for sn in by_scene:
        by_scene[sn].sort(key=lambda e: (int(e.episode_id), e.object_category))
    return dict(by_scene)


def cleanup(
    scene_loader=None,
    sam3_segmenter=None,
    images=None,
    episode_idx: int | None = None,
    interval: int = 20,
) -> None:
    """Flush caches + gc + CUDA.

    When *episode_idx* is given, only runs every *interval* episodes
    (prevents OOM on scenes with many episodes).  Without *episode_idx*,
    always runs (use at scene boundaries / end of eval).

    Clears (when provided):
    - ``scene_loader`` depth disk-read cache
    - ``sam3_segmenter`` mask cache
    - ``images`` (LazyImageList) RGB cache
    """
    if episode_idx is not None and (episode_idx + 1) % interval != 0:
        return

    if scene_loader is not None:
        scene_loader.clear_depth_cache()
    if sam3_segmenter is not None:
        sam3_segmenter.clear_cache()
    if images is not None and hasattr(images, 'clear'):
        images.clear()

    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # non-Linux or musl libc
