"""OVON-v1 dataset type for habitat.

Registers the ``OVON-v1`` dataset with habitat's registry so that
``habitat.Env`` can load HM3D-OVON episodes (379 open-vocabulary categories).

Import this module before creating ``habitat.Env``::

    import src.dataloaders.ovon  # noqa — registers OVON-v1

Based on goat-bench/goat_bench/dataset/ovon_dataset.py (minimal copy).

Key differences from standard ObjectNavDatasetV1:
- No ``category_to_task_category_id`` assertions
- Accepts ``OVONEpisode`` with ``children_object_categories``
- Handles ``goals_by_category`` resolution for empty ``goals: []``
- ``OVONObjectViewLocation`` with optional ``radius``
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class OVONObjectViewLocation(ObjectViewLocation):
    """View location with optional radius (OVON extension)."""
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class OVONEpisode(ObjectGoalNavEpisode):
    """OVON episode with children_object_categories field."""
    children_object_categories: Optional[List[str]] = []

    @property
    def goals_key(self) -> str:
        """Key to retrieve goals from goals_by_category."""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_dataset(name="OVON-v1")
class OVONDatasetV1(PointNavDatasetV1):
    """Open-Vocabulary Object Navigation dataset (379 categories).

    Loads HM3D-OVON episodes where ``goals: []`` is empty and goals
    are resolved from ``goals_by_category`` at the file level.
    """

    episodes: List[OVONEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Extract goals_by_category from episodes (for files without it)."""
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            scene_base = os.path.basename(ep["scene_id"])
            goals_key = f"{scene_base}_{ep['object_category']}"
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep["goals"]

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category
        return dataset

    def to_json(self) -> str:
        for ep in self.episodes:
            ep.goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for ep in self.episodes:
            goals = self.goals_by_category[ep.goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            ep.goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @property
    def category_to_task_category_id(self) -> Dict[str, int]:
        """Map category name → stable integer id for ObjectGoalSensor.

        Required by ``ObjectGoalSensor`` when ``goal_spec="TASK_CATEGORY_ID"``.
        Builds the mapping on demand from the object_category of all episodes.
        """
        categories = sorted(
            {ep.object_category for ep in self.episodes if ep.object_category}
        )
        return {cat: idx for idx, cat in enumerate(categories)}


    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = OVONObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = OVONEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            if not episode.goals:
                goals_key = episode.goals_key
                if goals_key in self.goals_by_category:
                    episode.goals = self.goals_by_category[goals_key]

            self.episodes.append(episode)  # type: ignore [attr-defined]
