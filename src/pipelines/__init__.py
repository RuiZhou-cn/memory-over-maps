"""Pipeline orchestration aligned with paper steps.

Step 1+2: Zero-shot retrieval + VLM re-ranking (retrieval.py)
Step 3:   3D localization — SAM3 + depth + projection (localization.py)
Step 4:   Navigation — goal determination + episode processing (navigation.py)
"""

from src.pipelines.localization import localize
from src.pipelines.retrieval import search_scene

__all__ = [
    "localize",
    "search_scene",
]
