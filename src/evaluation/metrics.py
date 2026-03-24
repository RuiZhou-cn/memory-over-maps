"""Metrics accumulators for localization and navigation benchmarks.

Two lightweight classes that replace duplicated dict + lambda patterns
across eval scripts:

- ``LocMetricsAccumulator``: SR@K for localization (GoatCore loc)
  Uses 2D navigation-plane distance (XZ in Habitat's Y-up frame).
- ``NavMetricsAccumulator``: Success/SPL/SoftSPL/DTG/Steps (HM3D/MP3D/OVON ObjectNav)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.utils.geometry import xz_dist


class LocMetricsAccumulator:
    """Accumulate SR@K metrics for localization benchmarks.

    Distance is computed on the navigation plane (XZ) — the Y (height)
    component is ignored so that predictions on different floors are not
    penalised for vertical offset alone.

    Usage::

        acc = LocMetricsAccumulator(k_values=[1, 3, 5], threshold=1.5)
        acc.update(preds, gt_goals, scene="TEEsavR23oF", category="chair")
        acc.print_scene_category_table()
        j = acc.to_json()
    """

    def __init__(self, k_values: Sequence[int], threshold: float = 1.5):
        self.k_values = sorted(k_values)
        self.threshold = threshold
        def _init():
            return {k: {"success": 0, "total": 0} for k in self.k_values}
        self._init = _init
        self.overall: Dict = _init()
        self.by_scene: Dict[str, Dict] = defaultdict(_init)
        self.by_category: Dict[str, Dict] = defaultdict(_init)
        # scene × category cross product for paper-style tables
        self.by_scene_category: Dict[str, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(self._init)
        )

    def update(
        self,
        predictions: List,
        gt_goals: List,
        scene: str = "",
        category: str = "",
    ) -> Dict[int, bool]:
        """Record one episode.

        Args:
            predictions: List of [x, y, z] predictions (ordered by rank).
            gt_goals: List of [x, y, z] ground-truth goal positions.
            scene: Scene identifier (for per-scene breakdown).
            category: Object category (for per-category breakdown).

        Returns:
            Dict mapping K → bool (success at K).
        """
        results = {}
        for k in self.k_values:
            top_k = predictions[:k]
            success = False
            for pred in top_k:
                pred_arr = np.asarray(pred)
                min_dist = min(
                    xz_dist(pred_arr, np.asarray(g))
                    for g in gt_goals
                )
                if min_dist <= self.threshold:
                    success = True
                    break
            results[k] = success
            buckets = [self.overall, self.by_scene[scene], self.by_category[category]]
            if scene and category:
                buckets.append(self.by_scene_category[scene][category])
            for bucket in buckets:
                bucket[k]["total"] += 1
                if success:
                    bucket[k]["success"] += 1
        return results

    def _fmt_row(self, label: str, buckets: Dict) -> str:
        parts = [f"{label:>18}"]
        n = buckets[self.k_values[0]]["total"]
        parts.append(f"{n:>5}")
        for k in self.k_values:
            s = buckets[k]
            rate = s["success"] / s["total"] * 100 if s["total"] > 0 else 0.0
            parts.append(f"{rate:>6.1f}%")
        return "  ".join(parts)

    def print_scene_category_table(
        self,
        title: str = "3D Localization Results",
        category_order: Optional[Sequence[str]] = None,
    ):
        """Print paper-style table: per-scene rows with per-category columns.

        Columns: Scene | Eps | Average | <category1> | <category2> | ...
        Rows: one per scene + Total row at the bottom.
        Uses the first (and typically only) K value for SR@K.
        """
        k = self.k_values[0]
        if category_order is None:
            category_order = sorted(self.by_category.keys())

        def _sr(bucket: Dict) -> str:
            s = bucket.get(k, {"success": 0, "total": 0})
            if s["total"] == 0:
                return "   -  "
            return f"{s['success'] / s['total'] * 100:>5.1f}%"

        def _sr_val(bucket: Dict) -> Optional[float]:
            s = bucket.get(k, {"success": 0, "total": 0})
            if s["total"] == 0:
                return None
            return s["success"] / s["total"] * 100

        cat_labels = [c.capitalize() for c in category_order]
        hdr = f"{'Scene':>12}  {'Eps':>4}  {'Average':>8}"
        for cl in cat_labels:
            hdr += f"  {cl:>8}"

        print(f"\n{'=' * (len(hdr) + 4)}")
        print(f"  {title} (SR@{k}, threshold={self.threshold}m)")
        print(f"{'=' * (len(hdr) + 4)}")
        print(f"\n{hdr}")
        print("-" * (len(hdr) + 4))

        scenes = sorted(self.by_scene.keys())
        for scene in scenes:
            n = self.by_scene[scene][k]["total"]
            cat_srs = []
            for cat in category_order:
                val = _sr_val(self.by_scene_category[scene].get(cat, {}))
                cat_srs.append(val)
            valid = [v for v in cat_srs if v is not None]
            avg = sum(valid) / len(valid) if valid else None
            avg_str = f"{avg:>5.1f}%" if avg is not None else "   -  "

            row = f"{scene:>12}  {n:>4}  {avg_str:>8}"
            for val in cat_srs:
                row += f"  {f'{val:>5.1f}%' if val is not None else '   -  ':>8}"
            print(row)

        print("-" * (len(hdr) + 4))
        n_total = self.overall[k]["total"]
        cat_srs_total = []
        for cat in category_order:
            cat_srs_total.append(_sr_val(self.by_category.get(cat, {})))
        valid_total = [v for v in cat_srs_total if v is not None]
        avg_total = sum(valid_total) / len(valid_total) if valid_total else None
        avg_total_str = f"{avg_total:>5.1f}%" if avg_total is not None else "   -  "

        row = f"{'Total':>12}  {n_total:>4}  {avg_total_str:>8}"
        for val in cat_srs_total:
            row += f"  {f'{val:>5.1f}%' if val is not None else '   -  ':>8}"
        print(row)
        print("=" * (len(hdr) + 4))

    def to_json(self) -> Dict:
        def _bucket_json(buckets):
            return {
                k: {
                    "total": buckets[k]["total"],
                    "success": buckets[k]["success"],
                    "sr": (buckets[k]["success"] / buckets[k]["total"] * 100
                           if buckets[k]["total"] > 0 else 0.0),
                }
                for k in self.k_values
            }
        by_scene_category = {}
        for scene, cats in self.by_scene_category.items():
            by_scene_category[scene] = {c: _bucket_json(d) for c, d in cats.items()}
        return {
            "threshold": self.threshold,
            "k_values": self.k_values,
            "overall": _bucket_json(self.overall),
            "by_scene": {s: _bucket_json(d) for s, d in self.by_scene.items()},
            "by_category": {c: _bucket_json(d) for c, d in self.by_category.items()},
            "by_scene_category": by_scene_category,
        }


class NavMetricsAccumulator:
    """Accumulate ObjectNav metrics (Success, SPL, SoftSPL, DTG, Steps).

    Usage::

        acc = NavMetricsAccumulator()
        acc.update(success=1.0, spl=0.8, soft_spl=0.7, dtg=0.05, steps=120,
                   scene="TEEsavR23oF", category="chair")
        acc.print_table()
        j = acc.to_json()
    """

    def __init__(self):
        def _init():
            return {"success": 0.0, "spl": 0.0, "soft_spl": 0.0,
                    "dtg": 0.0, "steps": 0, "total": 0}
        self.overall: Dict = _init()
        self.by_scene: Dict[str, Dict] = defaultdict(_init)
        self.by_category: Dict[str, Dict] = defaultdict(_init)

    def update(
        self,
        success: float,
        spl: float,
        soft_spl: float,
        dtg: float,
        steps: int,
        scene: str = "",
        category: str = "",
    ):
        for s in [self.overall, self.by_scene[scene], self.by_category[category]]:
            s["total"] += 1
            s["success"] += success
            s["spl"] += spl
            s["soft_spl"] += soft_spl
            s["dtg"] += dtg
            s["steps"] += steps

    def merge(self, other: "NavMetricsAccumulator"):
        """Merge another accumulator's counts into this one."""
        keys = ("success", "spl", "soft_spl", "dtg", "steps", "total")
        for key in keys:
            self.overall[key] += other.overall[key]
        for scene, vals in other.by_scene.items():
            for key in keys:
                self.by_scene[scene][key] += vals[key]
        for cat, vals in other.by_category.items():
            for key in keys:
                self.by_category[cat][key] += vals[key]

    @staticmethod
    def _fmt(s: Dict) -> str:
        n = s["total"]
        if n == 0:
            return "  -"
        sr = s["success"] / n * 100
        spl = s["spl"] / n
        sspl = s["soft_spl"] / n
        dtg = s["dtg"] / n
        steps = s["steps"] / n
        return (f"{n:>4}  {sr:>5.1f}%  {spl:>6.3f}  {sspl:>6.3f}  "
                f"{dtg:>5.2f}  {steps:>6.1f}")

    def print_table(
        self,
        success_distance: float = 1.0,
        distance_to: str = "POINT",
        title: str = "ObjectNav Results",
    ):
        hdr = (f"{'':>18}  {'Eps':>4}  {'SR':>6}  {'SPL':>6}  "
               f"{'SSPL':>6}  {'DTG':>5}  {'Steps':>6}")
        target = "object center" if distance_to == "POINT" else "nearest viewpoint"

        print(f"\n{'=' * 72}")
        print(f"  {title} (habitat.Env built-in metrics)")
        print(f"  Success: STOP + geodesic < {success_distance}m to {target}")
        print(f"{'=' * 72}")

        print(f"\n{hdr}")
        print("-" * 72)
        print(f"{'OVERALL':>18}  {self._fmt(self.overall)}")

        if len(self.by_scene) > 1:
            print("\n  Per-Scene:")
            print(f"{hdr}")
            print("-" * 72)
            for scene in sorted(self.by_scene):
                print(f"{scene:>18}  {self._fmt(self.by_scene[scene])}")

        print("\n  Per-Category:")
        print(f"{hdr}")
        print("-" * 72)
        for cat in sorted(self.by_category):
            print(f"{cat:>18}  {self._fmt(self.by_category[cat])}")

        print("=" * 72)

    def to_json(self) -> Dict:
        def _bucket_json(s):
            n = s["total"]
            if n == 0:
                return {"total": 0}
            return {
                "total": n,
                "success_rate": s["success"] / n * 100,
                "spl": s["spl"] / n,
                "soft_spl": s["soft_spl"] / n,
                "avg_dtg": s["dtg"] / n,
                "avg_steps": s["steps"] / n,
            }
        return {
            "overall": _bucket_json(self.overall),
            "by_scene": {s: _bucket_json(d) for s, d in self.by_scene.items()},
            "by_category": {c: _bucket_json(d) for c, d in self.by_category.items()},
        }
