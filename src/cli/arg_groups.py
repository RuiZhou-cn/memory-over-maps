"""Shared argparse argument groups for eval scripts.

All pipeline parameters (retrieval model, spatial fusion thresholds, navigation
params, etc.) are set in YAML config files. CLI args are limited to:
- Data selection: --config, --scene, --episode, --split, --data-root, --goatcore-root, --ovon-root
- Hardware: --device, --output
- Pipeline toggles: --no-vlm, --model
"""

import argparse


def add_hardware_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Hardware / Output")
    g.add_argument("--device", type=str, default="cuda",
                   help="Device for models (default: cuda)")
    g.add_argument("--output", type=str, default=None,
                   help="Output JSON path")


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add pipeline toggle flags shared across all eval scripts."""
    g = parser.add_argument_group("Pipeline")
    g.add_argument("--no-vlm", action="store_true", default=False,
                   help="Disable VLM re-ranking (Stage 2). "
                        "Run feature-only retrieval (Stage 1 only).")


def add_goatcore_data_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Goat-Core data")
    g.add_argument("--config", type=str, default="configs/goatcore.yaml",
                   help="YAML config file (all pipeline params)")
    g.add_argument("--goatcore-root", type=str, default=None,
                   help="Path to Goat-core root directory")
    g.add_argument("--scene", type=str, default=None,
                   help="Specific scene ID (e.g., 'nfv'). Default: all scenes")


_HABITAT_PRESETS = {
    "hm3d": dict(
        group="HM3D data",
        config="configs/hm3d.yaml",
        data_root_help="Path to hm3d root (with scenes/, meshes/, and episodes/)",
        scene_example="TEEsavR23oF",
        splits=["val", "train"],
    ),
    "mp3d": dict(
        group="MP3D data",
        config="configs/mp3d.yaml",
        data_root_help="Path to mp3d root (with {scene}/ dirs + objectnav_mp3d_v1/)",
        scene_example="8194nk5LbLH",
        splits=["val", "train", "val_mini"],
    ),
    "hm3d_ovon": dict(
        group="HM3D-OVON data",
        config="configs/hm3d_ovon.yaml",
        data_root_help="Path to hm3d root (meshes + scenes)",
        scene_example="TEEsavR23oF",
        splits=["val_seen", "val_seen_synonyms", "val_unseen"],
        split_help="OVON episode split to evaluate (default: all splits)",
        extra_args=lambda g: g.add_argument(
            "--ovon-root", type=str, default=None,
            help="Path to OVON episode data (with hm3d/{split}/ subdirs)"),
    ),
}


def add_habitat_data_args(parser: argparse.ArgumentParser, preset: str, *,
                          include_episode: bool = False) -> None:
    p = _HABITAT_PRESETS[preset]
    g = parser.add_argument_group(p["group"])
    g.add_argument("--config", type=str, default=p["config"],
                   help="YAML config file (all pipeline params)")
    g.add_argument("--data-root", type=str, default=None,
                   help=p["data_root_help"])
    if "extra_args" in p:
        p["extra_args"](g)
    g.add_argument("--scene", type=str, default=None,
                   help=f"Specific scene ID (e.g., '{p['scene_example']}'). Default: all scenes")
    g.add_argument("--split", type=str, default=None,
                   choices=p["splits"],
                   help=p.get("split_help", "Episode split to evaluate"))
    if include_episode:
        g.add_argument("--episode", type=int, nargs="+", default=None,
                       help="Episode IDs to evaluate (e.g., --episode 0 10 13). Default: all")
