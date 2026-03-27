#!/usr/bin/env python3
"""Unified dataset preparation script for HM3D, OVON, and MP3D.

Relocates files from the habitat-sim download layout into the project's
expected directory structure, then rewrites episode JSON files for
habitat-sim compatibility (e.g. stripping .basis.glb suffixes, fixing
scene_id prefixes).

All operations are idempotent and create .bak backups before rewriting files.

Usage:
    python scripts/prepare_data.py                  # All datasets
    python scripts/prepare_data.py --dataset hm3d   # Just one
"""

import argparse
import gzip
import json
import re
import shutil
import tarfile
from pathlib import Path

DATA_DIR = Path("data")

ALL_DATASETS = ["hm3d", "ovon", "mp3d"]


def log(msg: str) -> None:
    print(f"  {msg}")


def rewrite_gzip_json(
    path: Path,
    transform_fn,
    backup: bool = True,
) -> bool:
    """Read a .json.gz, apply transform_fn(data) -> data, write back.

    Returns True if changes were made.
    """
    if not path.exists():
        return False

    with gzip.open(path, "rt") as f:
        data = json.load(f)

    new_data = transform_fn(data)
    if new_data is None:
        return False

    log(f"REWRITE {path}")
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    with gzip.open(path, "wt") as f:
        json.dump(new_data, f)
    return True



def _relocate_hm3d_meshes() -> int:
    """Move meshes from data/versioned_data/hm3d-*/hm3d/val/ to data/hm3d/meshes/.

    Returns number of actions taken.
    """
    hm3d_root = DATA_DIR / "hm3d"
    meshes_dir = hm3d_root / "meshes"

    src_dir = None
    versioned = DATA_DIR / "versioned_data"
    if versioned.exists():
        for d in versioned.iterdir():
            if d.name.startswith("hm3d-"):
                candidate = d / "hm3d" / "val"
                if candidate.is_dir() and any(candidate.iterdir()):
                    src_dir = candidate
                    break

    if src_dir is None:
        return 0

    scene_dirs = sorted(
        d for d in src_dir.iterdir()
        if d.is_dir() and "-" in d.name
    )
    if not scene_dirs:
        return 0

    actions = 0

    if not meshes_dir.exists():
        log(f"MKDIR {meshes_dir}")
        meshes_dir.mkdir(parents=True, exist_ok=True)
        actions += 1

    for scene_d in scene_dirs:
        dst = meshes_dir / scene_d.name
        if dst.exists():
            continue
        log(f"MOVE {scene_d} -> {dst}")
        shutil.move(scene_d, dst)
        actions += 1

    cfg_name = "hm3d_annotated_basis.scene_dataset_config.json"
    dst_cfg = hm3d_root / cfg_name
    if not dst_cfg.exists():
        src_cfg = src_dir.parent / cfg_name
        if src_cfg.exists():
            log(f"COPY {src_cfg} -> {dst_cfg}")
            shutil.copy2(src_cfg, dst_cfg)
            actions += 1

    return actions


def _relocate_hm3d_episodes() -> int:
    """Move episodes from extracted zip layout to data/hm3d/episodes/.

    Detects objectnav_hm3d_v2/ in the project root (created by unzipping
    objectnav_hm3d_v2.zip) and moves its contents to data/hm3d/episodes/.

    Returns number of actions taken.
    """
    hm3d_root = DATA_DIR / "hm3d"
    episodes_dir = hm3d_root / "episodes"

    if episodes_dir.exists() and any(episodes_dir.iterdir()):
        return 0

    src_dir = Path("objectnav_hm3d_v2")
    if not src_dir.exists() or not src_dir.is_dir():
        return 0

    actions = 0

    if not episodes_dir.exists():
        log(f"MKDIR {episodes_dir}")
        episodes_dir.mkdir(parents=True, exist_ok=True)
        actions += 1

    for item in sorted(src_dir.iterdir()):
        dst = episodes_dir / item.name
        if dst.exists():
            continue
        log(f"MOVE {item} -> {dst}")
        shutil.move(item, dst)
        actions += 1

    if src_dir.exists() and not any(src_dir.iterdir()):
        log(f"RMDIR {src_dir}")
        src_dir.rmdir()
        actions += 1

    return actions


def _cleanup_hm3d_download_dirs() -> int:
    """Remove habitat-sim download artifacts after relocation.

    Removes data/scene_datasets/ (symlink created by habitat download)
    and data/versioned_data/ (source dir, now empty after mesh relocation).

    Returns number of actions taken.
    """
    actions = 0

    # Remove symlink: data/scene_datasets/hm3d -> versioned_data/...
    symlink = DATA_DIR / "scene_datasets"
    if symlink.exists():
        if symlink.is_symlink() or (symlink.is_dir() and not any(
            p for p in symlink.rglob("*") if p.is_file()
        )):
            log(f"RMTREE {symlink}")
            if symlink.is_symlink():
                symlink.unlink()
            else:
                shutil.rmtree(symlink)
            actions += 1

    # Remove versioned_data/ once meshes have been relocated to data/hm3d/meshes/
    versioned = DATA_DIR / "versioned_data"
    if versioned.exists():
        meshes_dir = DATA_DIR / "hm3d" / "meshes"
        if meshes_dir.exists() and any(meshes_dir.iterdir()):
            log(f"RMTREE {versioned}")
            shutil.rmtree(versioned)
            actions += 1

    return actions


def prepare_hm3d() -> int:
    """Prepare HM3D dataset. Returns number of actions taken."""
    print("\n--- HM3D ---")
    actions = 0

    hm3d_root = DATA_DIR / "hm3d"

    if not hm3d_root.exists():
        has_download = (
            (DATA_DIR / "versioned_data").exists()
            and any(
                (DATA_DIR / "versioned_data" / d / "hm3d" / "val").exists()
                for d in (DATA_DIR / "versioned_data").iterdir()
            )
        )
        has_episodes = Path("objectnav_hm3d_v2").exists()
        if has_download or has_episodes:
            log(f"MKDIR {hm3d_root}")
            hm3d_root.mkdir(parents=True, exist_ok=True)
            actions += 1
        else:
            log("SKIP: data/hm3d not found and no habitat download detected")
            return 0

    actions += _relocate_hm3d_meshes()
    actions += _relocate_hm3d_episodes()

    episodes_dir = hm3d_root / "episodes"
    if episodes_dir.exists():
        for gz_file in sorted(episodes_dir.rglob("*.json.gz")):
            changed = rewrite_gzip_json(gz_file, _strip_basis_glb)
            if changed:
                actions += 1
    else:
        log("No episodes/ dir found, skipping scene_id rewrite")

    scene_cfg = hm3d_root / "hm3d_annotated_basis.scene_dataset_config.json"
    if scene_cfg.exists():
        changed = _rewrite_scene_dataset_config(scene_cfg)
        if changed:
            actions += 1

    # Clean up habitat download artifacts (symlink + source dirs)
    actions += _cleanup_hm3d_download_dirs()

    if actions == 0:
        log("Already up to date")
    return actions


def _rewrite_scene_dataset_config(cfg_path: Path) -> bool:
    """Rewrite glob patterns to match the meshes/ directory layout.

    Rewrites split-relative globs (e.g. val/00800-TEEsavR23oF/*.basis.glb)
    to meshes-relative explicit paths (meshes/00800-.../TEEsavR23oF.basis.glb).
    Also removes *.basis.scene_instance.json patterns (files don't exist).
    """
    with open(cfg_path) as f:
        data = json.load(f)

    changed = False

    def _fix_paths(path_list):
        nonlocal changed
        new_list = []
        for p in path_list:
            if ".basis.scene_instance.json" in p:
                # These files don't exist; drop the entry
                changed = True
                continue
            if ".basis.glb" in p:
                # e.g. "val/00800-TEEsavR23oF/*.basis.glb"
                # -> "meshes/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
                m = re.match(r"[^/]+/(\d+-(\w+))/\*\.basis\.glb", p)
                if m:
                    new_list.append(f"meshes/{m.group(1)}/{m.group(2)}.basis.glb")
                    changed = True
                    continue
            new_list.append(p)
        return new_list

    for ext, paths in data.get("stages", {}).get("paths", {}).items():
        data["stages"]["paths"][ext] = _fix_paths(paths)

    for ext, paths in data.get("scene_instances", {}).get("paths", {}).items():
        data["scene_instances"]["paths"][ext] = _fix_paths(paths)

    if not changed:
        return False

    log(f"REWRITE {cfg_path}")
    bak = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def _strip_basis_glb(data: dict) -> dict:
    """Normalize scene_id paths and scene_dataset_config in episode data.

    Strips hm3d*/split/ prefix so scene_ids become e.g.
    '00800-TEEsavR23oF/TEEsavR23oF.basis.glb' (matching meshes/ layout).
    Also rewrites scene_dataset_config to point to data/hm3d/.
    """
    changed = False
    for ep in data.get("episodes", []):
        scene_id = ep.get("scene_id", "")
        new_id = scene_id

        # Strip version/split prefix: "hm3d_v0.2/val/00800-..." -> "00800-..."
        new_id = re.sub(r"^hm3d[^/]*/[^/]+/", "", new_id)

        if new_id != scene_id:
            ep["scene_id"] = new_id
            changed = True

        # Fix scene_dataset_config path to match project layout
        sdc = ep.get("scene_dataset_config", "")
        if sdc and "hm3d_annotated_basis.scene_dataset_config.json" in sdc:
            new_sdc = "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
            if sdc != new_sdc:
                ep["scene_dataset_config"] = new_sdc
                changed = True

    gbc = data.get("goals_by_category", {})
    if gbc:
        new_gbc = {}
        for key, val in gbc.items():
            new_key = re.sub(r"^hm3d[^/]*/[^/]+/", "", key)
            if new_key != key:
                changed = True
            new_gbc[new_key] = val
        if changed:
            data["goals_by_category"] = new_gbc

    return data if changed else None


def prepare_ovon() -> int:
    """Prepare OVON dataset. Returns number of actions taken."""
    print("\n--- OVON ---")
    actions = 0

    ovon_root = DATA_DIR / "hm3d_ovon"
    if not ovon_root.exists():
        log("SKIP: data/hm3d_ovon not found (download data first)")
        return 0

    # Extract hm3d.tar.gz if hm3d/ dir not yet present (HuggingFace clone layout)
    episodes_base = ovon_root / "hm3d"
    tarball = ovon_root / "hm3d.tar.gz"
    if not episodes_base.exists() and tarball.exists():
        log(f"EXTRACT {tarball} -> {ovon_root}/")
        with tarfile.open(tarball) as tf:
            members = [m for m in tf.getmembers() if not m.name.startswith("._") and "/._" not in m.name]
            tf.extractall(ovon_root, members=members)
        tarball.unlink()
        log(f"REMOVE {tarball}")
        actions += 1

    if episodes_base.exists():
        for gz_file in sorted(episodes_base.rglob("*.json.gz")):
            changed = rewrite_gzip_json(gz_file, _fix_ovon_scene_ids)
            if changed:
                actions += 1
    else:
        log("No hm3d/ episode dir found, skipping scene_id rewrite")

    if actions == 0:
        log("Already up to date")
    return actions


def _fix_ovon_scene_ids(data: dict) -> dict:
    """Strip hm3d/{split}/ prefix from OVON scene_ids."""
    changed = False
    for ep in data.get("episodes", []):
        scene_id = ep.get("scene_id", "")
        new_scene_id = scene_id

        # Strip "hm3d/{split}/" prefix (e.g. "hm3d/val//00800-TEEsavR23oF/...")
        if new_scene_id.startswith("hm3d/"):
            parts = new_scene_id.split("/", 2)
            if len(parts) >= 3:
                new_scene_id = parts[2]

        # Strip leading slash (original data has double-slash: "hm3d/val//00800-...")
        new_scene_id = new_scene_id.lstrip("/")

        if new_scene_id != scene_id:
            ep["scene_id"] = new_scene_id
            changed = True

        # Fix scene_dataset_config path to match project layout
        sdc = ep.get("scene_dataset_config", "")
        if sdc and "hm3d_annotated_basis.scene_dataset_config.json" in sdc:
            new_sdc = "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
            if sdc != new_sdc:
                ep["scene_dataset_config"] = new_sdc
                changed = True

    gbc = data.get("goals_by_category", {})
    if gbc:
        new_gbc = {}
        for key, val in gbc.items():
            new_key = key
            new_gbc[new_key] = val
        data["goals_by_category"] = new_gbc

    return data if changed else None


def _relocate_mp3d_meshes() -> int:
    """Move scene dirs from data/mp3d/v1/tasks/mp3d/<scene>/ to data/mp3d/<scene>/.

    The habitat download script places scene files under:
        data/mp3d/v1/tasks/mp3d/<scene_id>/<scene_id>.glb

    habitat-sim (and eval_mp3d.py) expect them at:
        data/mp3d/<scene_id>/<scene_id>.glb

    Leaves data/mp3d/v1/scans/ for the next steps (scan image relocation +
    non-val pruning). Removes the v1/tasks/ tree once empty.

    Returns number of actions taken.
    """
    mp3d_root = DATA_DIR / "mp3d"
    src_dir = mp3d_root / "v1" / "tasks" / "mp3d"

    if not src_dir.exists():
        return 0

    scene_dirs = sorted(d for d in src_dir.iterdir() if d.is_dir())
    if not scene_dirs:
        return 0

    actions = 0
    for scene_d in scene_dirs:
        dst = mp3d_root / scene_d.name
        if dst.exists():
            continue
        log(f"MOVE {scene_d} -> {dst}")
        shutil.move(str(scene_d), str(dst))
        actions += 1

    # Clean up empty v1/tasks/mp3d/ and parents (but leave v1/scans/ alone)
    for stale_dir in [src_dir, src_dir.parent, src_dir.parent.parent]:
        if stale_dir.exists() and not any(stale_dir.iterdir()):
            log(f"RMDIR {stale_dir}")
            stale_dir.rmdir()
            actions += 1

    return actions


_MP3D_SCAN_SUBDIRS = [
    "undistorted_color_images",
    "undistorted_depth_images",
    "matterport_camera_poses",
    "matterport_camera_intrinsics",
]


def _relocate_mp3d_scans(val_scenes: set) -> int:
    """Move image/pose subfolders from v1/scans/<scene>/<scene>/ to data/mp3d/<scene>/.

    The Matterport download extracts to:
        data/mp3d/v1/scans/<scene_id>/<scene_id>/undistorted_color_images/
        data/mp3d/v1/scans/<scene_id>/<scene_id>/matterport_camera_poses/
        ...

    The dataloader (MP3DSceneDatasetLoader) expects:
        data/mp3d/<scene_id>/undistorted_color_images/
        data/mp3d/<scene_id>/matterport_camera_poses/
        ...

    Only processes scenes in val_scenes (skips non-val). Idempotent.

    Returns number of actions taken.
    """
    mp3d_root = DATA_DIR / "mp3d"
    scans_root = mp3d_root / "v1" / "scans"
    if not scans_root.exists():
        return 0

    actions = 0
    for scene_id in sorted(val_scenes):
        inner = scans_root / scene_id / scene_id
        if not inner.is_dir():
            continue  # not extracted yet
        dst_scene = mp3d_root / scene_id
        for subdir in _MP3D_SCAN_SUBDIRS:
            src = inner / subdir
            dst = dst_scene / subdir
            if not src.exists() or dst.exists():
                continue
            log(f"MOVE {src} -> {dst}")
            shutil.move(str(src), str(dst))
            actions += 1

    return actions


def _prune_mp3d_nonval_scans(val_scenes: set) -> int:
    """Delete scan dirs in v1/scans/ that are not in val_scenes.

    After relocating the val scene subfolders, the remaining dirs in
    v1/scans/ are either non-val or already-emptied val entries. Both
    can be safely removed to reclaim disk space.

    Also removes the parent v1/scans/ and v1/ dirs if they end up empty.

    Returns number of actions taken.
    """
    scans_root = DATA_DIR / "mp3d" / "v1" / "scans"
    if not scans_root.exists():
        return 0

    actions = 0
    for scene_dir in sorted(scans_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_dir.name not in val_scenes:
            log(f"RMTREE {scene_dir} (non-val scene)")
            shutil.rmtree(scene_dir)
            actions += 1
        else:
            # Val scene: inner dir should now be empty (subfolders moved)
            inner = scene_dir / scene_dir.name
            if inner.exists() and not any(inner.iterdir()):
                log(f"RMDIR {inner} (emptied)")
                inner.rmdir()
                actions += 1
            if scene_dir.exists() and not any(scene_dir.iterdir()):
                log(f"RMDIR {scene_dir} (emptied)")
                scene_dir.rmdir()
                actions += 1

    for stale in [scans_root, scans_root.parent]:
        if stale.exists() and not any(stale.iterdir()):
            log(f"RMDIR {stale}")
            stale.rmdir()
            actions += 1

    return actions


def _prune_mp3d_root_scenes(mp3d_root: Path, val_scenes: set) -> int:
    """Delete non-val scene dirs from data/mp3d/ itself.

    After mesh relocation, data/mp3d/ contains scene dirs for all scenes
    (e.g. 90 total) but only 11 are val scenes. This removes the non-val
    ones to reclaim disk space.

    Only removes directories that look like scene IDs (alphanumeric/dash,
    not a known non-scene dir like 'objectnav_mp3d_v1' or 'v1').

    Returns number of actions taken.
    """
    reserved = {"objectnav_mp3d_v1", "v1"}

    actions = 0
    for d in sorted(mp3d_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in reserved:
            continue
        if d.name in val_scenes:
            continue
        log(f"RMTREE {d} (non-val scene)")
        shutil.rmtree(d)
        actions += 1

    return actions


def prepare_mp3d() -> int:
    """Prepare MP3D dataset. Returns number of actions taken."""
    print("\n--- MP3D ---")
    actions = 0

    mp3d_root = DATA_DIR / "mp3d"
    if not mp3d_root.exists():
        log("SKIP: data/mp3d not found (download data first)")
        return 0

    actions += _relocate_mp3d_meshes()

    episodes_dir = mp3d_root / "objectnav_mp3d_v1"
    if episodes_dir.exists():
        for gz_file in sorted(episodes_dir.rglob("*.json.gz")):
            changed = rewrite_gzip_json(gz_file, _fix_mp3d_scene_ids)
            if changed:
                actions += 1
    else:
        log("No objectnav_mp3d_v1/ dir found, skipping scene_id rewrite")

    val_scenes: set = set()
    content_dir = episodes_dir / "val" / "content" if episodes_dir.exists() else None
    if content_dir and content_dir.exists():
        for gz in content_dir.glob("*.json.gz"):
            val_scenes.add(gz.stem.replace(".json", "").split(".")[0])

    if val_scenes:
        actions += _relocate_mp3d_scans(val_scenes)
        actions += _prune_mp3d_nonval_scans(val_scenes)
        actions += _prune_mp3d_root_scenes(mp3d_root, val_scenes)
    else:
        log("No val scenes found, skipping scan relocation and pruning")

    if actions == 0:
        log("Already up to date")
    return actions


def _fix_mp3d_scene_ids(data: dict) -> dict:
    """Strip mp3d/ prefix from MP3D scene_ids.

    Original: 'mp3d/8194nk5LbLH/8194nk5LbLH.glb'
    New:      '8194nk5LbLH/8194nk5LbLH.glb'
    """
    changed = False
    for ep in data.get("episodes", []):
        scene_id = ep.get("scene_id", "")
        if scene_id.startswith("mp3d/"):
            ep["scene_id"] = scene_id[len("mp3d/"):]
            changed = True
    return data if changed else None


PREPARE_FNS = {
    "hm3d": prepare_hm3d,
    "ovon": prepare_ovon,
    "mp3d": prepare_mp3d,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Relocates files from the habitat-sim download layout into the project's
expected directory structure, then rewrites episode JSONs for compatibility.

  hm3d  Relocate meshes from habitat download + rewrite .basis.glb -> .glb
  ovon  Strip scene_id prefixes in data/hm3d_ovon/hm3d/{split}/
  mp3d  Relocate scene dirs from v1/tasks/mp3d/<scene>/ to <scene>/ +
        strip mp3d/ prefix in objectnav_mp3d_v1/ episode JSONs

All operations are idempotent. .bak backups are created for rewritten files.
""",
    )
    parser.add_argument(
        "--dataset",
        choices=ALL_DATASETS,
        default=None,
        help="Prepare only this dataset (default: all)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    print(f"Preparing datasets: {', '.join(datasets)}")
    print(f"Data directory: {DATA_DIR.resolve()}")

    total_actions = 0
    for ds in datasets:
        total_actions += PREPARE_FNS[ds]()

    print(f"\n{'=' * 50}")
    if total_actions > 0:
        print(f"Done! {total_actions} action(s) completed.")
    else:
        print("All datasets already up to date. Nothing to do.")


if __name__ == "__main__":
    main()
