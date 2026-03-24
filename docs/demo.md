# Real-time Interactive Demo

<p align="center">
  <img src="media/demo.gif" alt="Demo" width="800">
</p>

Real-time, interactive, open-vocabulary scene understanding from posed RGBD images alone — no 3D reconstruction or scene graph is required. Type any natural-language query — a rare object (*audio speaker*), a functional place (*where can I sleep?*, *where can I cook?*, *where can I eat?*), a material (*made of metal*), a physical property (*something that emits light*), an abstract concept (*cozy*, *festive*, *cluttered*), or a spatial relationship (*sofa next to the TV*, *door in the bedroom*) — and the corresponding regions are highlighted. The reconstructed mesh shown in the demo is purely for visualization.

---

## Prerequisites

```bash
pip install viser open3d trimesh scipy
```

## Run

```bash
python demo/demo.py --dataset hm3d --scene 4ok3usBNeis
python demo/demo.py --dataset mp3d --scene 8194nk5LbLH
python demo/demo.py --dataset goatcore --scene nfv
python demo/demo.py --dataset custom --scene /path/to/my_scene --pose-convention opencv
python demo/demo.py --dataset hm3d --list-scenes
```

**Datasets:** `hm3d`, `mp3d`, `goatcore`, `custom`

<details>
<summary><b>Flags</b></summary>

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset (`hm3d`, `mp3d`, `goatcore`, `custom`) |
| `--scene` | Scene ID, or path for custom datasets |
| `--list-scenes` | List available scenes and exit |
| `--pose-convention` | Pose coordinate convention for custom datasets: `habitat` (default) or `opencv` |
| `--no-cache` | Force rebuild mesh (ignore cached `.ply`) |
| `--query-res` | Max image dimension (px) for VLM re-ranking + SAM3 segmentation (default: `120`, `0` = original resolution) |

</details>

> **Tip:** Edit `configs/demo.yaml` to tune settings. If running out of GPU memory, try a smaller VLM (e.g. `vlm.model: Qwen/Qwen2.5-VL-3B-Instruct`) and reduce `sam3.batch_size`.

### Display Controls

- **VLM Re-ranking** — on by default; uncheck for faster feature-only retrieval.
- **Show Frustums / Show Object Cloud / Point Size** — toggle and resize visual overlays.
- **Spatial Fusion** — merges segmentation across nearby views into a denser point cloud. Off = single-view only.
- **Spatial Neighbors** (1–20) — more neighbors = denser clouds but slower queries.
- **HDBSCAN Cleaning** — removes outlier points from fused clouds. Adjust **Min Cluster Size** and **Min Samples** in real time after a query.

---

## Custom Dataset

<p align="center">
  <img src="media/custom.gif" alt="Custom Dataset Demo" width="800">
</p>

Organize your posed RGBD sequence as:

```
my_scene/
├── rgb/            # .jpg or .png
├── depth/          # uint16 PNG
├── intrinsics.json
└── poses.txt
```

RGB and depth filenames are matched by sorted order. Line *i* in `poses.txt` corresponds to the *i*-th sorted RGB image.

> **Pose Convention:** The pipeline uses Habitat/OpenGL convention (Y-up, −Z forward) internally. If your poses come from SLAM/VO (ORB-SLAM, COLMAP, etc.), use `--pose-convention opencv`. If the mesh looks flipped, switch conventions.

<details>
<summary><b>intrinsics.json</b></summary>

```json
{
  "fx": 600.0, "fy": 600.0,
  "cx": 320.0, "cy": 240.0,
  "width": 640, "height": 480,
  "depth_scale": 1000.0
}
```

`depth_scale` (default `1000.0`) divides raw uint16 depth to meters.
</details>

<details>
<summary><b>poses.txt</b></summary>

One line per frame: `tx ty tz qw qx qy qz` (camera-to-world). Lines starting with `#` are skipped.

| Convention | Up | Forward | Typical source |
|---|---|---|---|
| `habitat` (default) | +Y | -Z | Habitat, OpenGL, ARKit |
| `opencv` | -Y | +Z | ORB-SLAM, COLMAP, OpenCV-based SLAM/VO |

```
# tx ty tz qw qx qy qz
1.230 0.450 -2.100 0.707 0.000 0.707 0.000
1.250 0.450 -2.050 0.710 0.000 0.704 0.000
```
</details>
