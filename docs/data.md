# Dataset Setup

## 1. Goat-Core

From [Goat-Core](http://poss.pku.edu.cn/Goat-core.html).

```bash
cd /path/to/memory-over-maps
wget http://poss.pku.edu.cn/OpenDataResource/Goat-core.zip
unzip Goat-core.zip -d data/
rm Goat-core.zip
```

---

## 2. HM3D ObjectNav v2

Requires Matterport credentials for scene mesh download.

```bash
MATTERPORT_TOKEN_ID=<YOUR_MATTERPORT_TOKEN_ID>
MATTERPORT_TOKEN_SECRET=<YOUR_MATTERPORT_TOKEN_SECRET>

# Step 1: Create data directories
mkdir -p data/hm3d/scenes data/hm3d/episodes data/hm3d/meshes

# Step 2: Camera poses (36 scenes, bundled in repo)
tar xzf data/hm3d_poses.tar.gz -C data/hm3d/scenes/

# Step 3: Scene meshes + navmeshes (required for rendering + navigation)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 --data-path data

# Step 4: ObjectNav v2 episodes
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip objectnav_hm3d_v2.zip && rm objectnav_hm3d_v2.zip

# Step 5: Relocate meshes + fix episode paths
python scripts/prepare_data.py --dataset hm3d

# Step 6: Render RGB + depth from poses (~61 GB, requires habitat-sim + GPU)
python scripts/render_rgbd.py                    # All 36 scenes, 18 workers
```

---

## 3. HM3D-OVON

Requires HM3D ObjectNav setup (Section 2) first. OVON episodes from [HuggingFace](https://huggingface.co/datasets/nyokoyama/hm3d_ovon) ([OVON repo](https://github.com/naokiyokoyama/ovon)).

```bash
cd /path/to/memory-over-maps
git clone https://huggingface.co/datasets/nyokoyama/hm3d_ovon data/hm3d_ovon

# if you cannot extract the .tar file get the episodes manually
wget -O data/hm3d_ovon/hm3d.tar.gz "https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/main/hm3d.tar.gz?download=true"

# Rewrites scene_dataset_config and scene_id paths in episode JSONs
python scripts/prepare_data.py --dataset ovon
```

---

## 4. MP3D ObjectNav

Requires [Matterport3D Terms of Use](https://niessner.github.io/Matterport/) agreement.

| # | Component | Download |
|---|-----------|----------|
| 1 | Scene data (meshes, images, depth, poses) | `python download_mp.py --task habitat -o data/mp3d/` |
| 2 | Unzip scene data | `unzip data/mp3d/v1/tasks/mp3d_habitat.zip -d data/mp3d/v1/tasks/` |
| 3 | Undistorted images, depth, poses, intrinsics | `python download_mp.py --id <val_scene> --type undistorted_color_images undistorted_depth_images matterport_camera_poses matterport_camera_intrinsics -o data/mp3d/` |
| 4 | Unzip all data per scene | `unzip 'data/mp3d/v1/scans/<val_scene>/*.zip' -d data/mp3d/v1/scans/<val_scene>/` |
| 5 | ObjectNav MP3D v1 episode JSONs | Extract `objectnav_mp3d_v1.zip` into `data/mp3d/` |


> Download the following **11 val scenes** for step 3 and 4:
```
2azQ1b91cZZ  8194nk5LbLH  EU6Fwq7SyZv  oLBMNvg9in8  pLe4wQe7qrG
QUCTc6BB5sX  TbHJrupSAjP  X7HyMhZNoso  x8F5xyUWy9e  Z6MFQCViBuw
zsNo4HB9uLZ
```

After everything is set up, run the prepare script to relocate scene dirs and rewrite episode paths:

```bash
python scripts/prepare_data.py --dataset mp3d
```

---

## 5. SUN RGB-D

From [SUN RGB-D](https://rgbd.cs.princeton.edu/).

```bash
cd /path/to/memory-over-maps
wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip
unzip SUNRGBD.zip -d data/
rm SUNRGBD.zip
```
