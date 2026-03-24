#!/bin/bash
set -e

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This project requires an NVIDIA GPU with CUDA."
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. habitat-sim requires conda."
    exit 1
fi
CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
if [ -z "$CONDA_ENV" ] || [ "$CONDA_ENV" = "base" ]; then
    echo "Error: No conda env active (or 'base'). Run:"
    echo "  conda create -n MoM python=3.9 -y"
    echo "  conda activate MoM"
    echo "  bash scripts/install.sh"
    exit 1
fi

PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VER" != "3.9" ]; then
    echo "Error: Python $PY_VER detected, but habitat-sim requires Python 3.9."
    echo "Recreate env: conda create -n MoM python=3.9 -y"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Installing project (Python $PY_VER, env: $CONDA_ENV, CUDA)"
echo "=================================="

echo ""
echo "[1/5] Installing PyTorch (CUDA 12.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[2/5] Installing habitat-lab..."
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-}"

if [ -z "$HABITAT_LAB_DIR" ]; then
    if [ -d "$PROJECT_DIR/third_party/habitat-lab/habitat-lab" ]; then
        HABITAT_LAB_DIR="$PROJECT_DIR/third_party/habitat-lab/habitat-lab"
        echo "Found habitat-lab at $HABITAT_LAB_DIR"
    else
        echo "Cloning habitat-lab v0.3.3 into third_party/habitat-lab..."
        git clone --branch v0.3.3 --depth 1 https://github.com/facebookresearch/habitat-lab.git \
            "$PROJECT_DIR/third_party/habitat-lab"
        HABITAT_LAB_DIR="$PROJECT_DIR/third_party/habitat-lab/habitat-lab"
    fi
fi

if [ ! -d "$HABITAT_LAB_DIR" ]; then
    echo "Error: habitat-lab not found at $HABITAT_LAB_DIR"
    echo "Set HABITAT_LAB_DIR to your habitat-lab/habitat-lab directory."
    exit 1
fi

pip install -e "$HABITAT_LAB_DIR"

echo ""
echo "[3/5] Installing project package..."
pip install -e ".[dev]"

echo ""
echo "[4/5] Installing habitat-sim via conda..."
conda install habitat-sim=0.3.3 -c conda-forge -c aihabitat -y

echo ""
echo "[5/5] Installing SAM3..."
SAM3_DIR="$PROJECT_DIR/third_party/sam3"
pip install einops decord pycocotools psutil

if [ -d "$SAM3_DIR" ]; then
    echo "SAM3 already exists at $SAM3_DIR, pulling latest..."
    (cd "$SAM3_DIR" && git pull)
else
    echo "Cloning SAM3..."
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi

echo "Installing SAM3 in editable mode..."
(cd "$SAM3_DIR" && pip install -e .)

echo ""
echo "=================================="
echo "Verifying installation..."
echo "=================================="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available')
import transformers; print(f'Transformers {transformers.__version__}')
import habitat; print('habitat-lab: OK')
import habitat_sim; print('habitat-sim: OK')
from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')
"

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Download dataset (HM3D/MP3D/Goat-Core) — see docs/data.md for setup"
echo "  2. Run:"
echo "       python -m src.cli.eval_goatcore                 # Goat-Core: all scenes, all modes"
echo "       python -m src.cli.eval_hm3d                     # HM3D ObjectNav: 1000 episodes, 36 scenes, 6 categories"
echo "       python -m src.cli.eval_ovon                     # HM3D-OVON: 379 open-vocab categories, all splits"
echo "       python -m src.cli.eval_mp3d                     # MP3D ObjectNav: 2195 episodes, 11 scenes, 21 categories"
echo "       python -m src.cli.eval_sunrgbd                  # SUN RGB-D retrieval"
