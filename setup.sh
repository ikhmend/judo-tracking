#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo 'GPU probe failed; CPU remains supported.'
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    /usr/lib/wsl/lib/nvidia-smi || echo 'GPU probe failed; CPU remains supported.'
else
    echo 'NVIDIA GPU driver not visible; CPU remains supported.'
fi
python_bin="${PYTHON:-python3.11}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10–3.12 for the pinned inference stack"'
if [ ! -d .venv ]; then
    "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Recreate .venv with Python 3.10–3.12"'
.venv/bin/python -m pip install --upgrade pip
torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$torch_index"
.venv/bin/python -m pip install -r requirements.txt
mkdir -p weights
if [ ! -f weights/yolov8n.pt ]; then
    (cd weights && ../.venv/bin/python -c 'from ultralytics import YOLO; YOLO("yolov8n.pt")')
fi
.venv/bin/python track.py --check-device
.venv/bin/python -m unittest discover -s tests -v
