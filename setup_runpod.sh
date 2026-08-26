#!/usr/bin/env bash
# copublish phase-2 antislop fine-tune: RunPod pod setup.
# Run this on a fresh pod deployed from RunPod's PLAIN PyTorch template
# (NOT the "vLLM" template -- that auto-launches an unkillable vllm server
# that eats ~75GB of VRAM as PID 1's tracked child) with a CUDA 12.9+
# (ideally 13.0) filter applied at deploy time.
#
# Usage (paste this ONE line into the pod's web terminal -- do not paste
# the script body itself, multi-line pastes into that terminal have
# silently corrupted mid-word before):
#   curl -fsSL https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/setup_runpod.sh | bash
set -euo pipefail

cd /workspace

echo "=== [1/7] Cloning auto-antislop + submodules ==="
if [ ! -d auto-antislop ]; then
  git clone --recurse-submodules https://github.com/sam-paech/auto-antislop.git
fi
cd auto-antislop

echo "=== [2/7] Pulling copublish's configs (public repo, no paste needed) ==="
curl -fsSL -o configs_copublish.yaml https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/configs_copublish.yaml
curl -fsSL -o configs_copublish_test.yaml https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/configs_copublish_test.yaml

echo "=== [3/7] venv ==="
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel

echo "=== [4/7] Install vLLM FIRST, alone, so it settles torch's version once ==="
# Lesson from the 2026-08-26 failed attempt: installing torch==2.8 first
# (per the repo's own quickstart) then letting vllm silently bump it to
# whatever it's pinned to (2.13.0 as of this writing) wastes a step and,
# worse, if flash-attn or anything else builds against the pre-bump torch
# first, it ends up linked against the wrong version later. Settle torch
# via vllm's own dependency resolution FIRST, before anything else touches
# it.
pip install vllm

echo "--- torch/CUDA versions after vllm install (sanity check) ---"
python -c "import torch; print('torch', torch.__version__, '| torch CUDA build', torch.version.cuda)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
echo "--- If torch's CUDA build above is NEWER than what this pod's driver supports, STOP and redeploy a pod filtered for a higher CUDA version. Do not proceed past this point on a mismatch. ---"

echo "=== [5/7] Rest of requirements.txt, MINUS flash-attn (built separately, last, in step 7) ==="
grep -v '^flash-attn' requirements.txt > requirements.no-flash-attn.txt
pip install -r requirements.no-flash-attn.txt

echo "--- torch/CUDA versions after requirements.txt (confirm unchanged) ---"
python -c "import torch; print('torch', torch.__version__, '| torch CUDA build', torch.version.cuda)"

echo "=== [6/7] nvcc / CUDA_HOME fix ==="
# Lesson: the pod's baked-in nvcc toolkit can be older than torch's CUDA
# build even when the driver is fine (seen: 12.8 toolkit vs torch cu13).
# Fix is the UNVERSIONED nvidia-cuda-nvcc package -- the -cuXY suffixed
# one is deprecated and fails to build.
pip install nvidia-cuda-nvcc
NVCC_PATH=$(find "$(python -c 'import site; print(site.getsitepackages()[0])')" -iname 'nvcc' -path '*/bin/*' | head -1)
if [ -z "$NVCC_PATH" ]; then
  echo "Could not locate the pip-installed nvcc binary -- search manually with:"
  echo "  find / -iname 'nvcc' -path '*nvidia*' 2>/dev/null"
  exit 1
fi
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version

echo "=== [7/7] flash-attn, built from source LAST, against the now-settled torch ==="
pip install -U ninja packaging cmake
MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation --no-cache-dir

echo "=== Setup complete ==="
echo "Next: run the SMOKE TEST config first, not the full run:"
echo "  cd /workspace/auto-antislop && source venv/bin/activate"
echo "  export CUDA_HOME=$CUDA_HOME PATH=\"$CUDA_HOME/bin:\$PATH\""
echo "  python main.py -c configs_copublish_test.yaml"
echo ""
echo "Only after that completes end-to-end (check results/auto_antislop_runs/.../finetuned_model_*_copublish_TEST/ for output), run the real config:"
echo "  python main.py -c configs_copublish.yaml"
