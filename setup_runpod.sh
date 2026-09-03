#!/usr/bin/env bash
# copublish FTPO-on-SFT'd-model: RunPod pod setup.
# Run this on a fresh pod deployed from RunPod's PLAIN PyTorch template
# (NOT the "vLLM" template -- that auto-launches an unkillable vllm server
# that eats ~75GB of VRAM as PID 1's tracked child) with a CUDA 12.9+
# (ideally 13.0) filter applied at deploy time. Pick an A100 80GB -- the
# merge step below loads the full ~55GB bf16 base model, which needs the
# headroom (the P40's 24GB can't hold it at all, and this project has a
# real prior incident of that exact load freezing the P40's Windows host
# via system-RAM exhaustion -- don't attempt the merge there).
#
# 2026-09-03 REWRITE: this now runs FTPO against copublish's OWN SFT'd
# model (a LoRA adapter trained locally via local-ai-gateway/sft.train.ps1
# in the main copublish repo) instead of the stock sam-paech/gemma-3-27b-
# it-antislop -- per the 2026-08-29 reordering decision: SFT establishes
# voice first, FTPO strips remaining tics from THAT result second. Step
# [1/8] below merges that adapter onto the full-precision base before
# anything else happens. If you ever need to run FTPO against a stock
# base again (no SFT layer), skip step 1 and set configs_copublish.yaml's
# model_id back to an HF repo id directly.
#
# Also folds in every real fix discovered running this for real in
# 2026-08-26/27 (see project memory, project_local_ai_gateway) that
# previously had to be applied by hand, mid-session, on a live pod:
#   - flash-attn is NOT installed at all (see step [5/8]) -- confirmed
#     unnecessary, vllm's own flashinfer dependency covers this, and
#     flash-attn's from-source build was a real, repeated source of
#     wasted time chasing stale-binary/torch-mismatch errors.
#   - the nvcc pin below is deliberately EXACT versions, not the latest
#     available -- an earlier attempt at unpinned "pip install
#     nvidia-cuda-nvcc" grabbed a newer release than the auto-installed
#     cuda-toolkit package expected, producing a "CUDA compiler and CUDA
#     toolkit headers are incompatible" error deep inside flashinfer's
#     JIT compilation. These exact pins matched what was needed
#     2026-08-27; if vllm/torch have since moved to a different pinned
#     CUDA version, re-derive via:
#       python -c "import torch; print(torch.version.cuda)"
#       (find the matching cuda-toolkit release on PyPI, check ITS OWN
#        declared "nvcc" extra pin -- that's the version to match, not
#        whatever nvidia-cuda-nvcc's own latest release is)
#   - a libcudart symlink fix for flashinfer's JIT link step (separate
#     from flash-attn -- this is needed even with flash-attn skipped),
#     which failed with "ld: cannot find -lcudart" because libcudart.so.13
#     installs versioned-only, no unversioned symlink, and the linker
#     also checks a lib64/ directory that doesn't exist by default.
#
# Usage (paste this ONE line into the pod's web terminal -- do not paste
# the script body itself, multi-line pastes into that terminal have
# silently corrupted mid-word before; run inside `tmux new -s ftpo` so a
# terminal disconnect doesn't kill a multi-hour run -- this has happened
# for real once already):
#   curl -fsSL https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/setup_runpod.sh | bash
set -euo pipefail

cd /workspace

echo "=== [1/8] Merging copublish's SFT LoRA adapter onto the full-precision base ==="
# The adapter needs to already be at /workspace/copublish_sft_adapter before
# this runs -- see the runbook for how to get it here (HF Hub push/pull
# from the P40 box is the recommended path; direct SFTP is the fallback).
# This step is SKIPPED if the merged model already exists (e.g. a pod
# restart after this already ran once) -- delete
# /workspace/copublish_sft_merged first if you want to force a re-merge.
curl -fsSL -o merge_sft_adapter.py https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/merge_sft_adapter.py
if [ ! -d /workspace/copublish_sft_adapter ]; then
  echo "ERROR: /workspace/copublish_sft_adapter not found."
  echo "Get the LoRA adapter onto this pod first -- see the runbook's"
  echo "'getting the adapter to the pod' section -- then re-run this script."
  exit 1
fi

echo "=== [2/8] Cloning auto-antislop + submodules ==="
if [ ! -d auto-antislop ]; then
  git clone --recurse-submodules https://github.com/sam-paech/auto-antislop.git
fi
cd auto-antislop

echo "=== [3/8] Pulling copublish's configs (public repo, no paste needed) ==="
curl -fsSL -o configs_copublish.yaml https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/configs_copublish.yaml
curl -fsSL -o configs_copublish_test.yaml https://raw.githubusercontent.com/grantwyatt/antislop-configs/master/configs_copublish_test.yaml

echo "=== [3b/8] Bumping vLLM startup wait_timeout (720s -> 1440s) ==="
# This model's real cold-start (weights load + compile/warmup) took ~673s
# in 2026-08-27 testing -- right under the 720s default, causing the
# parent 'vllm serve' process to be timeout-killed while its EngineCore
# child was still alive and orphaned on the GPU, holding ~70GB VRAM with
# nothing left able to reach it. This edit was previously applied by hand
# on a live pod and lost on the next fresh provision -- baked in here so
# that never has to happen again.
sed -i 's/wait_timeout: int = 720/wait_timeout: int = 1440/' utils/vllm_manager.py
grep -n "wait_timeout: int = " utils/vllm_manager.py

echo "=== [3c/8] Removing a stale --disable-log-requests flag (vllm CLI drift) ==="
# Real failure 2026-09-03: whatever vllm version "pip install vllm" grabs
# now rejects this flag entirely ("unrecognized arguments"), crashing the
# managed server on startup with exit code 2 before it ever gets to load
# the model. auto-antislop's own code comment says this flag is purely
# cosmetic ("Cleaner logs during generation"), not load-bearing, so it's
# simply dropped here rather than chasing whatever the current vllm
# release renamed it to -- safer against this recurring on a future vllm
# release too.
sed -i '/"--disable-log-requests",/d' utils/vllm_manager.py
grep -n "disable-log-requests" utils/vllm_manager.py && echo "WARNING: still present" || echo "confirmed removed"

echo "=== [4/8] venv ==="
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel

echo "=== [5/8] requirements.txt (unsloth/bitsandbytes/etc), flash-attn EXCLUDED ==="
# Confirmed unnecessary 2026-08-26/27 -- vllm's own flashinfer dependency
# handles fast attention/sampling kernels, and both vllm and unsloth treat
# flash-attn as optional in recent releases. Installing it wasted real
# time chasing "undefined symbol" errors from stale binaries built
# against a torch version that got replaced by a later install step.
grep -v '^flash-attn' requirements.txt > requirements.no-flash-attn.txt
pip install -r requirements.no-flash-attn.txt

echo "=== [6/8] vLLM LAST of the python packages, so ITS pin wins ==="
# Lesson from repeated failed attempts: unsloth's own dependency
# resolution picks its own preferred torch version on its own pass (seen:
# downgraded to 2.11.0) with no hard error -- it's a soft preference, not
# a strict pin. vllm's torch pin IS strict and hard-errors on mismatch.
# Whichever of the two installs LAST wins the torch version pip's
# resolver leaves in place, so vllm has to go last.
pip install vllm

echo "--- torch/CUDA versions after vllm install (should be vllm's pinned version) ---"
python -c "import torch; print('torch', torch.__version__, '| torch CUDA build', torch.version.cuda)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
pip check || echo "--- pip check found conflicts above -- read them before continuing, don't ignore ---"

echo "--- confirming unsloth still imports under vllm's torch version (the real open question) ---"
python -c "import unsloth; print('unsloth ok')"

echo "=== [7/8] nvcc / CUDA_HOME fix (exact-pin, not latest-available) ==="
# See the header comment above for why this must be pinned, not just
# "pip install nvidia-cuda-nvcc". These exact versions matched what
# cuda-toolkit==13.0.3 (torch 2.13.0's own dependency) itself declared as
# its "nvcc" extra pin as of 2026-08-27 -- re-verify if this fails.
pip install "nvidia-cuda-nvcc==13.0.88.*" "nvidia-cuda-crt==13.0.88.*" "nvidia-nvvm==13.0.88.*" "nvidia-cuda-runtime==13.0.96.*"
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

echo "--- libcudart symlink fix (flashinfer's JIT link step needs this, independent of flash-attn) ---"
# flashinfer's lazy JIT-compiled kernels (e.g. sampling) failed to LINK
# with "ld: cannot find -lcudart" -- libcudart installs as the versioned
# libcudart.so.13 only (no unversioned symlink -lcudart needs), and only
# under lib/, not the lib64/ the linker also searches.
CUDART_DIR="$(dirname "$(dirname "$NVCC_PATH")")/lib"
if [ -f "$CUDART_DIR/libcudart.so.13" ] && [ ! -e "$CUDART_DIR/libcudart.so" ]; then
  ln -sf libcudart.so.13 "$CUDART_DIR/libcudart.so"
  mkdir -p "$(dirname "$CUDART_DIR")/lib64"
  ln -sf "../lib/libcudart.so.13" "$(dirname "$CUDART_DIR")/lib64/libcudart.so"
  echo "libcudart symlinks created under $CUDART_DIR and its sibling lib64/"
else
  echo "libcudart.so.13 not found at the expected path or symlink already exists -- check manually if flashinfer's JIT step fails later:"
  echo "  find / -iname 'libcudart.so.13' 2>/dev/null"
fi

echo "=== [8/8] Merging the SFT adapter now that the environment is ready ==="
# Absolute path -- merge_sft_adapter.py was downloaded to /workspace in
# step [1/8], but step [2/8]'s "cd auto-antislop" (and nothing since has
# cd'd back) means the working directory here is /workspace/auto-antislop,
# not /workspace. A bare "python merge_sft_adapter.py" failed with "No
# such file or directory" for exactly this reason on the first real run
# of this script (2026-09-03) -- fixed here rather than adding a cd,
# since this is robust regardless of any future reordering of the steps
# above.
python /workspace/merge_sft_adapter.py \
  --adapter /workspace/copublish_sft_adapter \
  --output /workspace/copublish_sft_merged
echo "Merged model at /workspace/copublish_sft_merged -- confirmed this is"
echo "what configs_copublish.yaml/configs_copublish_test.yaml's model_id"
echo "already points at (both pulled fresh in step 3 above)."

echo "=== Setup complete ==="
echo "Next: run the SMOKE TEST config first, not the full run -- this is a"
echo "NEW base model that has never been through auto-antislop before, so"
echo "prove the whole chain (generation, FTPO training, GGUF export) works"
echo "end to end before committing to the 1000-prompt run:"
echo "  cd /workspace/auto-antislop && source venv/bin/activate"
echo "  export CUDA_HOME=$CUDA_HOME PATH=\"$CUDA_HOME/bin:\$PATH\""
echo "  python main.py -c configs_copublish_test.yaml"
echo ""
echo "Only after that completes end-to-end (check results/auto_antislop_runs/.../finetuned_model_*_copublish_TEST/ for output), run the real config:"
echo "  python main.py -c configs_copublish.yaml"
