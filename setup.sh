#!/usr/bin/env bash
# Setup script for an ML environment with CUDA-aware PyTorch install.
# - Creates a Python venv
# - Logs in to Hugging Face & Weights & Biases
# - Detects CUDA and installs matching torch/torchvision/torchaudio
# - Installs transformers/accelerate/datasets
# - Tries to install flash-attn (optional; non-fatal if it fails)

set -euo pipefail

echo "=== [1/6] Create & activate virtualenv ==="
# add-apt-repository -y ppa:deadsnakes/ppa
# apt update
# apt install -y python3.10 python3.10-dev python3.10-venv

python3 -m venv venv
source venv/bin/activate


# python3 -m venv venv
# # shellcheck disable=SC1091
# source venv/bin/activate
python -m pip install -U pip setuptools wheel packaging ninja
echo "OK: venv ready."

echo "=== [2/6] CLI tools & logins (HF/W&B) ==="
pip install -U "huggingface_hub[cli]" git-lfs wandb
# git lfs install
# git config --global credential.helper store

# # Hugging Face login (interactive, or via HF_TOKEN env)
# if [[ -n "${HF_TOKEN:-}" ]]; then
#   echo "INFO: Using HF_TOKEN from env for login."
#   huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
# else
#   echo "INFO: Hugging Face login (press Ctrl-C to skip):"
#   huggingface-cli login || true
# fi

# # Weights & Biases login (interactive, or via WANDB_API_KEY env)
# if [[ -n "${WANDB_API_KEY:-}" ]]; then
#   echo "INFO: Using WANDB_API_KEY from env for login."
#   wandb login "$WANDB_API_KEY" || true
# else
#   echo "INFO: Weights & Biases login (press Ctrl-C to skip):"
#   wandb login || true
# fi
# echo "OK: CLI tools ready."

echo "=== [3/6] Detect CUDA & install PyTorch ==="
TORCH_VER="2.8.0"
TV_VER="0.23.0"
TA_VER="2.8.0"

detect_cuda() {
  local ver=""
  if command -v nvidia-smi &>/dev/null; then
    ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n 1 || true)
  elif command -v nvcc &>/dev/null; then
    ver=$(nvcc --version | grep -o 'release [0-9]\+\.[0-9]\+' | awk '{print $2}')
  fi
  echo "$ver"
}

CUDA_VER="$(detect_cuda)"
if [[ -z "${CUDA_VER}" ]]; then
  echo "WARN: No CUDA detected — installing CPU wheels for torch."
  pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VER}+cpu" "torchvision==${TV_VER}+cpu" "torchaudio==${TA_VER}+cpu"
else
  echo "INFO: Detected CUDA ${CUDA_VER}"
  CUDA_MAJOR="$(echo "$CUDA_VER" | cut -d. -f1)"
  CUDA_MINOR="$(echo "$CUDA_VER" | cut -d. -f2)"

  # If CUDA 13.0+, use CUDA 12.8 wheels (latest available PyTorch version)
  if [[ ${CUDA_MAJOR} -ge 13 ]]; then
    echo "INFO: CUDA ${CUDA_VER} detected. PyTorch wheels not available for CUDA 13.0+."
    echo "INFO: Using CUDA 12.8 wheels (compatible with CUDA 13.0+)"
    CUDA_TAG="cu128"
  else
    CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"  # e.g., cu128, cu121
  fi

  WHL_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
  echo "INFO: Trying PyTorch wheels for ${CUDA_TAG} (${WHL_INDEX})"

  set +e
  pip install --index-url "${WHL_INDEX}" --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==${TV_VER}+${CUDA_TAG}" \
    "torchaudio==${TA_VER}+${CUDA_TAG}"
  code=$?
  set -e

  if [[ $code -ne 0 ]]; then
    echo "WARN: ${CUDA_TAG} wheels not found. Falling back to cu121."
    pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
      "torch==${TORCH_VER}+cu121" \
      "torchvision==${TV_VER}+cu121" \
      "torchaudio==${TA_VER}+cu121"
  fi
fi

python - <<'PY'
import torch
print(f"OK: torch {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY

echo "=== [4/6] Core libs: transformers / accelerate / datasets ==="
pip install "transformers==4.56.0" "accelerate==1.10.1" "datasets==3.6.0" omegaconf
echo "OK: transformers/accelerate/datasets/omegaconf installed."

echo "=== [5/6] Optional: flash-attn 2.8.3 ==="
# Set TORCH_CUDA_ARCH_LIST automatically (helps build for correct SM).
ARCH="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")
PY
)"
if [[ -n "${ARCH}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${ARCH}"
  echo "INFO: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

set +e
pip install "flash-attn==2.8.3" --no-build-isolation
FA_CODE=$?
set -e
if [[ $FA_CODE -ne 0 ]]; then
  echo "WARN: flash-attn build failed. Continuing with PyTorch SDPA attention."
fi

echo "=== [6/6] Done ==="
python - <<'PY'
import torch, transformers, accelerate, datasets
print("ENV READY ✓")
print("torch      :", torch.__version__, "(cuda:", torch.version.cuda, ")")
print("transformers:", transformers.__version__)
print("accelerate :", accelerate.__version__)
print("datasets   :", datasets.__version__)
PY
echo "🎉 === Have a nice time! === 🎉"
