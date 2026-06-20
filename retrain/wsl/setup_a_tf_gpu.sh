#!/usr/bin/env bash
# Stage A: miniconda + env `dare2d-train` (py3.10) + CUDA 11.8 / cuDNN 8.6 + TF 2.12 GPU.
# Idempotent-ish; re-running skips already-done steps. Verifies TF sees the GPU.
set -euo pipefail
cd ~

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  echo "== installing miniconda =="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# use conda-forge only (avoids the defaults-channel Terms-of-Service gate)
if ! conda env list | grep -q "dare2d-train"; then
  echo "== creating env dare2d-train (py3.10) =="
  conda create -n dare2d-train -c conda-forge --override-channels python=3.10 -y
fi
conda activate dare2d-train
conda install -c conda-forge --override-channels pip -y   # conda-forge python lacks pip

echo "== CUDA 11.8 (conda) + cuDNN 8.6 + libdevice (pip) =="
conda install -c conda-forge --override-channels cudatoolkit=11.8.0 -y
python -m pip install --quiet nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-nvcc-cu11
# TF's XLA (the new experimental Keras optimizers use jit_compile=True) needs
# nvvm/libdevice/libdevice.10.bc on the GPU. The pip nvcc wheel ships it; symlink it
# where XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX looks ($CONDA_PREFIX/nvvm/libdevice).
NVCC_ROOT=$(python -c "import os,nvidia.cuda_nvcc as n; print(os.path.dirname(n.__file__))")
mkdir -p "$CONDA_PREFIX/nvvm"
ln -sfn "$NVCC_ROOT/nvvm/libdevice" "$CONDA_PREFIX/nvvm/libdevice"
# XLA also shells out to ptxas (PTX->SASS); put the wheel's ptxas on the env bin.
ln -sf "$NVCC_ROOT/bin/ptxas" "$CONDA_PREFIX/bin/ptxas"

# make TF find cuDNN/CUDA at activation (official TF 2.12 GPU recipe)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# /usr/lib/wsl/lib holds the WSL GPU driver's libcuda.so, which cuDNN's sublibs dlopen.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
EOF

echo "== installing tensorflow 2.12 (numpy pinned 1.23.5: DARE2D uses np.int) =="
python -m pip install --quiet "tensorflow==2.12.*" "numpy==1.23.5"

# re-activate so LD_LIBRARY_PATH from activate.d takes effect
conda deactivate
conda activate dare2d-train

echo "== verifying TF sees the GPU =="
TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("TF", tf.__version__, "| numpy", __import__("numpy").__version__)
print("GPUs:", gpus)
assert gpus, "NO GPU visible to TensorFlow"
# quick matmul on GPU
with tf.device('/GPU:0'):
    a = tf.random.normal((512, 512)); b = tf.matmul(a, a)
print("GPU matmul ok, mean=", float(tf.reduce_mean(b)))
print("STAGE_A_OK")
PY
