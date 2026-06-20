#!/usr/bin/env bash
# Stage B: DARE2D training deps into the dare2d-train env (TF backend, GPU via WSL).
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dare2d-train

echo "== installing DARE2D training deps =="
# numpy 1.23.5 (matches the Windows env): DARE2D uses np.int, removed in numpy>=1.24.
python -m pip install --quiet \
  "numpy==1.23.5" \
  "scikit-image<0.22" "scikit-learn<1.4" scipy \
  "opencv-python-headless==4.11.0.86" \
  "pandas<2.1" "matplotlib<3.9" tqdm rich \
  omegaconf hydra-core hydra-colorlog click \
  imageio "albumentations<1.4" segmentation-models tifffile

echo "== verifying imports =="
SM_FRAMEWORK=tf.keras TF_CPP_MIN_LOG_LEVEL=3 python - <<'PY'
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy, cv2, skimage, scipy, sklearn, hydra, omegaconf, albumentations, tifffile
import segmentation_models as sm
import tensorflow as tf
print("numpy", numpy.__version__, "| cv2", cv2.__version__, "| tf", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("STAGE_B_OK")
PY
