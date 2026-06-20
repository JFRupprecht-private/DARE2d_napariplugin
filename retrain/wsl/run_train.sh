#!/usr/bin/env bash
# Run the TF retraining driver inside the WSL `dare2d-train` env (GPU).
# Project root is derived from this script's location (PROJECT/retrain/wsl/run_train.sh),
# so it works from any /mnt/c path. All args are forwarded to train_split.py.
set -euo pipefail
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT=$(readlink -f "$SCRIPT_DIR/../..")

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dare2d-train
export SM_FRAMEWORK=tf.keras
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

cd "$PROJECT"
exec python retrain/train_split.py "$@"
