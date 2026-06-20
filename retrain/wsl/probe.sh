#!/usr/bin/env bash
# Probe the WSL2 Ubuntu environment for the TF-GPU training build.
set -u
echo "== disk (home) =="; df -h "$HOME" | tail -1
echo "== mem =="; free -h | head -2
echo "== tools =="
for t in wget curl bzip2 git nvidia-smi; do
  if command -v "$t" >/dev/null 2>&1; then echo "$t ok"; else echo "$t MISSING"; fi
done
echo "== gpu =="; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no gpu"
echo "== existing conda env (dare2d-train)? =="
if [ -x "$HOME/miniconda3/bin/conda" ]; then
  "$HOME/miniconda3/bin/conda" env list
else
  echo "no miniconda at ~/miniconda3"
fi
