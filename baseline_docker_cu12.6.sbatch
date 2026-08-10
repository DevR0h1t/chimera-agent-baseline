#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --nodelist=dlc-groudon,dlc-arceus,dlc-meowth
#SBATCH --job-name="chimera_built_img"
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --container-writable
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/test_built_image_gpu-%j.out
#SBATCH --container-image="dockerdex.umcn.nl:5005/hafsaakebli/chimeraagent:cu129-debug"
#SBATCH --container-mounts=/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/input:/input:ro,/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/output:/output,/data/pathology/projects/hafsa/baseline_models:/opt/ml/model:ro
#SBATCH --container-workdir=/opt/app

set -euo pipefail

echo "Node:"
hostname

echo "GPU:"
nvidia-smi

echo "Input:"
find /input -maxdepth 1 -type f -name "*.json" -print

echo "Clean output:"
rm -rf /output/*
mkdir -p /output

GPU_TOTAL_MIB=$(nvidia-smi \
  --query-gpu=memory.total \
  --format=csv,noheader,nounits |
  head -n 1 |
  tr -d ' ')

TARGET_VLLM_MIB=13824

GPU_MEMORY_UTILIZATION=$(awk \
  -v target="$TARGET_VLLM_MIB" \
  -v total="$GPU_TOTAL_MIB" \
  'BEGIN {printf "%.8f", target / total}')

export GPU_MEMORY_UTILIZATION

python3 - <<'PY'
import os
from omegaconf import OmegaConf

path = "/opt/app/configs/config.yaml"
cfg = OmegaConf.load(path)

OmegaConf.update(
    cfg,
    "generation.gpu_memory_utilization",
    float(os.environ["GPU_MEMORY_UTILIZATION"]),
)

OmegaConf.save(cfg, path)
PY

echo "Run inference:"
#python3 -u /opt/app/inference.py

VLLM_USE_FLASHINFER_SAMPLER=0 python3 -u /opt/app/inference.py
echo "Output:"
find /output -type f -name "*.json" -print

for f in /output/*.json; do
  [ -e "$f" ] || continue
  echo "===== $f ====="
  cat "$f"
  echo ""
done