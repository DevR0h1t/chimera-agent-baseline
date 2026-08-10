#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --job-name="chimera_built_img"
#SBATCH --nodes=1
#SBATCH --nodelist=dlc-groudon,dlc-arceus,dlc-meowth
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/test_built_image_gpu-%j.out
#SBATCH --container-image="dockerdex.umcn.nl:5005/hafsaakebli/chimeraagent:debug"
#SBATCH --container-mounts=/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/input:/input:ro,/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/output:/output,/data/pathology/projects/hafsa/baseline_models:/opt/ml/model:ro
#SBATCH --output=/data/pathology/projects/hafsa/chimera-agent-debug-kit/example_algorithm/baseline_docker_image-%j.out
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

echo "Run inference:"
python3 -u /opt/app/inference.py

echo "Output:"
find /output -type f -name "*.json" -print

for f in /output/*.json; do
  [ -e "$f" ] || continue
  echo "===== $f ====="
  cat "$f"
  echo ""
done