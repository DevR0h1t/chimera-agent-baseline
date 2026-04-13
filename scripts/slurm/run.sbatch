#!/bin/bash
# =============================================================================
# Agent run job — SLURM params injected via sbatch CLI flags from Makefile
#
# Environment variables (set by Makefile --export):
#   NAS_PATH        — project root on NAS
#   NAS_OUTPUTS     — output directory
#   RUN_ARGS        — Hydra overrides
# =============================================================================
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --no-container-entrypoint

set -euo pipefail

PROJECT_SLUG="chimera_agent_baseline"

echo "======================================================"
echo " Agent Run:     $SLURM_JOB_ID"
echo " Node:          $(hostname)"
echo " GPUs:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " Date:          $(date)"
echo "======================================================"

# --- Setup ---
export UV_CACHE_DIR="/tmp/uv-cache-${SLURM_JOB_ID}"
VENV="/tmp/venv-${SLURM_JOB_ID}"
uv venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
uv pip install -e . --no-deps -q

# --- Run agent ---
echo "Starting agent..."
python -m chimera_agent_baseline.run \
    paths.output_dir="${NAS_OUTPUTS}" \
    ${RUN_ARGS:-}
echo "Done at $(date)"
