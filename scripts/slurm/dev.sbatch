#!/bin/bash
# =============================================================================
# Dev session — SLURM params injected via sbatch CLI flags from Makefile
#
# Environment variables (set by Makefile --export):
#   NAS_PATH        — project root on NAS
#   NAS_OUTPUTS     — output directory
# =============================================================================
#SBATCH --open-mode=append
#SBATCH --no-container-entrypoint

set -euo pipefail

PROJECT_SLUG="chimera_agent_baseline"
PORT=${PORT:-58888}

echo "======================================================"
echo " Dev Session:   $SLURM_JOB_ID"
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

# --- Configure shell for SSH logins ---
NODE_SHORT=$(hostname)
cat >> /root/.bashrc <<EOF

# --- dev-session auto-setup ---
source "${VENV}/bin/activate"
cd "$(pwd)" 2>/dev/null
export NAS_PATH="${NAS_PATH}"
export NAS_OUTPUTS="${NAS_OUTPUTS}"
echo ""
echo "  Dev Session | Job \$SLURM_JOB_ID | \$(hostname)"
echo "  GPU:          \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Code:         $(pwd)"
echo ""
EOF

# --- Save connection info ---
mkdir -p "${HOME}/connection-info"
cat > "${HOME}/connection-info/${SLURM_JOB_ID}.txt" <<EOF
Job:          ${SLURM_JOB_ID}
Host:         $(hostname)
SSH:          ssh ${NODE_SHORT} -A
Code:         $(pwd)
Started:      $(date)
EOF

echo ""
echo "  SSH: ssh ${NODE_SHORT} -A"
echo "  Info saved to ~/connection-info/${SLURM_JOB_ID}.txt"
echo ""
echo "Starting SSH daemon on port ${PORT}..."

/usr/sbin/sshd -D -p ${PORT} &
wait $!
