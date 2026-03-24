#!/bin/bash
# eval.sh
#
# Wrapper executed by HTCondor for each evaluation job.
# Usage: eval.sh <path/to/checkpoint.pt>
#
# Adjust CONDA_ENV and SNN_DIR to match your installation.

set -euo pipefail

CKPT_PATH="${1:?ERROR: ckpt_path argument required}"
SNN_DIR="/lhome/ext/uovi123/uovi123j/SNN-Cal/SNN"
CONDA_ENV="snn_hgcal"

# ── Activate conda ─────────────────────────────────────────────────────────────
CONDA_BASE="/lhome/ext/uovi123/uovi123j/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Run evaluation ─────────────────────────────────────────────────────────────
cd "${SNN_DIR}"
echo "Starting eval: ${CKPT_PATH}  host=$(hostname)  $(date)"
python eval_job.py --ckpt_path "${CKPT_PATH}" --device cpu
echo "Finished eval: ${CKPT_PATH}  $(date)"
