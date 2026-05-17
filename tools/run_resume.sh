#!/usr/bin/env bash
# Disconnect-proof (run inside tmux) holdout sweep for a chosen patient set,
# then stitch ALL parts (H1 snapshot + every _resume* dir) into the canonical
# 12-row summary and render the holdout figure. Survives VSCode/SSH disconnect.
#
# Usage: bash tools/run_resume.sh <rheology> <gpu> <metrics_subdir> [patients...]
#   rheology       : newtonian | carreau_yasuda
#   gpu            : CUDA device index (0|1)
#   metrics_subdir : dir under reports/metrics/ for THIS part's CSV
#                    (must match _resume* so stitch globs it)
#   patients       : optional; default = the 11 non-H1 patients
set -uo pipefail
cd /home/olarinoyem/Project/CABG_WSS_PINN

RHE="${1:?rheology}"
GPU="${2:?gpu index}"
SUBDIR="${3:?metrics subdir}"
shift 3
PATIENTS=("$@")
if [ "${#PATIENTS[@]}" -eq 0 ]; then
  PATIENTS=(H2 H3 H4 BG1 BG2 BG3 BG4 BG5 D1 D2 D3)
fi
DT=/home/olarinoyem/miniconda3/envs/deep_tf/bin/python
LOG="logs/holdout_${RHE}_resume.log"

{
  echo "[run_resume] $(date -Is) START rheology=$RHE gpu=$GPU subdir=$SUBDIR patients=${PATIENTS[*]}"
  CUDA_VISIBLE_DEVICES="$GPU" "$DT" -m src.evaluate holdout \
    --rheology "$RHE" --epochs 3000 \
    --patients "${PATIENTS[@]}" \
    --metrics-dir "reports/metrics/${SUBDIR}"
  rc=$?
  echo "[run_resume] $(date -Is) sweep exited rc=$rc"
  echo "[run_resume] $(date -Is) stitching all parts -> canonical 12-row summary"
  "$DT" tools/stitch_holdout.py "$RHE"
  echo "[run_resume] $(date -Is) rendering holdout figure (--no-update-table)"
  "$DT" -m src.plots --rheology "$RHE" --no-update-table
  echo "[run_resume] $(date -Is) ALL DONE rheology=$RHE"
} >> "$LOG" 2>&1
