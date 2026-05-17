#!/usr/bin/env bash
# Disconnect-proof resume of a holdout sweep for the 11 non-H1 patients,
# then stitch H1 back and render the holdout figure. Runs inside tmux so it
# survives VSCode/SSH disconnect. All output appended to a per-rheology log.
#
# Usage: bash tools/run_resume.sh <newtonian|carreau_yasuda> <gpu_index>
set -uo pipefail
cd /home/olarinoyem/Project/CABG_WSS_PINN

RHE="${1:?rheology}"
GPU="${2:?gpu index}"
DT=/home/olarinoyem/miniconda3/envs/deep_tf/bin/python
LOG="logs/holdout_${RHE}_resume.log"

{
  echo "[run_resume] $(date -Is) START rheology=$RHE gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$DT" -m src.evaluate holdout \
    --rheology "$RHE" --epochs 3000 \
    --patients H2 H3 H4 BG1 BG2 BG3 BG4 BG5 D1 D2 D3 \
    --metrics-dir reports/metrics/_resume
  rc=$?
  echo "[run_resume] $(date -Is) sweep exited rc=$rc"
  echo "[run_resume] $(date -Is) stitching H1 back into canonical summary"
  "$DT" tools/stitch_holdout.py "$RHE"
  echo "[run_resume] $(date -Is) rendering holdout figure (--no-update-table)"
  "$DT" -m src.plots --rheology "$RHE" --no-update-table
  echo "[run_resume] $(date -Is) ALL DONE rheology=$RHE"
} >> "$LOG" 2>&1
