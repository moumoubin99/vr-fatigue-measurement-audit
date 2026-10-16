#!/bin/bash
# A3 BatchNorm diagnosis (review RV-G01): velocity, paper split and CV folds 0 and 2.
# Batch 1 = bf16 replicates of the reported fits, batch 2 = fp32-training controls. 3 fits in parallel (about 8 GB each).
# VRFAT_ROOT: analysis directory. PY: analysis interpreter. Needs out/pur_win_v3vel.npz (ekyt.py prep).
set -euo pipefail
: "${VRFAT_ROOT:?set VRFAT_ROOT to the analysis directory}"
cd "$VRFAT_ROOT/code"
P=${PY:-python}
N=../out/pur_win_v3vel.npz
O=../results/A3_diag; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for prec in bf16 fp32; do
  echo "$(date +%T) batch $prec"
  $P ekyt_diag.py $N $O --scheme paper --seed 0 --precision $prec > $O/paper_f0_$prec.log 2>&1 &
  $P ekyt_diag.py $N $O --scheme cv --seed 0 --repeat 0 --fold 0 --precision $prec > $O/cv_f0_$prec.log 2>&1 &
  $P ekyt_diag.py $N $O --scheme cv --seed 0 --repeat 0 --fold 2 --precision $prec > $O/cv_f2_$prec.log 2>&1 &
  wait
done
echo "$(date +%T) diag done"
