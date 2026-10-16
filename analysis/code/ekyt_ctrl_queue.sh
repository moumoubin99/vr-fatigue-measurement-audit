#!/bin/bash
# A3 follow-up on one GPU, three fits at a time (about 8 GB each):
#   1. controls (ekyt_ctrl.py): identity leakage and held-out positive controls, velocity and gaze-angle input
#   2. re-run of the original A3 set (ekyt.py, same commands as ekyt_queue_v3.sh) into new directories
# Skips finished fits. VRFAT_ROOT: analysis directory. PY: analysis interpreter.
set -euo pipefail
: "${VRFAT_ROOT:?set VRFAT_ROOT to the analysis directory}"
cd "$VRFAT_ROOT/code"
P=${PY:-python}
O=../out; T=$O/v3/tables
J=$O/ekyt_jobs.txt
: > $J
for inp in vel pos; do
  for s in window window_perm session identity paper_catchup paper_age; do
    d=$O/ekyt_ctrl_v3$inp
    echo "[ -f $d/${s}_s0.json ] || $P ekyt_ctrl.py $O/pur_win_v3$inp.npz $T $d --scheme $s > $d/${s}_s0.log 2>&1" >> $J
  done
done
for inp in vel pos; do
  d=$O/ekyt_rerun_v3$inp; n=$O/pur_win_v3$inp.npz
  echo "[ -f $d/paper_s0_r0_f0.json ] || $P ekyt.py run $n $d --scheme paper --seed 0 > $d/paper_s0_r0_f0.log 2>&1" >> $J
  for f in 0 1 2 3 4; do
    echo "[ -f $d/cv_s0_r0_f$f.json ] || $P ekyt.py run $n $d --scheme cv --seed 0 --repeat 0 --fold $f > $d/cv_s0_r0_f$f.log 2>&1" >> $J
  done
  echo "[ -f $d/overfit_s0_r0_f0.json ] || $P ekyt.py run $n $d --scheme overfit --seed 0 --epochs 100 > $d/overfit_s0_r0_f0.log 2>&1" >> $J
done
mkdir -p $O/ekyt_ctrl_v3vel $O/ekyt_ctrl_v3pos $O/ekyt_rerun_v3vel $O/ekyt_rerun_v3pos
echo "$(date +%T) $(wc -l < $J) jobs"
xargs -P 3 -I{} -d '\n' bash -c '{}' < $J
echo "$(date +%T) queue done"
