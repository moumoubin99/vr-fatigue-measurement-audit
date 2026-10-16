#!/bin/bash
# A3 round-1 review reruns (sequential, one GPU): learning-capacity control, then paper split and
# 5-fold CV (repeat 0) for v3-masked velocity and gaze-angle inputs. Skips finished fits.
P=${PY:-python}   # set PY to the analysis environment interpreter
O=../out
run() { [ -f "$2/$3.json" ] && return; echo "$(date +%T) start $1 $3"; $P ekyt.py run "$1" "$2" "${@:4}" > "$2/$3.log" 2>&1; }
mkdir -p $O/ekyt_v3vel $O/ekyt_v3pos
run $O/pur_win_v3vel.npz $O/ekyt_v3vel overfit_s0_r0_f0 --scheme overfit --seed 0 --epochs 100
run $O/pur_win_v3pos.npz $O/ekyt_v3pos overfit_s0_r0_f0 --scheme overfit --seed 0 --epochs 100
run $O/pur_win_v3vel.npz $O/ekyt_v3vel paper_s0_r0_f0 --scheme paper --seed 0
run $O/pur_win_v3pos.npz $O/ekyt_v3pos paper_s0_r0_f0 --scheme paper --seed 0
for f in 0 1 2 3 4; do run $O/pur_win_v3pos.npz $O/ekyt_v3pos cv_s0_r0_f$f --scheme cv --seed 0 --repeat 0 --fold $f; done
for f in 0 1 2 3 4; do run $O/pur_win_v3vel.npz $O/ekyt_v3vel cv_s0_r0_f$f --scheme cv --seed 0 --repeat 0 --fold $f; done
echo "$(date +%T) queue done"
