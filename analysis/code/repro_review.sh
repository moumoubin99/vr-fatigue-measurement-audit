#!/bin/bash
# Review-round re-runs (A1, A2_v4, A2_v5_lrcv, A4_v4, A4_rel, X4_v4, EKYT window files, REV1, REV2_VR).
# Run after run_v3.sh. CPU only. The EKYT fits themselves (A3) need a GPU: see ekyt_queue_v3.sh.
# VRFAT_ROOT: analysis directory (code/, data/, out/, results/). PY: analysis interpreter.
# The REV2 scripts read data/server_out/out/: point it at out/ to use this run's outputs.
set -euo pipefail
: "${VRFAT_ROOT:?set VRFAT_ROOT to the analysis directory}"
cd "$VRFAT_ROOT/code"
P=${PY:-python}
O=../out; R=../results; D=../data/gazebasevr/data
mkdir -p $R
log() { echo "$(date +%T) $*"; }
log A1;          $P partA1.py $O/v3/tables $O/A1 > $O/A1.log 2>&1 &
log X4_v4;       ( $P x4_tex.py $D $O/v3/tables $O/X4_v4 --jobs 20 > $O/X4_v4.log 2>&1 && $P x4_down_share.py $O/X4_v4 > $O/X4_down_share.log 2>&1 ) &
log A4_rel;      $P a4_reliability.py $O/v3 $R/A4_rel > $R/A4_rel.log 2>&1 &
log EKYT prep;   ( $P ekyt.py prep $D $O/v3/tables/meta.parquet $O/pur_win_v3vel.npz --input vel > $O/npz_v3vel.log 2>&1 &&
                   $P ekyt.py prep $D $O/v3/tables/meta.parquet $O/pur_win_v3pos.npz --input pos > $O/npz_v3pos.log 2>&1 ) &
log A2_v4;       $P partA_predict.py $O/v3/tables $O/A2_v4 --jobs 50 > $O/A2_v4.log 2>&1
log A4_v4;       $P partA4.py $O/v3/tables $O/A4_v4 --scores $O/A2_v4 > $O/A4_v4.log 2>&1 &
log A2_v5_lrcv;  $P a2_lrcv_nested.py $O/v3/tables $R/A2_v5_lrcv --jobs 30 > $R/A2_v5_lrcv.log 2>&1
wait
cp -r $O/A2_v4 $O/A4_v4 $O/X4_v4 $R/   # results/ keeps copies (rev1_table2.py and rev2_vr_si.py read them there)
log REV1
$P rev1_a1_null.py $O/v3/tables $R/REV1 > $R/rev1_a1_null.log 2>&1
$P rev1_a2_nb.py $O/v3/tables $O/A2_v4 $R/REV1 --power > $R/rev1_a2_nb.log 2>&1
$P rev1_a4_extra.py $O/v3 $O/A4_v4 $R/A4_rel $R/REV1 > $R/rev1_a4_extra.log 2>&1
$P rev1_table2.py > $R/rev1_table2.log 2>&1
$P rev1_partB_extra.py $O/v3 $O/full/ran_saccades.parquet $O/B_v3 $R/A4_rel $R/REV1 > $R/rev1_partB_extra.log 2>&1
$P rev1_b5_reweight.py $O/v3/tables/primary.parquet $O/B5_v3 $R/REV1 > $R/rev1_b5_reweight.log 2>&1
$P rev1_x4.py $O/X4_v4/X4_recordings.parquet $R/REV1 > $R/rev1_x4.log 2>&1
log A3 summary
[ -d $O/ekyt_v3vel ] && $P ekyt_v3_summary.py $R/A3_ekyt_v3 > $R/A3_summary.log 2>&1
log REV2
$P rev2_vr_tables.py .. > $R/rev2_vr_tables.log 2>&1
$P rev2_vr_si.py .. > $R/rev2_vr_si.log 2>&1
log done
