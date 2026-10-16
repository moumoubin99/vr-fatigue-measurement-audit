#!/bin/bash
# v3 re-run after the code-review fixes (run02). Outputs go to new *_v3 directories.
# VRFAT_ROOT: analysis directory holding code/, data/gazebasevr/, data/gazebase/RAN and out/.
# PY: Python interpreter of the analysis environment.
set -euo pipefail
: "${VRFAT_ROOT:?set VRFAT_ROOT to the analysis directory}"
cd "$VRFAT_ROOT/code"
P=${PY:-python}
D=../data/gazebasevr/data; X=../data/gazebasevr/participant_details.xlsx; O=../out
mkdir -p $O
ext() { $P extract_all.py $D $O/$1 --jobs $2 ${@:3} > $O/$1.log 2>&1 && $P prep.py $O/$1 $X $O/$1/tables > $O/$1_prep.log 2>&1; }
ext v3 40 & ext v3_ivt 30 --detector fixed & ext v3_sph 30 --geometry sphere &
wait
echo "extraction done $(date)"
$P partA_predict.py $O/v3/tables $O/A2_v3 --jobs 50 > $O/A2_v3.log 2>&1 &
A2=$!
$P partB.py $O/v3/tables $O/B_v3 > $O/B_v3.log 2>&1 &
$P partB.py $O/v3_ivt/tables $O/B_v3_ivt --brief > $O/B_v3_ivt.log 2>&1 &
$P partB.py $O/v3_sph/tables $O/B_v3_sph --brief > $O/B_v3_sph.log 2>&1 &
$P b5_sim.py ../data/gazebase/RAN $D $O/v3/tables/primary.parquet $O/B5_v3 --jobs 40 > $O/B5_v3.log 2>&1 &
$P x4_tex.py $D $O/v3/tables $O/X4_v3 --jobs 20 > $O/X4_v3.log 2>&1 &
wait $A2
$P partA4.py $O/v3/tables $O/A4_v3s --scores $O/A2_v3 > $O/A4_v3s.log 2>&1
wait
echo "ALL DONE $(date)"
