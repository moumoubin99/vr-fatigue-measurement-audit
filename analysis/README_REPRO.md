# Analysis package: headset gaze and visual fatigue

This selected analysis package accompanies the study "Evaluating headset gaze measures of visual fatigue: a benchmark audit and paired-session analysis" and Supplementary Sections S1–S15. It contains machine-readable derived outputs, selected implementations, and a minimal dependency specification. Original public recordings and their source questionnaire workbook must be obtained separately. This analysis directory is not a complete raw-data archive. See the repository-level DATA_SOURCES.md and DOWNLOAD_AND_REASSEMBLE.md for original-data attribution and planned release assets.

## Inspection and scope

Start with `MANIFEST.json`, then inspect the reported result files. `MANIFEST.json` contains relative portable paths, file sizes and SHA-256 values only. `code/verify_package.py` performs lightweight manifest, syntax, CSV, JSON, Parquet and NPZ reading checks. It does not run study analyses, train models, call servers or download data.

```
python code/verify_package.py
```

The preparation checked syntax, file formats, recursive structured strings and invariance of the retained scientific numeric fields. These checks are package checks. They are not an independent reproduction of the experiments. Supplementary Section S11 describes earlier recorded reproduction runs and their exceptions. Selected historical comparison tables are in `results/REPRO/`; they refer to the original study's output scope, not to a new run of this package.

## Result map

| Analysis | Files to inspect | Implementations and external requirements |
|---|---|---|
| A1, S1/S4/S5 | `results/A1/*.csv`, `results/REV1/A1*.csv`, `results/REV2_VR/A1_a001_table8_cells.csv`, `results/REV2_VR/labels_round1.csv` | `partA1.py`, `rev1_a1_null.py`, `rev2_vr_tables.py`; derived metadata retained |
| A2, S2/S3/S8/S12 | `results/A2_v4/*.csv`, `results/A2_v5_lrcv/*.csv`, `results/REV1/A2*.csv`, `results/REV2_VR/A2_cv_splits.csv` | `partA_predict.py`, `a2_lrcv_nested.py`, `rev1_a2_nb.py`; frozen OOF/scores in `data/server_out/out/A2_v4/` |
| A3, S10/S13 | `results/A3_ekyt_v3/*.csv`, `results/A3_diag/*.json`, `results/A3_ctrl/`, `results/A3_rerun/*.csv`, `data/server_out/out/ekyt_v3vel/`, `data/server_out/out/ekyt_v3pos/` | `ekyt.py`, `ekyt_diag.py`, `ekyt_ctrl.py`; raw PUR needed to rebuild windows, and a CUDA GPU needed to refit |
| A4, S14 | `results/A4_v4/`, `results/A4_rel/`, `results/REV1/A4*.csv` | `partA4.py`, `a4_reliability.py`, `rev1_a4_extra.py`, `rev1_table2.py`; paired derived changes and compact v3 feature tables retained |
| B, S14 | `results/B_v3/`, `results/B_v3_ivt/`, `results/B_v3_sph/`, `results/REV1/B_*.csv`, `results/REV1/SDC.csv` | `partB.py`, `rev1_partB_extra.py`; compact derived v3 inputs and earlier per-jump saccade table retained |
| B5, S9/S15 | `results/B5_v3/*.csv`, `results/REV1/B5*.csv` | `b5_sim.py` needs original GazeBase RAN and GazeBaseVR PUR; `rev1_b5_reweight.py` uses retained derived outputs |
| Historical X4, outside the current manuscript | `results/X4_v4/`, `results/X4_double/`, `results/REV1/X4_left.csv` | `x4_tex.py`, `x4_double_sweep.py` need public raw TEX; derived recording summaries retained for historical wrapper/comparison compatibility |

The earlier reading return-sweep branch X4 is retained as historical additional output because scripts and recorded comparisons reference it. It is not a contribution of the current manuscript and has no section in the current supplement. Existing file and script names preserve their historical numbering; the result-map references above use the current supplement numbering.

## Environment

`code/requirements_review.txt` gives selected exact package versions from the recorded execution environment. `code/environment_review.yml` gives Python and these dependencies, without a machine prefix or host configuration. The filenames are retained for compatibility. They specify an environment, not a claim that it was newly installed or tested during release preparation. Earlier CPU analyses used a second recorded environment as described in S11. Hardware logs, checkpoints and the complete machine package freeze are omitted.

## Full raw-data route

Use a separate working copy for a new run, because the scripts write results. Retain this frozen package for comparison. Create the layout below with the original public data from `THIRD_PARTY_NOTICES.md`:

```
analysis/
  code/
  data/gazebasevr/data/             original public gaze files
  data/gazebasevr/participant_details.xlsx
  data/gazebase/RAN/                original 1000 Hz RAN files, B5 only
  out/                             generated outputs
  results/                         generated summaries
```

Set `VRFAT_ROOT` to that separate analysis directory and `PY` to the selected Python interpreter. Run `bash code/run_v3.sh`, then `bash code/repro_review.sh`, from the analysis root. The retained wrappers use relative input paths. They preserve original worker counts, so choose local resources before starting them. `repro_review.sh` and the REV2 generators expect `data/server_out/out/` to resolve to the newly generated `out/` tree. In a fresh run, create that relative layout or link it to `out/`. The earlier per-jump input `out/full/ran_saccades.parquet` is included among frozen derived inputs and needs copying into that working output layout for the descriptive within-session proxy.

A3 is a separate GPU step. From `code/`, use `ekyt.py prep` to create both input-window NPZ files from public PUR recordings and the regenerated `out/v3/tables/meta.parquet`. `ekyt_queue_v3.sh` runs the original A3 split/fold set, `ekyt_diag_queue.sh` the six BN diagnosis fits, and `ekyt_ctrl_queue.sh` the twelve controls and fourteen-fit rerun. No A3 refit was executed during release preparation. Raw input windows and GPU checkpoints are not distributed here.

The frozen `A2_v4` contains the historical `LogisticRegressionCV` secondary rows. The current predictor script uses a fully nested secondary learner, and the manuscript reports `A2_v5_lrcv`. Fixed-penalty LR and LightGBM remain the reported comparisons. The frozen `A4_v4` includes an extra pre-review EKYT score not reported in the manuscript. S11 documents the corresponding historical reproduction exceptions.

## Randomness and interpretation

A2 uses 10 repeats of stratified participant-level 5-fold CV with random states 0-9. The exported folds are in `results/REV2_VR/A2_cv_splits.csv`. A1 uses seeds 20261015 and 20261016. EKYT uses one seed per fit and one CV repeat, with nondeterministic GPU kernels and bf16 training. Saved-prediction confidence intervals exclude refit uncertainty. BN batch-statistic scoring on held-out data is transductive and diagnostic only.

A1 establishes label compatibility, not historical identification. A2 does not exclude a small signal in a few gaze features. A3 is an unsuccessful adaptation. A4 cannot resolve unknown rating-change reliability. B does not separate fatigue from practice or task order. B5 ratios are relative to REF25 and do not calibrate absolute HMD accuracy. Its reweighting leaves the least noisy VR decile unsupported. X4 measures gaze proxies without public text coordinates and does not establish a validated cognitive cost.

## Rights and omissions

See `THIRD_PARTY_NOTICES.md` and the unchanged `refs.bib`. Raw recordings, the original participant workbook, input windows, complete training logs, hardware logs, host paths, private author notes, internal assessments, private traces, backups, and withdrawn results are not included in this analysis directory. Selected fit/diagnostic summaries and saved-prediction arrays are scientific outputs retained for the reported controls. Public source-dataset citations and creator names are retained. The repository-level documentation distinguishes planned raw-data release assets from files already present. A manifest/readability pass is not a new reproduction of the study.
