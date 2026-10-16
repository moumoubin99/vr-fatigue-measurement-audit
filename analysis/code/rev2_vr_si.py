"""REV2 (Virtual Reality revision): Supplementary Tables S5-S9 and Sections S10-S12 as Markdown.

CPU only. Reads frozen outputs (results/REV1, results/REV2_VR, results/A2_v4, results/A2_v5_lrcv,
results/A3_ekyt_v3, results/A3_diag, results/A3_ctrl, results/A3_rerun, results/REPRO, results/X4_double,
data/server_out/out/v3/features.parquet). No refits. Writes results/REV2_VR/SI_S5_S12.md.

Usage: python rev2_vr_si.py [ROOT]   (ROOT = project directory, default ..)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "REV2_VR")
FEAT = os.path.join(ROOT, "data/server_out/out/v3/features.parquet")
md = []


def table(df):
    return df.to_markdown(index=False, disable_numparse=True) + "\n"


def f3(v):
    return "–" if pd.isna(v) else f"{v:.3f}"


# ---- S5: Table VIII cell-level check, assignment 001 -----------------------------------------
cells = pd.read_csv(os.path.join(OUT, "A1_a001_table8_cells.csv"))
conv = cells.groupby("convention").match.sum().sort_values(ascending=False)
name = lambda c: c.replace("post=after", "After S2").replace("post=between", "Between") \
    .replace("|welch", ", Welch").replace("|student", ", Student").replace("|", ", ")
s5a = pd.DataFrame({"Convention": [name(c) for c in conv.index], "Cells matched": conv.values,
                    "Cells": cells.groupby("convention").size()[conv.index].values})
best = "post=after|welch|paired-subset"
mis = cells[(cells.convention == best) & (~cells.match)]
s5b = pd.DataFrame({"Measure": mis.measure, "Cell": mis.cell.str.replace("_", " "),
                    "Reported": mis.reported.map(lambda v: f"{v:.2f}"),
                    "Computed": mis.computed.map(lambda v: f"{v:.3f}"),
                    "Difference": mis.difference.map(lambda v: f"{v:+.3f}")})
md += ["## Supplementary Table S5. Table VIII cell-level check for the reconstructed label\n",
       "The reconstructed label is assignment 001: F0 ≥ 2, with F1 as fallback for the three participants "
       "without F0, and no rating counted as negative. It gives the published class sizes (230 positive, "
       "177 negative). A cell matched if it differed from the reported value by at most 0.005, which is half "
       "a unit of the second decimal. (a) Matches by reporting convention.\n", table(s5a),
       f"(b) Cells that did not match under the after-S2, Welch, paired-subset convention ({len(mis)} of 45). "
       "All of them are between-group t statistics. All other cells matched.\n", table(s5b),
       "The full cell-by-cell file is `results/REV2_VR/A1_a001_table8_cells.csv` in the analysis package. "
       "It gives, for all 8 conventions and 45 cells, the label, convention, measure, cell, rating columns, "
       "reported value, computed value, difference, tolerance and match flag.\n"]

# ---- S6: sample ---------------------------------------------------------------------------------
lab = pd.read_csv(os.path.join(OUT, "labels_round1.csv"))
samp = pd.read_csv(os.path.join(OUT, "sample_table.csv"))


def sorted_counts(s):
    vc = s.value_counts().sort_index()
    return "; ".join(f"{int(k)}: {v}" for k, v in vc.items()) + f"; missing: {int(s.isna().sum())}"


for col, key in (("F0", "F0,"), ("F1", "F1,"), ("F2", "F2,")):
    samp.loc[samp.Characteristic.str.startswith(key), "Value"] = sorted_counts(lab[col])
samp.loc[samp.Characteristic.str.startswith("A1 label"), "Value"] += " (" + "; ".join(
    f"{k.replace('missing -> negative', 'no rating, counted negative')} {v}" for k, v in lab.label_source.value_counts().items()) + ")"
md += ["## Supplementary Table S6. Round-1 sample, label rules and analysis sets\n",
       "GazeBaseVR round 1 (n = 407). Ratings are on a 1–5 scale, given as rating: count. F0 is before "
       "session 1, F1 between sessions and F2 after session 2. The A2 label is F0 ≥ 2, and participants "
       "without F0 are left out of A2. The A4 set needs both F1 and F2. The per-participant label file is "
       "`results/REV2_VR/labels_round1.csv`, and the A2 folds are in `results/REV2_VR/A2_cv_splits.csv`.\n",
       table(samp)]

# ---- S7: quality --------------------------------------------------------------------------------
qual = pd.read_csv(os.path.join(OUT, "quality_table.csv"))
f = pd.read_parquet(FEAT)
f = f[f["round"] == 1]
tot = f.groupby(["pid", "session"]).dur_s.sum().unstack()
rec = " ".join(f"In S{s}, the mean was {tot[s].mean():.1f} s (SD {tot[s].std():.1f}), median {tot[s].median():.0f} s."
                for s in (1, 2))
md += ["## Supplementary Table S7. RAN and PUR recording quality and event counts by session\n",
       "Median [IQR] over round-1 recordings. Tracking loss is the share of samples without a valid "
       "cyclopean gaze value. RMS-S2S precision is the root mean square of sample-to-sample differences in "
       "inter-saccadic intervals. The binocular offset and SD use the left and right channels. All other outcomes "
       f"use the cyclopean channels. We also summed the recorded time of the five tasks per session. {rec}\n",
       table(qual)]

# ---- S8: A2 estimators --------------------------------------------------------------------------
s1 = pd.read_csv(os.path.join(RES, "A2_v4/A2_summary_s1.csv"))
lr = s1[s1.learner == "lr"].set_index("set").auc
lgb = s1[s1.learner == "lgb"].set_index("set").auc
nest = pd.read_csv(os.path.join(RES, "A2_v5_lrcv/A2_lrcv_nested_auc.csv")).set_index("set")["mean"]
nbs = pd.read_csv(os.path.join(RES, "REV1/A2_nb_single.csv")).set_index("set")
order = ["X", "D", "D+X", "D+Q", "D+Q+X", "D+Qt", "D+Qt+X", "D+S", "D+S+X", "D+S+Q", "D+S+Q+X",
         "D+S+Qt", "D+S+Qt+X"]
s8a = pd.DataFrame({
    "Set": order,
    "LR fold mean [95% CI]": [f"{nbs.auc_fold_mean[k]:.3f} [{nbs.nb_lo[k]:.3f}, {nbs.nb_hi[k]:.3f}]" for k in order],
    "LR pooled OOF": [f3(nbs.auc_pooled_oof[k]) for k in order],
    "Absolute difference": [f"{abs(nbs.auc_fold_mean[k] - nbs.auc_pooled_oof[k]):.3f}" for k in order],
    "LightGBM": [f3(lgb.get(k)) for k in order],
    "Nested LR": [f3(nest.get(k)) for k in order]})
nb = pd.read_csv(os.path.join(RES, "REV1/A2_nb.csv"))
inc = pd.read_csv(os.path.join(RES, "A2_v4/A2_increments_s1.csv"))[["base", "full", "d_auc_lgb"]]
inc_n = pd.read_csv(os.path.join(RES, "A2_v5_lrcv/A2_lrcv_nested_inc.csv"))
nb = nb.merge(inc, on=["base", "full"], how="left").merge(inc_n, on=["base", "full"], how="left")
s8b = pd.DataFrame({
    "Increment": nb.base + " → " + nb.full,
    "LR fold mean [95% CI]": [f"{r.d_auc_fold_mean:+.3f} [{r.nb_lo:+.3f}, {r.nb_hi:+.3f}]" for r in nb.itertuples()],
    "LR pooled OOF": nb.d_auc_pooled_oof.map(lambda v: f"{v:+.3f}"),
    "LightGBM": nb.d_auc_lgb.map(lambda v: "–" if pd.isna(v) else f"{v + 0.0:+.3f}".replace("-0.000", "+0.000")),
    "Nested LR": nb.d_auc_lrcv_nested.map(lambda v: "–" if pd.isna(v) else f"{v:+.3f}")})
gap = (nbs.auc_fold_mean - nbs.auc_pooled_oof).abs().max()
md += ["## Supplementary Table S8. A2 fold-mean and pooled out-of-fold AUCs\n",
       "Reconstructed label, n = 404, 10 repeats of stratified 5-fold participant CV (J = 50 folds). The LR "
       "fold mean averages the 50 per-fold AUCs. The pooled out-of-fold (OOF) AUC pools the held-out "
       "predictions of each repeat and averages over the 10 repeats. The Nadeau–Bengio corrected resampled-t "
       "interval (df 49) is centered on the fold mean. It is approximate and pointwise, with no adjustment for "
       "multiple sets. LightGBM and nested regularized LR are pooled-OOF point estimates only. "
       f"The largest absolute difference between the two LR estimators is {gap:.3f}. (a) Single sets.\n",
       table(s8a), "(b) Increments. A dash marks an increment that was not run for that learner.\n", table(s8b)]

# ---- S9: B5 support and reweighting ------------------------------------------------------------
bins = pd.read_csv(os.path.join(RES, "REV1/B5_reweight_bins.csv"))
s9a = pd.DataFrame({"VR precision decile": bins.vr_decile,
                    "RMS-S2S range (°)": [f"{r.lo:.4f}–{r.hi:.4f}" for r in bins.itertuples()],
                    "Attainable GazeBase recordings": bins.n_attainable,
                    "All GazeBase recordings": bins.n_all})
rw = pd.read_csv(os.path.join(RES, "REV1/B5_reweight.csv"))
rw["pipeline"] = rw["pipe"].str.replace("P250_", "")
g = rw.groupby(["outcome", "pipeline", "variant"])
s9b = g.agg(n=("n", "first"), ess=("ess", "first"), unmatched=("unmatched_vr_mass", "first"),
            ratio=("ratio_ref25", "mean"), rmin=("ratio_ref25", "min"), rmax=("ratio_ref25", "max")).reset_index()
vorder = {"all, unweighted": 0, "attainable, unweighted": 1, "attainable, VR-reweighted (10 bins)": 2,
          "attainable, VR-reweighted (5 bins)": 3}
s9b = s9b[s9b.pipeline != "none"].assign(o=s9b.variant.map(vorder)).sort_values(["outcome", "pipeline", "o"])
s9b = pd.DataFrame({"Outcome": s9b.outcome, "Noise": s9b.pipeline, "Sample": s9b.variant, "n": s9b.n,
                    "ESS": s9b.ess.map(lambda v: f"{v:.1f}"),
                    "Unmatched VR mass": s9b.unmatched.map(lambda v: "–" if pd.isna(v) else f"{v:.2f}"),
                    "Mean ratio to REF25 (range over conditions)":
                        [f"{r.ratio:.3f} ({r.rmin:.3f}–{r.rmax:.3f})" for r in s9b.itertuples()]})
md += ["## Supplementary Table S9. B5 reweighting support by VR precision bin\n",
       "Attainable recordings are GazeBase 1000 Hz RAN recordings whose precision can reach the GazeBaseVR "
       "target after added noise. The per-recording precision was not saved, so this analysis reconstructs it "
       "(`rev1_b5_reweight.py`). A recording counts as attainable if the added noise changed its O2. This proxy "
       f"gives {int(bins.n_attainable.sum())} attainable recordings, against 154 in the original B5 run. Achieved precision is set to the "
       "drawn target. O1 keeps the main-sequence line of the original run, so the intervals do not include a "
       "refit of this line. The reweighting is therefore a proxy-based sensitivity analysis. (a) GazeBase recordings per GazeBaseVR RAN precision decile, binned by their drawn precision "
       "target. The lowest decile has no attainable recording, so 10% of the VR distribution is unmatched.\n", table(s9a),
       "(b) Recovered change at 250 Hz divided by the change recovered by the 1000 Hz reference with identical "
       "smoothing (REF25). The mean is over the injected conditions of each outcome, four for O1 and two "
       "for O2. ESS is the Kish effective "
       "sample size. With 5 bins every bin has support, but the lowest bin holds 1 attainable recording. "
       "Per-condition values with 95% CIs are in `results/REV1/B5_reweight.csv`.\n", table(s9b)]

# ---- S10: A3 -----------------------------------------------------------------------------------
cur = pd.read_csv(os.path.join(RES, "A3_ekyt_v3/A3v3_curves.csv"))
pooled = pd.read_csv(os.path.join(RES, "A3_ekyt_v3/A3v3_pooled.csv")).set_index("input")
last = cur[cur.epoch == cur.groupby(["input", "fit"]).epoch.transform("max")]
last = last[~last.fit.str.startswith("overfit")]
s10 = pd.DataFrame({"Input": last.input.map({"vel": "velocity", "pos": "gaze angle"}), "Fit": last.fit,
                    "Final epoch": last.epoch,
                    "Train-mode batch accuracy": last.train_batch_acc.map(f3),
                    "Eval-mode training accuracy": last.train_eval_acc.map(f3),
                    "Eval-mode training window AUC": last.train_eval_win_auc.map(f3),
                    "Held-out participant AUC": last.test_part_auc.map(f3)})
md += ["## Supplementary Section S10. A3 details\n",
       "**Inputs and split.** The adapted EKYT (Eye Know You Too) DenseNet used 10 s non-overlapping PUR "
       "windows with two input channels. These were horizontal and vertical velocity, or horizontal and vertical "
       "gaze angle. The benchmark does not document its input channels, so both were tested. The \"paper split\" "
       "is our reconstruction of the published description (323 training and 81 test participants). It is not "
       "the original partition, which is not public.\n",
       "**Train–eval gap.** Under velocity input, the training loss went close to zero, and the train-mode "
       "batch accuracy reached 0.98–1.00. In eval mode, the same training windows gave a lower accuracy at the "
       "0.5 threshold in some folds. The window AUC stayed at 0.99–1.00. The table "
       "gives the final checkpoint of each fit.\n", table(s10),
       "A gap between train-mode and eval-mode accuracy with an unchanged AUC is a threshold shift. The ranking "
       "of held-out participants stayed near chance in all fits, so the gap does not explain the failure to "
       "generalize. The diagnosis below tests its cause.\n",
       "**Pooled participant AUC.** We pooled the held-out predictions of the five CV folds. The participant AUC was "
       f"{pooled.pooled_part_auc['vel']:.3f} [{pooled.ci_lo['vel']:.3f}, {pooled.ci_hi['vel']:.3f}] for velocity "
       f"and {pooled.pooled_part_auc['pos']:.3f} [{pooled.ci_lo['pos']:.3f}, {pooled.ci_hi['pos']:.3f}] for gaze "
       "angle. The intervals are participant bootstraps over the saved predictions. They do not include the "
       "uncertainty from refitting the network.\n",
       "**Overfit control.** On 40 participants, the same network reached a training participant AUC of 1.0 "
       "for both inputs, so it can fit the label.\n"]

diag = []
for f in sorted(glob.glob(os.path.join(RES, "A3_diag/*.json"))):
    r = json.load(open(f))
    fit = "paper split" if r["scheme"] == "paper" else f"CV fold {r['fold']}"
    for c, lab in [("eval_fp32", "eval, stored statistics"), ("batchstat_fp32", "batch statistics"),
                   ("recal_fp32", "eval, recomputed statistics")]:
        tr, te = r["conditions"]["train_subset"][c], r["conditions"]["heldout"][c]
        diag.append({"Fit": fit, "Training": r["precision"], "Condition": lab,
                     "Training accuracy": f3(tr["win_acc"]), "Training mean p": f3(tr["mean_p"]),
                     "Held-out window AUC": f3(te["win_auc"]), "Held-out participant AUC": f3(te["part_auc"]),
                     "Held-out mean p": f3(te["mean_p"]), "_o": r["precision"] != "bf16",
                     "_lo": min(b["var_ratio_min"] for b in r["bn"]), "_hi": max(b["var_ratio_max"] for b in r["bn"]),
                     "_sh": max(b["mean_shift_sd_max"] for b in r["bn"])})
if diag:
    s10d = pd.DataFrame(diag).sort_values("_o", kind="stable")
    bn_lo, bn_hi, bn_sh = s10d._lo.min(), s10d._hi.max(), s10d._sh.max()
    s10d = s10d.drop(columns=["_o", "_lo", "_hi", "_sh"])
    md += ["**BatchNorm diagnosis.** We retrained the velocity fits for the paper split and CV "
           "folds 0 and 2 with the same code path and seed. Each fit ran once with bf16 autocast and once in fp32 "
           "(`ekyt_diag.py`, about 1 h per fit). The saved fit records name the GPU as one NVIDIA GeForce RTX 4080 "
       "SUPER (PyTorch 2.14.1+cu130). The final checkpoint was scored on the "
           "4,000-window training subset and on the held-out windows. Batch statistics used shuffled batches of "
           "256 with the stored statistics frozen. Recomputed statistics were exact averages over one pass of all "
           "training windows. On held-out windows, batch statistics are transductive and serve only as a "
           "diagnostic. The base rate is 0.57 in both sets. Scoring was in fp32.\n", table(s10d),
           "With stored statistics, the bf16 fits scored 0.71–0.87 on their own training windows. With batch or "
           "recomputed statistics, the same weights scored 0.99–1.00, and the mean probability matched the base "
           "rate. Scoring in bf16 instead of fp32 changed the training accuracy by less than 0.04 (values in "
           "`results/A3_diag/`). With fp32 training, the gap was absent in two fits and present in CV fold 2. "
           "Thus the gap comes from the BatchNorm running statistics, and bf16 training is not necessary for it.\n",
           f"Per channel, the stored running variance was {bn_lo:.2f}–{bn_hi:.2f} times the recomputed variance, "
           f"and the mean shift was at most {bn_sh:.2f} SD. These small differences moved the training accuracy by "
           "up to 0.28, so the overfit network is very sensitive to normalization. The gap also varied between "
           "runs. CV fold 0 gave 0.479 in the original fit and 0.708 in the replicate, and the paper split gave "
           "1.000 and 0.835. GPU training is not bit-reproducible, so some run-to-run variation is expected.\n",
           "Under all conditions, the held-out window AUC was 0.48–0.54. The held-out participant AUC (81 "
           "participants) was 0.42–0.63. Batch or recomputed statistics moved the held-out mean probability "
           "toward the base rate, but the ranking did not improve. The BatchNorm effect therefore explains the "
           "train–eval gap but not the failure to generalize.\n"]

ctrl_f = os.path.join(RES, "A3_ctrl/A3_ctrl_summary.csv")
if os.path.exists(ctrl_f):
    C = pd.read_csv(ctrl_f)
    lab = {"window": "window split", "window_perm": "window split, permuted labels",
           "session": "session split", "identity": "identity, session split",
           "paper_catchup": "paper split, catch-up rate label", "paper_age": "paper split, age label"}
    s10c = pd.DataFrame({"Input": C.input.map({"vel": "velocity", "pos": "gaze angle"}),
                         "Control": C.scheme.map(lab),
                         "Train / test participants (shared)": [f"{a} / {b} ({c})" for a, b, c in
                                                                zip(C.n_train_p, C.n_test_p, C.n_shared_p)],
                         "Window accuracy (majority)": [f"{f3(a)} ({f3(b)})" for a, b in zip(C.win_acc, C.win_majority)],
                         "Window AUC": C.win_auc.map(f3),
                         "Participant AUC [95% CI]": [f"{f3(a)} [{f3(b)}, {f3(c)}]" for a, b, c in
                                                      zip(C.part_auc, C.part_auc_lo, C.part_auc_hi)],
                         "Participant accuracy": C.part_acc.map(f3)})
    g = C.set_index(["input", "scheme"])
    md += ["**Controls.** We ran six controls for each input with the network, optimizer, batch size, epochs and "
           "bf16 autocast of the main fits (`ekyt_ctrl.py`, one seed). Only the split or the target changed. The "
           "window split put a random, label-stratified 80% of windows into training, so the same people were in "
           "training and test. The permuted version gave each participant a random label from the permuted label "
           "vector (seed 0). The session split trained on session 1 and tested on session 2 of the same 404 "
           "participants. The identity control used the session split with one output class per participant. Its "
           "fatigue score was the summed probability of the label-1 participants. The positive controls used the "
           "paper split with a new participant label. This label marked people above the median of the round-1 "
           "PUR catch-up saccade rate (mean of both sessions) or of age. For these two, the majority rate refers to the new label. Participant "
           "CIs are bootstraps over saved predictions.\n", table(s10c),
           f"The catch-up label transferred to new people (participant AUC {g.part_auc['vel', 'paper_catchup']:.3f} "
           f"and {g.part_auc['pos', 'paper_catchup']:.3f}). The pipeline therefore can rank held-out participants on a "
           "trait that the PUR windows carry. This label comes from the same recordings, so it is an easy target. "
           "The age and fatigue-label analyses did not establish transfer. With the same people in training and test, the participant "
           f"AUC for the fatigue label rose to {g.part_auc['vel', 'window']:.3f} and {g.part_auc['pos', 'window']:.3f}. "
           f"Permuted labels gave {g.part_auc['vel', 'window_perm']:.3f} and {g.part_auc['pos', 'window_perm']:.3f}. "
           "The permuted-label fits thus also exceeded chance, which is consistent with participant reuse. Across sessions, identity transferred weakly. By majority "
           f"vote over windows, the identity network named the right participant for "
           f"{g.id_top1_part['vel', 'identity'] * 100:.1f}% and {g.id_top1_part['pos', 'identity'] * 100:.1f}% of "
           f"session-2 participants (chance 0.25%). For single windows, the values were "
           f"{g.id_top1_win['vel', 'identity'] * 100:.1f}% and {g.id_top1_win['pos', 'identity'] * 100:.1f}%. None of the controls reached the "
           "published 91–94.5% accuracy. Values per checkpoint are in `results/A3_ctrl/`.\n"]

# ---- S11: reproducibility map ------------------------------------------------------------------
smap = pd.DataFrame([
    ("GazeBaseVR recordings, participant_details.xlsx", "extract_all.py → gazefeat.py; prep.py",
     "out/v3/features.parquet, out/v3/tables/*.parquet", "S2, S3, S6, S7"),
    ("out/v3/tables", "partA1.py; rev1_a1_null.py; rev2_vr_tables.py",
     "out/A1, results/REV1/A1*, results/REV2_VR/A1_a001_table8_cells.csv", "Fig. 2, S1, S4, S5"),
    ("out/v3/tables", "partA_predict.py; a2_lrcv_nested.py; rev1_a2_nb.py",
     "out/A2_v4, results/A2_v5_lrcv, results/REV1/A2_nb*.csv", "Table 1, Fig. 3, S8"),
    ("GazeBaseVR PUR recordings", "ekyt.py prep/run (GPU); ekyt_queue_v3.sh; ekyt_v3_summary.py",
     "out/ekyt_v3vel, out/ekyt_v3pos, results/A3_ekyt_v3", "Fig. 4, S10"),
    ("out/pur_win_v3vel.npz", "ekyt_diag.py (GPU)", "results/A3_diag", "A3 text, S10"),
    ("out/pur_win_v3{vel,pos}.npz, out/v3/tables", "ekyt_ctrl.py; ekyt.py; ekyt_ctrl_queue.sh (GPU)",
     "results/A3_ctrl, results/A3_rerun", "A3 text, S10, S11"),
    ("out/v3/tables, out/A2_v3 scores", "partA4.py; a4_reliability.py; rev1_a4_extra.py; rev1_table2.py",
     "out/A4_v3s, results/A4_v3e, results/A4_v4, results/REV1/A4*.csv", "Table 2, Fig. 5"),
    ("out/v3/tables (also v3_ivt, v3_sph)", "partB.py; rev1_partB_extra.py",
     "out/B_v3\\*, results/REV1/B_\\*.csv, results/REV1/SDC.csv", "Table 3, Fig. 6"),
    ("GazeBase 1000 Hz RAN, out/v3/tables/primary.parquet", "b5_sim.py; rev1_b5_reweight.py",
     "out/B5_v3, results/REV1/B5_reweight*.csv", "Fig. 7, S9"),
    ("GazeBaseVR TEX recordings", "x4_tex.py; x4_down_share.py; rev1_x4.py",
     "out/X4_v4, results/REV1/X4_left.csv", "Fig. 8, X4 text"),
    ("GazeBaseVR TEX recordings, out/X4_v4", "x4_double_sweep.py", "results/X4_double", "X4 text, S12"),
    ("results/*", "figures/gen_fig2–7_*.py; draft_fig1_overview.py; draft_fig8_x4_schematic.py",
     "figures/*.pdf", "Figs. 1–8"),
    ("results/REV2_VR, results/REV1", "rev2_vr_si.py", "results/REV2_VR/SI_S5_S12.md", "S5–S12"),
    ("frozen and re-run outputs", "repro_review.sh; repro_compare.py", "results/REPRO", "S11"),
], columns=["Input", "Script (code/)", "Output", "Used in"])
md += ["## Supplementary Section S11. Reproducibility map\n",
       "All scripts are in `code/`. `run_v3.sh` runs extraction, A2, Part B, B5, X4 and A4 in order. Set "
       "`VRFAT_ROOT` to the analysis directory and `PY` to the Python interpreter. The EKYT fits (A3) need one "
       "CUDA GPU and run separately through `ekyt_queue_v3.sh`. `ekyt_ctrl_queue.sh` runs the A3 controls and the "
       "re-run of the EKYT fits. `repro_review.sh` re-runs the review-round "
       "analyses after `run_v3.sh`. All other steps run on CPU. Seeds are fixed in "
       "each script. `code/README_REPRO.md` gives the commands. `out/` stands for the server output directory, "
       "copied to `data/server_out/out/` in the package.\n", table(smap),
       "**Environment.** The original extraction, A2–A4, Part B, B5, X4 and EKYT runs used a Linux conda "
       "environment with Python 3.11. EKYT ran on one NVIDIA RTX 5090 (32 GB). This environment was later moved "
       "to a second server (one NVIDIA RTX 4080 SUPER) and exported there as `code/environment_vrfat.yml` "
       "and `code/requirements_vrfat_freeze.txt`. Its main versions are Python 3.11, PyTorch 2.14 (CUDA 13.0), "
       "NumPy 2.4, pandas 3.0, scikit-learn 1.9, SciPy 1.17, statsmodels 0.15 and LightGBM 4.7. The REV1/REV2 "
       "analyses and the figures ran on CPU with Python 3.13. Their main packages were NumPy 2.4, pandas 3.0, scikit-learn 1.8, SciPy "
       "1.17, statsmodels 0.14, LightGBM 4.6 and Matplotlib 3.10.\n"]

rep = sorted(glob.glob(os.path.join(RES, "REPRO/cmp_*.csv")))
if rep:
    st = pd.concat([pd.read_csv(f).assign(_f=os.path.basename(f)) for f in rep], ignore_index=True)
    st["_s"] = st.status.str.split(":").str[0]
    n_all = len(st)
    n_id = int(st._s.str.match(r"identical").sum())
    n_close = int(st._s.str.match(r"close").sum())
    md += [f"**Reproduction check.** On the second server, we re-ran `run_v3.sh` and `repro_review.sh` from the raw "
           f"recordings into a new directory. `repro_compare.py` then compared {n_all} output tables with the frozen "
           f"outputs. Of these, {n_id} were identical on all shared columns and {n_close} agreed within a relative "
           "tolerance of 1e-6. The others have known causes. Newer script versions add rows or columns to A2_v3, "
           "A4_v3s, X4_v3 and X4_v4. The frozen A2_v4 has an older LR-CV learner that the paper does not report "
           "(Table 1 uses A2_v5_lrcv, which matched). The frozen A4_v4 also includes a pre-review EKYT score that "
           "the paper does not use. With that score supplied, a second run matched all 7 A4_v4 files. "
           "One withdrawn A4 file is not produced. Per-file results are in `results/REPRO/`.\n"]

rr = os.path.join(RES, "A3_rerun")
if os.path.exists(os.path.join(rr, "A3_rerun_pooled.csv")):
    W = pd.read_csv(os.path.join(rr, "A3_rerun_compare.csv"))
    Pp = pd.read_csv(os.path.join(rr, "A3_rerun_pooled.csv")).set_index(["input", "run"])
    H = W[~W.fit.str.startswith("overfit")]
    O = W[W.fit.str.startswith("overfit")]

    def rng(c, d=2):
        return f"{H[c].min():.{d}f}–{H[c].max():.{d}f}"

    def pa(inp, run):
        r = Pp.loc[(inp, run)]
        return f"{r.pooled_part_auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
    md += [f"**EKYT re-run.** On the second server, `ekyt_ctrl_queue.sh` re-ran all {len(W)} EKYT fits with the "
           f"original commands and seeds into new directories. This set has {len(H)} held-out fits (paper split and "
           f"five CV folds for each input) and {('no', 'one', 'two')[len(O)] if len(O) < 3 else len(O)} overfit checks. GPU training is not bit-reproducible, so we "
           f"compared summary values. The held-out window AUC was {rng('win_auc_rerun')} in the re-run and "
           f"{rng('win_auc_frozen')} in the frozen fits. The held-out participant AUC was "
           f"{rng('part_auc_rerun')} and {rng('part_auc_frozen')}, in the same order. The pooled CV participant AUC was {pa('vel', 'rerun')} for velocity and "
           f"{pa('pos', 'rerun')} for gaze angle. The frozen values were {pa('vel', 'frozen')} and "
           f"{pa('pos', 'frozen')}. In eval mode, the re-run fits scored {rng('train_eval_acc_rerun')} on their own "
           f"training windows (frozen {rng('train_eval_acc_frozen')}). This accuracy changed between runs of the same "
           f"fit, as the BatchNorm results in S10 predict. The final training-batch accuracy was "
           f"{rng('final_batch_acc_rerun', 3)} (frozen {rng('final_batch_acc_frozen', 3)}). The overfit checks "
           f"reached a final batch accuracy of {O.final_batch_acc_rerun.min():.3f}–{O.final_batch_acc_rerun.max():.3f}. "
           "Per-fit values are in `results/A3_rerun/`.\n"]

# ---- S12: X4 double-sweep count ---------------------------------------------------------------
xd = os.path.join(RES, "X4_double")
if os.path.exists(os.path.join(xd, "X4_double_desc.csv")):
    D = pd.read_csv(os.path.join(xd, "X4_double_desc.csv")).set_index(["session", "measure"])
    P = pd.read_csv(os.path.join(xd, "X4_double_paired.csv")).set_index("measure")
    rows = [("double_share", "Share of sweeps followed by a second return sweep", lambda v: f"{v:.4f}"),
            ("corr_rate_left", "Leftward post-sweep saccade rate", f3),
            ("corr_rate_left_x", "Leftward rate, second sweeps excluded", f3),
            ("corr_rate", "Any post-sweep saccade rate", f3),
            ("corr_rate_x", "Any rate, second sweeps excluded", f3),
            ("rs_rate", "Return sweeps per min", lambda v: f"{v:.2f}"),
            ("rs_rate_merged", "Return sweeps per min, pairs merged", lambda v: f"{v:.2f}")]

    def ci(sess, m, fm):
        r = D.loc[(sess, m)]
        return f"{fm(r['mean'])} [{fm(r.ci_lo)}, {fm(r.ci_hi)}]"
    s12 = pd.DataFrame([{"Measure": lab, "Session 1": ci(1, m, fm), "Session 2": ci(2, m, fm),
                         "S2 − S1": f"{fm(P.at[m, 'diff'])} [{fm(P.at[m, 'ci_lo'])}, {fm(P.at[m, 'ci_hi'])}]"}
                        for m, lab, fm in rows])
    n1, n2 = int(D.loc[(1, "pooled_double_share"), "n"]), int(D.loc[(2, "pooled_double_share"), "n"])
    md += ["## Supplementary Section S12. X4 double-sweep count\n",
           "The X4 correction rule takes the next saccade within 400 ms after a return sweep. If this saccade is "
           "leftward or smaller than 5°, it counts as a post-sweep saccade. A second return sweep (dx ≤ −15°) "
           "meets this rule, and it also counts as a sweep of its own. `x4_double_sweep.py` uses the same detector "
           "and event rules as `x4_tex.py` and counts these cases directly in all round-1 TEX recordings. Its "
           "sweep counts and correction rates matched `x4_tex.py` in all 814 recordings.\n",
           "The table gives participant means with participant-bootstrap 95% CIs. All rows, including the sweep rates, use "
           "the 406 recordings per session with at least one return sweep. S2 − S1 uses the 405 participants with "
           "sweeps in both sessions. The main-text sweep rate also counts the recording without a sweep. It uses all "
           "407 recordings and pairs and gives 19.67 and 20.38 per min (S2 − S1 0.71).\n", table(s12),
           f"Pooled over sweeps, {D.loc[(1, 'pooled_double_share'), 'mean'] * 100:.2f}% of {n1:,} session-1 sweeps "
           f"and {D.loc[(2, 'pooled_double_share'), 'mean'] * 100:.2f}% of {n2:,} session-2 sweeps were followed by "
           "a second return sweep. This share did not differ between sessions "
           f"(Wilcoxon p = {P.at['double_share', 'p_wilcoxon']:.2f}). Without these cases, both correction rates "
           "fell by 0.4 points, and the S2 − S1 drop in the leftward rate was unchanged. Per-recording values are "
           "in `results/X4_double/`.\n"]

with open(os.path.join(OUT, "SI_S5_S12.md"), "w") as fh:
    fh.write("\n".join(md))
print("\n".join(md))
