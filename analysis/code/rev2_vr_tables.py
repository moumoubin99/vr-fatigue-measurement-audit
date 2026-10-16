"""REV2 (Virtual Reality revision): reproducibility attachments and sample/quality tables.

CPU only. Reads the frozen v3 tables and the REV1 outputs; writes results/REV2_VR/.
No new exclusions, no refits of the reported models.

  1. A1_a001_table8_cells.csv  cell-by-cell Table VIII check for the fallback label (assignment 001)
  2. labels_round1.csv         reconstructed label vector, missing-label handling, A2/A4 membership
  3. A2_cv_splits.csv          the 10 x 5 stratified participant folds of A2 (regenerated from the seeds)
  4. sample_table.md/.csv      round-1 sample description and predictor missingness
  5. quality_table.md/.csv     RAN/PUR recording quality and event counts by session
  6. A2_table1.csv             fold-mean LR increment with its NB interval, and pooled-OOF points

Usage: python rev2_vr_tables.py [ROOT]   (ROOT = project directory, default ..)
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import partA1  # noqa: E402  (TABLE8 targets and the frozen cell rules)

TAB = os.path.join(ROOT, "data/server_out/out/v3/tables")
FEAT = os.path.join(ROOT, "data/server_out/out/v3/features.parquet")
A4 = os.path.join(ROOT, "data/server_out/out/A4_v3s/A4_changes.parquet")
OUT = os.path.join(ROOT, "results/REV2_VR")
os.makedirs(OUT, exist_ok=True)

meta = pd.read_parquet(os.path.join(TAB, "meta.parquet"))
m = meta[meta["round"] == 1].reset_index(drop=True)
assert len(m) == 407

# ---- 2. label vector ------------------------------------------------------------------
base = np.where(m.eye0.isna(), np.nan, (m.eye0 >= 2).astype(float))
a001 = base.copy()
miss = np.isnan(base)
fb = m.eye1.where(m.eye1.notna())
a001[miss] = np.where(fb[miss].notna() & (fb[miss] >= 2), 1.0, 0.0)
assert int(a001.sum()) == 230 and int((a001 == 0).sum()) == 177, (a001.sum(), (a001 == 0).sum())
chg = pd.read_parquet(A4).set_index("pid")
lab = pd.DataFrame({
    "pid": m.pid, "F0": m.eye0, "F1": m.eye1, "F2": m.eye2,
    "label_F0ge2": base,                       # A2 target; NaN = no F0 (left out of A2)
    "label_a001": a001,                        # A1 fallback rule: F0>=2, else F1>=2, else negative
    "label_source": "F0",
})
lab.loc[miss, "label_source"] = ["F1 fallback" if pd.notna(v) else "missing -> negative" for v in m.eye1[miss]]
lab["in_A2"] = lab.label_F0ge2.notna()
lab["in_A4"] = lab.pid.map(chg["d_eye"]).notna().to_numpy()
lab.to_csv(os.path.join(OUT, "labels_round1.csv"), index=False)

# ---- 1. Table VIII cells for assignment 001 -------------------------------------------
cells = []
for post_choice, welch, paired in itertools.product((0, 1), (False, True), (False, True)):
    d = partA1.table8(m, a001, post_choice, welch, paired)
    d["convention"] = (f"post={'after' if post_choice == 0 else 'between'}|{'welch' if welch else 'student'}|"
                       f"{'paired-subset' if paired else 'available-case'}")
    cells.append(d)
cells = pd.concat(cells, ignore_index=True)
cells.insert(0, "label", "a001 (F0>=2, F1 fallback, missing negative)")
cells["difference"] = cells.computed - cells.reported
cells["tolerance"] = 0.005                    # half a unit of the second decimal (all targets have 2 dp)
cells = cells[["label", "convention", "measure", "cell", "pre", "post", "reported", "computed", "difference",
               "tolerance", "match"]]
cells.to_csv(os.path.join(OUT, "A1_a001_table8_cells.csv"), index=False)
conv = cells.groupby("convention").match.sum().sort_values(ascending=False)
assert conv.max() == 36, conv

# ---- 3. A2 CV splits -----------------------------------------------------------------
d2 = m[m.label.notna()].sort_values("pid").reset_index(drop=True)
y = d2.label.to_numpy(int)
assert len(d2) == 404 and y.sum() == 229
rows = []
for r in range(10):
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(d2, y)):
        rows += [dict(pid=int(p), repeat=r, fold=k) for p in d2.pid.iloc[te]]
pd.DataFrame(rows).sort_values(["repeat", "pid"]).to_csv(os.path.join(OUT, "A2_cv_splits.csv"), index=False)

# ---- 4. sample table -------------------------------------------------------------------
def mean_sd(s):
    s = s.dropna()
    return f"{s.mean():.1f} ({s.std():.1f}), range {s.min():.0f}–{s.max():.0f}; n = {len(s)}"


def counts(s, labels=None):
    vc = s.value_counts(dropna=False)
    parts = []
    for k, v in vc.items():
        name = "missing" if pd.isna(k) else (labels.get(k, k) if labels else k)
        parts.append(f"{name} {v}")
    return "; ".join(parts)


yn = {0.0: "no", 1.0: "yes"}
srows = [
    ("Participants, round 1", str(len(m))),
    ("Age, years, mean (SD)", mean_sd(m.age)),
    ("Gender", counts(m.gender)),
    ("Vision correction reported", counts(m.vision_corr, yn)),
    ("Glasses / contact lenses", f"glasses {int(m.glasses.sum())}; contacts {int(m.contacts.sum())}"),
    ("Eye condition reported", counts(m.eye_cond, yn)),
    ("Prior eye-tracking experience", counts(m.prior_et, yn)),
    ("Dominant eye", counts(m.dom_eye)),
    ("Sleep, hours, mean (SD)", mean_sd(m.sleep_h)),
    ("Caffeine in past 24 h", counts(m.caffeine, yn)),
    ("Headache history", counts(m.headache_hist, yn)),
    ("F0, eye fatigue before S1 (1–5)", counts(m.eye0.astype("Int64").astype(object).where(m.eye0.notna()))),
    ("F1, between sessions (1–5)", counts(m.eye1.astype("Int64").astype(object).where(m.eye1.notna()))),
    ("F2, after S2 (1–5)", counts(m.eye2.astype("Int64").astype(object).where(m.eye2.notna()))),
    ("A1 label (assignment 001)", "positive 230; negative 177"),
    ("A2 set: F0 present", f"{int(lab.in_A2.sum())} (positive {int(np.nansum(base))})"),
    ("A4 set: F1 and F2 present", f"{int(lab.in_A4.sum())}"),
    ("In A2 but not A4 / in A4 but not A2", f"{int((lab.in_A2 & ~lab.in_A4).sum())} / {int((~lab.in_A2 & lab.in_A4).sum())}"),
    ("In both A2 and A4", f"{int((lab.in_A2 & lab.in_A4).sum())}"),
]
D_ITEMS = ["age", "gender", "lens_wear", "lens_type", "sleep_h", "caffeine", "sleep_med", "headache24",
           "alcohol24", "prior_et", "vision_corr", "eye_cond", "headache_hist", "dom_eye", "admin",
           "sleep0", "neck0", "shoulder0", "comfort0"]
missing = {c: int(m[c].isna().sum()) for c in D_ITEMS if c in m}
srows.append(("Missing values in D/S items (count of 407)",
              "; ".join(f"{k} {v}" for k, v in missing.items() if v) or "none"))
sample = pd.DataFrame(srows, columns=["Characteristic", "Value"])
sample.to_csv(os.path.join(OUT, "sample_table.csv"), index=False)
with open(os.path.join(OUT, "sample_table.md"), "w") as fh:
    fh.write(sample.to_markdown(index=False) + "\n")

# ---- 5. quality table ------------------------------------------------------------------
f = pd.read_parquet(FEAT)
f = f[f["round"] == 1]
f["valid_s"] = f.dur_s * (1 - f.loss)
f["ran_n_primary"] = f.ran_n_jumps * f.ran_p_primary
QCOLS = {
    "RAN": [("dur_s", "Duration (s)"), ("valid_s", "Valid time (s)"), ("loss", "Tracking loss (share)"),
            ("precision_rms", "RMS-S2S precision (°)"), ("binoc_v_off", "Binocular vertical offset (°)"),
            ("binoc_v_sd", "Binocular vertical SD (°)"), ("ran_track_err", "Target-tracking error (°)"),
            ("ran_n_jumps", "Target jumps (n)"), ("ran_n_primary", "Primary saccades (n)"),
            ("ran_n_lat", "Eligible latency trials (n)"), ("ran_p_anticipatory", "Anticipatory share")],
    "PUR": [("dur_s", "Duration (s)"), ("valid_s", "Valid time (s)"), ("loss", "Tracking loss (share)"),
            ("precision_rms", "RMS-S2S precision (°)"), ("binoc_v_off", "Binocular vertical offset (°)"),
            ("binoc_v_sd", "Binocular vertical SD (°)"), ("pur_track_err", "Target-tracking error (°)"),
            ("pur_catchup_rate", "Catch-up saccades (per s)")],
}


def med_iqr(s):
    s = s.dropna()
    fmt = (lambda v: f"{v:.0f}") if s.abs().median() >= 100 else (lambda v: f"{v:.3g}")
    return f"{fmt(s.median())} [{fmt(s.quantile(.25))}, {fmt(s.quantile(.75))}]"


qrows = []
for task, cols in QCOLS.items():
    t = f[f.task == task]
    for c, name in cols:
        if c not in t:
            continue
        r = {"Task": task, "Measure": name}
        for s in (1, 2):
            r[f"S{s} median [IQR]"] = med_iqr(t[t.session == s][c])
        r["n recordings (S1/S2)"] = f"{t[(t.session == 1)][c].notna().sum()}/{t[(t.session == 2)][c].notna().sum()}"
        qrows.append(r)
qual = pd.DataFrame(qrows)
qual.to_csv(os.path.join(OUT, "quality_table.csv"), index=False)
with open(os.path.join(OUT, "quality_table.md"), "w") as fh:
    fh.write(qual.to_markdown(index=False) + "\n")

# ---- 6. A2 Table 1 ---------------------------------------------------------------------
nb = pd.read_csv(os.path.join(ROOT, "results/REV1/A2_nb.csv"))
nb = nb[nb.full.str.endswith("+X")]
inc = pd.read_csv(os.path.join(ROOT, "data/server_out/out/A2_v4/A2_increments_s1.csv"))
nb.to_csv(os.path.join(OUT, "A2_table1_lr.csv"), index=False)
inc.to_csv(os.path.join(OUT, "A2_table1_all_learners_source.csv"), index=False)

print("A1 a001 Table VIII matches by convention:\n", conv.to_string())
print(sample.to_string(index=False))
print(qual.to_string(index=False))
print(nb[["base", "full", "d_auc_fold_mean", "nb_lo", "nb_hi", "d_auc_pooled_oof"]].round(4).to_string(index=False))
print(inc.head(30).to_string())
