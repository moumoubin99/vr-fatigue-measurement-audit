"""Build analysis tables from extracted features + participant metadata.

Outputs (OUT_DIR):
  primary.parquet   - one row per (round, pid, session): O1-O3, technical quality, exclusion flags
  ran_primary.parquet - primary RAN saccades (trial level) used for O1 (main sequence is refit in partB)
  wide.parquet      - one row per (round, pid, session): all task features prefixed by task
  meta.parquet      - one row per (round, pid): covariates and ratings
  mainseq.json      - pooled main-sequence fit (round 1, session 1, equal weight per recording)
  exclusions.md     - exclusion counts by session, baseline label and transition
"""
import json
import os
import sys

import numpy as np
import pandas as pd

AMP_LO, AMP_HI = 5.0, 25.0
MIN_N_O1 = 10   # frozen: >= 10 primary 5-25 deg saccades
MIN_N_O2 = 1    # frozen: >= 1 eligible primary saccade

META_MAP = {
    "Recording round": "round", "Participant ID": "pid", "Date": "date", "Age": "age",
    "Gender": "gender", "Wear corrective lenses": "lens_wear", "Lens type": "lens_type",
    "Hours slept last night": "sleep_h", "Caffeine in past 24 hours": "caffeine",
    "Medicine to aid sleep in past 24 hours": "sleep_med", "Headache in past 24 hours": "headache24",
    "Alcohol in past 24 hours": "alcohol24", "Previously involved in an eye tracking study": "prior_et",
    "Require vision correction": "vision_corr",
    "Glaucoma/ambylopia/eye injury/blindess/strabismus/nystagmus/dry eyes/retinitis pigmentosa/macular degeneration": "eye_cond",
    "Suffer from headaches": "headache_hist", "Dominant eye": "dom_eye",
    "Recording administrator ID": "admin",
}
for when, tag in [("before", 0), ("between", 1), ("after", 2)]:
    META_MAP[f"Eye fatigue {when} sessions"] = f"eye{tag}"
    META_MAP[f"Sleepiness {when} sessions"] = f"sleep{tag}"
    META_MAP[f"Neck fatigue {when} sessions"] = f"neck{tag}"
    META_MAP[f"Shoulder fatigue {when} sessions"] = f"shoulder{tag}"
    META_MAP[f"Physical comfort {when} sessions"] = f"comfort{tag}"
for when, tag in [("between", 1), ("after", 2)]:
    META_MAP[f"Mental effort exerted {when} sessions"] = f"mental{tag}"
    META_MAP[f"Physical effort exerted {when} sessions"] = f"physical{tag}"


def fit_mainseq(ps):
    """log V = a + b log A, each recording weighted equally (weights 1/n_rec)."""
    w = 1.0 / ps.groupby("rec").rec.transform("size").to_numpy()
    b, a = np.polyfit(np.log(ps.amp), np.log(ps.vpeak), 1, w=np.sqrt(w))
    return a, b


def o1_table(ps, ref_mask):
    a, b = fit_mainseq(ps[ref_mask])
    r = np.log(ps.vpeak) - (a + b * np.log(ps.amp))
    return r.groupby(ps.rec).median().rename("O1"), (a, b)


def main(feat_dir, xlsx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    f = pd.read_parquet(os.path.join(feat_dir, "features.parquet"))
    tr = pd.read_parquet(os.path.join(feat_dir, "ran_trials.parquet"))
    key = f[["rec", "round", "pid", "session"]]
    tr = tr.merge(key, on="rec")

    # ---- O1: main-sequence residual on primary saccades with realised amplitude 5-25 deg
    ps = tr[(tr.cls == "primary") & (tr.amp >= AMP_LO) & (tr.amp <= AMP_HI)].copy()
    ps.to_parquet(os.path.join(out_dir, "ran_primary.parquet"))
    ref = (ps["round"] == 1) & (ps.session == 1)
    o1, (a, b) = o1_table(ps, ref)
    # sensitivity: no anticipatory precedence (first candidate in the window evaluated)
    pn = tr[(tr.cls_np == "primary") & (tr.amp_np >= AMP_LO) & (tr.amp_np <= AMP_HI)].copy()
    pn = pn.assign(amp=pn.amp_np, vpeak=pn.vpeak_np)
    o1_np, _ = o1_table(pn, (pn["round"] == 1) & (pn.session == 1))
    n_np = pn.groupby("rec").size().rename("ran_n_ms_np")
    res = np.log(ps.vpeak) - (a + b * np.log(ps.amp))
    diag = {"a": a, "b": b, "n_saccades": int(ref.sum()), "n_recordings": int(ps[ref].rec.nunique()),
            "resid_sd": float(res[ref].std()),
            "resid_by_amp_bin": res[ref].groupby(pd.cut(ps[ref].amp, [5, 8, 12, 16, 20, 25])).median()
            .round(4).astype(float).to_dict()}
    diag["resid_by_amp_bin"] = {str(k): v for k, v in diag["resid_by_amp_bin"].items()}
    json.dump(diag, open(os.path.join(out_dir, "mainseq.json"), "w"), indent=1)

    # ---- primary table (RAN, PUR only)
    ran = f[f.task == "RAN"].set_index("rec")
    ran = ran.join(o1).join(o1_np.rename("O1_np")).join(n_np)
    ran = ran.reset_index().rename(columns={
        "loss": "ran_loss", "precision_rms": "ran_prec", "binoc_v_off": "ran_binoc_off",
        "binoc_v_sd": "ran_binoc_sd", "ran_lat_med": "O2"})
    ran["ran_logprec"] = np.log(ran.ran_prec)
    ran["ok_O1"] = (ran.ran_n_ms >= MIN_N_O1) & ran.O1.notna()
    ran["ok_O2"] = (ran.ran_n_lat >= MIN_N_O2) & ran.O2.notna()
    ran.loc[~(ran.ran_n_ms_np.fillna(0) >= MIN_N_O1), "O1_np"] = np.nan
    rcols = ["round", "pid", "session", "O1", "O2", "ok_O1", "ok_O2", "ran_loss", "ran_prec", "ran_logprec",
             "ran_binoc_off", "ran_binoc_sd", "ran_track_err", "ran_n_ms", "ran_n_lat", "ran_lat_wide_med",
             "ran_p_primary", "ran_p_anticipatory", "ran_p_misdirected", "ran_p_late", "ran_p_none",
             "ran_p_missing", "vel_thr", "O1_np", "ran_lat_np_med", "ran_p_ant_flag"]
    ran = ran[rcols].rename(columns={"vel_thr": "ran_vel_thr"})
    pur = f[f.task == "PUR"].rename(columns={"pur_gain": "O3", "loss": "pur_loss", "precision_rms": "pur_prec",
                                             "binoc_v_off": "pur_binoc_off", "binoc_v_sd": "pur_binoc_sd",
                                             "vel_thr": "pur_vel_thr"})
    pur["pur_logprec"] = np.log(pur.pur_prec)
    pur["ok_O3"] = pur.O3.notna()                       # frozen: all 6 cells present
    pcols = ["round", "pid", "session", "O3", "ok_O3", "pur_loss", "pur_prec", "pur_logprec", "pur_binoc_off",
             "pur_binoc_sd", "pur_track_err", "pur_smooth_s", "pur_catchup_rate", "pur_vel_thr",
             "pur_gain_5", "pur_gain_10", "pur_gain_20"]
    prim = ran.merge(pur[pcols], on=["round", "pid", "session"], how="outer")
    for o in ("O1", "O2", "O3"):
        prim[f"ok_{o}"] = prim[f"ok_{o}"].fillna(False).astype(bool)
        prim.loc[~prim[f"ok_{o}"], o] = np.nan
    prim.to_parquet(os.path.join(out_dir, "primary.parquet"))

    # ---- wide table of every feature, prefixed by task
    drop = {"rec", "round", "pid", "session", "task", "dur_s", "ran_n_jumps"}
    parts = []
    for task, g in f.groupby("task"):
        cols = [c for c in g.columns if c not in drop and g[c].notna().any()]
        w = g[["round", "pid", "session"] + cols].copy()
        w.columns = ["round", "pid", "session"] + [f"{task}__{c}" for c in cols]
        parts.append(w.set_index(["round", "pid", "session"]))
    wide = pd.concat(parts, axis=1).reset_index()
    wide = wide.merge(prim[["round", "pid", "session", "O1"]], on=["round", "pid", "session"], how="left")
    wide = wide.rename(columns={"O1": "RAN__ms_resid_pooled"})
    wide.to_parquet(os.path.join(out_dir, "wide.parquet"))

    # ---- metadata
    m = pd.read_excel(xlsx)
    m = m[list(META_MAP)].rename(columns=META_MAP)
    m["female"] = (m.gender == "Female").astype(float)
    m["contacts"] = (m.lens_type == "Contacts").astype(float)
    m["glasses"] = (m.lens_type == "Glasses").astype(float)
    m["label"] = np.where(m.eye0.isna(), np.nan, (m.eye0 >= 2).astype(float))
    m["date_ord"] = (m.date - m.date.min()).dt.days
    m.to_parquet(os.path.join(out_dir, "meta.parquet"))

    # ---- exclusion report (round 1)
    r1 = prim[prim["round"] == 1].merge(m[m["round"] == 1][["pid", "label", "eye0", "eye1"]], on="pid", how="left")
    r1["trans"] = np.sign(r1.eye1 - r1.eye0).map({-1: "decrease", 0: "same", 1: "increase"})
    lines = ["# Exclusions (round 1)\n"]
    for o in ("O1", "O2", "O3"):
        t = r1.groupby(["session"])[f"ok_{o}"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "kept"})
        lines.append(f"## {o} by session\n\n" + t.to_markdown() + "\n")
        t = r1.groupby(["label"], dropna=False)[f"ok_{o}"].agg(["size", "sum"])
        lines.append(f"## {o} by baseline label\n\n" + t.to_markdown() + "\n")
        t = r1.groupby(["trans"], dropna=False)[f"ok_{o}"].agg(["size", "sum"])
        lines.append(f"## {o} by eye-fatigue transition (before -> between)\n\n" + t.to_markdown() + "\n")
    rr = r1.groupby("session")[["ran_p_primary", "ran_p_anticipatory", "ran_p_misdirected", "ran_p_late",
                                "ran_p_none", "ran_p_missing"]].mean()
    lines.append("## RAN trial classes (mean proportion)\n\n" + rr.round(4).to_markdown() + "\n")
    open(os.path.join(out_dir, "exclusions.md"), "w").write("\n".join(lines))

    print("main sequence: logV = %.3f + %.3f logA (n=%d)" % (a, b, int(ref.sum())))
    print("round1 kept:", {o: int(r1[f"ok_{o}"].sum()) for o in ("O1", "O2", "O3")})
    print(r1.groupby("session")[["O1", "O2", "O3"]].describe().T.round(3))
    print("label counts (round 1):", m[m["round"] == 1].label.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:4])
