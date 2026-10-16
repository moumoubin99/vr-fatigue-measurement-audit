"""REV1 / Part B additions (Stage-3 review) + SDC.

(a) d_z and ICC(A,1) (and ICC(3,1)) for O1-O3, round 1 (n = 407), with participant-bootstrap 95% CIs;
    point estimates checked against results/B_v3/B1B2.csv and B4.csv.
(b) Calibration-gain check: per participant, change (S2 - S1) in log RAN primary saccadic gain
    (RAN ran_gain_med: median amp/step of primary saccades with 5-25 deg steps) and in log RAN median saccade
    amplitude (sac_amp_med, all RAN saccades). A pure gaze-scale (calibration-gain) change g multiplies
    amplitude and peak velocity by g, shifting the main-sequence residual by (1 - b) log g (b = 0.640,
    mainseq.json). Reported: implied O1 shift (1 - b) * mean dlog g; gain-adjusted O1 contrast
    mean(dO1 - (1 - b) dlog g); and OLS dO1 ~ dlog g with the intercept as the change at zero gain change.
(c) Within-session halves of O1 (S1a, S1b, S2a, S2b; first vs second half of the 79 RAN jumps, by jump
    index). The v3 per-trial table (ran_primary.parquet) is not available locally; the only local per-jump
    data are the pre-review v1 extraction out/full/ran_saccades.parquet (no v3 trial classification). A v1
    proxy is used (saccades with a measured response, realised amplitude 5-25 deg; residual against the v3
    mainseq line; >= 10 per recording, >= 5 per half) and validated against v3 O1 before use.
(d) Round 2 / round 3 O1 change in percent (from results/B_v3/B_rep.csv).
(7) Smallest detectable change, SDC = 1.96 * sqrt(2) * SEM, for O1 and O2 (and O3) with three SEM choices:
    SEM_diff = SD(S2 - S1) / sqrt(2) (=> SDC = 1.96 SD_diff, the LoA half-width), SEM_ICC = SD_pooled
    sqrt(1 - ICC(A,1)), SEM_split = SD_level sqrt(1 - R_split-half,SB) (within-session error only; O1/O2 from
    results/A4_rel, O3 from rev1_a4_extra balanced cell split).
Participant bootstrap: 4000 resamples, seed 20261019.
Usage: python rev1_partB_extra.py V3_DIR V1_SACC_PARQUET B_DIR A4REL_DIR REV1_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partB import icc_all  # noqa: E402

SEED = 20261019
NB = 4000


def pct(x):
    return 100 * (np.exp(x) - 1)


def ci(b):
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def main(v3, v1_sacc, b_dir, rel_dir, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    T = os.path.join(v3, "tables")
    ms = json.load(open(os.path.join(T, "mainseq.json")))
    a_ms, b_ms = ms["a"], ms["b"]
    prim = pd.read_parquet(os.path.join(T, "primary.parquet"))
    r1 = prim[prim["round"] == 1]
    md = ["# REV1 Part B additions and SDC\n", f"Participant bootstrap {NB} resamples, seed {SEED}. "
          f"Main sequence (v3, frozen): a = {a_ms:.4f}, b = {b_ms:.4f}.\n"]

    # (a) dz, ICC
    b1 = pd.read_csv(os.path.join(b_dir, "B1B2.csv")).set_index("outcome")
    b4 = pd.read_csv(os.path.join(b_dir, "B4.csv")).set_index("outcome")
    rows, W = [], {}
    for o in ("O1", "O2", "O3"):
        w = r1[r1[f"ok_{o}"]].pivot(index="pid", columns="session", values=o).dropna()
        W[o] = w
        Y = w[[1, 2]].to_numpy()
        d = Y[:, 1] - Y[:, 0]
        st = icc_all(Y)
        dz = d.mean() / d.std(ddof=1)
        idx = rng.integers(0, len(Y), (NB, len(Y)))
        bdz = [d[i].mean() / d[i].std(ddof=1) for i in idx]
        bia = [icc_all(Y[i])["iccA1"] for i in idx]
        bi3 = [icc_all(Y[i])["icc31"] for i in idx]
        rows.append(dict(outcome=o, n=len(Y), dz=dz, dz_lo=ci(bdz)[0], dz_hi=ci(bdz)[1],
                         dz_B1B2=b1.loc[o, "dz"], iccA1=st["iccA1"], iccA1_lo=ci(bia)[0], iccA1_hi=ci(bia)[1],
                         iccA1_B4=b4.loc[o, "iccA1"], icc31=st["icc31"], icc31_boot_lo=ci(bi3)[0],
                         icc31_boot_hi=ci(bi3)[1], icc31_F_lo=st["icc31_lo"], icc31_F_hi=st["icc31_hi"]))
    ta = pd.DataFrame(rows)
    ta.to_csv(os.path.join(out, "B_dz_icc.csv"), index=False)
    md += ["## (a) d_z and ICC (round 1, session 1 vs 2)\n", ta.round(4).to_markdown(index=False), ""]

    # (b) calibration-gain check
    f = pd.read_parquet(os.path.join(v3, "features.parquet"))
    fr = f[(f.task == "RAN") & (f["round"] == 1)].pivot(index="pid", columns="session",
                                                         values=["ran_gain_med", "sac_amp_med"])
    w1 = W["O1"]
    g = pd.DataFrame({"dO1": w1[2] - w1[1],
                      "dlg": np.log(fr[("ran_gain_med", 2)]) - np.log(fr[("ran_gain_med", 1)]),
                      "dla": np.log(fr[("sac_amp_med", 2)]) - np.log(fr[("sac_amp_med", 1)]),
                      "g1": fr[("ran_gain_med", 1)], "g2": fr[("ran_gain_med", 2)]}).reindex(w1.index)
    rows = []
    for nm, col in (("primary saccadic gain (ran_gain_med)", "dlg"), ("median saccade amplitude (sac_amp_med)", "dla")):
        x = g[["dO1", col]].dropna().to_numpy()
        n = len(x)
        idx = rng.integers(0, n, (NB, n))

        def stats_(z):
            dO, dl = z[:, 0], z[:, 1]
            sl, ic = np.polyfit(dl, dO, 1)
            return dict(mean_dlog=dl.mean(), mean_ratio_pct=pct(dl.mean()), implied=(1 - b_ms) * dl.mean(),
                        dO1=dO.mean(), adj=(dO - (1 - b_ms) * dl).mean(), slope=sl, intercept=ic,
                        r=np.corrcoef(dO, dl)[0, 1])
        s0 = stats_(x)
        bs = pd.DataFrame([stats_(x[i]) for i in idx])
        row = dict(proxy=nm, n=n)
        for k, v in s0.items():
            lo, hi = ci(bs[k])
            row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
        row.update(dO1_pct=pct(s0["dO1"]), adj_pct=pct(s0["adj"]), adj_pct_lo=pct(row["adj_lo"]),
                   adj_pct_hi=pct(row["adj_hi"]), intercept_pct=pct(s0["intercept"]),
                   intercept_pct_lo=pct(row["intercept_lo"]), intercept_pct_hi=pct(row["intercept_hi"]),
                   expected_slope_if_pure_gain=1 - b_ms)
        rows.append(row)
    tb = pd.DataFrame(rows)
    tb.to_csv(os.path.join(out, "B_gaincheck.csv"), index=False)
    md += ["## (b) Calibration-gain check (round 1, O1 sample)\n",
           "`implied` = (1 - b) * mean dlog(gain) = O1 shift expected if the whole gain change were a gaze-scale "
           "artefact; `adj` = mean(dO1 - (1 - b) dlog gain); `intercept` = OLS dO1 ~ dlog gain at dlog gain = 0. "
           "Log units; *_pct = 100(exp - 1).\n",
           tb.T.reset_index().rename(columns={"index": "quantity"}).to_markdown(index=False), ""]

    # (c) within-session halves from the v1 per-jump extraction
    s = pd.read_parquet(v1_sacc)
    key = f[["rec", "round", "pid", "session"]].drop_duplicates()
    n_seq = s.pivot(index="rec", columns="jump", values="step").round(1).astype(str).agg("|".join, axis=1).nunique()
    s = s.merge(key, on="rec", how="inner")
    s = s[(s["round"] == 1) & s.amp.between(5, 25) & np.isfinite(s.vpeak) & (s.vpeak > 0)].copy()
    s["res"] = np.log(s.vpeak) - (a_ms + b_ms * np.log(s.amp))
    s["half"] = np.where(s.jump < 39.5, "a", "b")
    gr = s.groupby(["pid", "session"]).res
    o1p = gr.median().where(gr.size() >= 10).rename("O1_v1")
    v3o = r1[r1.ok_O1].set_index(["pid", "session"]).O1
    cmp_ = pd.concat([o1p, v3o], axis=1, join="inner").dropna()
    wv = cmp_.unstack("session")
    dv1 = (wv[("O1_v1", 2)] - wv[("O1_v1", 1)]).reindex(w1.index)
    dv3 = w1[2] - w1[1]
    val = dict(n_recordings=len(cmp_), r_level=np.corrcoef(cmp_.O1_v1, cmp_.O1)[0, 1],
               mean_level_diff=(cmp_.O1_v1 - cmp_.O1).mean(), n_participants=int(dv1.notna().sum()),
               r_change=np.corrcoef(dv1.dropna(), dv3.reindex(dv1.dropna().index))[0, 1],
               mean_change_v1proxy=dv1.mean(), mean_change_v3=dv3.mean(), n_distinct_jump_sequences=n_seq,
               n_recordings_v1=int(s.rec.nunique()))
    gh = s.groupby(["pid", "session", "half"]).res
    h = gh.median().where(gh.size() >= 5).unstack(["session", "half"])
    h = h.reindex(w1.index)[[(1, "a"), (1, "b"), (2, "a"), (2, "b")]].dropna()
    H = h.to_numpy()
    comps = {"S1a": lambda z: z[:, 0], "S1b": lambda z: z[:, 1], "S2a": lambda z: z[:, 2], "S2b": lambda z: z[:, 3],
             "drift within S1 (S1b-S1a)": lambda z: z[:, 1] - z[:, 0],
             "drift within S2 (S2b-S2a)": lambda z: z[:, 3] - z[:, 2],
             "mean within-session drift": lambda z: ((z[:, 1] - z[:, 0]) + (z[:, 3] - z[:, 2])) / 2,
             "between-session step (S2a-S1b)": lambda z: z[:, 2] - z[:, 1],
             "same-half change S2a-S1a": lambda z: z[:, 2] - z[:, 0],
             "same-half change S2b-S1b": lambda z: z[:, 3] - z[:, 1],
             "total S2-S1 (half means)":lambda z: (z[:, 2] + z[:, 3]) / 2 - (z[:, 0] + z[:, 1]) / 2,
             "step minus mean drift": lambda z: (z[:, 2] - z[:, 1]) - ((z[:, 1] - z[:, 0]) + (z[:, 3] - z[:, 2])) / 2}
    idx = rng.integers(0, len(H), (NB, len(H)))
    rows = []
    for k, fn in comps.items():
        v0 = fn(H).mean()
        bs = np.array([fn(H[i]).mean() for i in idx])
        rows.append(dict(quantity=k, n=len(H), mean_log=v0, lo=ci(bs)[0], hi=ci(bs)[1], pct=pct(v0),
                         pct_lo=pct(ci(bs)[0]), pct_hi=pct(ci(bs)[1])))
    tc = pd.DataFrame(rows)
    tc.to_csv(os.path.join(out, "B_halves.csv"), index=False)
    pd.DataFrame([val]).to_csv(os.path.join(out, "B_halves_validation.csv"), index=False)
    md += ["## (c) Within-session halves of O1 (v1 per-jump proxy; see caveats)\n",
           "Validation of the v1 proxy against v3 O1 (round 1):\n",
           pd.DataFrame([val]).round(4).to_markdown(index=False), "",
           "Halves: jumps 0-39 (a) vs 40-78 (b) within each recording; level values are residuals against "
           "the v3 line (so absolute levels differ from v3 O1 by the validation offset).\n",
           tc.round(4).to_markdown(index=False), ""]

    # (d) rounds 2/3
    rep = pd.read_csv(os.path.join(b_dir, "B_rep.csv"))
    td = rep[rep.outcome == "O1"][["round", "outcome", "n", "diff", "ci95_lo", "ci95_hi", "pct_diff", "pct_ci95_lo",
                                   "pct_ci95_hi", "dz", "p", "p_tost"]]
    r1row = b1.loc[["O1"]].reset_index()[["outcome", "n", "diff", "ci95_lo", "ci95_hi", "pct_diff", "pct_ci95_lo",
                                          "pct_ci95_hi", "dz", "p", "p_tost"]].assign(round=1)
    td = pd.concat([r1row[td.columns], td], ignore_index=True)
    td.to_csv(os.path.join(out, "B_rounds_pct.csv"), index=False)
    md += ["## (d) O1 change by round, percent (100[exp(mean dlog) - 1])\n", td.round(4).to_markdown(index=False), ""]

    # (7) SDC
    spl = pd.read_csv(os.path.join(rel_dir, "A4_rel_split.csv"))
    o3s = pd.read_csv(os.path.join(out, "A4_O3_splithalf.csv"))
    rsplit = {o: spl[(spl.measure == o) & spl.quantity.str.startswith("level")].sb_r.mean() for o in ("O1", "O2")}
    rsplit["O3"] = o3s[(o3s.split.str.startswith("balanced")) & (o3s.quantity == "level, session 1")].sb_r.iloc[0]
    rows = []
    for o in ("O1", "O2", "O3"):
        Y = W[o][[1, 2]].to_numpy()

        def sdcs(Y):
            d = Y[:, 1] - Y[:, 0]
            sd_d = d.std(ddof=1)
            sd_lvl = np.sqrt((Y[:, 0].var(ddof=1) + Y[:, 1].var(ddof=1)) / 2)
            iccA = icc_all(Y)["iccA1"]
            sem = dict(diff=sd_d / np.sqrt(2), icc=sd_lvl * np.sqrt(1 - iccA), split=sd_lvl * np.sqrt(1 - rsplit[o]))
            return {k: 1.96 * np.sqrt(2) * v for k, v in sem.items()}, sem
        s0, sem0 = sdcs(Y)
        idx = rng.integers(0, len(Y), (NB, len(Y)))
        bs = pd.DataFrame([sdcs(Y[i])[0] for i in idx])
        for k in s0:
            lo, hi = ci(bs[k])
            row = dict(outcome=o, sem_source=k, n=len(Y), SEM=sem0[k], SDC=s0[k], SDC_lo=lo, SDC_hi=hi,
                       mean_change=(Y[:, 1] - Y[:, 0]).mean(),
                       frac_abs_change_gt_SDC=float(np.mean(np.abs(Y[:, 1] - Y[:, 0]) > s0[k])))
            if o == "O1":
                row.update(SDC_pct_up=pct(s0[k]), SDC_pct_down=pct(-s0[k]))
            rows.append(row)
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(out, "SDC.csv"), index=False)
    md += ["## (7) Smallest detectable change, SDC = 1.96 sqrt(2) SEM\n",
           "SEM sources: diff = SD(S2-S1)/sqrt(2) (all session-to-session error incl. true individual change "
           "variation); icc = SD_pooled sqrt(1 - ICC(A,1)); split = SD_level sqrt(1 - split-half SB "
           f"reliability of a single-session level; O1 {rsplit['O1']:.3f}, O2 {rsplit['O2']:.3f}, O3 "
           f"{rsplit['O3']:.3f}) = within-session sampling error only. O1 in log units (SDC_pct = percent).\n",
           t7.round(4).to_markdown(index=False), ""]
    open(os.path.join(out, "B_extra.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:6])
