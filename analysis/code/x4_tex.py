"""X4 companion: return-sweep and line-entry cost on the wide GazeBaseVR text panel (TEX), round 1.

Fully automatic, uses the frozen gazefeat protocol (4 ms grid, SG 2/7, adaptive detector).
Events (per recording):
  return sweep    detected saccade with dx <= -15 deg (panel ~51 deg wide)
  corrective      the immediately next detected saccade, if it starts within 400 ms after the sweep's
                  offset and is leftward (dx < 0) or small (|dx| < 5 deg); the frozen rule.
                  Leftward-only is reported as a variant.
  line entry      first inter-saccadic interval (ISI) starting after the sweep (or its correction)
  mid-line ISI    any other ISI that does not immediately precede a sweep
Per-recording measures:
  rs_rate         return sweeps per minute of valid reading time
  rs_amp          median sweep amplitude (deg)
  corr_rate       share of sweeps followed by a corrective saccade (variant: leftward only)
  undershoot      median (sweep landing x - recording's median line-entry x), deg; > 0 = lands right
  rs_time_share   duration of the union of sweep and correction intervals (valid samples only) /
                  valid reading time
  Recordings with valid data but no sweep get rs_rate = rs_time_share = 0; event-specific measures stay
  missing. Paired-table session means use the complete pairs of each measure.
  entry_cost_ms   median line-entry ISI duration - median mid-line ISI duration
  entry_ratio     log(median line-entry ISI / median mid-line ISI)
Construct checks (review round 1; automatic, descriptive only; event definitions unchanged):
  line_dy_med     median vertical change (deg) from the ISI before a sweep to its line-entry ISI
  line_sign_share share of sweeps whose vertical change has the sign of the recording median
  line_down_share share of sweeps with vertical change dy < 0 (fixed sign; review round 3)
  n_dy            eligible transitions: sweeps with a preceding ISI, a following line-entry ISI and finite dy
  line_band_share share of sweeps whose vertical change lies within +-50% of the recording median (same sign),
                  i.e. consistent with a one-line transition
  corr_improve_share  share of candidate corrections that bring the horizontal position closer to the
                  recording's line-start estimate than the sweep landing was
  corr_improve_deg    median reduction (deg) of that horizontal distance
Analyses: descriptives per session (participant bootstrap); paired S2 - S1 contrast (content-dependent:
excerpts differ by session); exploratory change regression d_eye ~ d_measure (HC3), BH-FDR.
Usage: python x4_tex.py VR_DATA_DIR TABLE_DIR OUT_DIR [--jobs 60]
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

import gazefeat as gf

warnings.filterwarnings("ignore")
RS_DX = -15.0
CORR_WIN_MS = 400.0
CORR_SMALL = 5.0
CHECKS = ["line_dy_med", "line_sign_share", "line_band_share", "corr_improve_share", "corr_improve_deg"]
MEASURES = ["rs_rate", "rs_amp", "corr_rate", "corr_rate_left", "undershoot", "rs_time_share",
            "entry_cost_ms", "entry_ratio"]


def one(path, cfg=gf.DEFAULT):
    meta = gf.parse_name(path)
    sig = gf.load(path, cfg, "vr")
    n = len(sig["t"])
    x, y = sig["x"], sig["y"]
    vx, vy, speed = gf.velocity(x, y, cfg)
    sac, pt, on = gf.detect_saccades(x, y, speed, cfg)
    fx, insac = gf.intervals(x, y, sac, n, cfg)
    valid_min = np.isfinite(x).sum() / cfg.fs / 60.0
    row = dict(rec=os.path.basename(path)[:-4], **meta, valid_min=valid_min,
               loss=float(np.mean(~np.isfinite(x))))
    s = sac.reset_index(drop=True)
    rs_idx = np.where(s.dx <= RS_DX)[0]
    row["n_rs"] = len(rs_idx)
    if valid_min > 0 and len(rs_idx) == 0:
        row.update(rs_rate=0.0, rs_time_share=0.0)
    if len(rs_idx) == 0 or len(fx) == 0 or valid_min <= 0:
        return row
    in_rs = np.zeros(n, bool)
    win = int(round(CORR_WIN_MS / 1000 * cfg.fs))
    fx_on = fx["on"].to_numpy()
    corr, corr_left, land, entry_on, entry_dur = [], [], [], [], []
    dy_line, x_corr = [], []
    fx_off = fx["off"].to_numpy()
    entry_set = set()
    pre_rs = set()
    for i in rs_idx:
        off = int(s.at[i, "off"])
        land.append(x[min(off, n - 1)])
        in_rs[int(s.at[i, "on"]):off] = True
        c = c_left = False
        end = off
        if i + 1 < len(s) and int(s.at[i + 1, "on"]) - off <= win:
            nx_on, nx_off, nx_dx = int(s.at[i + 1, "on"]), int(s.at[i + 1, "off"]), float(s.at[i + 1, "dx"])
            c_left = nx_dx < 0                    # immediately next detected saccade
            c = c_left or abs(nx_dx) < CORR_SMALL
            if c:
                end = nx_off
                in_rs[nx_on:end] = True
        x_corr.append(x[min(end, n - 1)] if c else np.nan)
        corr.append(c)
        corr_left.append(c_left)
        j = np.searchsorted(fx_on, end)
        if j < len(fx):
            entry_set.add(j)
            entry_on.append(fx_on[j])
            entry_dur.append(fx.at[j, "dur_ms"])
        # ISI that immediately precedes the sweep
        k = np.searchsorted(fx_off, s.at[i, "on"], side="right") - 1
        if k >= 0:
            pre_rs.add(k)
        if k >= 0 and j < len(fx):
            y_pre = np.nanmedian(y[fx.at[k, "on"]:fx.at[k, "off"]]) if fx.at[k, "off"] > fx.at[k, "on"] else np.nan
            y_post = np.nanmedian(y[fx.at[j, "on"]:fx.at[j, "off"]]) if fx.at[j, "off"] > fx.at[j, "on"] else np.nan
            dy_line.append(y_post - y_pre)
        else:
            dy_line.append(np.nan)
    mid = [k for k in range(len(fx)) if k not in entry_set and k not in pre_rs]
    entry_x = [np.nanmedian(x[fx.at[j, "on"]:fx.at[j, "off"]]) for j in sorted(entry_set)]
    line_start = np.nanmedian(entry_x) if entry_x else np.nan
    mid_dur = fx.loc[mid, "dur_ms"].median() if mid else np.nan
    dy = np.array(dy_line, float)
    dy = dy[np.isfinite(dy)]
    if len(dy):
        med = np.median(dy)
        row.update(line_dy_med=float(med), line_sign_share=float(np.mean(np.sign(dy) == np.sign(med))),
                   line_down_share=float(np.mean(dy < 0)), n_dy=int(len(dy)),
                   line_band_share=float(np.mean((np.sign(dy) == np.sign(med)) & (np.abs(dy - med) <= 0.5 * abs(med)))))
    xc, la = np.array(x_corr, float), np.array(land, float)
    ok = np.isfinite(xc) & np.isfinite(la)
    if ok.any() and np.isfinite(line_start):
        red = np.abs(la[ok] - line_start) - np.abs(xc[ok] - line_start)
        row.update(corr_improve_share=float(np.mean(red > 0)), corr_improve_deg=float(np.median(red)))
    e_dur = np.median(entry_dur) if entry_dur else np.nan
    row.update(rs_rate=len(rs_idx) / valid_min, rs_amp=float(s.loc[rs_idx, "amp"].median()),
               corr_rate=float(np.mean(corr)), corr_rate_left=float(np.mean(corr_left)),
               undershoot=float(np.nanmedian(np.array(land) - line_start)),
               rs_time_share=float((in_rs & np.isfinite(x)).sum() / np.isfinite(x).sum()),
               entry_dur_ms=e_dur, mid_dur_ms=mid_dur, entry_cost_ms=e_dur - mid_dur,
               entry_ratio=float(np.log(e_dur / mid_dur)) if e_dur > 0 and mid_dur > 0 else np.nan,
               line_start_x=line_start, n_entry=len(entry_set), n_mid=len(mid))
    return row


def boot_mean(v, nb=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(nb)])
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vr_dir")
    ap.add_argument("table_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.vr_dir, "S_1*_S[12]_*_TEX.csv")))
    R = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one)(f) for f in files))
    R.to_parquet(os.path.join(a.out_dir, "X4_recordings.parquet"))
    meta = pd.read_parquet(os.path.join(a.table_dir, "meta.parquet"))
    m = meta[meta["round"] == 1].set_index("pid")

    desc = []
    for sess in (1, 2):
        z = R[R.session == sess]
        for c in MEASURES + ["n_rs", "entry_dur_ms", "mid_dur_ms"] + CHECKS:
            mu, lo, hi, nn = boot_mean(z[c])
            desc.append(dict(session=sess, measure=c, n=nn, mean=mu, ci_lo=lo, ci_hi=hi,
                             median=z[c].median(), q25=z[c].quantile(.25), q75=z[c].quantile(.75)))
    D = pd.DataFrame(desc)

    w = R.pivot_table(index="pid", columns="session", values=MEASURES)
    pair = []
    for c in MEASURES:
        d = (w[(c, 2)] - w[(c, 1)]).dropna()
        mu, lo, hi, nn = boot_mean(d, seed=1)
        pair.append(dict(measure=c, n=nn, s1=w[(c, 1)][d.index].mean(), s2=w[(c, 2)][d.index].mean(), diff=mu, ci_lo=lo,
                         ci_hi=hi, dz=d.mean() / d.std(), p_wilcoxon=stats.wilcoxon(d).pvalue))
    P = pd.DataFrame(pair)
    P["p_bh"] = multipletests(P.p_wilcoxon, method="fdr_bh")[1]

    assoc = []
    for rating in ("eye", "sleep"):
        dF = (m[f"{rating}2"] - m[f"{rating}1"]).rename("dF")
        for c in MEASURES:
            dx = (w[(c, 2)] - w[(c, 1)]).rename("dX")
            z = pd.concat([dF, dx], axis=1).dropna()
            sd = z.dX.std()
            f = sm.OLS(z.dF, sm.add_constant(z.dX / sd)).fit(cov_type="HC3")
            lo, hi = f.conf_int().loc["dX"]
            assoc.append(dict(rating=rating, measure=c, n=len(z), slope_per_sd=f.params["dX"], ci_lo=lo,
                              ci_hi=hi, p=f.pvalues["dX"]))
    A = pd.DataFrame(assoc)
    A["p_bh"] = np.nan
    for r in ("eye", "sleep"):
        k = A.rating == r
        A.loc[k, "p_bh"] = multipletests(A.loc[k, "p"], method="fdr_bh")[1]

    for nm, t in (("desc", D), ("paired", P), ("assoc", A)):
        t.to_csv(os.path.join(a.out_dir, f"X4_{nm}.csv"), index=False)
    with open(os.path.join(a.out_dir, "X4_summary.md"), "w") as fh:
        fh.write("# X4 return sweeps and line entry on the VR text panel (round 1)\n\n")
        fh.write(f"Recordings: {len(R)}; with >=1 return sweep: {(R.n_rs > 0).sum()}\n\n")
        fh.write("## Descriptives (participant bootstrap 95% CI)\n\n" + D.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Session contrast S2 - S1 (content-dependent: excerpts differ by session)\n\n"
                 + P.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Exploratory: d_rating(after - between) ~ d_measure (per SD), HC3, BH within rating\n\n"
                 + A.round(4).to_markdown(index=False) + "\n")
    print(open(os.path.join(a.out_dir, "X4_summary.md")).read())


if __name__ == "__main__":
    main()
