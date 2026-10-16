"""X4 direct double-sweep count (review follow-up), round 1 TEX.

The X4 correction rule counts the next saccade within 400 ms after a return sweep as a post-sweep saccade if it is
leftward or small. A second return sweep (dx <= -15 deg) therefore counts as a correction, and it is also counted
as a sweep of its own. This script uses the same detector and event definitions as x4_tex.py and counts, per
recording:
  n_rs              return sweeps (as in x4_tex.py)
  n_double          sweeps whose post-sweep saccade is itself a return sweep
  double_share      n_double / n_rs
  corr_rate_left    share of sweeps followed by a leftward post-sweep saccade (as in x4_tex.py; sanity check)
  corr_rate_left_x  as corr_rate_left, but the post-sweep saccade must not be a return sweep (-15 < dx < 0)
  corr_rate_x       any post-sweep saccade (leftward or |dx| < 5 deg) that is not a return sweep
  rs_rate_merged    (n_rs - n_double) / valid minutes: a sweep pair counted as one line transition
Outputs: X4_double_recordings.parquet, X4_double_desc.csv (session means, participant bootstrap),
X4_double_paired.csv (S2 - S1), X4_double_summary.md.
Usage: python x4_double_sweep.py VR_DATA_DIR OUT_DIR [--jobs 60] [--x4 X4_recordings.parquet]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import gazefeat as gf
from x4_tex import CORR_SMALL, CORR_WIN_MS, RS_DX, boot_mean

MEAS = ["n_rs", "n_double", "double_share", "corr_rate_left", "corr_rate_left_x", "corr_rate", "corr_rate_x",
        "rs_rate", "rs_rate_merged"]


def one(path, cfg=gf.DEFAULT):
    meta = gf.parse_name(path)
    sig = gf.load(path, cfg, "vr")
    n = len(sig["t"])
    x, y = sig["x"], sig["y"]
    _, _, speed = gf.velocity(x, y, cfg)
    sac, _, _ = gf.detect_saccades(x, y, speed, cfg)
    valid_min = np.isfinite(x).sum() / cfg.fs / 60.0
    row = dict(rec=os.path.basename(path)[:-4], **meta, valid_min=valid_min)
    s = sac.reset_index(drop=True)
    rs_idx = np.where(s.dx <= RS_DX)[0]
    row["n_rs"] = len(rs_idx)
    if len(rs_idx) == 0 or valid_min <= 0:
        return row
    win = int(round(CORR_WIN_MS / 1000 * cfg.fs))
    dbl, cl, clx, c, cx = [], [], [], [], []
    for i in rs_idx:
        off = int(s.at[i, "off"])
        nx = i + 1 < len(s) and int(s.at[i + 1, "on"]) - off <= win
        dx = float(s.at[i + 1, "dx"]) if nx else np.nan
        left = nx and dx < 0
        anyc = nx and (dx < 0 or abs(dx) < CORR_SMALL)
        is_rs = nx and dx <= RS_DX
        dbl.append(is_rs)
        cl.append(left)
        clx.append(left and not is_rs)
        c.append(anyc)
        cx.append(anyc and not is_rs)
    nd = int(np.sum(dbl))
    row.update(n_double=nd, double_share=nd / len(rs_idx), corr_rate_left=float(np.mean(cl)),
               corr_rate_left_x=float(np.mean(clx)), corr_rate=float(np.mean(c)), corr_rate_x=float(np.mean(cx)),
               rs_rate=len(rs_idx) / valid_min, rs_rate_merged=(len(rs_idx) - nd) / valid_min)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vr_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--x4", default=None, help="X4_recordings.parquet of x4_tex.py, for the sanity check")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.vr_dir, "S_1*_S[12]_*_TEX.csv")))
    R = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one)(f) for f in files))
    R.to_parquet(os.path.join(a.out_dir, "X4_double_recordings.parquet"))
    check = ""
    if a.x4:
        X = pd.read_parquet(a.x4).set_index("rec")
        z = R.set_index("rec").join(X[["n_rs", "corr_rate_left", "corr_rate"]], rsuffix="_x4")
        check = ("Sanity check against x4_tex.py: n_rs equal in {}/{} recordings; max |diff| corr_rate_left {:.2e}, "
                 "corr_rate {:.2e}.\n\n").format(int((z.n_rs == z.n_rs_x4).sum()), len(z),
                                                 np.nanmax(np.abs(z.corr_rate_left - z.corr_rate_left_x4)),
                                                 np.nanmax(np.abs(z.corr_rate - z.corr_rate_x4)))
    desc = []
    for sess in (1, 2):
        g = R[R.session == sess]
        for c in MEAS:
            mu, lo, hi, nn = boot_mean(g[c])
            desc.append(dict(session=sess, measure=c, n=nn, mean=mu, ci_lo=lo, ci_hi=hi, median=g[c].median()))
        tot = dict(session=sess, measure="pooled_double_share", n=int(g.n_rs.sum()),
                   mean=g.n_double.sum() / g.n_rs.sum())
        desc.append(tot)
    D = pd.DataFrame(desc)
    w = R.pivot_table(index="pid", columns="session", values=MEAS)
    pair = []
    for c in ["double_share", "corr_rate_left", "corr_rate_left_x", "corr_rate", "corr_rate_x", "rs_rate",
              "rs_rate_merged"]:
        d = (w[(c, 2)] - w[(c, 1)]).dropna()
        mu, lo, hi, nn = boot_mean(d, seed=1)
        pair.append(dict(measure=c, n=nn, s1=w[(c, 1)][d.index].mean(), s2=w[(c, 2)][d.index].mean(), diff=mu,
                         ci_lo=lo, ci_hi=hi, p_wilcoxon=stats.wilcoxon(d).pvalue))
    P = pd.DataFrame(pair)
    D.to_csv(os.path.join(a.out_dir, "X4_double_desc.csv"), index=False)
    P.to_csv(os.path.join(a.out_dir, "X4_double_paired.csv"), index=False)
    with open(os.path.join(a.out_dir, "X4_double_summary.md"), "w") as fh:
        fh.write("# X4 direct double-sweep count (round 1 TEX)\n\n" + __doc__.split("Outputs")[0].strip() + "\n\n"
                 + check + "## Session means (participant bootstrap 95% CI)\n\n" + D.round(4).to_markdown(index=False)
                 + "\n\n## S2 - S1 (content-dependent)\n\n" + P.round(4).to_markdown(index=False) + "\n")
    print(open(os.path.join(a.out_dir, "X4_double_summary.md")).read())


if __name__ == "__main__":
    main()
