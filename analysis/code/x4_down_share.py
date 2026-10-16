"""X4 fixed-sign downward share (review round 3), computed from the saved per-recording table.

The X4_v4 run predates the `line_down_share` column in x4_tex.py. It saved, per recording, the median vertical
step line_dy_med and line_sign_share = mean(sign(dy) == sign(median dy)) over that recording's eligible
transitions (sweeps with a preceding ISI, a following line-entry ISI and finite dy). Hence, exactly up to
dy == 0 ties (a zero-measure event for medians of continuous gaze):
    share(dy < 0) = line_sign_share        if line_dy_med < 0
                  = 1 - line_sign_share    if line_dy_med > 0
The per-recording count of eligible transitions was not saved; n_rs (all detected sweeps, an upper bound on
it) is used as the weight for the sweep-weighted figure.
Usage: python x4_down_share.py results/X4_v4
"""
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
d = sys.argv[1]
R = pd.read_parquet(os.path.join(d, "X4_recordings.parquet"))
R = R[np.isfinite(R.line_dy_med) & np.isfinite(R.line_sign_share)].copy()
R["down_share"] = np.where(R.line_dy_med < 0, R.line_sign_share, 1 - R.line_sign_share)
rows = []
for (rnd, ses), g in R.groupby(["round", "session"]):
    if rnd != 1:
        continue
    pm = g.groupby("pid").down_share.mean()
    bs = [pm.iloc[rng.integers(0, len(pm), len(pm))].mean() for _ in range(2000)]
    rows.append(dict(session=ses, recordings=len(g), participants=len(pm), sweeps_n_rs=int(g.n_rs.sum()),
                     recording_mean=g.down_share.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                     recording_median=g.down_share.median(),
                     sweep_weighted=np.average(g.down_share, weights=g.n_rs),
                     rec_median_dy_neg=int((g.line_dy_med < 0).sum()),
                     rec_share_ge_0p8=float((g.down_share >= 0.8).mean())))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(d, "X4_down_share.csv"), index=False)
with open(os.path.join(d, "X4_down_share.md"), "w") as fh:
    fh.write("# X4 fixed-sign downward share (dy < 0), round 1 TEX\n\n" + __doc__.split("Usage")[0].strip()
             + "\n\nAggregation unit: recording (share of that recording's eligible transitions with dy < 0); "
               "recording_mean is the unweighted mean over recordings, CI = participant bootstrap (2000). "
               "sweep_weighted weights recordings by n_rs.\n\n" + T.round(4).to_markdown(index=False) + "\n")
print(open(os.path.join(d, "X4_down_share.md")).read())
