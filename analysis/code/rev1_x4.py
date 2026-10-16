"""REV1 / X4: leftward-only correction rate and the 'correction is itself a sweep' question (Stage-3 review).

Local input: results/X4_v4/X4_recordings.parquet (per-recording aggregates from code/x4_tex.py; round 1).
(1) corr_rate_left (share of return sweeps, dx <= -15 deg, whose immediately next saccade starts within 400 ms
    and is leftward): recording-level mean per session (as in X4_desc.csv) with participant bootstrap CI,
    sweep-weighted rate, and session 1 + 2 pooled.
(2) How often a 'correction' is itself a >= 15 deg leftward sweep needs per-sweep saccade dx (raw TEX gaze or a
    per-event table); neither is local, so the direct count is NOT computed. Indirect indicator only:
    n_rs - n_entry (a line-entry ISI shared by two sweeps arises when a sweep's correction is itself an
    uncorrected sweep, or when the last sweep has no following ISI).
Participant bootstrap 4000 resamples, seed 20261020.
Usage: python rev1_x4.py X4_RECORDINGS_PARQUET OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

SEED = 20261020
NB = 4000


def main(path, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    r = pd.read_parquet(path)
    r = r[r["round"] == 1]
    rows = []
    for nm, g in (("session 1", r[r.session == 1]), ("session 2", r[r.session == 2]), ("sessions 1+2", r)):
        g = g[g.corr_rate_left.notna()]
        pids = g.pid.unique()
        by = {p: d for p, d in g.groupby("pid")}
        v_mean = g.corr_rate_left.mean()
        v_w = (g.corr_rate_left * g.n_rs).sum() / g.n_rs.sum()
        v_all = (g.corr_rate * g.n_rs).sum() / g.n_rs.sum()
        bm, bw = [], []
        for _ in range(NB):
            d = pd.concat([by[p] for p in rng.choice(pids, len(pids))])
            bm.append(d.corr_rate_left.mean())
            bw.append((d.corr_rate_left * d.n_rs).sum() / d.n_rs.sum())
        gap = (g.n_rs - g.n_entry).sum()
        rows.append(dict(set=nm, n_recordings=len(g), n_participants=len(pids), n_sweeps=int(g.n_rs.sum()),
                         left_rate_recording_mean=v_mean, lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5),
                         left_rate_sweep_weighted=v_w, w_lo=np.percentile(bw, 2.5), w_hi=np.percentile(bw, 97.5),
                         frozen_rate_sweep_weighted=v_all, n_rs_minus_n_entry=int(gap),
                         share_rs_minus_entry=gap / g.n_rs.sum()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "X4_left.csv"), index=False)
    md = ["# REV1 X4: leftward-only correction rate\n",
          "Recording-level mean (as reported in X4_desc.csv) and sweep-weighted rate, participant bootstrap 95% "
          "CI. `frozen_rate_sweep_weighted` = frozen rule (leftward or |dx| < 5 deg).\n",
          t.round(4).to_markdown(index=False), "",
          "Whether a 'correction' is itself a >= 15 deg leftward sweep cannot be counted from local data "
          "(needs per-sweep dx; raw TEX gaze not local). `n_rs_minus_n_entry` is only an indirect indicator "
          "(sweeps sharing a line-entry ISI: a sweep whose correction is itself an uncorrected sweep, or a final "
          "sweep without a following ISI); it is not the requested count.\n"]
    open(os.path.join(out, "X4_left.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:3])
