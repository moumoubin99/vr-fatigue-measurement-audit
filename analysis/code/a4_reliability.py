"""A4 measurement-uncertainty checks (review round 1).

1. Split-half reliability (round 1) of recording-level O1/O2 and of their session change: RAN jumps split
   odd/even by jump index within each recording. O1 half = median main-sequence residual against the frozen
   session-1 reference line (mainseq.json), >= 5 saccades per half; O2 half = median latency of primary
   saccades with a 5-25 deg target step. Spearman-Brown corrected. Restricted to the A4 eligible sample.
2. Cross-round retest of change scores (round 1 vs 2 and 1 vs 3; same participants, different days) for O1-O3
   and for the eye/sleep rating change F2 - F1. True change can differ between days, so these correlations
   measure the temporal stability of change, NOT change-score reliability.
3. Observed association r(dF, dX) in round 1 (eye-rating change vs feature change), participant bootstrap.
Review round 3: the former attenuation step (ceiling sqrt(rel_dX * rel_dF) with rel_dF = max cross-round
rating-change r, and disattenuated intervals) is removed. Its rel_dF was a temporal-stability correlation,
not a reliability, so the ceiling was invalid (withdrawn in review round 2). Rating-change reliability and
O3 change-score reliability are not estimated here.
Usage: python a4_reliability.py V3_DIR OUT_DIR      (full run; needs the v3 tables)
       python a4_reliability.py --report OUT_DIR   (rewrite the report from the saved CSVs)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N_BOOT = 2000


def sb(r):
    return 2 * r / (1 + r)


def boot_r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    r = np.corrcoef(a, b)[0, 1]
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        bs.append(np.corrcoef(a[i], b[i])[0, 1])
    return dict(n=len(a), r=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5))


def main(v3, out):
    os.makedirs(out, exist_ok=True)
    T = os.path.join(v3, "tables")
    ms = json.load(open(os.path.join(T, "mainseq.json")))
    prim = pd.read_parquet(os.path.join(T, "primary.parquet"))
    meta = pd.read_parquet(os.path.join(T, "meta.parquet"))
    ps = pd.read_parquet(os.path.join(T, "ran_primary.parquet"))
    key = pd.read_parquet(os.path.join(v3, "features.parquet"))[["rec", "round", "pid", "session"]].drop_duplicates()
    tr = pd.read_parquet(os.path.join(v3, "ran_trials.parquet")).merge(key, on="rec")

    r1 = prim[prim["round"] == 1]
    elig = {o: set(r1[r1[f"ok_{o}"]].groupby("pid").session.nunique().loc[lambda s: s == 2].index)
            for o in ("O1", "O2", "O3")}

    ps = ps.assign(res=np.log(ps.vpeak) - (ms["a"] + ms["b"] * np.log(ps.amp)), half=ps.jump % 2)
    g = ps.groupby(["round", "pid", "session", "half"]).res
    o1h = g.median().where(g.size() >= 5).unstack("half")
    p = tr[(tr.cls == "primary") & (tr.step >= 5) & (tr.step <= 25)].assign(half=lambda z: z.jump % 2)
    o2h = p.groupby(["round", "pid", "session", "half"]).lat_ms.median().unstack("half")
    rows = []
    rel_change = {}
    for nm, h in (("O1", o1h), ("O2", o2h)):
        h1 = h.loc[1]
        h1 = h1[h1.index.get_level_values("pid").isin(elig[nm])]
        for s in (1, 2):
            x = h1.xs(s, level="session")
            b = boot_r(x[0].to_numpy(), x[1].to_numpy())
            rows.append(dict(measure=nm, quantity=f"level, session {s}", **b))
        w = h1.unstack("session")
        b = boot_r((w[(0, 2)] - w[(0, 1)]).to_numpy(), (w[(1, 2)] - w[(1, 1)]).to_numpy())
        rows.append(dict(measure=nm, quantity="change s2-s1", **b))
        rel_change[nm] = sb(b["r"])
    split = pd.DataFrame(rows)
    for c in ("r", "lo", "hi"):
        split[f"sb_{c}"] = sb(split[c])

    # cross-round retest of change scores
    rows = []
    for o in ("O1", "O2", "O3"):
        q = prim[prim[f"ok_{o}"]].pivot_table(index="pid", columns=["round", "session"], values=o)
        for rr in (2, 3):
            if (rr, 1) in q and (rr, 2) in q:
                b = boot_r((q[(1, 2)] - q[(1, 1)]).to_numpy(), (q[(rr, 2)] - q[(rr, 1)]).to_numpy())
                rows.append(dict(measure=o, rounds=f"1 vs {rr}", **b))
    for f in ("eye", "sleep"):
        m = meta.assign(dF=meta[f"{f}2"] - meta[f"{f}1"]).pivot_table(index="pid", columns="round", values="dF")
        for rr in (2, 3):
            if rr in m:
                b = boot_r(m[1].to_numpy(), m[rr].to_numpy())
                rows.append(dict(measure=f"d_{f} (F2-F1)", rounds=f"1 vs {rr}", **b))
    cross = pd.DataFrame(rows)

    # observed association in round 1 (no reliability correction)
    w = r1.pivot(index="pid", columns="session", values=["O1", "O2", "O3"])
    m1 = meta[meta["round"] == 1].set_index("pid")
    rows = []
    for o in ("O1", "O2", "O3"):
        dx = (w[(o, 2)] - w[(o, 1)]).reindex(sorted(elig[o]))
        dF = (m1.eye2 - m1.eye1).reindex(dx.index)
        b = boot_r(dx.to_numpy(), dF.to_numpy())
        rows.append(dict(feature=o, n=b["n"], r_obs=b["r"], r_lo=b["lo"], r_hi=b["hi"]))
    obs = pd.DataFrame(rows)

    for n_, t in (("split", split), ("cross", cross), ("obs", obs)):
        t.to_csv(os.path.join(out, f"A4_rel_{n_}.csv"), index=False)
    report(out)


def report(out):
    split, cross, obs = (pd.read_csv(os.path.join(out, f"A4_rel_{n_}.csv")) for n_ in ("split", "cross", "obs"))
    with open(os.path.join(out, "A4_reliability.md"), "w") as fh:
        fh.write("# A4 measurement uncertainty\n\n"
                 "Review round 3: the attenuation ceiling and disattenuated intervals are removed from this analysis "
                 "(withdrawn in round 2; the cross-round rating-change correlation is temporal stability, not "
                 "reliability). The former output is kept only as `A4_rel_atten_WITHDRAWN.csv`. Rating-change "
                 "reliability is unknown, and O3 change-score reliability has not been estimated here.\n\n")
        fh.write("## Split-half reliability, round 1 (odd/even RAN jumps; sb_* = Spearman-Brown)\n\n"
                 + split.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Cross-round correlation of change scores (temporal stability across days, not reliability)\n\n"
                 + cross.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Observed association: eye-rating change F2-F1 vs feature change, round 1 (no correction)\n\n"
                 + obs.round(3).to_markdown(index=False) + "\n")
    print(open(os.path.join(out, "A4_reliability.md")).read())


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main(*sys.argv[1:3])
