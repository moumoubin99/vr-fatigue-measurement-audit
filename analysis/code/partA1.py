"""A1: label-rule compatibility check against the published summaries of Zafar et al. (2025).

The published summaries are compared with every single-item threshold rule (and a few
composite rules) over the GazeBaseVR round-1 rating columns. Matching establishes
compatibility, not provenance.

Published targets (arXiv 2510.12994 v1):
  Table II  : group n 177 (no fatigue) / 230 (fatigue); age 21.18+-4.55 / 20.62+-3.78;
              hours slept 7.32+-1.42 / 6.83+-1.49.
  Table VIII: group means "Pre-Sess"/"Post-Sess" and t statistics (see TABLE8 below).
Rounding tolerance: a reported value r with d decimals matches if |x - r| <= 0.5*10^-d (+1e-9).

Usage: python partA1.py TABLE_DIR OUT_DIR
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

T2 = {"n": (177, 230), "age": ((21.18, 4.55), (20.62, 3.78)), "sleep_h": ((7.32, 1.42), (6.83, 1.49))}
# measure: (pre col, post col candidates), means (noF pre, noF post, F pre, F post),
#          t: pre-group, post-group, paired noF, paired F, delta-group
TABLE8 = {
    "Sleepiness": (("sleep0",), ("sleep2", "sleep1"), (1.86, 2.16, 2.43, 2.74),
                   (-6.26, -5.71, -3.82, -4.62, -0.15)),
    "Neck fatigue": (("neck0",), ("neck2", "neck1"), (1.39, 1.75, 1.91, 2.27),
                     (-6.21, -5.84, -6.34, -5.53, -0.06)),
    "Physical comfort": (("comfort0",), ("comfort2", "comfort1"), (5.62, 4.94, 5.39, 4.63),
                         (1.76, 2.07, 6.05, 8.29, 0.51)),
    "Mental effort": (("mental1",), ("mental2",), (1.94, 2.01, 2.36, 2.51),
                      (-4.54, -5.03, -1.58, -3.05, -1.19)),
    "Physical effort": (("physical1",), ("physical2",), (1.49, 1.54, 1.74, 1.85),
                        (-2.94, -3.51, -1.29, -2.16, -0.85)),
}
RATING_COLS = [f"{m}{k}" for m in ("eye", "sleep", "neck", "shoulder", "comfort") for k in (0, 1, 2)] + \
              ["mental1", "mental2", "physical1", "physical2"]


def ok(x, r, d=2):
    return bool(np.isfinite(x) and abs(x - r) <= 0.5 * 10 ** -d + 1e-9)


def table2(m, lab):
    out = {}
    for g, name in ((0, "noF"), (1, "F")):
        s = m[lab == g]
        out[f"n_{name}"] = len(s)
        for c in ("age", "sleep_h"):
            out[f"{c}_mean_{name}"] = s[c].mean()
            out[f"{c}_sd_{name}"] = s[c].std(ddof=1)
    return out


def score_t2(r):
    hits = [r["n_noF"] == 177, r["n_F"] == 230]
    for c in ("age", "sleep_h"):
        for gi, name in ((0, "noF"), (1, "F")):
            hits += [ok(r[f"{c}_mean_{name}"], T2[c][gi][0]), ok(r[f"{c}_sd_{name}"], T2[c][gi][1])]
    return int(sum(hits)), len(hits)


def rules(m):
    for c in RATING_COLS:
        v = m[c]
        for k in sorted(v.dropna().unique())[1:]:
            yield f"{c}>={k:g}", np.where(v.isna(), np.nan, (v >= k).astype(float))
    for base in ("eye", "sleep"):
        cols = [f"{base}{k}" for k in (0, 1, 2)]
        mx = m[cols].max(axis=1, skipna=False)
        mn = m[cols].mean(axis=1, skipna=False)
        for k in range(2, 6):
            yield f"max({base}0..2)>={k}", np.where(mx.isna(), np.nan, (mx >= k).astype(float))
        for k in (1.5, 2.0, 2.5, 3.0):
            yield f"mean({base}0..2)>={k}", np.where(mn.isna(), np.nan, (mn >= k).astype(float))
        d = m[f"{base}2"] - m[f"{base}0"]
        yield f"{base}2-{base}0>0", np.where(d.isna(), np.nan, (d > 0).astype(float))


def table8(m, lab, post_choice, welch, paired_subset):
    rows = []
    for meas, (pre_c, post_cs, means, ts) in TABLE8.items():
        pre, post = pre_c[0], post_cs[min(post_choice, len(post_cs) - 1)]
        g0, g1 = m[lab == 0], m[lab == 1]
        if paired_subset:
            g0, g1 = g0.dropna(subset=[pre, post]), g1.dropna(subset=[pre, post])
        est = [g0[pre].mean(), g0[post].mean(), g1[pre].mean(), g1[post].mean()]
        tt = [stats.ttest_ind(g0[pre].dropna(), g1[pre].dropna(), equal_var=not welch).statistic,
              stats.ttest_ind(g0[post].dropna(), g1[post].dropna(), equal_var=not welch).statistic]
        for g in (g0, g1):
            p = g[[pre, post]].dropna()
            tt.append(stats.ttest_rel(p[pre], p[post]).statistic)
        d0 = (g0[post] - g0[pre]).dropna()
        d1 = (g1[post] - g1[pre]).dropna()
        tt.append(stats.ttest_ind(d0, d1, equal_var=not welch).statistic)
        for i, (e, r) in enumerate(zip(est, means)):
            rows.append(dict(measure=meas, cell=["mean_noF_pre", "mean_noF_post", "mean_F_pre", "mean_F_post"][i],
                             pre=pre, post=post, reported=r, computed=e, match=ok(e, r)))
        for i, (e, r) in enumerate(zip(tt, ts)):
            rows.append(dict(measure=meas, cell=["t_pre_group", "t_post_group", "t_paired_noF", "t_paired_F",
                                                 "t_delta_group"][i],
                             pre=pre, post=post, reported=r, computed=e, match=ok(e, r)))
    return pd.DataFrame(rows)


def main(table_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_parquet(os.path.join(table_dir, "meta.parquet"))
    m = meta[meta["round"] == 1].reset_index(drop=True)
    # 1) candidate rules vs Table II (missing rule values kept missing)
    res = []
    for name, lab in rules(m):
        r = table2(m, lab)
        h, n = score_t2(r)
        res.append(dict(rule=name, hits=h, of=n, n_missing=int(np.isnan(lab).sum()), **r))
    cand = pd.DataFrame(res).sort_values(["hits", "n_missing"], ascending=[False, True])
    cand.to_csv(os.path.join(out_dir, "A1_rules_table2.csv"), index=False)

    # 2) primary rule eye0>=2; enumerate assignments of the missing labels (summary sensitivity only)
    base = np.where(m.eye0.isna(), np.nan, (m.eye0 >= 2).astype(float))
    miss = np.where(np.isnan(base))[0]
    assign = []
    for bits in itertools.product((0, 1), repeat=len(miss)):
        lab = base.copy()
        lab[miss] = bits
        r = table2(m, lab)
        h, n = score_t2(r)
        assign.append(dict(assignment="a" + "".join(map(str, bits)), hits=h, of=n, **r))
    assign = pd.DataFrame(assign)
    assign.to_csv(os.path.join(out_dir, "A1_missing_assignments.csv"), index=False)

    # 3) Table VIII under eye0>=2 for each convention; conventions are reported, never tuned per cell
    t8 = []
    for post_choice, welch, paired in itertools.product((0, 1), (False, True), (False, True)):
        d = table8(m, base, post_choice, welch, paired)
        d["convention"] = f"post={'after' if post_choice == 0 else 'between'}|{'welch' if welch else 'student'}|" \
                          f"{'paired-subset' if paired else 'available-case'}"
        t8.append(d)
    t8 = pd.concat(t8)
    t8.to_csv(os.path.join(out_dir, "A1_table8_cells.csv"), index=False)
    conv = t8.groupby("convention").match.agg(["sum", "size"]).sort_values("sum", ascending=False)

    # Table VIII again with each 230/177-consistent assignment of the missing labels, all conventions
    t8a = []
    cons = assign[(assign.n_noF == 177) & (assign.n_F == 230)]
    for code in cons.assignment:
        lab = base.copy()
        lab[miss] = [int(c) for c in str(code)[1:]]
        for post_choice, welch, paired in itertools.product((0, 1), (False, True), (False, True)):
            d = table8(m, lab, post_choice, welch, paired)
            t8a.append(dict(assignment=str(code), post=("after" if post_choice == 0 else "between"),
                            test=("welch" if welch else "student"),
                            subset=("paired-subset" if paired else "available-case"),
                            t8_hits=int(d.match.sum()), of=len(d),
                            mean_hits=int(d[d.cell.str.startswith("mean")].match.sum()),
                            t_hits=int(d[d.cell.str.startswith("t_")].match.sum())))
    t8a = pd.DataFrame(t8a).sort_values("t8_hits", ascending=False)
    t8a.to_csv(os.path.join(out_dir, "A1_table8_assignments.csv"), index=False)
    best = conv.index[0]

    with open(os.path.join(out_dir, "A1_summary.md"), "w") as fh:
        fh.write("# A1 label-rule compatibility (round 1, n=%d)\n\n" % len(m))
        fh.write("Rounding tolerance: |computed - reported| <= half a unit in the last reported digit.\n\n")
        fh.write("## Candidate rules vs Table II (top 15 of %d)\n\n" % len(cand))
        fh.write(cand.head(15).round(3).to_markdown(index=False) + "\n\n")
        fh.write("## eye0>=2: assignments of the %d missing labels\n\n" % len(miss))
        fh.write(assign.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Table VIII cells matched, by convention (eye0>=2, missing unlabelled)\n\n")
        fh.write(conv.to_markdown() + "\n\n")
        fh.write(f"## Best convention `{best}`: cell detail\n\n")
        fh.write(t8[t8.convention == best].round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Table VIII with 230/177-consistent missing assignments (all conventions)\n\n")
        fh.write(t8a.head(12).to_markdown(index=False) + "\n")
    print(open(os.path.join(out_dir, "A1_summary.md")).read()[:5000])


if __name__ == "__main__":
    main(*sys.argv[1:3])
