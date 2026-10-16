"""REV1 / A1 null calibration (Stage-3 review: EIC M1c, R1 item 8, DA M1).

(a) Re-runs the current code/partA1.py on the v3 meta.parquet into results/REV1/A1 and checks that
    assignment 001 equals the rule eye0.fillna(eye1) >= 2 (no rating -> 0).
(b) Random-partition null: N_PART random 230/177 partitions of the 407 round-1 participants. For each,
    Table II matches (of 10, same half-unit rule as partA1) and Table VIII matches (of 45) under each of
    the same 8 reporting conventions; the maximum over conventions is the null for the "best convention".
(c) Label perturbation: starting from the 001 labels, swap k random positive/negative pairs
    (k = 1, 2, 5, 10; N_PERT draws each), re-score Table II and Table VIII (max over 8 conventions, and the
    fixed paper convention post=after|welch|paired-subset).

The Table II/VIII statistics are computed in vectorised form; the script verifies them against
partA1.table2/table8 (all 8 conventions, labels 001 and 'missing unlabelled') before using them.
Usage: python rev1_a1_null.py TABLE_DIR OUT_DIR
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partA1  # noqa: E402

N_PART = 100000
N_PERT = 2000
KS = (1, 2, 5, 10)
SEED = 20261015
TOL = 0.5e-2 + 1e-9
CONVS = list(itertools.product((0, 1), (False, True), (False, True)))  # post_choice, welch, paired


def conv_name(post_choice, welch, paired):
    return (f"post={'after' if post_choice == 0 else 'between'}|{'welch' if welch else 'student'}|"
            f"{'paired-subset' if paired else 'available-case'}")


def hit(x, r):
    return np.isfinite(x) & (np.abs(x - r) <= TOL)


def gstats(G, x):
    """Group n, mean, sd (ddof=1) for each row of the 0/1 matrix G over the finite entries of x."""
    v = np.isfinite(x).astype(float)
    c = np.nanmean(x)
    x0 = np.where(np.isfinite(x), x - c, 0.0)
    n = G @ v
    s = G @ x0
    ss = G @ (x0 ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
        var = (ss - n * m ** 2) / (n - 1)
    return n, m + c, var


def tind(n0, m0, v0, n1, m1, v1, welch):
    with np.errstate(invalid="ignore", divide="ignore"):
        if welch:
            return (m0 - m1) / np.sqrt(v0 / n0 + v1 / n1)
        sp = ((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2)
        return (m0 - m1) / np.sqrt(sp * (1 / n0 + 1 / n1))


def score_t2(m, G1, G0):
    hits = (G0.sum(1) == 177).astype(int) + (G1.sum(1) == 230).astype(int)
    for c in ("age", "sleep_h"):
        x = m[c].to_numpy(float)
        for gi, G in ((0, G0), (1, G1)):
            n, mu, var = gstats(G, x)
            hits += hit(mu, partA1.T2[c][gi][0]).astype(int) + hit(np.sqrt(var), partA1.T2[c][gi][1]).astype(int)
    return hits


def score_t8(m, G1, G0):
    """Returns array (B, 8) of Table VIII hits per convention (order CONVS)."""
    out = np.zeros((G1.shape[0], len(CONVS)), int)
    for meas, (pre_c, post_cs, means, ts) in partA1.TABLE8.items():
        pre = m[pre_c[0]].to_numpy(float)
        for ci, (pc, welch, paired) in enumerate(CONVS):
            post = m[post_cs[min(pc, len(post_cs) - 1)]].to_numpy(float)
            both = np.isfinite(pre) & np.isfinite(post)
            if paired:
                pre_u, post_u = np.where(both, pre, np.nan), np.where(both, post, np.nan)
            else:
                pre_u, post_u = pre, post
            s = {}
            for g, G in (("0", G0), ("1", G1)):
                s["pre" + g] = gstats(G, pre_u)
                s["post" + g] = gstats(G, post_u)
                dd = np.where(both, pre - post, np.nan)
                nd, md, vd = gstats(G, dd)
                with np.errstate(invalid="ignore", divide="ignore"):
                    s["tp" + g] = md / np.sqrt(vd / nd)
                s["d" + g] = gstats(G, np.where(both, post - pre, np.nan))
            est = [s["pre0"][1], s["post0"][1], s["pre1"][1], s["post1"][1]]
            tt = [tind(*s["pre0"], *s["pre1"], welch), tind(*s["post0"], *s["post1"], welch),
                  s["tp0"], s["tp1"], tind(*s["d0"], *s["d1"], welch)]
            for e, r in zip(est + tt, list(means) + list(ts)):
                out[:, ci] += hit(e, r).astype(int)
    return out


def mats(L):
    L = np.atleast_2d(L)
    G1 = (L == 1).astype(float)
    G0 = (L == 0).astype(float)
    return G1, G0


def main(table_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # (a) regenerate A1 with the current code
    partA1.main(table_dir, os.path.join(out_dir, "A1"))
    meta = pd.read_parquet(os.path.join(table_dir, "meta.parquet"))
    m = meta[meta["round"] == 1].reset_index(drop=True)
    base = np.where(m.eye0.isna(), np.nan, (m.eye0 >= 2).astype(float))
    miss = np.where(np.isnan(base))[0]
    a001 = base.copy()
    a001[miss] = [0, 0, 1]
    rule = (m.eye0.fillna(m.eye1) >= 2).astype(float).to_numpy()   # NaN >= 2 is False -> 0
    lines = ["# REV1 A1 null calibration\n",
             f"Table dir: `{table_dir}`; n = {len(m)} round-1 participants.\n",
             "## (a) Regeneration and assignment 001\n",
             f"- Missing eye0 for pids {m.pid[miss].tolist()}; their eye1 = {m.eye1[miss].tolist()}.",
             f"- Assignment 001 == eye0.fillna(eye1) >= 2 (no rating -> 0): **{bool(np.array_equal(a001, rule))}** "
             f"(class sizes {int((rule == 0).sum())}/{int((rule == 1).sum())}).",
             f"- meta.parquet `label` column == eye0>=2 with NaN kept: "
             f"{bool(np.array_equal(np.nan_to_num(m.label.to_numpy(float), nan=-1), np.nan_to_num(base, nan=-1)))}."]

    # verify the vectorised scorers against partA1
    ver = []
    for nm, lab in (("missing-unlabelled", base), ("a001", a001)):
        G1, G0 = mats(lab)
        t2v = int(score_t2(m, G1, G0)[0])
        t2r = partA1.score_t2(partA1.table2(m, lab))[0]
        t8v = score_t8(m, G1, G0)[0]
        for ci, (pc, w, p) in enumerate(CONVS):
            t8r = int(partA1.table8(m, lab, pc, w, p).match.sum())
            ver.append(dict(label=nm, convention=conv_name(pc, w, p), t2_vec=t2v, t2_ref=t2r,
                            t8_vec=int(t8v[ci]), t8_ref=t8r))
    ver = pd.DataFrame(ver)
    ver.to_csv(os.path.join(out_dir, "A1_vector_check.csv"), index=False)
    assert (ver.t2_vec == ver.t2_ref).all() and (ver.t8_vec == ver.t8_ref).all(), ver
    obs = ver[ver.label == "a001"]
    lines += [f"- Vectorised scorer reproduces partA1 for all 8 conventions x 2 labelings (`A1_vector_check.csv`). "
              f"Under 001: Table II {obs.t2_ref.iloc[0]}/10; Table VIII by convention "
              + ", ".join(f"{r.convention} {r.t8_ref}" for r in obs.itertuples()) + "."]
    a1 = pd.read_csv(os.path.join(out_dir, "A1", "A1_table8_assignments.csv"))
    oth = a1[a1.assignment != "a001"]
    for sub in ("paired-subset", "available-case"):
        same_c = oth[(oth.post == "after") & (oth.test == "welch") & (oth.subset == sub)]
        lines += [f"- Other 230/177-consistent assignments (010, 100) under after|welch|{sub} (001 gives 36/45 "
                  f"there): {sorted(same_c.t8_hits.tolist())}/45."]
    lines += [f"- Other assignments, maximum over all 8 conventions: {oth.t8_hits.max()}/45 "
              f"({oth.sort_values('t8_hits').iloc[-1][['assignment', 'post', 'test', 'subset']].tolist()}).", ""]

    # (b) random partitions
    rng = np.random.default_rng(SEED)
    N = len(m)
    t2_all, t8_all = [], []
    for s in range(0, N_PART, 2000):
        B = min(2000, N_PART - s)
        L = np.zeros((B, N))
        for b in range(B):
            L[b, rng.choice(N, 230, replace=False)] = 1
        G1, G0 = mats(L)
        t2_all.append(score_t2(m, G1, G0))
        t8_all.append(score_t8(m, G1, G0))
    t2 = np.concatenate(t2_all)
    t8 = np.concatenate(t8_all)
    t8max = t8.max(1)
    t8fix = t8[:, CONVS.index((0, True, True))]
    pd.DataFrame({"t2_hits": t2, "t8_max8": t8max, "t8_after_welch_paired": t8fix}
                 | {f"t8_{conv_name(*c)}": t8[:, i] for i, c in enumerate(CONVS)}) \
        .to_csv(os.path.join(out_dir, "A1_null_partitions.csv"), index=False)

    def ub(k, n):  # one-sided 95% Clopper-Pearson upper bound
        from scipy.stats import beta
        return beta.ppf(0.95, k + 1, n - k) if k < n else 1.0

    dist2 = pd.Series(t2).value_counts().sort_index()
    q8 = np.percentile(t8max, [50, 95, 99, 99.9])
    p9 = int((t2 >= 9).sum())
    p36 = int((t8max >= 36).sum())
    lines += ["## (b) Random 230/177 partitions (null)\n",
              f"{N_PART} partitions, seed {SEED}. Table II: the two class-size cells match by construction, so the "
              f"informative cells are 8.\n",
              "Table II hits (of 10), count of partitions:\n",
              pd.DataFrame({"hits": dist2.index, "n": dist2.values,
                            "share": dist2.values / N_PART}).to_markdown(index=False), "",
              f"- P(Table II >= 9/10) = {p9}/{N_PART} = {p9 / N_PART:.5f} (one-sided 95% upper bound "
              f"{ub(p9, N_PART):.5f}). Max observed {t2.max()}/10. P(>= 7/10) = {(t2 >= 7).mean():.5f}; "
              f"P(>= 6/10) = {(t2 >= 6).mean():.5f}.",
              f"- Table VIII, max over 8 conventions: mean {t8max.mean():.2f}, SD {t8max.std(ddof=1):.2f}, median "
              f"{q8[0]:.0f}, 95th pct {q8[1]:.0f}, 99th {q8[2]:.0f}, 99.9th {q8[3]:.0f}, max {t8max.max()}/45.",
              f"- P(Table VIII >= 36/45, best of 8) = {p36}/{N_PART} = {p36 / N_PART:.5f} (one-sided 95% upper "
              f"bound {ub(p36, N_PART):.5f}). P(>= 26/45) = {(t8max >= 26).mean():.5f}; P(>= 17/45) = "
              f"{(t8max >= 17).mean():.5f}; P(>= 10/45) = {(t8max >= 10).mean():.5f}.",
              f"- Fixed convention after|welch|paired-subset: mean {t8fix.mean():.2f}, max {t8fix.max()}/45.",
              f"- Joint P(Table II >= 9 and Table VIII >= 36) = {int(((t2 >= 9) & (t8max >= 36)).sum())}/{N_PART}.", ""]

    # (c) label perturbation from 001
    pos, neg = np.where(a001 == 1)[0], np.where(a001 == 0)[0]
    rows, draws = [], []
    rng = np.random.default_rng(SEED + 1)
    for k in KS:
        L = np.tile(a001, (N_PERT, 1))
        for b in range(N_PERT):
            L[b, rng.choice(pos, k, replace=False)] = 0
            L[b, rng.choice(neg, k, replace=False)] = 1
        G1, G0 = mats(L)
        a2, a8 = score_t2(m, G1, G0), score_t8(m, G1, G0)
        mx, fx = a8.max(1), a8[:, CONVS.index((0, True, True))]
        draws.append(pd.DataFrame({"k": k, "t2": a2, "t8_max8": mx, "t8_after_welch_paired": fx}))
        rows.append(dict(k=k, draws=N_PERT, t2_mean=a2.mean(), t2_sd=a2.std(ddof=1), t2_max=a2.max(),
                         p_t2_ge9=(a2 >= 9).mean(), t8max_mean=mx.mean(), t8max_sd=mx.std(ddof=1),
                         t8max_p95=np.percentile(mx, 95), t8max_max=mx.max(), p_t8max_ge36=(mx >= 36).mean(),
                         t8fix_mean=fx.mean(), t8fix_max=fx.max()))
    pert = pd.DataFrame(rows)
    pert.to_csv(os.path.join(out_dir, "A1_perturb.csv"), index=False)
    pd.concat(draws).to_csv(os.path.join(out_dir, "A1_perturb_draws.csv"), index=False)
    lines += ["## (c) Label perturbation from assignment 001 (k positive/negative pair swaps, 230/177 kept)\n",
              f"{N_PERT} draws per k, seed {SEED + 1}. t8max = best of the 8 conventions; t8fix = "
              "post=after|welch|paired-subset. Unperturbed: Table II 9/10, Table VIII 36/45.\n",
              pert.round(3).to_markdown(index=False), ""]
    open(os.path.join(out_dir, "A1_null.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
