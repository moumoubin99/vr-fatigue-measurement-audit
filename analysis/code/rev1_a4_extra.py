"""REV1 / A4 additions (Stage-3 review): sensitivity and measurement-bound analyses for the null association
between the eye-fatigue rating change dF = F2 - F1 (after minus between sessions, round 1) and the
oculomotor change dX = session 2 - session 1 (O1, O2, O3).

(a) minimum detectable |r| (n = 404, power 0.80; alpha .05 and .0167 two-sided; Fisher z approximation)
(b) bound on the disattenuated correlation, r_hi / sqrt(Rx * Ry), with Rx at the split-half (Spearman-Brown)
    change-reliability lower CI limits (O1 0.493, O2 0.621; also point estimates and O3 from (g)),
    Ry = 0.05 ... 0.7 (rating-change reliability is unknown)
(c) correlations of d_eye with the change in the other rated items (CTT: r12 <= sqrt(R1 R2) so
    R_eye >= r12^2 / R2 >= r12^2 if errors are uncorrelated)
(d) standardized composite rating change (z-scored item changes, comfort reversed) vs dO1-dO3
(e) dF distribution; max attainable Pearson r for a 6-level dF: (i) latent-normal L thresholded at the observed
    cumulative proportions, corr(L, dF) (simulated, 10^6 draws); (ii) comonotone (sorted) pairing of the
    observed dF and each observed dX marginal
(f) sleepiness change vs dO1-dO3
(g) O3 change-score split-half reliability from the six per-cell pursuit gains (features.parquet)
All CIs: participant bootstrap, 4000 resamples, percentile, seed 20261017.
Usage: python rev1_a4_extra.py V3_DIR A4_DIR A4_REL_DIR OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261017
NB = 4000
ITEMS = ["eye", "sleep", "neck", "shoulder", "comfort", "mental", "physical"]
CELLS = [f"pur_gain_{s}{d}" for s in (5, 10, 20) for d in "LR"]


def sb(r):
    return 2 * r / (1 + r)


def boot_r(a, b, rng, nb=NB):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    r = np.corrcoef(a, b)[0, 1]
    idx = rng.integers(0, len(a), (nb, len(a)))
    A, B = a[idx], b[idx]
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    bs = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return dict(n=len(a), r=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5))


def min_r(n, alpha, power=0.80):
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3)))


def exact_power(r, n, alpha):
    """Power of the two-sided Pearson test (Fisher z approximation)."""
    zc = stats.norm.ppf(1 - alpha / 2)
    m = np.arctanh(r) * np.sqrt(n - 3)
    return float(stats.norm.sf(zc - m) + stats.norm.cdf(-zc - m))


def main(v3, a4_dir, rel_dir, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    T = os.path.join(v3, "tables")
    ch = pd.read_parquet(os.path.join(a4_dir, "A4_changes.parquet")).set_index("pid")
    meta = pd.read_parquet(os.path.join(T, "meta.parquet"))
    m1 = meta[meta["round"] == 1].set_index("pid")
    dI = pd.DataFrame({i: m1[f"{i}2"] - m1[f"{i}1"] for i in ITEMS}).reindex(ch.index)
    assert np.allclose(dI.eye.dropna(), ch.d_eye.dropna()) and np.allclose(dI.sleep.dropna(), ch.d_sleep.dropna())
    dO = ch[["d_O1", "d_O2", "d_O3"]]
    md = ["# REV1 A4 additions: sensitivity and measurement bounds for r(dF_eye, dX)\n",
          f"Round 1; dF = F2 (after) - F1 (between); dX = session 2 - session 1. A4 change sample: {len(ch)} "
          f"participants with dO1-dO3; d_eye available for {int(ch.d_eye.notna().sum())}. Participant bootstrap "
          f"{NB} resamples, seed {SEED}.\n"]

    # (a) minimum detectable r
    n = int((ch.d_eye.notna() & dO.notna().all(1)).sum())
    a_rows = [dict(n=n, alpha=a, power=0.80, min_detectable_abs_r=min_r(n, a),
                   power_at_r_0_10=exact_power(0.10, n, a), power_at_r_0_15=exact_power(0.15, n, a),
                   power_at_r_0_20=exact_power(0.20, n, a)) for a in (0.05, 0.05 / 3)]
    ta = pd.DataFrame(a_rows)
    ta.to_csv(os.path.join(out, "A4_mindet.csv"), index=False)
    md += ["## (a) Minimum detectable |r| (two-sided, Fisher z)\n", ta.round(4).to_markdown(index=False), ""]

    # (g) O3 change split-half from per-cell pursuit gains
    f = pd.read_parquet(os.path.join(v3, "features.parquet"))
    f = f[(f.task == "PUR") & (f["round"] == 1)].set_index(["pid", "session"])
    assert np.allclose(f[CELLS].mean(1).dropna(), f.pur_gain.dropna())
    elig3 = ch.index
    w = f[CELLS].unstack("session")
    dc = pd.DataFrame({c: w[(c, 2)] - w[(c, 1)] for c in CELLS}).reindex(elig3)
    lev1 = f.xs(1, level="session")[CELLS].reindex(elig3)
    assert np.allclose(dc.mean(1).dropna(), ch.d_O3.reindex(dc.dropna().index))
    splits = {"direction (L cells vs R cells)": (["pur_gain_5L", "pur_gain_10L", "pur_gain_20L"],
                                                 ["pur_gain_5R", "pur_gain_10R", "pur_gain_20R"]),
              "balanced (5L,10R,20L vs 5R,10L,20R)": (["pur_gain_5L", "pur_gain_10R", "pur_gain_20L"],
                                                      ["pur_gain_5R", "pur_gain_10L", "pur_gain_20R"])}
    g_rows = []
    for nm, (h1, h2) in splits.items():
        for q, src in (("change s2-s1", dc), ("level, session 1", lev1)):
            b = boot_r(src[h1].mean(1), src[h2].mean(1), rng)
            g_rows.append(dict(measure="O3", split=nm, quantity=q, **b, sb_r=sb(b["r"]), sb_lo=sb(b["lo"]),
                               sb_hi=sb(b["hi"])))

    def alpha(x):
        x = x.dropna()
        k = x.shape[1]
        return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))

    al = alpha(dc)
    idx = rng.integers(0, len(dc.dropna()), (NB, len(dc.dropna())))
    X = dc.dropna().to_numpy()
    al_bs = [alpha(pd.DataFrame(X[i])) for i in idx[:1000]]
    g_rows.append(dict(measure="O3", split="Cronbach alpha over 6 cells", quantity="change s2-s1",
                       n=len(X), r=np.nan, lo=np.nan, hi=np.nan, sb_r=al, sb_lo=np.percentile(al_bs, 2.5),
                       sb_hi=np.percentile(al_bs, 97.5)))
    tg = pd.DataFrame(g_rows)
    tg.to_csv(os.path.join(out, "A4_O3_splithalf.csv"), index=False)

    # (b) attenuation bound table
    obs = pd.read_csv(os.path.join(rel_dir, "A4_rel_obs.csv")).set_index("feature")
    spl = pd.read_csv(os.path.join(rel_dir, "A4_rel_split.csv"))
    spl = spl[spl.quantity == "change s2-s1"].set_index("measure")
    o3 = tg[(tg.split.str.startswith("balanced")) & (tg.quantity == "change s2-s1")].iloc[0]
    o3d = tg[(tg.split.str.startswith("direction")) & (tg.quantity == "change s2-s1")].iloc[0]
    rx = {("O1", "SB lower CI"): spl.loc["O1", "sb_lo"], ("O1", "SB point"): spl.loc["O1", "sb_r"],
          ("O2", "SB lower CI"): spl.loc["O2", "sb_lo"], ("O2", "SB point"): spl.loc["O2", "sb_r"],
          ("O3", "SB lower CI (balanced cell split)"): o3.sb_lo, ("O3", "SB point (balanced cell split)"): o3.sb_r,
          ("O3", "SB lower CI (L vs R cell split)"): o3d.sb_lo}
    b_rows = []
    for (o, how), R in rx.items():
        row = dict(feature=o, Rx_source=how, Rx=R, r_obs=obs.loc[o, "r_obs"], r_hi=obs.loc[o, "r_hi"])
        for ry in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
            row[f"Ry={ry}"] = min(1.0, obs.loc[o, "r_hi"] / np.sqrt(R * ry))
        b_rows.append(row)
    tb = pd.DataFrame(b_rows)
    tb.to_csv(os.path.join(out, "A4_atten_bound.csv"), index=False)

    # (c) d_eye vs other item changes
    c_rows = []
    for it in ITEMS[1:]:
        b = boot_r(dI.eye, dI[it], rng)
        c_rows.append(dict(item=f"d_{it}", **b, R_eye_lower_bound_r2=b["r"] ** 2,
                           R_eye_lower_bound_lo2=max(b["lo"], 0) ** 2))
    tc = pd.DataFrame(c_rows)
    tc.to_csv(os.path.join(out, "A4_item_corr.csv"), index=False)

    # (d) composite
    zI = dI.copy()
    zI["comfort"] = -zI["comfort"]          # higher comfort = less fatigue; reversed (r(d_eye, d_comfort) < 0)
    zI = (zI - zI.mean()) / zI.std(ddof=1)
    comp_rows = []
    for nm, cols in (("all 7 items (eye incl.)", ITEMS), ("6 items excl. eye", ITEMS[1:]),
                     ("4 fatigue/sleepiness items (eye, sleep, neck, shoulder)", ["eye", "sleep", "neck", "shoulder"])):
        zz = zI[cols].dropna()
        comp = zz.mean(axis=1)
        a_c = alpha(dI[cols].assign(**({"comfort": -dI["comfort"]} if "comfort" in cols else {})))
        for o in ("O1", "O2", "O3"):
            b = boot_r(comp, dO[f"d_{o}"].reindex(comp.index), rng)
            comp_rows.append(dict(composite=nm, cronbach_alpha=a_c, feature=o, **b))
    td = pd.DataFrame(comp_rows)
    td.to_csv(os.path.join(out, "A4_composite.csv"), index=False)

    # (e) dF distribution + max attainable r
    de = ch.d_eye.dropna()
    vc = de.value_counts().sort_index()
    p = (vc / vc.sum()).to_numpy()
    thr = stats.norm.ppf(np.cumsum(p)[:-1])
    L = np.random.default_rng(SEED + 1).standard_normal(1_000_000)
    D = vc.index.to_numpy()[np.searchsorted(thr, L)]
    r_lat = np.corrcoef(L, D)[0, 1]
    # analytic check: polyserial-to-point relation, r = sum phi(thr_k) * (v_{k+1} - v_k) / sd(D)
    v = vc.index.to_numpy(float)
    sdD = np.sqrt(np.sum(p * v ** 2) - np.sum(p * v) ** 2)
    r_an = np.sum(stats.norm.pdf(thr) * np.diff(v)) / sdD
    e_rows = [dict(quantity="latent normal (observed thresholds), corr(L, dF)", n=len(de), value_sim=r_lat,
                   value_analytic=r_an)]
    for o in ("O1", "O2", "O3"):
        x = dO[f"d_{o}"].reindex(de.index).dropna()
        dd = de.reindex(x.index)
        e_rows.append(dict(quantity=f"comonotone max Pearson r(dF, d{o}) given both observed marginals",
                           n=len(x), value_sim=np.corrcoef(np.sort(dd.to_numpy()), np.sort(x.to_numpy()))[0, 1],
                           value_analytic=np.nan))
    te = pd.DataFrame(e_rows)
    te.to_csv(os.path.join(out, "A4_dF_maxr.csv"), index=False)
    dist = pd.DataFrame({"dF": vc.index, "count": vc.to_numpy(), "prop": p})
    dist.to_csv(os.path.join(out, "A4_dF_dist.csv"), index=False)

    # (f) sleepiness
    f_rows = []
    for o in ("O1", "O2", "O3"):
        f_rows.append(dict(rating="d_sleep", feature=o, **boot_r(dI.sleep, dO[f"d_{o}"], rng)))
        f_rows.append(dict(rating="d_eye", feature=o, **boot_r(dI.eye, dO[f"d_{o}"], rng)))
    tf = pd.DataFrame(f_rows)
    tf.to_csv(os.path.join(out, "A4_sleep.csv"), index=False)

    md += ["## (b) Upper bound on the disattenuated correlation: min(1, r_hi / sqrt(Rx * Ry))\n",
           "r_hi = upper 95% bootstrap limit of the observed r (results/A4_rel/A4_rel_obs.csv). Rx = split-half "
           "change reliability (Spearman-Brown); Ry = assumed rating-change reliability (unknown).\n",
           tb.round(3).to_markdown(index=False), "",
           "## (c) d_eye vs change in the other rated items (F2 - F1)\n",
           "Under classical test theory with uncorrelated errors, r12 <= sqrt(R1 R2), so R_eye >= r12^2 (taking "
           "R2 = 1). Shared response style / common-method error inflates r12, so this is only an indicative bound.\n",
           tc.round(4).to_markdown(index=False), "",
           "## (d) Standardized composite rating change vs dO (comfort reversed; complete cases)\n",
           td.round(4).to_markdown(index=False), "",
           "## (e) Distribution of dF (eye) and maximum attainable Pearson r\n",
           dist.round(4).to_markdown(index=False), "", te.round(4).to_markdown(index=False), "",
           "## (f) Sleepiness change vs dO (eye shown for comparison, same bootstrap)\n",
           tf.round(4).to_markdown(index=False), "",
           "## (g) O3 split-half reliability from the six per-cell pursuit gains (round 1)\n",
           "Halves are means of 3 cells; cells differ in speed/direction, so halves are not strictly parallel "
           "(an alpha-type estimate). Cronbach alpha bootstrap: 1000 resamples.\n",
           tg.round(4).to_markdown(index=False), ""]
    open(os.path.join(out, "A4_extra.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:5])
