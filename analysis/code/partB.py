"""Part B: session-associated change in O1-O3 (round 1), equivalence, repeatability.

Frozen in refine-logs/FINAL_PROPOSAL.md:
  B1 primary: unadjusted mean paired change (S2 - S1), 95% CI, Holm across O1-O3.
     O1 reported as 100[exp(mean dO1) - 1] (transformed mean log change).
     Adjusted: dY ~ 1 + d loss + d log precision (task-specific technical quality, HC3);
     intercept = change at zero technical-quality change. Tracking-error proxies only in a
     separately labelled sensitivity.
  B2: TOST, alpha .05 (90% CI inside bounds), Holm across the three TOST p-values.
     Bounds: O1 [log .97, log 1.03]; O2 +-10 ms; O3 +-0.03 (+-0.05 and O2 +-15 ms as comparators).
     Operational thresholds, not validated fatigue thresholds. d_z descriptive.
  B4: ICC(3,1) consistency + ICC(A,1) absolute agreement + Bland-Altman limits of agreement.
Usage: python partB.py TABLE_DIR OUT_DIR [--brief]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

BOUNDS = {"O1": (np.log(0.97), np.log(1.03)), "O2": (-10.0, 10.0), "O3": (-0.03, 0.03)}
COMPARATOR = {"O2": (-15.0, 15.0), "O3": (-0.05, 0.05)}
TECH = {"O1": ["ran_loss", "ran_logprec"], "O2": ["ran_loss", "ran_logprec"], "O3": ["pur_loss", "pur_logprec"]}
TRACK = {"O1": "ran_track_err", "O2": "ran_track_err", "O3": "pur_track_err"}
N_BOOT = 2000


def deltas(df, o):
    w = df.pivot(index="pid", columns="session", values=o).dropna()
    return w, (w[2] - w[1]).to_numpy()


def tost(dlt, lo, hi):
    n = len(dlt)
    m, se = dlt.mean(), dlt.std(ddof=1) / np.sqrt(len(dlt))
    return max(stats.t.sf((m - lo) / se, n - 1), stats.t.cdf((m - hi) / se, n - 1))


def paired(df, o):
    w, dlt = deltas(df, o)
    n = len(dlt)
    m, sd = dlt.mean(), dlt.std(ddof=1)
    se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    lo, hi = BOUNDS[o]
    r = dict(outcome=o, n=n, s1_mean=w[1].mean(), s2_mean=w[2].mean(), diff=m,
             ci95_lo=m - t95 * se, ci95_hi=m + t95 * se, ci90_lo=m - t90 * se, ci90_hi=m + t90 * se,
             p=2 * stats.t.sf(abs(m / se), n - 1), dz=m / sd, bound_lo=lo, bound_hi=hi, p_tost=tost(dlt, lo, hi))
    if o in COMPARATOR:
        r["p_tost_comparator"] = tost(dlt, *COMPARATOR[o])
    if o == "O1":
        for k in ("diff", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi"):
            r["pct_" + k] = 100 * (np.exp(r[k]) - 1)
    q = np.percentile(dlt, [5, 25, 50, 75, 95])
    r.update({f"delta_q{p}": v for p, v in zip((5, 25, 50, 75, 95), q)})
    r["frac_beyond_bounds"] = float(np.mean((dlt < lo) | (dlt > hi)))
    return r


def adjusted(df, o, extra=()):
    cols = TECH[o] + list(extra)
    w = df.pivot(index="pid", columns="session", values=[o] + cols).dropna()
    x = pd.DataFrame({"dy": w[(o, 2)] - w[(o, 1)]})
    for c in cols:
        x["d_" + c] = w[(c, 2)] - w[(c, 1)]
    fit = smf.ols("dy ~ " + " + ".join("d_" + c for c in cols), data=x).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["Intercept"]
    out = dict(outcome=o, covariates="+".join(cols), n=len(x), adj_diff=fit.params["Intercept"],
               adj_lo=lo, adj_hi=hi, adj_p=fit.pvalues["Intercept"])
    for c in cols:
        out["coef_d_" + c] = fit.params["d_" + c]
    return out


# ---------------------------------------------------------------- O1 refit / sensitivities
def _fit_line(ps):
    w = 1.0 / ps.groupby("rec").rec.transform("size").to_numpy()
    b, a = np.polyfit(np.log(ps.amp.to_numpy()), np.log(ps.vpeak.to_numpy()), 1, w=np.sqrt(w))
    return a, b


def o1_from(ps, ref):
    a, b = _fit_line(ref)
    r = np.log(ps.vpeak) - (a + b * np.log(ps.amp))
    return r.groupby([ps.pid, ps.session]).median().unstack()


def o1_bootstrap(ps, pids, keep, seed=0):
    """Participant bootstrap that refits the reference main sequence in each resample.
    Resamples the full reference population (pids); the contrast uses resampled participants
    with an accepted O1 in both sessions (keep), as in the point estimate."""
    rng = np.random.default_rng(seed)
    by = {p: g for p, g in ps.groupby("pid")}
    pids = np.array([p for p in pids if p in by])
    keep = set(keep)
    out = []
    for _ in range(N_BOOT):
        s = rng.choice(pids, len(pids))
        parts = []
        for k, p in enumerate(s):
            g = by[p].copy()
            g["pid"] = k
            g["ok"] = p in keep
            g["rec"] = g["rec"] + f"#{k}"
            parts.append(g)
        b = pd.concat(parts)
        w = o1_from(b, b[b.session == 1])
        ok_k = b.groupby("pid").ok.first()
        w = w[ok_k.reindex(w.index).fillna(False).to_numpy()].dropna()
        out.append((w[2] - w[1]).mean())
    return np.array(out)


def o1_variants(ps):
    import patsy
    ref = ps[ps.session == 1]
    res = {}
    # (a) unweighted pooled line
    b, a = np.polyfit(np.log(ref.amp), np.log(ref.vpeak), 1)
    res["O1_unweighted"] = np.log(ps.vpeak) - (a + b * np.log(ps.amp))
    # (b) natural cubic spline in log amplitude, df=4, equal weight per recording
    w = 1.0 / ref.groupby("rec").rec.transform("size")
    X = patsy.dmatrix("cr(np.log(amp), df=4)", ref, return_type="dataframe")
    fit = smf.wls("np.log(vpeak) ~ cr(np.log(amp), df=4)", data=ref, weights=w).fit()
    res["O1_spline"] = np.log(ps.vpeak) - fit.predict(ps)
    # (c) direction and start-eccentricity adjusted line
    d = ps.assign(horiz=np.abs(np.cos(np.deg2rad(ps.ang))), ecc=np.hypot(ps.sx, ps.sy))
    fit = smf.wls("np.log(vpeak) ~ np.log(amp) + horiz + ecc", data=d[d.session == 1], weights=w).fit()
    res["O1_dir_ecc"] = np.log(d.vpeak) - fit.predict(d)
    out = {}
    for k, r in res.items():
        out[k] = r.groupby([ps.pid, ps.session]).median()
    return pd.DataFrame(out).reset_index()


# ---------------------------------------------------------------- repeatability
def icc_all(Y):
    """ICC(3,1) consistency and ICC(A,1) absolute agreement (McGraw & Wong), F-based 95% CI for
    ICC(3,1); Bland-Altman bias and 95% limits of agreement for k=2."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm
    mse = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    iccA = (msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse))
    F = msr / mse
    df1, df2 = n - 1, (n - 1) * (k - 1)
    fl, fu = F / stats.f.ppf(0.975, df1, df2), F * stats.f.ppf(0.975, df2, df1)
    d = Y[:, 1] - Y[:, 0]
    return dict(n=n, icc31=icc3, icc31_lo=(fl - 1) / (fl + k - 1), icc31_hi=(fu - 1) / (fu + k - 1),
                iccA1=iccA, ba_bias=d.mean(), loa_lo=d.mean() - 1.96 * d.std(ddof=1),
                loa_hi=d.mean() + 1.96 * d.std(ddof=1))


def icc(df, o, by="session"):
    w = df.pivot(index="pid", columns=by, values=o).dropna()
    return dict(outcome=o, **icc_all(w.to_numpy()))


def b1b2(r1):
    B = pd.DataFrame([paired(r1, o) for o in BOUNDS])
    B["p_holm"] = multipletests(B.p, method="holm")[1]
    B["p_tost_holm"] = multipletests(B.p_tost, method="holm")[1]
    B["equivalent"] = B.p_tost_holm < 0.05
    return B


def main(table_dir, out_dir, brief=False):
    os.makedirs(out_dir, exist_ok=True)
    prim = pd.read_parquet(os.path.join(table_dir, "primary.parquet"))
    r1 = prim[prim["round"] == 1]
    B = b1b2(r1)
    B.to_csv(os.path.join(out_dir, "B1B2.csv"), index=False)
    if brief:
        print(B.round(4).to_string())
        return
    Badj = pd.DataFrame([adjusted(r1, o) for o in BOUNDS] +
                        [adjusted(r1, o, extra=[TRACK[o]]) for o in BOUNDS])
    # O1 participant bootstrap with main-sequence refit
    ps = pd.read_parquet(os.path.join(table_dir, "ran_primary.parquet"))
    ps1 = ps[ps["round"] == 1]
    ok = r1[r1.ok_O1].groupby("pid").size()
    keep = ok[ok == 2].index
    boot = o1_bootstrap(ps1, ps1.pid.unique(), keep)
    o1boot = dict(outcome="O1", n=len(keep), boot_mean=boot.mean(), boot_lo=np.percentile(boot, 2.5),
                  boot_hi=np.percentile(boot, 97.5), boot90_lo=np.percentile(boot, 5),
                  boot90_hi=np.percentile(boot, 95),
                  pct_lo=100 * (np.exp(np.percentile(boot, 2.5)) - 1),
                  pct_hi=100 * (np.exp(np.percentile(boot, 97.5)) - 1))
    # O1 / O2 / O3 measurement sensitivities
    var = o1_variants(ps1)
    var = var[var.pid.isin(keep)]
    sens = []
    for c in ("O1_unweighted", "O1_spline", "O1_dir_ecc"):
        sens.append(dict(variant=c, **{k: v for k, v in paired(var.rename(columns={c: "O1"}), "O1").items()
                                       if k in ("n", "diff", "ci95_lo", "ci95_hi", "pct_diff", "p_tost")}))
    np_ = r1.drop(columns="O1").rename(columns={"O1_np": "O1"})
    sens.append(dict(variant="O1_no_anticipatory_precedence", **{
        k: v for k, v in paired(np_, "O1").items() if k in ("n", "diff", "ci95_lo", "ci95_hi", "pct_diff", "p_tost")}))
    for c, o in (("ran_lat_np_med", "O2"), ("ran_lat_wide_med", "O2"), ("pur_gain_5", "O3"), ("pur_gain_10", "O3"), ("pur_gain_20", "O3")):
        d = r1[r1[f"ok_{o}"]].rename(columns={o: "_orig"}).rename(columns={c: o})
        sens.append(dict(variant=c, **{k: v for k, v in paired(d, o).items()
                                       if k in ("n", "diff", "ci95_lo", "ci95_hi", "p_tost")}))
    rv = r1.copy()
    rv["O1v10"] = np.log(rv.merge(pd.read_parquet(os.path.join(table_dir, "wide.parquet"))
                                  [["round", "pid", "session", "RAN__ran_ms_v10"]],
                                  on=["round", "pid", "session"], how="left")["RAN__ran_ms_v10"].to_numpy())
    rv.loc[~rv.ok_O1, "O1v10"] = np.nan
    sens.append(dict(variant="O1_per_recording_logV10", **{
        k: v for k, v in paired(rv.drop(columns="O1").rename(columns={"O1v10": "O1"}), "O1").items()
        if k in ("n", "diff", "ci95_lo", "ci95_hi", "pct_diff", "p_tost")}))
    sens = pd.DataFrame(sens)
    # technical quality change (descriptive)
    qual = []
    for c in ("ran_loss", "ran_logprec", "ran_binoc_off", "ran_binoc_sd", "ran_track_err", "pur_loss",
              "pur_logprec", "pur_binoc_off", "pur_binoc_sd", "pur_track_err", "ran_p_anticipatory",
              "ran_p_none", "ran_p_misdirected"):
        w = r1.pivot(index="pid", columns="session", values=c).dropna()
        dl = w[2] - w[1]
        qual.append(dict(measure=c, n=len(w), s1=w[1].mean(), s2=w[2].mean(), diff=dl.mean(),
                         p=stats.ttest_1samp(dl, 0).pvalue))
    qual = pd.DataFrame(qual)
    B4 = pd.DataFrame([icc(r1, o) for o in BOUNDS])
    meta = pd.read_parquet(os.path.join(table_dir, "meta.parquet"))
    acr = []
    for rr in (2, 3):
        ids = set(prim[prim["round"] == rr].pid)
        sub = prim[(prim["round"].isin([1, rr])) & (prim.session == 1) & prim.pid.isin(ids)]
        dd = meta[meta.pid.isin(ids)].pivot(index="pid", columns="round", values="date")
        for o in BOUNDS:
            used = sub.pivot(index="pid", columns="round", values=o).dropna().index
            gap = (dd.loc[dd.index.isin(used), rr] - dd.loc[dd.index.isin(used), 1]).dt.days.dropna()
            acr.append(dict(rounds=f"1 vs {rr}", interval_days_median=gap.median(),
                            interval_days_range=f"{gap.min():.0f}-{gap.max():.0f}", **icc(sub, o, by="round")))
    B4r = pd.DataFrame(acr)
    rep = pd.DataFrame([dict(round=rr, **paired(prim[prim["round"] == rr], o)) for rr in (2, 3) for o in BOUNDS])

    # exploratory: every extracted feature (VID/TEX content differs across sessions)
    wide = pd.read_parquet(os.path.join(table_dir, "wide.parquet"))
    w1 = wide[wide["round"] == 1]
    ex = []
    for c in [c for c in w1.columns if c not in ("round", "pid", "session")]:
        w = w1.pivot(index="pid", columns="session", values=c).dropna()
        if len(w) < 50:
            continue
        dl = w[2] - w[1]
        ex.append(dict(feature=c, task=c.split("__")[0], n=len(w), s1=w[1].median(), s2=w[2].median(),
                       dz=dl.mean() / dl.std(ddof=1), p=stats.wilcoxon(dl).pvalue))
    ex = pd.DataFrame(ex)
    ex["q_bh"] = multipletests(ex.p, method="fdr_bh")[1]
    ex = ex.sort_values("p")

    with open(os.path.join(out_dir, "B_summary.md"), "w") as fh:
        fh.write("# Part B: session-associated change (round 1, session 2 minus session 1)\n\n")
        fh.write("## B1/B2 unadjusted paired change and equivalence (Holm within each family)\n\n"
                 + B.round(4).T.to_markdown() + "\n\n")
        fh.write("## O1 participant bootstrap with main-sequence refit (2000)\n\n"
                 + pd.DataFrame([o1boot]).round(4).to_markdown(index=False) + "\n\n")
        fh.write("## B1 adjusted for technical-quality change (HC3; rows 4-6 add the tracking-error proxy)\n\n"
                 + Badj.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Measurement-definition sensitivities\n\n" + sens.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Technical quality and RAN response classes, S2 - S1\n\n" + qual.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## B4 repeatability across sessions\n\n" + B4.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## B4 across rounds (returners, session 1 of each round)\n\n" + B4r.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Session contrast within rounds 2 and 3 (same people as round 1, not independent)\n\n"
                 + rep[["round", "outcome", "n", "diff", "ci95_lo", "ci95_hi", "p", "p_tost"]].round(4)
                 .to_markdown(index=False) + "\n\n")
        fh.write("## Exploratory: all features (Wilcoxon, BH-FDR). VID/TEX stimuli differ between sessions.\n\n"
                 + ex.round(4).to_markdown(index=False) + "\n")
    for n_, t in [("B1adj", Badj), ("B_sens", sens), ("B_quality", qual), ("B4", B4), ("B4_rounds", B4r),
                  ("B_rep", rep), ("B_explore", ex)]:
        t.to_csv(os.path.join(out_dir, f"{n_}.csv"), index=False)
    pd.DataFrame([o1boot]).to_csv(os.path.join(out_dir, "B_O1_boot.csv"), index=False)
    print(open(os.path.join(out_dir, "B_summary.md")).read()[:9000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], brief="--brief" in sys.argv)
