"""A4 / B3: within-person responsiveness of gaze measures and classifier scores to rating change.

Primary model (static participant fixed effects written as a change regression):
    F2 - F1 = alpha + beta * (X_s2 - X_s1) + e        OLS, HC3; participant bootstrap (2000)
  F1 = rating "between sessions", F2 = rating "after sessions" (successive session-associated
  endpoints; exact timing within the interval is not documented). X = O1, O2, O3 fitted separately.
  Eye fatigue primary (Holm across O1-O3); sleepiness secondary (own Holm family).
  Slopes are reported per 3% peak-velocity change (log 1.03), per 10 ms, per 0.03 gain.
Sensitivities: + task-specific technical-quality change (d loss, d log precision); ordinal model on
sign(dF) (cumulative logit); classifier scores (EKYT trained on session-1 windows of training
participants, LR gaze-only PUR/RAN trained on session-1 features), labelled as conditional on the
fitted classifiers. A non-significant slope is reported as inconclusive, not as absent responsiveness.
Review round 1 additions: standardized effects (Pearson r with participant-bootstrap CI, Spearman rho) for every
rating change x feature change; timing sensitivities with F1 - F0 and F2 - F0 as the rating change (different
estimands: they bracket session 1 alone, or both sessions, instead of the interval between recordings).
Usage: python partA4.py TABLE_DIR OUT_DIR [--ekyt EKYT_DIR] [--scores A2_DIR]
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

UNIT = {"O1": np.log(1.03), "O2": 10.0, "O3": 0.03, "ekyt_score": 0.1, "lr_purran_score": 0.1,
        "lr_all_score": 0.1}
TECH = {"O1": ["ran_loss", "ran_logprec"], "O2": ["ran_loss", "ran_logprec"], "O3": ["pur_loss", "pur_logprec"]}
N_BOOT = 2000


def change_table(prim, meta, scores):
    r1 = prim[prim["round"] == 1]
    cols = ["O1", "O2", "O3", "ran_loss", "ran_logprec", "pur_loss", "pur_logprec"]
    w = r1.pivot(index="pid", columns="session", values=cols)
    d = pd.DataFrame(index=w.index)
    for c in cols:
        d["d_" + c] = w[(c, 2)] - w[(c, 1)]
    m = meta[meta["round"] == 1].set_index("pid")
    for r in ("eye", "sleep"):
        d[f"d_{r}"] = m[f"{r}2"] - m[f"{r}1"]
        d[f"d_{r}_10"] = m[f"{r}1"] - m[f"{r}0"]
        d[f"d_{r}_20"] = m[f"{r}2"] - m[f"{r}0"]
        for k in (0, 1, 2):
            d[f"{r}{k}"] = m[f"{r}{k}"]
    for nm, s in scores.items():
        d["d_" + nm] = s.set_index("pid").reindex(d.index).pipe(lambda z: z.score_s2 - z.score_s1)
    return d.reset_index()


def ols(d, y, x, extra=()):
    z = d[[y, x] + list(extra)].dropna()
    X = sm.add_constant(z[[x] + list(extra)])
    f = sm.OLS(z[y], X).fit(cov_type="HC3")
    return f, z


def boot_slope(z, y, x, extra=(), seed=0):
    rng = np.random.default_rng(seed)
    zz = z[[y, x] + list(extra)].to_numpy()
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = zz[rng.integers(0, len(zz), len(zz))]
        X = np.column_stack([np.ones(len(s)), s[:, 1:]])
        out[b] = np.linalg.lstsq(X, s[:, 0], rcond=None)[0][1]
    return out


def boot_slope_o1(ps, d, rating, extra=(), seed=0):
    """O1 participant bootstrap that refits the session-1 reference main sequence in each
    resample (frozen protocol): resample participants of the full reference population, refit
    the equal-recording-weight line with multiplicity weights, recompute recording median
    residuals and changes, then refit the change regression on resampled eligible participants."""
    rng = np.random.default_rng(seed)
    ps = ps.reset_index(drop=True)
    la, lv = np.log(ps.amp.to_numpy()), np.log(ps.vpeak.to_numpy())
    rec_codes, recs = pd.factorize(ps.rec)
    rec_pid = ps.groupby(rec_codes).pid.first().to_numpy()
    rec_ses = ps.groupby(rec_codes).session.first().to_numpy()
    n_in_rec = np.bincount(rec_codes)
    pids = np.unique(ps.pid)
    pid_pos = {p: i for i, p in enumerate(pids)}
    sac_pidx = np.array([pid_pos[p] for p in ps.pid])
    is_ref = ps.session.to_numpy() == 1
    order = np.argsort(rec_codes, kind="stable")
    bounds = np.cumsum(np.r_[0, n_in_rec])
    cols = ["d_" + e for e in extra]
    z = d.set_index("pid")[[f"d_{rating}", "d_O1"] + cols].dropna()
    elig = z.index.to_numpy()
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        mult = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids)).astype(float)
        w = mult[sac_pidx] / n_in_rec[rec_codes]
        m = is_ref & (w > 0)
        bb, aa = np.polyfit(la[m], lv[m], 1, w=np.sqrt(w[m]))
        r = (lv - (aa + bb * la))[order]
        med = np.array([np.median(r[bounds[i]:bounds[i + 1]]) for i in range(len(n_in_rec))])
        o1 = pd.Series(med, index=pd.MultiIndex.from_arrays([rec_pid, rec_ses])).unstack()
        dO1 = (o1[2] - o1[1]).reindex(elig)
        mm = mult[[pid_pos.get(p, -1) for p in elig]] if len(elig) else np.array([])
        keep = np.isfinite(dO1.to_numpy()) & (mm > 0)
        rep = np.repeat(np.arange(len(elig))[keep], mm[keep].astype(int))
        y = z[f"d_{rating}"].to_numpy()[rep]
        X = np.column_stack([np.ones(len(rep)), dO1.to_numpy()[rep]] + [z[c].to_numpy()[rep] for c in cols])
        out[b] = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return out


def fit_row(d, rating, x, extra=(), boot=True, ps=None):
    f, z = ols(d, f"d_{rating}", "d_" + x, ["d_" + e for e in extra])
    u = UNIT[x]
    lo, hi = f.conf_int().loc["d_" + x]
    row = dict(rating=rating, feature=x, adjust="+".join(extra) or "none", n=len(z),
               slope_per_unit=f.params["d_" + x] * u, ci_lo=lo * u, ci_hi=hi * u, p=f.pvalues["d_" + x],
               intercept=f.params["const"], r2=f.rsquared)
    if boot and x == "O1" and ps is not None:
        bs = boot_slope_o1(ps, d, rating, extra) * u
        row.update(boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5))
    elif boot:
        bs = boot_slope(z, f"d_{rating}", "d_" + x, ["d_" + e for e in extra]) * u
        row.update(boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5))
    return row


def std_row(d, rating, x, seed=0):
    from scipy.stats import spearmanr
    z = d[[f"d_{rating}", "d_" + x]].dropna().to_numpy()
    r = np.corrcoef(z[:, 0], z[:, 1])[0, 1]
    rng = np.random.default_rng(seed)
    bs = [np.corrcoef(*z[rng.integers(0, len(z), len(z))].T)[0, 1] for _ in range(N_BOOT)]
    return dict(rating=rating, feature=x, n=len(z), pearson_r=r, boot_lo=np.nanpercentile(bs, 2.5),
                boot_hi=np.nanpercentile(bs, 97.5), spearman_rho=spearmanr(z[:, 0], z[:, 1])[0])


def ordinal_row(d, rating, x):
    z = d[[f"d_{rating}", "d_" + x]].dropna()
    c = np.sign(z[f"d_{rating}"]).astype(int)
    xs = z["d_" + x] / UNIT[x]
    try:
        f = OrderedModel(pd.Categorical(c, categories=[-1, 0, 1], ordered=True), xs.to_frame(),
                         distr="logit").fit(method="bfgs", disp=False)
        b, p = f.params.iloc[0], f.pvalues.iloc[0]
    except Exception:
        b, p = np.nan, np.nan
    return dict(rating=rating, feature=x, n=len(z), n_decrease=int((c < 0).sum()), n_same=int((c == 0).sum()),
                n_increase=int((c > 0).sum()), logodds_per_unit=b, p=p)


def load_ekyt(ekyt_dir):
    rows = []
    for f in glob.glob(os.path.join(ekyt_dir, "cv_s1_*.json")):
        j = json.load(open(f))
        for r in j["rec_scores"]:
            rows.append(dict(r, repeat=j["repeat"]))
    if not rows:
        return None
    s = pd.DataFrame(rows).groupby(["pid", "session"]).score.mean().unstack()
    return pd.DataFrame({"pid": s.index, "score_s1": s[1].values, "score_s2": s[2].values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ekyt", default=None)
    ap.add_argument("--scores", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    prim = pd.read_parquet(os.path.join(a.table_dir, "primary.parquet"))
    meta = pd.read_parquet(os.path.join(a.table_dir, "meta.parquet"))
    scores = {}
    if a.ekyt:
        e = load_ekyt(a.ekyt)
        if e is not None:
            scores["ekyt_score"] = e
    if a.scores:
        for nm, fn in (("lr_purran_score", "scores_X_PURRAN.parquet"), ("lr_all_score", "scores_X_all.parquet")):
            p = os.path.join(a.scores, fn)
            if os.path.exists(p):
                scores[nm] = pd.read_parquet(p)
    d = change_table(prim, meta, scores)
    d.to_parquet(os.path.join(a.out_dir, "A4_changes.parquet"))

    ps = pd.read_parquet(os.path.join(a.table_dir, "ran_primary.parquet"))
    ps = ps[ps["round"] == 1]
    prim_rows = [fit_row(d, r, x, ps=ps) for r in ("eye", "sleep") for x in ("O1", "O2", "O3")]
    P = pd.DataFrame(prim_rows)
    P["p_holm"] = np.nan
    for r in ("eye", "sleep"):
        m = P.rating == r
        P.loc[m, "p_holm"] = multipletests(P.loc[m, "p"], method="holm")[1]
    Q = pd.DataFrame([fit_row(d, r, x, TECH[x], ps=ps) for r in ("eye", "sleep") for x in ("O1", "O2", "O3")])
    O = pd.DataFrame([ordinal_row(d, r, x) for r in ("eye", "sleep") for x in ("O1", "O2", "O3")])
    C = pd.DataFrame([fit_row(d, r, nm) for r in ("eye", "sleep") for nm in scores])
    T = pd.DataFrame([fit_row(d, r, x, ps=ps) for r in ("eye_10", "eye_20", "sleep_10", "sleep_20")
                      for x in ("O1", "O2", "O3")])
    S = pd.DataFrame([std_row(d, r, x) for r in ("eye", "sleep", "eye_10", "eye_20")
                      for x in ("O1", "O2", "O3", *scores)])
    # descriptive: rating transitions and change distributions
    trans = {}
    for r in ("eye", "sleep"):
        trans[f"{r} F0->F1"] = pd.crosstab(d[f"{r}0"], d[f"{r}1"])
        trans[f"{r} F1->F2"] = pd.crosstab(d[f"{r}1"], d[f"{r}2"])
    dist = pd.DataFrame({c: d[c].describe() for c in d.columns if c.startswith("d_")}).T

    with open(os.path.join(a.out_dir, "A4_summary.md"), "w") as fh:
        fh.write("# A4 within-person responsiveness: dF(after - between) ~ dX(session 2 - session 1)\n\n")
        fh.write("Slopes in rating points per 3% peak velocity (O1), 10 ms (O2), 0.03 gain (O3), 0.1 score.\n\n")
        fh.write("## Primary (OLS HC3, participant bootstrap; Holm within rating family)\n\n"
                 + P.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Technical-quality-adjusted\n\n" + Q.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Ordinal sensitivity on sign(dF)\n\n" + O.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Classifier scores (held-out; conditional on fitted classifiers)\n\n"
                 + (C.round(4).to_markdown(index=False) if len(C) else "not available") + "\n\n")
        fh.write("## Standardized effects (Pearson r, participant bootstrap 95% CI; Spearman rho)\n\n"
                 + S.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Timing sensitivities (rating change F1-F0 = '_10', F2-F0 = '_20'; different estimands)\n\n"
                 + T.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Change distributions\n\n" + dist.round(4).to_markdown() + "\n\n")
        for k, t in trans.items():
            fh.write(f"## Transitions {k}\n\n" + t.to_markdown() + "\n\n")
    for n_, t in [("primary", P), ("quality", Q), ("ordinal", O), ("scores", C), ("std", S), ("timing", T)]:
        t.to_csv(os.path.join(a.out_dir, f"A4_{n_}.csv"), index=False)
    print(open(os.path.join(a.out_dir, "A4_summary.md")).read()[:7000])


if __name__ == "__main__":
    main()
