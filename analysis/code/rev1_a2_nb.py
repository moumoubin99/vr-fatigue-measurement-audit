"""REV1 / A2 calibrated intervals for incremental AUC (Stage-3 review R1 M4).

Same data, label (reconstructed, n = 404), LR pipeline (median impute, scale, C = 1, balanced) and the same
10 x stratified 5-fold participant CV (StratifiedKFold(5, shuffle=True, random_state=rep), rep = 0..9) as
code/partA_predict.py. For every increment the fold-level dAUC (50 values: AUC on each held-out fold, full minus
base, identical folds) is summarised with the Nadeau-Bengio corrected resampled t:
    var_corr = (1/J + n_test/n_train) * S^2,   J = 50, n_test/n_train = 1/4,   t_{J-1}.
Point estimates are also computed exactly as in A2 (mean over repeats of the pooled-OOF AUC) and compared
with results/A2_v4/A2_increments_s1.csv; the refit OOF predictions are compared with the saved A2_v4 OOFs.
Optional power check (--power): a synthetic feature z ~ N(delta * y, 1) is added to D+S (alone, and together
with the real X block); detection = NB lower limit > 0.
Usage: python rev1_a2_nb.py TABLE_DIR A2_DIR OUT_DIR [--power]
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partA_predict as A  # noqa: E402

REPS, K = 10, 5
RATIO = 1 / (K - 1)
INC = [("D", "D+S"), ("D", "D+X"), ("D+S", "D+S+X"), ("D+Q", "D+Q+X"), ("D+Qt", "D+Qt+X"),
       ("D+S+Q", "D+S+Q+X"), ("D+S+Qt", "D+S+Qt+X"), ("D", "D+Q"), ("D+S", "D+S+Q")]


def nb(x, ratio=RATIO):
    x = np.asarray(x, float)
    J = len(x)
    m, s2 = x.mean(), x.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * s2)
    t = stats.t.ppf(0.975, J - 1)
    p = 2 * stats.t.sf(abs(m / se), J - 1) if se > 0 else np.nan
    return m, m - t * se, m + t * se, se, p, np.sqrt(s2 / J)


def cv_preds(X, y, rep):
    splits = list(StratifiedKFold(K, shuffle=True, random_state=rep).split(X, y))
    p = A.oof(X, y, splits, "lr", rep)
    return p, splits


def run_set(d, cols, y, rep):
    p, splits = cv_preds(d[cols].to_numpy(float), y, rep)
    return p, [roc_auc_score(y[te], p[te]) for _, te in splits]


def power_draw(d, sets, y, delta, i):
    rng = np.random.default_rng(777_000 + i)
    z = rng.normal(delta * y, 1.0)
    dd = d.assign(_z=z)
    out = {}
    for nm, base, full in (("D+S -> D+S+z", sets["D+S"], sets["D+S"] + ["_z"]),
                           ("D+S -> D+S+X+z", sets["D+S"], sets["D+S+X"] + ["_z"])):
        fd, pooled = [], []
        for r in range(REPS):
            pb, fb = run_set(dd, base, y, r)
            pf, ff = run_set(dd, full, y, r)
            fd += list(np.subtract(ff, fb))
            pooled.append(roc_auc_score(y, pf) - roc_auc_score(y, pb))
        m, lo, hi, *_ = nb(fd)
        out[nm] = dict(delta=delta, draw=i, comparison=nm, d_auc_pooled=np.mean(pooled), d_auc_fold=m, nb_lo=lo,
                       nb_hi=hi, auc_z_alone=roc_auc_score(y, z))
    return list(out.values())


def main(table_dir, a2_dir, out_dir, power=False):
    os.makedirs(out_dir, exist_ok=True)
    d, sets, b, _ = A.build(table_dir, 1)
    y = d.label.to_numpy(int)
    names = sorted({s for pair in INC for s in pair} | {"X"})
    res = Parallel(n_jobs=min(32, os.cpu_count()))(delayed(run_set)(d, sets[n], y, r) for n in names
                                                      for r in range(REPS))
    P, F = {}, {}
    k = 0
    for n in names:
        for r in range(REPS):
            P[(n, r)], F[(n, r)] = res[k]
            k += 1
    # check against saved OOFs
    saved = pd.read_parquet(os.path.join(a2_dir, "oof_s1.parquet"))
    assert (saved.pid.to_numpy() == d.pid.to_numpy()).all()
    chk = [dict(set=n, rep=r, max_abs_diff=float(np.nanmax(np.abs(P[(n, r)] - saved[f"{n}|lr|{r}"].to_numpy()))))
           for n in names for r in range(REPS)]
    chk = pd.DataFrame(chk)
    ref = pd.read_csv(os.path.join(a2_dir, "A2_increments_s1.csv")).set_index(["base", "full"])
    rows, folds = [], []
    for lo_s, hi_s in INC:
        fd = np.concatenate([np.subtract(F[(hi_s, r)], F[(lo_s, r)]) for r in range(REPS)])
        pooled = np.mean([roc_auc_score(y, P[(hi_s, r)]) - roc_auc_score(y, P[(lo_s, r)]) for r in range(REPS)])
        m, lo, hi, se, p, se_naive = nb(fd)
        tq = stats.t.ppf(0.975, len(fd) - 1)
        refv = ref.loc[(lo_s, hi_s), "d_auc_lr"] if (lo_s, hi_s) in ref.index else np.nan
        rows.append(dict(base=lo_s, full=hi_s, n=len(y), J=len(fd), d_auc_pooled_oof=pooled, d_auc_A2_v4=refv,
                         match_A2_v4=bool(abs(pooled - refv) < 1e-9) if np.isfinite(refv) else None,
                         d_auc_fold_mean=m, nb_lo=lo, nb_hi=hi, nb_se=se, nb_p=p,
                         naive_lo=m - tq * se_naive, naive_hi=m + tq * se_naive,
                         boot_lo_A2_v4=ref.loc[(lo_s, hi_s), "boot_lo"] if np.isfinite(refv) else np.nan,
                         boot_hi_A2_v4=ref.loc[(lo_s, hi_s), "boot_hi"] if np.isfinite(refv) else np.nan))
        folds += [dict(base=lo_s, full=hi_s, rep=j // K, fold=j % K, d_auc=v) for j, v in enumerate(fd)]
    inc = pd.DataFrame(rows)
    # single-set AUCs (fold-level, vs 0.5)
    single = []
    for n in names:
        fa = np.concatenate([F[(n, r)] for r in range(REPS)])
        m, lo, hi, se, p, _ = nb(fa - 0.5)
        single.append(dict(set=n, auc_pooled_oof=np.mean([roc_auc_score(y, P[(n, r)]) for r in range(REPS)]),
                           auc_fold_mean=m + 0.5, nb_lo=lo + 0.5, nb_hi=hi + 0.5, nb_p_vs_0_5=p))
    single = pd.DataFrame(single)
    inc.to_csv(os.path.join(out_dir, "A2_nb.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(out_dir, "A2_nb_folds.csv"), index=False)
    single.to_csv(os.path.join(out_dir, "A2_nb_single.csv"), index=False)
    chk.to_csv(os.path.join(out_dir, "A2_nb_oofcheck.csv"), index=False)
    lines = ["# REV1 A2: Nadeau-Bengio corrected resampled-t intervals (LR, C = 1)\n",
             f"n = {len(y)} (label = 1: {int(y.sum())}); 10 x stratified 5-fold CV with the A2 seeds; J = 50 "
             f"fold-level dAUC per increment; n_test/n_train = 1/4; t_49. Refit OOF predictions vs saved A2_v4 OOFs: "
             f"max |diff| = {chk.max_abs_diff.max():.2e} over {len(chk)} set x repeat vectors.\n",
             "`d_auc_pooled_oof` is the A2 statistic (mean over repeats of the pooled-OOF AUC difference); "
             "`d_auc_fold_mean` is the mean of the 50 fold-level differences, which the NB interval is centred on. "
             "`naive_*` is the uncorrected t interval (shown only to illustrate the correction). "
             "`boot_*_A2_v4` are the earlier descriptive bootstrap ranges.\n",
             inc.round(4).to_markdown(index=False), "",
             "## Single feature sets: fold-level AUC with NB interval\n",
             single.round(4).to_markdown(index=False), ""]
    if power:
        deltas = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
        n_draw = 200
        pw = Parallel(n_jobs=min(32, os.cpu_count()))(delayed(power_draw)(d, sets, y, dl, i)
                                                        for dl in deltas for i in range(n_draw))
        pw = pd.DataFrame([x for r in pw for x in r])
        pw.to_csv(os.path.join(out_dir, "A2_nb_power_draws.csv"), index=False)
        ps = pw.groupby(["comparison", "delta"]).agg(
            draws=("draw", "size"), auc_z_alone=("auc_z_alone", "mean"), mean_d_auc_pooled=("d_auc_pooled", "mean"),
            mean_d_auc_fold=("d_auc_fold", "mean"), power_nb_lo_gt0=("nb_lo", lambda v: float((v > 0).mean())),
            mean_nb_halfwidth=("nb_hi", "mean")).reset_index()
        ps["mean_nb_halfwidth"] = (pw.groupby(["comparison", "delta"]).apply(
            lambda g: ((g.nb_hi - g.nb_lo) / 2).mean())).to_numpy()
        ps.to_csv(os.path.join(out_dir, "A2_nb_power.csv"), index=False)
        lines += ["## Power check: synthetic feature z ~ N(delta*y, 1) added to D+S\n",
                  f"{n_draw} draws per delta (seeds 777000+i); the 'known' increment is the pipeline's mean dAUC "
                  "across draws (its expectation at n = 404), since a population AUC for D+S is not available.\n",
                  ps.round(4).to_markdown(index=False), ""]
    open(os.path.join(out_dir, "A2_nb.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4], power="--power" in sys.argv)
