"""A2: incremental between-person prediction of the benchmark label (baseline eye fatigue >= 2).

Predictor blocks (frozen in refine-logs/FINAL_PROPOSAL.md)
  D  demographics / context available before session 1      S  other baseline symptom reports
  Q  technical quality only (loss, precision, binocular, velocity thresholds)
  X  frozen gaze vector (all remaining per-task features, incl. target-tracking error proxies;
     pooled main-sequence residual excluded because its reference would need refitting per fold;
     per-recording main-sequence summaries are used instead)
Branches: context D, D+Q, D+Q+X; questionnaire D+S, D+S+Q, D+S+Q+X; standalone X.
Review round 1 additions: direct D -> D+X and D+S -> D+S+X; Q split into Qt (technical: loss, precision,
binocular) and Qd (gaze-derived detector thresholds), with D+Qt(+X) and D+S+Qt(+X); a training-fold
regularization sensitivity (LR with C chosen by inner 5-fold CV on the training fold, point estimates);
bootstrap centering (bootstrap mean vs point estimate); training-size sensitivity (point statistic on
participant subsamples of 256 without replacement, about the number of distinct participants per bootstrap draw).
Learners: logistic regression, C=1 fixed after training-fold scaling (primary); LightGBM fixed
settings (secondary). Stratified participant 5-fold CV, 10 fixed repeats, identical folds for all
sets. Threshold 0.5 with balanced class weights (fixed in advance).
Uncertainty: participant bootstrap with full refitting (folds grouped by original participant).
Permutation: participant-level label shuffles with the whole pipeline refitted; tests overall
predictability, not the conditional increment.
Also writes session-1-trained gaze-only (PUR/RAN) held-out scores for both sessions (A4).
Usage: python partA_predict.py TABLE_DIR OUT_DIR [--reps 10] [--boot 500] [--perm 500] [--session 1]
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

D_COLS = ["age", "female", "gender_other", "sleep_h", "caffeine", "alcohol24", "headache24", "sleep_med",
          "lens_wear", "contacts", "glasses", "vision_corr", "eye_cond", "prior_et", "headache_hist"]
S_COLS = ["sleep0", "neck0", "shoulder0", "comfort0"]
Q_KEYS = ("__loss", "__precision_rms", "__binoc_v_off", "__binoc_v_sd", "__vel_thr", "__onset_thr")
QD_KEYS = ("__vel_thr", "__onset_thr")
# not part of the frozen primary feature set (O1 input; sensitivity-only RAN variables added after review)
X_DROP = ("RAN__ms_resid_pooled", "RAN__ran_lat_np_med", "RAN__ran_p_ant_flag")
BRANCHES = ["D", "D+Q", "D+Q+X", "D+S", "D+S+Q", "D+S+Q+X", "X",
            "D+X", "D+S+X", "D+Qt", "D+Qt+X", "D+S+Qt", "D+S+Qt+X"]
INCREMENTS = [("D+Q", "D+Q+X"), ("D+S+Q", "D+S+Q+X"), ("D", "D+Q"), ("D", "D+S"), ("D+S", "D+S+Q"),
              ("D", "D+X"), ("D+S", "D+S+X"), ("D+Qt", "D+Qt+X"), ("D+S+Qt", "D+S+Qt+X")]
N_SUB = 256


def blocks(wide_cols):
    feat = [c for c in wide_cols if c not in ("round", "pid", "session") + X_DROP]
    q = [c for c in feat if c.endswith(Q_KEYS)]
    x = [c for c in feat if c not in q]
    qd = [c for c in q if c.endswith(QD_KEYS)]
    return dict(D=D_COLS, S=S_COLS, Q=q, X=x, Qt=[c for c in q if c not in qd], Qd=qd)


def build(table_dir, session):
    meta = pd.read_parquet(os.path.join(table_dir, "meta.parquet"))
    wide = pd.read_parquet(os.path.join(table_dir, "wide.parquet"))
    meta = meta[(meta["round"] == 1) & meta.label.notna()].copy()
    meta["gender_other"] = (meta.gender == "Other").astype(float)
    w1 = wide[wide["round"] == 1]
    if session in (1, 2):
        w = w1[w1.session == session].drop(columns=["round", "session"])
    else:
        w = w1.drop(columns=["round", "session"]).groupby("pid").mean().reset_index()
    d = meta.merge(w, on="pid", how="left").sort_values("pid").reset_index(drop=True)
    b = blocks(wide.columns)
    sets = {name: sum((b[k] for k in name.split("+")), []) for name in BRANCHES}
    return d, sets, b, w1


def make_model(kind, seed):
    if kind == "lrcv":
        # imputation, scaling and C all chosen inside the inner CV (review round 2: fully nested)
        from sklearn.model_selection import GridSearchCV
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000))
        return GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}, scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight="balanced", random_state=seed, verbose=-1, n_jobs=1)


def oof(X, y, splits, kind, seed):
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        p[te] = make_model(kind, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def one_rep(d, sets, kind, rep):
    y = d.label.to_numpy(int)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(d, y))
    return [dict(set=n, learner=kind, rep=rep, oof=oof(d[c].to_numpy(float), y, splits, kind, rep))
            for n, c in sets.items()]


def boot_one(d, sets, i, reps):
    """Participant bootstrap: resample participants, then compute the same statistic as the
    point estimate (mean AUC over `reps` repetitions of 5-fold CV grouped by original id,
    identical splits across feature sets)."""
    rng = np.random.default_rng(20_000 + i)
    idx = rng.integers(0, len(d), len(d))
    b = d.iloc[idx].reset_index(drop=True)
    y = b.label.to_numpy(int)
    acc = {n: [] for n in sets}
    for r in range(reps):
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=1000 * i + r).split(b, y, b.pid))
        for n, c in sets.items():
            acc[n].append(roc_auc_score(y, oof(b[c].to_numpy(float), y, splits, "lr", r)))
    return {n: float(np.mean(v)) for n, v in acc.items()}


def sub_one(d, sets, i, reps):
    """Training-size sensitivity: point statistic on a participant subsample without replacement."""
    rng = np.random.default_rng(30_000 + i)
    b = d.iloc[np.sort(rng.choice(len(d), N_SUB, replace=False))].reset_index(drop=True)
    y = b.label.to_numpy(int)
    acc = {n: [] for n in sets}
    for r in range(reps):
        splits = list(StratifiedKFold(5, shuffle=True, random_state=r).split(b, y))
        for n, c in sets.items():
            acc[n].append(roc_auc_score(y, oof(b[c].to_numpy(float), y, splits, "lr", r)))
    return {n: float(np.mean(v)) for n, v in acc.items()}


def perm_one(d, sets, names, i, reps):
    """Permuted labels; same statistic as observed (mean AUC over the `reps` repeat splits)."""
    rng = np.random.default_rng(10_000 + i)
    y = rng.permutation(d.label.to_numpy(int))
    acc = {n: [] for n in names}
    for r in range(reps):
        splits = list(StratifiedKFold(5, shuffle=True, random_state=r).split(d, y))
        for n in names:
            acc[n].append(roc_auc_score(y, oof(d[sets[n]].to_numpy(float), y, splits, "lr", r)))
    return {n: float(np.mean(v)) for n, v in acc.items()}


def date_blocks(date_ord, k=5):
    """Whole acquisition dates assigned to k chronological blocks of ~equal participant count."""
    cnt = date_ord.value_counts().sort_index()
    mid = (cnt.cumsum() - cnt / 2) / cnt.sum()
    blk = np.minimum((mid * k).astype(int), k - 1)
    return date_ord.map(blk).to_numpy()


def grouped_eval(d, sets, group_col, scheme):
    y = d.label.to_numpy(int)
    g = d[group_col].to_numpy()
    groups = pd.Series(y).groupby(g).agg(["size", "mean"])
    single = groups[(groups["mean"] == 0) | (groups["mean"] == 1)]
    splits = list(GroupKFold(len(np.unique(g))).split(d, y, g))
    out = []
    for n, c in sets.items():
        p = oof(d[c].to_numpy(float), y, splits, "lr", 0)
        out.append(dict(set=n, scheme=scheme, n_groups=len(groups), single_class_groups=len(single),
                        single_class_n=int(single["size"].sum()), auc=roc_auc_score(y, p),
                        bacc=balanced_accuracy_score(y, p >= 0.5)))
    return out


def session_scores(d, w1, cols, reps):
    """Train on session-1 features of training participants; score both sessions of held-out ones."""
    y = d.label.to_numpy(int)
    s1 = d[["pid"]].merge(w1[w1.session == 1], on="pid", how="left")
    s2 = d[["pid"]].merge(w1[w1.session == 2], on="pid", how="left")
    X1, X2 = s1[cols].to_numpy(float), s2[cols].to_numpy(float)
    sc = np.zeros((reps, len(d), 2))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(d, y):
            m = make_model("lr", r).fit(X1[tr], y[tr])
            sc[r, te, 0] = m.predict_proba(X1[te])[:, 1]
            ok2 = ~np.all(np.isnan(X2[te]), axis=1)
            sc[r, te, 1] = np.where(ok2, m.predict_proba(X2[te])[:, 1], np.nan)
    avg = np.nanmean(sc, 0)
    return pd.DataFrame({"pid": d.pid, "label": y, "score_s1": avg[:, 0], "score_s2": avg[:, 1],
                         "auc_s1_rep_mean": np.mean([roc_auc_score(y, sc[r, :, 0]) for r in range(reps)])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--session", type=int, default=1, help="1, 2, or 0 = mean of both sessions")
    ap.add_argument("--jobs", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d, sets, b, w1 = build(a.table_dir, a.session)
    tag = f"s{a.session}"
    y = d.label.to_numpy(int)
    pd.Series({k: ", ".join(v) for k, v in b.items()}).to_csv(os.path.join(a.out_dir, f"blocks_{tag}.csv"))
    miss = d[sum(b.values(), [])].isna().mean().sort_values(ascending=False)
    print(f"n={len(d)} label=1:{int(y.sum())}", {k: len(v) for k, v in sets.items()})

    # repeated CV, identical folds across sets
    res = Parallel(n_jobs=a.jobs)(delayed(one_rep)(d, sets, k, r) for k in ("lr", "lgb", "lrcv")
                                  for r in range(a.reps))
    rep = pd.DataFrame([x for r in res for x in r])
    rep["auc"] = rep.oof.apply(lambda p: roc_auc_score(y, p))
    rep["bacc"] = rep.oof.apply(lambda p: balanced_accuracy_score(y, p >= 0.5))
    summ = rep.groupby(["set", "learner"]).agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                                               bacc=("bacc", "mean")).reset_index()
    pd.DataFrame({f"{r.set}|{r.learner}|{r.rep}": r.oof for r in rep.itertuples()} |
                 {"pid": d.pid, "label": y}).to_parquet(os.path.join(a.out_dir, f"oof_{tag}.parquet"))

    # participant bootstrap with refitting (LR)
    boots = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(boot_one)(d, sets, i, a.reps) for i in range(a.boot)))
    subs = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(sub_one)(d, sets, i, a.reps) for i in range(a.boot // 5)))
    lr = summ[summ.learner == "lr"].set_index("set")
    summ = summ.merge(pd.DataFrame({"set": list(sets), "boot_lo": boots.quantile(0.025).values,
                                    "boot_hi": boots.quantile(0.975).values, "boot_mean": boots.mean().values,
                                    f"sub{N_SUB}_mean": subs.mean().values}).assign(learner="lr"),
                      on=["set", "learner"], how="left")
    inc = []
    for lo, hi in INCREMENTS:
        dd = boots[hi] - boots[lo]
        ds = subs[hi] - subs[lo]
        row = dict(base=lo, full=hi, d_auc_lr=lr.auc[hi] - lr.auc[lo], boot_lo=dd.quantile(0.025),
                   boot_hi=dd.quantile(0.975), boot_mean=dd.mean(), boot_p_le0=float((dd <= 0).mean()),
                   sub_mean=ds.mean(), sub_lo=ds.quantile(0.025), sub_hi=ds.quantile(0.975))
        for k in ("lgb", "lrcv"):
            g = summ[summ.learner == k].set_index("set").auc
            row[f"d_auc_{k}"] = g[hi] - g[lo]
        inc.append(row)
    inc = pd.DataFrame(inc)

    # acquisition-context sensitivity: leave-administrator-out and date blocks
    d["date_block"] = date_blocks(d.date_ord)
    grp = pd.DataFrame([x for col, sch in [("admin", "leave-admin-out"), ("date_block", "date-block")]
                        for x in grouped_eval(d, sets, col, sch)])

    # permutation null with full refit
    pn = ["X", "D+S", "D+S+Q+X", "D+Q+X"]
    perm = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(perm_one)(d, sets, pn, i, a.reps) for i in range(a.perm)))
    pt = pd.DataFrame([dict(set=n, obs=lr.auc[n], null_mean=perm[n].mean(), null_95=perm[n].quantile(.95),
                            p=(1 + (perm[n] >= lr.auc[n]).sum()) / (1 + len(perm))) for n in pn])

    # session-1-trained held-out scores for A4: gaze-only PUR/RAN (principal) and all-task X (secondary)
    if a.session == 1:
        xpr = [c for c in b["X"] if c.startswith(("PUR__", "RAN__"))]
        for nm, cols in (("X_PURRAN", xpr), ("X_all", b["X"])):
            session_scores(d, w1, cols, a.reps).to_parquet(os.path.join(a.out_dir, f"scores_{nm}.parquet"))

    with open(os.path.join(a.out_dir, f"A2_summary_{tag}.md"), "w") as fh:
        fh.write(f"# A2 incremental prediction (features from session {a.session or 'mean'})\n\n")
        fh.write(f"n = {len(d)}, label=1: {int(y.sum())}; {a.reps} repeats x stratified 5-fold; "
                 f"LR C=1 (primary), LightGBM, LR with inner-CV C (lrcv); {a.boot} participant bootstraps "
                 f"with refit; {a.boot // 5} subsamples of {N_SUB} participants (sub_*); "
                 f"{a.perm} permutations with refit\n\n")
        fh.write("Blocks: " + ", ".join(f"{k}={len(v)}" for k, v in b.items()) + "\n\n")
        fh.write(summ.round(3).to_markdown(index=False) + "\n\n## Increments (AUC)\n\n")
        fh.write(inc.round(4).to_markdown(index=False) + "\n\n## Acquisition-context grouped CV (LR)\n\n")
        fh.write(grp.round(3).to_markdown(index=False) + "\n\n## Permutation null (LR, full refit)\n\n")
        fh.write(pt.round(4).to_markdown(index=False) + "\n\n## Highest missingness\n\n")
        fh.write(miss.head(10).round(3).to_markdown() + "\n")
    for n_, t in [("summary", summ), ("increments", inc), ("grouped", grp), ("perm", pt), ("boot", boots),
                  ("sub", subs)]:
        t.to_csv(os.path.join(a.out_dir, f"A2_{n_}_{tag}.csv"), index=False)
    print(open(os.path.join(a.out_dir, f"A2_summary_{tag}.md")).read())


if __name__ == "__main__":
    main()
