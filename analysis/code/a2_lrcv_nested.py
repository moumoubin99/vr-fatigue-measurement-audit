"""A2 review round 2: rerun only the inner-CV regularization sensitivity (lrcv) with the whole
preprocessing pipeline inside the inner search (see make_model). Same outer folds and repeats as A2.
Usage: python a2_lrcv_nested.py TABLE_DIR OUT_DIR [--reps 10] [--jobs 30]
"""
import argparse, os
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from partA_predict import build, one_rep, INCREMENTS

ap = argparse.ArgumentParser()
ap.add_argument("table_dir"); ap.add_argument("out_dir")
ap.add_argument("--reps", type=int, default=10); ap.add_argument("--jobs", type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
d, sets, b, w1 = build(a.table_dir, 1)
y = d.label.to_numpy(int)
res = Parallel(n_jobs=a.jobs)(delayed(one_rep)(d, sets, "lrcv", r) for r in range(a.reps))
rep = pd.DataFrame([x for r in res for x in r])
rep["auc"] = rep.oof.apply(lambda p: roc_auc_score(y, p))
auc = rep.groupby("set").auc.agg(["mean", "std"])
inc = pd.DataFrame([dict(base=lo, full=hi, d_auc_lrcv_nested=auc["mean"][hi] - auc["mean"][lo]) for lo, hi in INCREMENTS])
auc.to_csv(os.path.join(a.out_dir, "A2_lrcv_nested_auc.csv")); inc.to_csv(os.path.join(a.out_dir, "A2_lrcv_nested_inc.csv"), index=False)
with open(os.path.join(a.out_dir, "A2_lrcv_nested.md"), "w") as fh:
    fh.write(f"# A2 lrcv, fully nested (n = {len(d)}, {a.reps} repeats x stratified 5-fold)\n\n"
             + auc.round(4).to_markdown() + "\n\n" + inc.round(4).to_markdown(index=False) + "\n")
print(open(os.path.join(a.out_dir, "A2_lrcv_nested.md")).read())
