"""A3 v3 summary (review round 1): learning curves, overfit control, velocity vs gaze-angle input,
window accuracy against the majority rate, and pooled out-of-fold participant AUC for the cv repeat.
Usage: python ekyt_v3_summary.py OUT_DIR   (reads ../out/ekyt_v3{vel,pos}/*.json and the window npz files)
"""
import json, glob, sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(0)
fits, curves, pooled = [], [], []
for inp in ("vel", "pos"):
    d = np.load(f"../out/pur_win_v3{inp}.npz")
    lab = pd.Series(d["y"]).groupby(d["pid"]).first()
    for f in sorted(glob.glob(f"../out/ekyt_v3{inp}/*.json")):
        j = json.load(open(f))
        name = os.path.basename(f)[:-5]
        fits.append(dict(input=inp, fit=name, scheme=j["scheme"], n_train_p=j["n_train_p"], n_test_p=j["n_test_p"],
                         win_acc=j["win_acc"], win_majority=j["win_majority"], win_auc=j["win_auc"],
                         part_auc=j["part_auc"], part_bacc=j["part_bacc"],
                         final_loss=j["epoch_loss"][-1], min_loss=min(j["epoch_loss"]),
                         final_batch_acc=j["epoch_batch_acc"][-1]))
        for c in j["curve"]:
            curves.append(dict(input=inp, fit=name, **c))
    rows = []
    for f in glob.glob(f"../out/ekyt_v3{inp}/cv_s0_r0_f*.json"):
        rows += json.load(open(f))["rec_scores"]
    if not rows:
        continue
    s = pd.DataFrame(rows).groupby("pid").score.mean()
    yl = lab.loc[s.index].to_numpy()
    sc = s.to_numpy()
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(s), len(s))
        if len(np.unique(yl[i])) == 2:
            bs.append(roc_auc_score(yl[i], sc[i]))
    pooled.append(dict(input=inp, n_folds=len(glob.glob(f"../out/ekyt_v3{inp}/cv_s0_r0_f*.json")), n_part=len(s),
                       pooled_part_auc=roc_auc_score(yl, sc), ci_lo=np.percentile(bs, 2.5),
                       ci_hi=np.percentile(bs, 97.5), label_rate=yl.mean()))

F, C, P = pd.DataFrame(fits), pd.DataFrame(curves), pd.DataFrame(pooled)
for n_, t in (("fits", F), ("curves", C), ("pooled", P)):
    t.to_csv(f"{out_dir}/A3v3_{n_}.csv", index=False)
cv = F[F.scheme == "cv"].groupby("input")[["win_acc", "win_majority", "win_auc", "part_auc", "final_loss",
                                            "final_batch_acc"]].agg(["mean", "min", "max"])
with open(f"{out_dir}/A3v3_summary.md", "w") as fh:
    fh.write("# A3 v3: EKYT learning validity and input sensitivity (review round 1)\n\n"
             "Inputs: v3 (SG boundary mask fixed) 10 s PUR windows; vel = 2 velocity channels, pos = x/y gaze angle "
             "(interpolated, clipped +/-40 deg). Window accuracy is the unit comparable to the published 94.5%; "
             "participant AUROC is reported separately. CIs for pooled AUC are participant bootstraps over saved "
             "predictions (exclude refitting uncertainty).\n\n## Per fit\n\n"
             + F.round(3).to_markdown(index=False) + "\n\n## CV (repeat 0, 5 folds) by input\n\n"
             + cv.round(3).to_markdown() + "\n\n## Pooled out-of-fold participant AUC\n\n"
             + P.round(3).to_markdown(index=False) + "\n\n## Learning curves (checkpoints)\n\n"
             + C.round(3).to_markdown(index=False) + "\n")
print(open(f"{out_dir}/A3v3_summary.md").read())
