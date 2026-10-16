"""A3 follow-up summary (CPU): identity-leakage and positive controls, and the re-run of the original A3 set.

Reads the outputs of ekyt_ctrl_queue.sh (copied to results/A3_ctrl/{vel,pos} and results/A3_rerun/{vel,pos}) and
the frozen A3 fits (data/server_out/out/ekyt_v3{vel,pos}). Participant bootstrap CIs (2000 draws, seed 0; for the pooled CV AUC one generator per run, used for vel
then pos, as in ekyt_v3_summary.py) are conditional on the saved predictions and do not include refit uncertainty.
Writes results/A3_ctrl/A3_ctrl_summary.csv, results/A3_rerun/A3_rerun_compare.csv, results/A3_rerun/A3_rerun_pooled.csv
and results/A3_ctrl/A3_followup_summary.md.

Usage: python a3_ctrl_summary.py [ROOT]   (ROOT = project directory, default ..)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
CTRL, RERUN = os.path.join(RES, "A3_ctrl"), os.path.join(RES, "A3_rerun")
FROZEN = os.path.join(ROOT, "data/server_out/out")
SCHEMES = ["window", "window_perm", "session", "identity", "paper_catchup", "paper_age"]
LAB = pd.read_csv(os.path.join(RES, "REV2_VR/labels_round1.csv")).set_index("pid")["label_a001"]


def boot_auc(y, s, n=2000, seed=0, rng=None):
    rng = np.random.default_rng(seed) if rng is None else rng
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ctrl_table():
    rows = []
    for inp in ("vel", "pos"):
        for sc in SCHEMES:
            f = os.path.join(CTRL, inp, f"{sc}_s0.json")
            if not os.path.exists(f):
                continue
            j = json.load(open(f))
            p = np.load(os.path.join(CTRL, inp, f"{sc}_s0_pred.npz"))
            g = pd.DataFrame(dict(pid=p["pid"], s=p["score"], y=p["target"])).groupby("pid").agg(s=("s", "mean"),
                                                                                                    y=("y", "first"))
            lo, hi = boot_auc(g.y.to_numpy(), g.s.to_numpy())
            last = j["curve"][-1]
            rows.append(dict(input=inp, scheme=sc, n_train_p=j["n_train_p"], n_test_p=j["n_test_p"],
                             n_shared_p=j["n_shared_p"], n_train_win=j["n_train_win"], n_test_win=j["n_test_win"],
                             final_loss=j["epoch_loss"][-1], final_batch_acc=j["epoch_batch_acc"][-1],
                             train_win_acc=last["train_win_acc"], win_acc=j["win_acc"],
                             win_majority=j["win_majority"], win_auc=j["win_auc"], part_auc=j["part_auc"],
                             part_auc_lo=lo, part_auc_hi=hi, part_bacc=j["part_bacc"], part_acc=j["part_acc"],
                             id_top1_win=j.get("id_top1_win"), id_top1_part=j.get("id_top1_part"),
                             best_epoch_part_auc=max(c["test_part_auc"] for c in j["curve"])))
    return pd.DataFrame(rows)


def fits(d, inp, tag, rng):
    rows, rec = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        name = os.path.basename(f)[:-5]
        last = j["curve"][-1]
        rows.append(dict(input=inp, fit=name, run=tag, win_acc=j["win_acc"], win_auc=j["win_auc"],
                         part_auc=j["part_auc"], part_bacc=j["part_bacc"], final_loss=j["epoch_loss"][-1],
                         final_batch_acc=j["epoch_batch_acc"][-1], train_eval_acc=last["train_eval_acc"],
                         train_eval_win_auc=last["train_eval_win_auc"]))
        if name.startswith("cv_"):
            rec += j["rec_scores"]
    pooled = {}
    if rec:
        s = pd.DataFrame(rec).groupby("pid").score.mean()
        y = LAB.loc[s.index].to_numpy().astype(int)
        lo, hi = boot_auc(y, s.to_numpy(), rng=rng)
        pooled = dict(input=inp, run=tag, n_part=len(s), pooled_part_auc=roc_auc_score(y, s.to_numpy()), ci_lo=lo,
                      ci_hi=hi)
    return rows, pooled


def main():
    md = ["# A3 follow-up: controls and re-run\n", __doc__.split("Usage")[0].strip() + "\n"]
    C = ctrl_table()
    if len(C):
        C.to_csv(os.path.join(CTRL, "A3_ctrl_summary.csv"), index=False)
        md += ["## Controls (final epoch, held-out set)\n", C.round(3).to_markdown(index=False) + "\n"]
    rows, pooled = [], []
    # one generator per run, used for vel then pos, as in ekyt_v3_summary.py (reproduces the frozen CIs)
    rngs = {"frozen": np.random.default_rng(0), "rerun": np.random.default_rng(0)}
    for inp in ("vel", "pos"):
        for d, tag in [(os.path.join(FROZEN, f"ekyt_v3{inp}"), "frozen"), (os.path.join(RERUN, inp), "rerun")]:
            r, p = fits(d, inp, tag, rngs[tag])
            rows += r
            if p:
                pooled.append(p)
    R = pd.DataFrame(rows)
    if (R.run == "rerun").any():
        W = R.pivot_table(index=["input", "fit"], columns="run",
                          values=["win_auc", "part_auc", "train_eval_acc", "final_batch_acc"])
        W.columns = [f"{a}_{b}" for a, b in W.columns]
        W = W.reset_index()
        W.to_csv(os.path.join(RERUN, "A3_rerun_compare.csv"), index=False)
        P = pd.DataFrame(pooled)
        P.to_csv(os.path.join(RERUN, "A3_rerun_pooled.csv"), index=False)
        md += ["## Re-run vs frozen fits (final epoch)\n", W.round(3).to_markdown(index=False) + "\n",
               "## Pooled CV participant AUC\n", P.round(3).to_markdown(index=False) + "\n"]
    with open(os.path.join(CTRL, "A3_followup_summary.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
