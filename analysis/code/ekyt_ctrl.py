"""A3 controls (review follow-up): identity leakage and held-out positive controls for the adapted EKYT.

Same network, optimizer, batch, epochs, bf16 autocast and checkpoint curve as ekyt.py run. Only the split or the
target changes. Schemes:
  window        random 80/20 split of WINDOWS (stratified by label): the same people are in training and test
  window_perm   as window, with the fatigue label permuted across participants (seed 0): identity carries a
                random participant label
  session       train on all session-1 windows, test on session-2 windows of the same people
  identity      session split; the network learns participant identity (one class per person). Fatigue score =
                summed probability of the label-1 participants. "Train on identity, test the fatigue label."
  paper_catchup paper split (same people as ekyt.py --scheme paper); label = participant mean PUR catch-up
                saccade rate (round 1, both sessions) above the median: a gaze trait, expected to generalize
  paper_age     paper split; label = age above the median
Sessions used: participants with windows in both sessions for session and identity; all for the others.

Usage: python ekyt_ctrl.py NPZ TABLES_DIR OUT_DIR --scheme S [--seed 0] [--epochs 200]
"""
import argparse
import json
import os

import numpy as np

from ekyt import CHECKPOINTS, build_net, splits_for


def make_task(scheme, X, pid, sess, y, tables, seed):
    """Return (target, train mask, test mask, n_out, fatigue label per window)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    n = len(y)
    both = np.isin(pid, np.intersect1d(pid[sess == 1], pid[sess == 2]))
    if scheme in ("window", "window_perm"):
        t = y.copy()
        if scheme == "window_perm":
            up = np.unique(pid)
            lab = np.array([y[pid == p][0] for p in up])
            perm = dict(zip(up, np.random.RandomState(seed).permutation(lab)))
            t = np.array([perm[p] for p in pid])
        tr_i, te_i = train_test_split(np.arange(n), test_size=0.2, stratify=t, random_state=42)
        tr, te = np.zeros(n, bool), np.zeros(n, bool)
        tr[tr_i], te[te_i] = True, True
        return t, tr, te, 2
    if scheme in ("session", "identity"):
        tr, te = both & (sess == 1), both & (sess == 2)
        if scheme == "session":
            return y.copy(), tr, te, 2
        up = np.unique(pid[both])
        idx = {p: k for k, p in enumerate(up)}
        t = np.array([idx.get(p, -1) for p in pid])
        return t, tr, te, len(up)
    tr_p, te_p = splits_for(pid, y, "paper", seed, 0, 0)
    tr, te = np.isin(pid, tr_p), np.isin(pid, te_p)
    if scheme == "paper_catchup":
        w = pd.read_parquet(os.path.join(tables, "wide.parquet"))
        v = w[w["round"] == 1].groupby("pid")["PUR__pur_catchup_rate"].mean()
    elif scheme == "paper_age":
        m = pd.read_parquet(os.path.join(tables, "meta.parquet"))
        v = m[m["round"] == 1].set_index("pid")["age"].astype(float)
    else:
        raise ValueError(scheme)
    v = v.reindex(np.unique(pid))
    ok = v.notna()
    thr = v[ok].median()
    lab = (v > thr).astype(int)
    keep = np.isin(pid, v.index[ok])
    t = np.array([lab.get(p, 0) for p in pid])
    return t, tr & keep, te & keep, 2


def main(npz, tables, out_dir, scheme, seed, epochs, batch=256):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(npz)
    X, pid, sess, y = d["X"], d["pid"], d["sess"], d["y"]
    t, tr, te, n_out = make_task(scheme, X, pid, sess, y, tables, seed)
    sd = X[tr].std()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    Xt = torch.tensor(X / sd, dtype=torch.float32, device=dev)
    tt = torch.tensor(np.maximum(t, 0), dtype=torch.long, device=dev)
    net = build_net(n_out=n_out).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    idx_tr, idx_te = np.where(tr)[0], np.where(te)[0]
    sub_tr = np.random.RandomState(seed).permutation(idx_tr)[:4000]
    pos_cls = None
    if scheme == "identity":  # classes whose participant has fatigue label 1
        up = np.unique(pid[tr | te])
        pos_cls = torch.tensor([int(y[pid == p][0] == 1) for p in up], dtype=torch.float32, device=dev)

    def predict(idx):
        """Return (score for target-1 or fatigue-1, predicted class)."""
        net.eval()
        sc, cl = [], []
        with torch.no_grad():
            for b in range(0, len(idx), 256):
                i = torch.as_tensor(idx[b:b + 256], device=dev)
                pr = torch.softmax(net(Xt[i]).float(), 1)
                sc.append((pr @ pos_cls if pos_cls is not None else pr[:, 1]).cpu().numpy())
                cl.append(pr.argmax(1).cpu().numpy())
        return np.concatenate(sc), np.concatenate(cl)

    lab = y if scheme == "identity" else t  # label that the score is evaluated against

    def summary(idx, p, cl):
        u = np.unique(pid[idx])
        pp = np.array([p[pid[idx] == k].mean() for k in u])
        yl = np.array([lab[idx][pid[idx] == k][0] for k in u])
        r = dict(win_acc=float(((p >= 0.5) == lab[idx]).mean()), win_auc=float(roc_auc_score(lab[idx], p)),
                 part_auc=float(roc_auc_score(yl, pp)), part_bacc=float(balanced_accuracy_score(yl, pp >= 0.5)),
                 part_acc=float(((pp >= 0.5) == yl).mean()), n_part=int(len(u)),
                 win_majority=float(max(lab[idx].mean(), 1 - lab[idx].mean())))
        if scheme == "identity":
            r["id_top1_win"] = float((cl == t[idx]).mean())
            # participant identified by majority vote over windows
            r["id_top1_part"] = float(np.mean([np.bincount(cl[pid[idx] == k]).argmax() == t[idx][pid[idx] == k][0]
                                               for k in u]))
        return r

    ep_loss, ep_acc, curve = [], [], []
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(idx_tr)
        ls, ac, nn_ = 0.0, 0.0, 0
        for b in range(0, len(perm), batch):
            i = torch.as_tensor(perm[b:b + batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(Xt[i]).float()
                loss = F.cross_entropy(logit, tt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls += float(loss) * len(i)
            ac += float((logit.argmax(1) == tt[i]).sum())
            nn_ += len(i)
        ep_loss.append(ls / nn_)
        ep_acc.append(ac / nn_)
        if ep + 1 in CHECKPOINTS or ep + 1 == epochs:
            ptr, ctr = predict(sub_tr)
            pte, cte = predict(idx_te)
            c = dict(epoch=ep + 1, train_loss=ep_loss[-1], train_batch_acc=ep_acc[-1])
            c.update({"train_" + k: v for k, v in summary(sub_tr, ptr, ctr).items()})
            c.update({"test_" + k: v for k, v in summary(idx_te, pte, cte).items()})
            curve.append(c)
            print(json.dumps(c), flush=True)
    p, cl = predict(idx_te)
    res = dict(scheme=scheme, seed=seed, epochs=epochs, batch=batch, n_out=n_out,
               n_train_win=int(tr.sum()), n_test_win=int(te.sum()),
               n_train_p=int(len(np.unique(pid[tr]))), n_test_p=int(len(np.unique(pid[te]))),
               n_shared_p=int(len(np.intersect1d(pid[tr], pid[te]))), **summary(idx_te, p, cl),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{scheme}_s{seed}"
    np.savez_compressed(os.path.join(out_dir, tag + "_pred.npz"), idx=idx_te, score=p, cls=cl,
                        pid=pid[idx_te], sess=sess[idx_te], target=lab[idx_te])
    json.dump(dict(res, curve=curve, epoch_loss=ep_loss, epoch_batch_acc=ep_acc),
              open(os.path.join(out_dir, tag + ".json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("tables")
    ap.add_argument("out_dir")
    ap.add_argument("--scheme", required=True,
                    choices=["window", "window_perm", "session", "identity", "paper_catchup", "paper_age"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    main(a.npz, a.tables, a.out_dir, a.scheme, a.seed, a.epochs)
