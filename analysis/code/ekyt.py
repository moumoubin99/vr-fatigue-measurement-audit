"""A3: EKYT-style DenseNet reproduction of the Zafar et al. (2025) PUR / 10 s configuration.

Steps
  prep : build PUR input arrays -> npz. --input vel: horizontal/vertical velocity (250 Hz, Savitzky-Golay with
         the v3 missing-data mask, clamped +-1000 deg/s). --input pos: horizontal/vertical gaze angle (deg; gaps
         linearly interpolated, clamped +-40 deg), the input type described by Zafar et al.
  run  : train/evaluate one split specification and save window- and recording-level scores, plus a learning
         curve (mean training loss / batch accuracy every epoch; eval-mode training and held-out metrics at
         checkpoint epochs). Scheme "overfit": 40 participants, trained and evaluated on the same windows
         (learning-capacity control).

Documented choices where the benchmark is silent:
  input channels = horizontal and vertical velocity (EKYT); window = 10 s non-overlapping;
  both sessions of round 1 used; label = participant label repeated on every window;
  Adam lr 1e-3, batch 256, 200 epochs, no early stopping, last epoch evaluated. The paper reports 200 epochs and
  100-150 s per model on an RTX 4090 but no batch size; batch 256 is our documented choice (64 as sensitivity).
Usage:
  python ekyt.py prep DATA_DIR META_PARQUET OUT_NPZ [--input vel|pos]
  python ekyt.py run NPZ OUT_DIR --scheme paper|cv|cv_s1|random|overfit --seed S [--fold F --repeat R] [--batch B]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

WIN = 2500  # 10 s at 250 Hz
CHECKPOINTS = (1, 10, 25, 50, 100, 150, 200)


def prep(data_dir, meta_parquet, out_npz, inp="vel"):
    import pandas as pd
    import gazefeat
    meta = pd.read_parquet(meta_parquet)
    lab = meta[meta["round"] == 1].set_index("pid").label
    X, P, S, Y = [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "S_1*_PUR.csv"))):
        m = gazefeat.parse_name(f)
        if m["round"] != 1 or not np.isfinite(lab.get(m["pid"], np.nan)):
            continue
        sig = gazefeat.load(f)
        if inp == "vel":
            vx, vy, _ = gazefeat.velocity(sig["x"], sig["y"])
            v = np.nan_to_num(np.clip(np.vstack([vx, vy]), -1000, 1000)).astype(np.float32)
        else:
            ch = []
            for c in (sig["x"], sig["y"]):
                ok = np.isfinite(c)
                if ok.sum() < 2:
                    break
                ii = np.arange(len(c))
                ch.append(np.clip(np.interp(ii, ii[ok], c[ok]), -40, 40))
            if len(ch) < 2:
                continue
            v = np.vstack(ch).astype(np.float32)
        for k in range(v.shape[1] // WIN):
            X.append(v[:, k * WIN:(k + 1) * WIN])
            P.append(m["pid"])
            S.append(m["session"])
            Y.append(int(lab[m["pid"]]))
    np.savez_compressed(out_npz, X=np.stack(X), pid=np.array(P), sess=np.array(S), y=np.array(Y))
    print("windows", len(X), "participants", len(set(P)))


def build_net(in_ch=2, growth=32, depth=8, n_out=2):
    import torch.nn as nn
    import torch

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            c = in_ch
            for i in range(depth):
                self.layers.append(nn.Sequential(
                    nn.BatchNorm1d(c), nn.ReLU(),
                    nn.Conv1d(c, growth, 3, padding=2 ** (i % 7), dilation=2 ** (i % 7))))
                c += growth
            self.head = nn.Sequential(nn.BatchNorm1d(c), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, n_out))

        def forward(self, x):
            for l in self.layers:
                x = torch.cat([x, l(x)], 1)
            return self.head(x)
    return Dense()


def splits_for(pids, labels, scheme, seed, fold, repeat):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    up = np.unique(pids)
    ul = np.array([labels[pids == p][0] for p in up])
    if scheme == "overfit":
        _, sub = train_test_split(up, test_size=40, stratify=ul, random_state=seed)
        return sub, sub
    if scheme in ("paper", "random"):
        tr, te = train_test_split(up, test_size=0.2, stratify=ul,
                                  random_state=(42 if scheme == "paper" else 1000 + seed))
        return tr, te
    sk = list(StratifiedKFold(5, shuffle=True, random_state=repeat).split(up, ul))
    return up[sk[fold][0]], up[sk[fold][1]]


def run(npz, out_dir, scheme, seed, fold, repeat, epochs, batch=256):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(npz)
    X, pid, sess, y = d["X"], d["pid"], d["sess"], d["y"]
    tr_p, te_p = splits_for(pid, y, scheme, seed, fold, repeat)
    tr, te = np.isin(pid, tr_p), np.isin(pid, te_p)
    if scheme == "cv_s1":  # A4: train on session-1 windows only; score both sessions of held-out people
        tr &= sess == 1
    sd = X[tr].std()  # global scale from training windows only
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    Xt = torch.tensor(X / sd, dtype=torch.float32, device=dev)  # whole set fits in GPU memory
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    net = build_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    idx_tr = np.where(tr)[0]
    idx_te = np.where(te)[0]
    sub_tr = np.random.RandomState(seed).permutation(idx_tr)[:4000]  # eval-mode training subset

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), 256):
                i = torch.as_tensor(idx[b:b + 256], device=dev)
                out.append(torch.softmax(net(Xt[i]).float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def part_auc(idx, p):
        u = np.unique(pid[idx])
        if len(np.unique(y[idx])) < 2:
            return float("nan")
        return float(roc_auc_score([y[idx][pid[idx] == k][0] for k in u], [p[pid[idx] == k].mean() for k in u]))

    ep_loss, ep_acc, curve = [], [], []
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(idx_tr)
        ls, ac, nn_ = 0.0, 0.0, 0
        for b in range(0, len(perm), batch):
            i = torch.as_tensor(perm[b:b + batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(Xt[i]).float()
                loss = F.cross_entropy(logit, yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls += float(loss) * len(i)
            ac += float((logit.argmax(1) == yt[i]).sum())
            nn_ += len(i)
        ep_loss.append(ls / nn_)
        ep_acc.append(ac / nn_)
        if ep + 1 in CHECKPOINTS or ep + 1 == epochs:
            ptr, pte = predict(sub_tr), predict(idx_te)
            curve.append(dict(epoch=ep + 1, train_loss=ep_loss[-1], train_batch_acc=ep_acc[-1],
                              train_eval_acc=float(((ptr >= 0.5) == y[sub_tr]).mean()),
                              train_eval_win_auc=float(roc_auc_score(y[sub_tr], ptr)),
                              train_part_auc=part_auc(sub_tr, ptr),
                              test_win_acc=float(((pte >= 0.5) == y[idx_te]).mean()),
                              test_win_auc=float(roc_auc_score(y[idx_te], pte)), test_part_auc=part_auc(idx_te, pte)))
            print(json.dumps(curve[-1]), flush=True)
    p = predict(idx_te)
    yw = y[te]
    res = dict(scheme=scheme, seed=seed, fold=fold, repeat=repeat, epochs=epochs, batch=batch,
               n_train_p=len(tr_p), n_test_p=len(te_p), n_test_win=int(te.sum()),
               win_acc=float(((p >= 0.5) == yw).mean()), win_auc=float(roc_auc_score(yw, p)))
    # participant level: mean window probability
    tp = pid[te]
    up = np.unique(tp)
    pp = np.array([p[tp == u].mean() for u in up])
    yl = np.array([yw[tp == u][0] for u in up])
    res.update(part_auc=float(roc_auc_score(yl, pp)), part_bacc=float(balanced_accuracy_score(yl, pp >= 0.5)),
               win_majority=float(max(yw.mean(), 1 - yw.mean())))
    # recording-level held-out scores (for A4)
    ts = sess[te]
    rec = [dict(pid=int(u), session=int(s), score=float(p[(tp == u) & (ts == s)].mean()))
           for u in up for s in (1, 2) if ((tp == u) & (ts == s)).any()]
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{scheme}_s{seed}_r{repeat}_f{fold}" + ("" if batch == 256 else f"_b{batch}")
    json.dump(dict(res, rec_scores=rec, curve=curve, epoch_loss=ep_loss, epoch_batch_acc=ep_acc),
              open(os.path.join(out_dir, tag + ".json"), "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--scheme", default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--input", default="vel", choices=["vel", "pos"])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(*a.args, inp=a.input)
    else:
        run(a.args[0], a.args[1], a.scheme, a.seed, a.fold, a.repeat, a.epochs, a.batch)
