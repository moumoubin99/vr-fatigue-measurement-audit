"""A3 diagnosis (review RV-G01): train-mode/eval-mode gap of the adapted EKYT on velocity windows.

Trains one fit with the same code path as ekyt.py run (same split, seed, optimizer, batch order and checkpoint
curve), with bf16 autocast (as in the reported fits) or in fp32 (precision control), and saves the final weights.
Then it scores the SAME checkpoint on the SAME windows under paired conditions:
  eval_fp32        eval-mode BatchNorm (running statistics), fp32 forward      = the reported predict()
  eval_bf16        eval-mode BatchNorm, bf16 autocast forward
  batchstat_fp32   BatchNorm with batch statistics (train mode, running statistics frozen), fp32, batch 256
  batchstat_bf16   as above with bf16 autocast (the forward used for the reported training batch accuracy)
  recal_fp32       eval mode after recomputing the running statistics as exact averages over all training windows
Sets: the 4,000-window eval-mode training subset of ekyt.py, and the held-out windows. Batch statistics on
held-out windows are transductive and serve only as a diagnostic.
Metrics: window accuracy at 0.5, window AUC, participant AUC (mean window probability), mean probability,
share predicted positive. BatchNorm summary: ratio of reported to recalibrated running variance per layer.

Usage: python ekyt_diag.py NPZ OUT_DIR --scheme paper|cv --seed 0 [--fold F --repeat R] [--precision bf16|fp32]
       [--epochs 200]
"""
import argparse
import copy
import json
import os
import time

import numpy as np

from ekyt import CHECKPOINTS, build_net, splits_for


def main(npz, out_dir, scheme, seed, fold, repeat, epochs, precision, batch=256):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(2)
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(npz)
    X, pid, sess, y = d["X"], d["pid"], d["sess"], d["y"]
    tr_p, te_p = splits_for(pid, y, scheme, seed, fold, repeat)
    tr, te = np.isin(pid, tr_p), np.isin(pid, te_p)
    sd = X[tr].std()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    Xt = torch.tensor(X / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    net = build_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    idx_tr = np.where(tr)[0]
    idx_te = np.where(te)[0]
    sub_tr = np.random.RandomState(seed).permutation(idx_tr)[:4000]
    bf16 = precision == "bf16"

    def fwd(m, i, amp):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            return m(Xt[i]).float()

    def scores(m, idx, amp):
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), 256):
                i = torch.as_tensor(idx[b:b + 256], device=dev)
                out.append(torch.softmax(fwd(m, i, amp), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def predict(idx):
        net.eval()
        return scores(net, idx, False)

    def part_auc(idx, p):
        u = np.unique(pid[idx])
        if len(np.unique(y[idx])) < 2:
            return float("nan")
        return float(roc_auc_score([y[idx][pid[idx] == k][0] for k in u], [p[pid[idx] == k].mean() for k in u]))

    def metrics(idx, p):
        u = np.unique(pid[idx])
        pp = np.array([p[pid[idx] == k].mean() for k in u])
        yl = np.array([y[idx][pid[idx] == k][0] for k in u])
        return dict(win_acc=float(((p >= 0.5) == y[idx]).mean()), win_auc=float(roc_auc_score(y[idx], p)),
                    part_auc=float(roc_auc_score(yl, pp)), part_bacc=float(balanced_accuracy_score(yl, pp >= 0.5)),
                    mean_p=float(p.mean()), share_pos=float((p >= 0.5).mean()), base_rate=float(y[idx].mean()))

    # ---- training: identical to ekyt.run, with optional fp32 ------------------------------------------
    ep_loss, ep_acc, curve = [], [], []
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(idx_tr)
        ls, ac, nn_ = 0.0, 0.0, 0
        for b in range(0, len(perm), batch):
            i = torch.as_tensor(perm[b:b + batch], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
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
    t_train = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{scheme}_s{seed}_r{repeat}_f{fold}_{precision}"
    torch.save(net.state_dict(), os.path.join(out_dir, tag + ".pt"))

    # ---- paired conditions on the same checkpoint ----------------------------------------------------
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]

    def batchstat_net():
        m = copy.deepcopy(net)
        m.train()
        for b in m.modules():
            if isinstance(b, nn.BatchNorm1d):
                b.momentum = 0.0  # use batch statistics, leave the running statistics unchanged
        return m

    recal = copy.deepcopy(net)
    for b in recal.modules():
        if isinstance(b, nn.BatchNorm1d):
            b.reset_running_stats()
            b.momentum = None  # exact cumulative average
    recal.train()
    with torch.no_grad():
        rp = np.random.RandomState(seed + 1).permutation(idx_tr)
        for b in range(0, len(rp), 256):
            fwd(recal, torch.as_tensor(rp[b:b + 256], device=dev), False)
    recal.eval()
    rbns = [m for m in recal.modules() if isinstance(m, nn.BatchNorm1d)]
    bn_summary = []
    for k, (a, r) in enumerate(zip(bns, rbns)):
        ratio = (a.running_var / r.running_var.clamp_min(1e-12)).cpu().numpy()
        dmean = ((a.running_mean - r.running_mean).abs() / r.running_var.clamp_min(1e-12).sqrt()).cpu().numpy()
        bn_summary.append(dict(layer=k, n_ch=int(len(ratio)), var_ratio_median=float(np.median(ratio)),
                               var_ratio_min=float(ratio.min()), var_ratio_max=float(ratio.max()),
                               mean_shift_sd_median=float(np.median(dmean)), mean_shift_sd_max=float(dmean.max())))

    te_perm = np.random.RandomState(seed + 2).permutation(idx_te)  # mixed batches, as in training
    sets = {"train_subset": sub_tr, "heldout": te_perm}
    conds = {}
    net.eval()
    bs = batchstat_net()
    for s_name, idx in sets.items():
        conds[s_name] = {
            "eval_fp32": metrics(idx, scores(net, idx, False)),
            "eval_bf16": metrics(idx, scores(net, idx, True)),
            "batchstat_fp32": metrics(idx, scores(bs, idx, False)),
            "batchstat_bf16": metrics(idx, scores(bs, idx, True)),
            "recal_fp32": metrics(idx, scores(recal, idx, False)),
        }
    res = dict(scheme=scheme, seed=seed, fold=fold, repeat=repeat, epochs=epochs, batch=batch, precision=precision,
               n_train_p=len(tr_p), n_test_p=len(te_p), n_train_win=int(tr.sum()), n_test_win=int(te.sum()),
               train_seconds=round(t_train, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    json.dump(dict(res, conditions=conds, bn=bn_summary, curve=curve, epoch_loss=ep_loss, epoch_batch_acc=ep_acc),
              open(os.path.join(out_dir, tag + ".json"), "w"), indent=1)
    print(json.dumps(res))
    print(json.dumps(conds, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out_dir")
    ap.add_argument("--scheme", default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    main(a.npz, a.out_dir, a.scheme, a.seed, a.fold, a.repeat, a.epochs, a.precision)
