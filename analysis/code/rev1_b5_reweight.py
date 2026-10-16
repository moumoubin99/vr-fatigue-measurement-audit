"""REV1 / B5: precision-reweighted injected-effect recovery (Stage-3 review).

Local inputs only: results/B5_v3/B5_recording_outcomes.csv (per-recording O1/O2 per pipeline x condition),
B5_recovery.csv (nominal effects, published recovery), v3 primary.parquet (GazeBaseVR round-1 RAN precision,
the distribution b5_sim.py drew targets from).
Per-recording achieved precision was NOT saved locally (it lives in B5_trials.parquet on the server). It is
reconstructed as follows:
  * p_target: b5_sim.py draws np.random.default_rng(crc32(f"{rec}|draw")).choice(vr_prec); reproduced exactly
    (check: median of the 322 draws must equal B5_precision.csv target_median 0.2358).
  * attainable: unattainable recordings get sigma = 0, so P250_white and P250_ar1 outcomes equal P250_none
    exactly; a recording is classed unattainable if O2 is identical (|diff| < 1e-9) across all 8 conditions
    for both noise models (O1 cannot be used: per-pipeline main-sequence lines differ).
  * achieved precision of attainable recordings = p_target (B5 calibration: 97-98 % within 2 %); for
    unattainable recordings it is the recording's own 250 Hz precision, which is only known to be > p_target.
Reweighting: attainable recordings (P250_white, P250_ar1) are importance-weighted so that their achieved
precision distribution matches the full GazeBaseVR distribution, using VR quantile bins (deciles; quintiles as
sensitivity); w_rec = (1/K) / (share of attainable recordings in bin). Bins without attainable recordings
cannot be matched (unmatched VR mass reported). Recovery: weighted mean dO (cond - base, per recording, fixed
per-pipeline line as in B5_recording_outcomes) / nominal, and paired ratio to REF25 (sum w dP / sum w dREF25).
CIs: recording (= participant) bootstrap, 2000 resamples, seed 20261018, weights recomputed in each resample.
Usage: python rev1_b5_reweight.py PRIMARY_PARQUET B5_DIR OUT_DIR
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

SEED = 20261018
NB = 2000
CONDS = {"O1": ["vel0.95", "vel0.97", "vel1.03", "vel1.05"], "O2": ["shift-10", "shift+10"]}


def weights(p, att, edges):
    K = len(edges) - 1
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, K - 1)
    w = np.zeros(len(p))
    cnt = np.bincount(b[att], minlength=K)
    for k in range(K):
        sel = att & (b == k)
        if cnt[k]:
            w[sel] = (1 / K) / (cnt[k] / att.sum())
    return w, float((cnt == 0).sum() / K)


def stat(dp, dref, w):
    ok = np.isfinite(dp) & (w > 0)
    m = np.sum(w[ok] * dp[ok]) / np.sum(w[ok])
    ok2 = ok & np.isfinite(dref)
    rr = np.sum(w[ok2] * dp[ok2]) / np.sum(w[ok2] * dref[ok2])
    return m, rr


def main(primary, b5_dir, out):
    os.makedirs(out, exist_ok=True)
    prim = pd.read_parquet(primary)
    vr = prim[(prim["round"] == 1) & prim.ran_prec.notna()].ran_prec.to_numpy()
    r = pd.read_csv(os.path.join(b5_dir, "B5_recording_outcomes.csv"))
    rec = np.array(sorted(r.rec.unique()))
    pt = np.array([float(np.random.default_rng(zlib.crc32(f"{q}|draw".encode())).choice(vr)) for q in rec])
    calib = pd.read_csv(os.path.join(b5_dir, "B5_precision.csv")).set_index("pipe")
    assert abs(np.median(pt) - calib.loc["P250_white", "target_median"]) < 1e-12
    W2 = r.pivot_table(index="rec", columns=["pipe", "cond"], values="O2").reindex(rec)
    W1 = r.pivot_table(index="rec", columns=["pipe", "cond"], values="O1").reindex(rec)
    conds = [c for c in r.cond.unique()]
    same = np.all([np.abs(W2[(p, c)] - W2[("P250_none", c)]).to_numpy() < 1e-9
                   for p in ("P250_white", "P250_ar1") for c in conds], axis=0)
    att = ~same
    rv = pd.read_csv(os.path.join(b5_dir, "B5_recovery.csv"))
    nominal = rv.groupby("cond").nominal.first()
    # validation against the published attainable-subset recovery
    val = []
    for o, W in (("O1", W1), ("O2", W2)):
        for c in CONDS[o]:
            for p in ("P250_white", "P250_ar1"):
                d = (W[(p, c)] - W[(p, "base")]).to_numpy()
                pub = rv[(rv.cond == c) & (rv.outcome == o) & (rv["pipe"] == p) & (rv.subset == "attainable")]
                val.append(dict(outcome=o, cond=c, pipe=p, n_att_local=int((att & np.isfinite(d)).sum()),
                                recovered_att_local=np.nanmean(d[att]), n_att_pub=int(pub.n.iloc[0]),
                                recovered_att_pub=float(pub.recovered.iloc[0])))
    val = pd.DataFrame(val)
    rows = []
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(rec), (NB, len(rec)))
    for K in (10, 5):
        edges = np.quantile(vr, np.linspace(0, 1, K + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        w_full, unmatched = weights(pt, att, edges)
        for o, W in (("O1", W1), ("O2", W2)):
            for c in CONDS[o]:
                nom = nominal[c]
                dref = (W[("REF25", c)] - W[("REF25", "base")]).to_numpy()
                for p in ("P250_white", "P250_ar1", "P250_none"):
                    dp = (W[(p, c)] - W[(p, "base")]).to_numpy()
                    variants = {"all, unweighted": np.ones(len(rec)),
                                "attainable, unweighted": att.astype(float),
                                f"attainable, VR-reweighted ({K} bins)": w_full}
                    if p == "P250_none":
                        variants = {"all, unweighted": np.ones(len(rec))}
                    for vn, w in variants.items():
                        if K == 5 and "reweighted" not in vn:
                            continue
                        m, rr = stat(dp, dref, w)
                        bm, br = [], []
                        for idx in boot_idx:
                            if "reweighted" in vn:
                                wb, _ = weights(pt[idx], att[idx], edges)
                            else:
                                wb = w[idx]
                            a_, b_ = stat(dp[idx], dref[idx], wb)
                            bm.append(a_)
                            br.append(b_)
                        ok = np.isfinite(dp) & (w > 0)
                        ess = w[ok].sum() ** 2 / (w[ok] ** 2).sum()
                        rows.append(dict(outcome=o, cond=c, pipe=p, variant=vn, n=int(ok.sum()), ess=ess,
                                         unmatched_vr_mass=unmatched if "reweighted" in vn else np.nan,
                                         nominal=nom, recovered=m, rec_lo=np.nanpercentile(bm, 2.5),
                                         rec_hi=np.nanpercentile(bm, 97.5), ratio_nominal=m / nom,
                                         ratio_nom_lo=np.nanpercentile(np.array(bm) / nom, 2.5),
                                         ratio_nom_hi=np.nanpercentile(np.array(bm) / nom, 97.5),
                                         ratio_ref25=rr, ratio_ref25_lo=np.nanpercentile(br, 2.5),
                                         ratio_ref25_hi=np.nanpercentile(br, 97.5)))
    res = pd.DataFrame(rows)
    # sort ratio_nom CI for negative nominals
    lo = np.minimum(res.ratio_nom_lo, res.ratio_nom_hi)
    hi = np.maximum(res.ratio_nom_lo, res.ratio_nom_hi)
    res["ratio_nom_lo"], res["ratio_nom_hi"] = lo, hi
    # pooled over conditions (mean ratio across the 4 velocity / 2 latency conditions)
    pool = res.groupby(["outcome", "pipe", "variant"])[["ratio_nominal", "ratio_ref25"]].mean().reset_index()
    prec = pd.DataFrame(dict(set=["GazeBaseVR round-1 RAN (n=%d)" % len(vr), "drawn targets, all 322",
                                  "drawn targets, attainable", "drawn targets, unattainable"],
                             n=[len(vr), len(pt), int(att.sum()), int((~att).sum())],
                             median=[np.median(vr), np.median(pt), np.median(pt[att]), np.median(pt[~att])],
                             q25=[np.quantile(x, .25) for x in (vr, pt, pt[att], pt[~att])],
                             q75=[np.quantile(x, .75) for x in (vr, pt, pt[att], pt[~att])]))
    edges10 = np.quantile(vr, np.linspace(0, 1, 11))
    bins = pd.DataFrame(dict(vr_decile=np.arange(1, 11), lo=edges10[:-1], hi=edges10[1:],
                             n_attainable=np.bincount(np.clip(np.searchsorted(edges10, pt[att], side="right") - 1,
                                                              0, 9), minlength=10),
                             n_all=np.bincount(np.clip(np.searchsorted(edges10, pt, side="right") - 1, 0, 9),
                                               minlength=10)))
    res.to_csv(os.path.join(out, "B5_reweight.csv"), index=False)
    val.to_csv(os.path.join(out, "B5_reweight_validation.csv"), index=False)
    bins.to_csv(os.path.join(out, "B5_reweight_bins.csv"), index=False)
    show = res[res["pipe"] != "P250_none"][["outcome", "cond", "pipe", "variant", "n", "ess", "nominal", "recovered",
                                         "ratio_nominal", "ratio_nom_lo", "ratio_nom_hi", "ratio_ref25",
                                         "ratio_ref25_lo", "ratio_ref25_hi"]]
    md = ["# REV1 B5: precision-reweighted recovery\n",
          f"p_target reconstructed for all {len(rec)} recordings (median {np.median(pt):.4f}, matches "
          f"B5_precision.csv). Attainable (inferred from O2 identity): {int(att.sum())} "
          f"(B5_precision.csv: 154; 2 attainable recordings probably had noise that left every O2 unchanged). "
          f"Achieved precision for attainable recordings taken = p_target.\n",
          "## Precision distributions\n", prec.round(4).to_markdown(index=False), "",
          "## VR deciles: attainable recordings per bin\n", bins.round(4).to_markdown(index=False), "",
          "## Validation: local attainable-subset recovery vs published B5_recovery (attainable)\n",
          val.round(5).to_markdown(index=False), "",
          "## Recovery (per condition)\n",
          "`ratio_nominal` = recovered / nominal injected effect; `ratio_ref25` = paired ratio to the 1000 Hz "
          "REF25 pipeline. Recording bootstrap 95% CIs (weights recomputed per resample). O1 uses each "
          "pipeline's fixed base-condition line (B5_recording_outcomes), not the refit-per-bootstrap line of "
          "B5_recovery, so CIs are narrower than a refit bootstrap would give.\n",
          show.round(4).to_markdown(index=False), "",
          "## Mean ratio across conditions\n", pool.round(4).to_markdown(index=False), "",
          "## P250_none (no added noise) reference\n",
          res[res["pipe"] == "P250_none"][["outcome", "cond", "recovered", "ratio_nominal", "ratio_ref25"]]
          .round(4).to_markdown(index=False), ""]
    open(os.path.join(out, "B5_reweight.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:4])
