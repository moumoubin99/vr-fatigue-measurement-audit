"""B5: conditional measurement sensitivity at 250 Hz (GazeBase 1000 Hz RAN as reference; GazeBaseVR PUR).

Pipelines (all through gazefeat, same detector and trial rules)
  REF25 : 1000 Hz, SG 25 samples (24 ms span)          - reference
  REF7  : 1000 Hz, SG 7 samples (6 ms span)            - "same pipeline" reference for total degradation
  P250_none / P250_white / P250_ar1 : every 4th sample (random phase per recording), SG 7 (24 ms span);
          noise added after decimation so that the precision achieved by the complete pipeline matches a
          precision drawn from the GazeBaseVR round-1 RAN distribution (iterative calibration; draws at or
          below the recording's own 250 Hz precision are flagged unattainable and get no noise).
          AR(1) rho = 0.8 per 4 ms. Jump times always come from the full-rate target log.
          The same noise realisation (common random numbers) is used for every injection condition.
Conditions (local time warps on the 1000 Hz signal around every reference primary saccade with >= 60 ms
clearance from neighbouring saccades before and after the warp; endpoints, amplitude and target timestamps
unchanged)
  base; zero (base with another decimation phase and noise seed); vel x0.95/0.97/1.03/1.05;
  onset -10/+10 ms (saccade shifted, neighbouring intersaccadic intervals locally rescaled).
Events are matched by target jump. Reported per pipeline: detection agreement with REF25, matched-event
bias, and injected-effect recovery both among matched events and under the complete pipeline (recording
outcome O1 = median log peak-velocity residual from the pipeline's own base-condition main sequence,
O2 = median latency of primary responses to 5-25 deg steps). Uncertainty clustered by participant.
PUR (O3): GazeBaseVR PUR recordings; eye position += dg * target displacement within each moving segment
(decaying over 250 ms after it), dg in {-0.05,-0.03,0,+0.03,+0.05}; recovered change in pur_gain.
Usage: python b5_sim.py GB_RAN_DIR VR_DATA_DIR VR_PRIMARY_PARQUET OUT_DIR [--n 322] [--npur N] [--jobs 60]
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter, savgol_filter

import gazefeat as gf

C1K = gf.replace(gf.DEFAULT, fs=1000.0, sg_win=25)
C1K7 = gf.replace(gf.DEFAULT, fs=1000.0, sg_win=7)
C250 = gf.DEFAULT
VEL = (0.95, 0.97, 1.03, 1.05)
SHIFT = (-10, 10)
RHO = 0.8
W_MS, P_MS = 60, 60


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def run_pipe(df, cfg, tgt=None):
    """tgt = (t_ms, xT, yT): the full-rate logged target record. Jump times and target positions are
    always taken from it, so decimated pipelines keep the original jump timestamps."""
    sig = gf.load(None, cfg, raw=df)
    if tgt is not None:
        sig["tgt_t"] = tgt[0] / 1000.0 - sig["t0"]
        sig["xT_raw"], sig["yT_raw"] = tgt[1], tgt[2]
    x, y = sig["x"], sig["y"]
    _, _, sp = gf.velocity(x, y, cfg)
    sac, _, _ = gf.detect_saccades(x, y, sp, cfg)
    fx, insac = gf.intervals(x, y, sac, len(x), cfg)
    tr = gf.ran_trials(sig, sac, insac, cfg)
    if len(tr):
        tr["t_abs"] = tr["t_jump"] + sig["t0"]
    return tr, gf.precision_rms(sig, fx, None, cfg), sac


def grid_df(t_ms, x, y, xT, yT):
    return pd.DataFrame({"n": t_ms, "x": x, "y": y, "xT": xT, "yT": yT})


def warp(arr, tau, lo, hi):
    return np.interp(tau, np.arange(lo, hi + 1), arr[lo:hi + 1])


def inject(x, y, events, kind, val):
    """events: list of (jump, s, e, prev_off, next_on) in 1 kHz samples. Returns warped copies and
    the set of jump indices whose saccade was actually modified. An event is modified only if the
    warped saccade keeps >= 60 ms clearance from the preceding and following detected saccades."""
    x2, y2 = x.copy(), y.copy()
    done = set()
    for jmp, s, e, prev_off, next_on in events:
        if kind == "vel":
            k, D = val, e - s
            Dk = D / k
            T = int(np.ceil(max(e, s + Dk))) + P_MS       # 60 ms after the (possibly slowed) offset
            if s - W_MS <= prev_off or T >= next_on or T - s - Dk < 10:
                continue
            t = np.arange(s, T)
            tau = np.where(t < s + Dk, s + k * (t - s), e + (t - s - Dk) * (T - e) / (T - s - Dk))
            lo, hi = s, T
        else:
            d = int(val)
            A, B = s - W_MS - abs(d), e + P_MS + abs(d)
            if A <= prev_off or B >= next_on:
                continue
            t = np.arange(A, B)
            tau = np.where(t < s + d, A + (t - A) * (s - A) / (s + d - A),
                           np.where(t < e + d, t - d, e + (t - e - d) * (B - e) / (B - e - d)))
            lo, hi = A, B
        if not (np.isfinite(x[lo:hi + 1]).all() and np.isfinite(y[lo:hi + 1]).all()):
            continue
        x2[lo:hi] = warp(x, tau, lo, hi)
        y2[lo:hi] = warp(y, tau, lo, hi)
        done.add(jmp)
    return x2, y2, done


def add_noise(x, y, sigma, kind, rng):
    if kind == "none" or sigma <= 0:
        return x, y
    n = len(x)
    e = rng.standard_normal((2, n))
    if kind == "ar1":
        e[:, 0] /= np.sqrt(1 - RHO ** 2)            # stationary start
        e = lfilter([np.sqrt(1 - RHO ** 2)], [1, -RHO], e, axis=1)
    return x + sigma * e[0], y + sigma * e[1]


def noise_sigma(p_base, p_target, kind):
    extra = max(p_target ** 2 - p_base ** 2, 0.0)
    if kind == "white":
        return np.sqrt(extra / 4)
    if kind == "ar1":
        return np.sqrt(extra / (4 * (1 - RHO)))
    return 0.0


def one_recording(path, vr_prec):
    rec = os.path.basename(path)[:-4]
    raw = gf.read_raw(path, "gb")
    tgt = (raw["n"].to_numpy(float), raw["xT"].to_numpy(float), raw["yT"].to_numpy(float))
    base = gf.load(None, C1K, raw=raw)
    t_ms = np.round((base["t"] + base["t0"]) * 1000).astype(np.int64)
    x, y, xT, yT = base["x"], base["y"], base["xT"], base["yT"]
    tr_ref, _, sac_ref = run_pipe(grid_df(t_ms, x, y, xT, yT), C1K, tgt)
    # reference primary saccades -> events with neighbour limits
    on_all, off_all = sac_ref["on"].to_numpy(), sac_ref["off"].to_numpy()
    events = []
    for r in tr_ref[tr_ref.cls == "primary"].itertuples():
        j = int(round(r.t_jump * 1000))
        s = j + int(round(r.lat_ms))
        k = np.searchsorted(on_all, s)
        if k >= len(on_all) or on_all[k] != s:
            continue
        e = off_all[k]
        prev_off = off_all[k - 1] if k > 0 else 0
        next_on = on_all[k + 1] if k + 1 < len(on_all) else len(x)
        events.append((int(r.jump), s, e, prev_off, next_on))
    rng0 = np.random.default_rng(seed_of(rec, "draw"))
    p_target = float(rng0.choice(vr_prec))
    phase0 = int(rng0.integers(0, 4))
    conds = [("base", None, None), ("zero", None, None)] + [(f"vel{v}", "vel", v) for v in VEL] + \
            [(f"shift{d:+d}", "shift", d) for d in SHIFT]
    # noise scale calibrated on the base-condition decimated signal so that the precision achieved by
    # the complete pipeline (noise -> redetection -> fixation mask -> RMS-S2S) matches the drawn target;
    # the same sigma and noise realisation are then used for every injection condition. A draw at or
    # below the recording's own 250 Hz precision is unattainable: no noise is added and it is flagged.
    sl0 = slice(phase0, None, 4)

    def p250(xs, ys, sl):
        return run_pipe(grid_df(t_ms[sl], xs, ys, xT[sl], yT[sl]), C250, tgt)

    _, p_base, _ = p250(x[sl0], y[sl0], sl0)
    attain = bool(np.isfinite(p_base) and p_target > p_base)
    sig_n, p_ach = {"none": 0.0}, {"none": p_base}
    for nk in ("white", "ar1"):
        sg, pa = (noise_sigma(p_base, p_target, nk), np.nan) if attain else (0.0, p_base)
        for it in range(6 if attain else 0):
            rng = np.random.default_rng(seed_of(rec, nk, "crn"))
            _, pa, _ = p250(*add_noise(x[sl0], y[sl0], sg, nk, rng), sl0)
            if not np.isfinite(pa) or abs(pa / p_target - 1) < 0.02 or it == 5:
                break
            sg *= np.sqrt((p_target ** 2 - p_base ** 2) / max(pa ** 2 - p_base ** 2, 1e-12))
        sig_n[nk], p_ach[nk] = sg, pa
    out = []
    for cname, kind, val in conds:
        xi, yi, done = (x, y, set()) if kind is None else inject(x, y, events, kind, val)
        if cname != "zero":
            for pname, cfg in (("REF25", C1K), ("REF7", C1K7)):
                tr, prec, _ = run_pipe(grid_df(t_ms, xi, yi, xT, yT), cfg, tgt)
                out.append(tr.assign(rec=rec, cond=cname, pipe=pname, prec=prec, p_target=p_target,
                                     p_base=p_base, attainable=attain, calib_ok=attain, sigma=0.0,
                                     inj=tr["jump"].isin(done) if len(tr) else False))
        # "zero": base signal, other decimation phase and other noise draw (null replicate)
        phase = (phase0 + 2) % 4 if cname == "zero" else phase0
        sl = slice(phase, None, 4)
        for nk in ("none", "white", "ar1"):
            rng = np.random.default_rng(seed_of(rec, nk, "zero" if cname == "zero" else "crn"))
            xn, yn = add_noise(xi[sl], yi[sl], sig_n[nk], nk, rng)
            tr, prec, _ = run_pipe(grid_df(t_ms[sl], xn, yn, xT[sl], yT[sl]), C250, tgt)
            out.append(tr.assign(rec=rec, cond=cname, pipe=f"P250_{nk}", prec=prec, p_target=p_target,
                                 p_base=p_base, attainable=attain, sigma=sig_n[nk],
                                 calib_ok=attain and bool(np.isfinite(p_ach[nk])),
                                 inj=tr["jump"].isin(done) if len(tr) else False))
    d = pd.concat(out, ignore_index=True)
    keep = ["rec", "cond", "pipe", "prec", "p_target", "p_base", "attainable", "calib_ok", "sigma", "jump", "t_abs", "inj", "step", "cls", "lat_ms", "amp", "vpeak", "dur_ms"]
    return d[[c for c in keep if c in d]]


# ------------------------------------------------------------------ PUR gain injection (GazeBaseVR)
def pur_one(path, dgs=(-0.05, -0.03, 0.0, 0.03, 0.05)):
    raw = gf.read_raw(path, "vr")
    sig = gf.load(None, C250, raw=raw)
    fs = C250.fs
    xT = sig["xT"]
    starts, ends = gf.pur_motion_segments(sig, C250)          # from the logged target trajectory
    disp = np.zeros(len(xT))
    for s, e in zip(starts, ends):
        disp[s:e] = xT[s:e] - xT[s]
        n = min(int(0.25 * fs), len(xT) - e)
        if n > 0:
            disp[e:e + n] = (xT[e - 1] - xT[s]) * (1 - np.arange(1, n + 1) / n)
    t_ms = np.round(sig["t"] * 1000).astype(np.int64)
    rows = []
    for dg in dgs:
        df = pd.DataFrame({"n": t_ms, "x": sig["x"] + dg * disp, "y": sig["y"], "xT": xT, "yT": sig["yT"]})
        s2 = gf.load(None, C250, raw=df)
        vx, vy, sp = gf.velocity(s2["x"], s2["y"], C250)
        sac, _, _ = gf.detect_saccades(s2["x"], s2["y"], sp, C250)
        pf, _ = gf.pur_features(s2, sac, vx, C250)
        rows.append(dict(rec=os.path.basename(path)[:-4], dg=dg, pur_gain=pf.get("pur_gain", np.nan),
                         **{f"pur_gain_{k}": pf.get(f"pur_gain_{k}", np.nan) for k in (5, 10, 20)}))
    return rows


# ------------------------------------------------------------------ summaries
def o1_o2(d, lines):
    p = d[d.cls == "primary"]
    ms = p[(p.amp >= 5) & (p.amp <= 25)].copy()
    a, b = lines.loc[ms["pipe"]].to_numpy().T if len(ms) else (np.array([]), np.array([]))
    ms["res"] = np.log(ms.vpeak.to_numpy()) - (a + b * np.log(ms.amp.to_numpy()))
    g = ms.groupby(["pipe", "cond", "rec"]).res
    o1 = g.median().where(g.size() >= 10).rename("O1")          # frozen minimum: 10 events
    o2 = p[(p.step >= 5) & (p.step <= 25)].groupby(["pipe", "cond", "rec"]).lat_ms.median().rename("O2")
    return pd.concat([o1, o2], axis=1).reset_index()


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 1.96 * se, m + 1.96 * se, v.std(ddof=1), len(v)


def ratio_ci(vp, vr, nb=2000, seed=0):
    """Recovery ratio pipeline / REF25 on recordings with finite changes in both (paired bootstrap)."""
    c = pd.concat([vp, vr], axis=1, keys=["p", "r"]).dropna()
    if len(c) < 3 or c.r.mean() == 0:
        return np.nan, np.nan, np.nan, len(c)
    rng = np.random.default_rng(seed)
    P, R = c.p.to_numpy(), c.r.to_numpy()
    bs = [P[i].mean() / R[i].mean() for i in (rng.integers(0, len(c), len(c)) for _ in range(nb))]
    return P.mean() / R.mean(), *np.quantile(bs, [0.025, 0.975]), len(c)


def o1_refit_boot(d, conds, nb=1000, seed=0):
    """Participant bootstrap for complete-pipeline O1 recovery that refits every pipeline's base-condition
    main sequence in each replicate (resampled recordings, multiplicity-weighted, 1/n per recording), then
    recomputes recording O1 (>= 10 events) and changes. Returns per (pipe, cond, subset) replicate means and
    ratios to REF25 on recordings with finite changes in both pipelines."""
    e = d[(d.cls == "primary") & (d.amp >= 5) & (d.amp <= 25) & d.cond.isin(["base"] + conds)]
    recs = np.array(sorted(d.rec.unique()))
    rid = {r: i for i, r in enumerate(recs)}
    G = {}
    for (p, c, r), g in e.groupby(["pipe", "cond", "rec"]):
        G.setdefault((p, c), {})[rid[r]] = (np.log(g.amp.to_numpy()), np.log(g.vpeak.to_numpy()))
    pipes = sorted(e["pipe"].unique())
    att = {p: d[d["pipe"] == p].groupby("rec").calib_ok.first().reindex(recs).fillna(False).to_numpy(bool)
           for p in pipes}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(nb):
        cnt = np.bincount(rng.integers(0, len(recs), len(recs)), minlength=len(recs))
        sel = np.nonzero(cnt)[0]
        w = cnt[sel].astype(float)
        chg = {}
        for p in pipes:
            base = G.get((p, "base"), {})
            ii = [i for i in sel if i in base]
            if not ii:                                   # no eligible base events in this draw
                for c in conds:
                    chg[(p, c)] = np.full(len(sel), np.nan)
                continue
            la = np.concatenate([base[i][0] for i in ii])
            lv = np.concatenate([base[i][1] for i in ii])
            ww = np.concatenate([np.full(len(base[i][0]), cnt[i] / len(base[i][0])) for i in ii])
            bb, aa = np.polyfit(la, lv, 1, w=np.sqrt(ww))

            def o1(dct, i):
                if i not in dct or len(dct[i][0]) < 10:
                    return np.nan
                return np.median(dct[i][1] - aa - bb * dct[i][0])
            o1b = np.array([o1(base, i) for i in sel])
            for c in conds:
                dc = G.get((p, c), {})
                chg[(p, c)] = np.array([o1(dc, i) for i in sel]) - o1b
        for p in pipes:
            for c in conds:
                for subset in (["all", "attainable"] if p in ("P250_white", "P250_ar1") else ["all"]):
                    v = chg[(p, c)]
                    ok = np.isfinite(v) & (att[p][sel] if subset == "attainable" else True)
                    row = dict(b=b, pipe=p, cond=c, subset=subset,
                               mean=np.sum(w[ok] * v[ok]) / np.sum(w[ok]) if ok.any() else np.nan, ratio=np.nan)
                    if p != "REF25":
                        r = chg[("REF25", c)]
                        ok2 = ok & np.isfinite(r)
                        den = np.sum(w[ok2] * r[ok2])
                        row["ratio"] = np.sum(w[ok2] * v[ok2]) / den if ok2.any() and den != 0 else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def summarise(d, out_dir):
    base = d[(d.cond == "base") & (d.cls == "primary") & (d.amp >= 5) & (d.amp <= 25)]
    lines = {}
    for p, g in base.groupby("pipe"):
        w = 1.0 / g.groupby("rec").rec.transform("size")
        b, a = np.polyfit(np.log(g.amp), np.log(g.vpeak), 1, w=np.sqrt(w))
        lines[p] = (a, b)
    lines = pd.DataFrame(lines, index=["a", "b"]).T
    rec = o1_o2(d, lines)
    calib_ok = d.groupby(["pipe", "rec"]).calib_ok.first()     # attainable and calibration produced a value

    # (0) precision calibration: target vs achieved (base condition)
    pb = d[d.cond == "base"].groupby(["pipe", "rec"]).agg(prec=("prec", "first"), p_target=("p_target", "first"),
                                                          attainable=("attainable", "first")).reset_index()
    pb["att"] = pb.attainable.astype(bool) & pb["pipe"].isin(["P250_white", "P250_ar1"])
    pb["rel_err"] = (pb.prec / pb.p_target - 1).where(pb.att)
    pb["hit"] = pb.rel_err.abs() < .02                  # NaN achieved precision counts as a failure
    pb["nan_ach"] = pb.att & pb.prec.isna()
    calib = pb.groupby("pipe").agg(n=("rec", "size"), share_attainable=("attainable", "mean"),
                                   target_median=("p_target", "median"), achieved_median=("prec", "median"),
                                   n_attainable=("att", "sum"), n_nan_achieved=("nan_ach", "sum"),
                                   relerr_median_attainable=("rel_err", "median"),
                                   n_within2pct=("hit", "sum")
                                   ).reset_index()
    calib["within2pct_attainable"] = calib.n_within2pct / calib.n_attainable.where(calib.n_attainable > 0)

    # (1) agreement with a reference pipeline (base condition). Denominator = every reference jump
    #     (or reference primary); events absent or mistimed (> 5 ms) in the pipeline count as failures.
    agree = []
    for refp in ("REF25", "REF7"):
        ref = d[(d.cond == "base") & (d["pipe"] == refp)].set_index(["rec", "jump"])
        for p in sorted(d["pipe"].unique()):
            if p == refp:
                continue
            q = d[(d.cond == "base") & (d["pipe"] == p)].set_index(["rec", "jump"])
            j = ref.join(q, lsuffix="_ref", rsuffix="_p", how="left")
            ok_t = (j.t_abs_ref - j.t_abs_p).abs() < 0.005
            rp = j.cls_ref == "primary"
            mm = rp & ok_t & (j.cls_p == "primary")
            m = j[mm]
            lv = np.log(m.vpeak_p / m.vpeak_ref)
            dl = m.lat_ms_p - m.lat_ms_ref
            per = pd.DataFrame({"lv": lv, "dl": dl}).groupby(level=0).mean()
            agree.append(dict(reference=refp, pipe=p, n_ref_jumps=len(j), n_ref_primary=int(rp.sum()),
                              recovered=float(mm.sum() / max(rp.sum(), 1)),
                              cls_agreement=float(((j.cls_ref == j.cls_p) & ok_t).mean()),
                              logv_bias_median=float(lv.median()), logv_bias_mean=float(per.lv.mean()),
                              logv_diff_sd_events=float(lv.std()), lat_bias_median=float(dl.median()),
                              lat_bias_mean=float(per.dl.mean()), lat_diff_sd_events=float(dl.std()),
                              amp_ratio_median=float((m.amp_p / m.amp_ref).median()),
                              prec_median=float(d[(d["pipe"] == p) & (d.cond == "base")]
                                                .groupby("rec").prec.first().median())))
    agree = pd.DataFrame(agree)

    # (2) injected-effect recovery, complete pipeline (recording outcomes; O1 needs >= 10 events).
    #     ratio_ref25 is computed on recordings with finite changes in both pipelines.
    nominal = {f"vel{v}": ("O1", np.log(v)) for v in VEL} | {f"shift{s:+d}": ("O2", float(s)) for s in SHIFT} | \
              {"zero": ("both", 0.0)}
    rb = rec[rec.cond == "base"].set_index(["pipe", "rec"])
    recov = []
    for cond, (o, nom) in nominal.items():
        ri = rec[rec.cond == cond].set_index(["pipe", "rec"])
        dd = (ri[["O1", "O2"]] - rb[["O1", "O2"]].reindex(ri.index))
        pipes = dd.index.get_level_values(0).unique()
        for p in sorted(pipes):
            for subset in (["all", "attainable"] if p in ("P250_white", "P250_ar1") else ["all"]):
                for oo in (["O1", "O2"] if o == "both" else [o]):
                    v = dd.loc[p][oo]
                    if subset == "attainable":
                        v = v[calib_ok.loc[p].reindex(v.index).fillna(False).to_numpy(bool)]
                    m, lo, hi, sd, n = ci(v)
                    row = dict(cond=cond, outcome=oo, pipe=p, subset=subset, nominal=nom, recovered=m,
                               ci_lo=lo, ci_hi=hi, sd_rec=sd, n=n, ratio_nominal=m / nom if nom else np.nan)
                    if nom and p != "REF25" and "REF25" in pipes:
                        r, rlo, rhi, rn = ratio_ci(v, dd.loc["REF25"][oo])
                        row.update(ratio_ref25=r, ratio_lo=rlo, ratio_hi=rhi, n_paired=rn)
                    recov.append(row)
    recov = pd.DataFrame(recov)
    # O1 rows: replace analytic CIs with the refit bootstrap (frozen measurement rule 5)
    bo = o1_refit_boot(d, [c for c, (o, _) in nominal.items() if o in ("O1", "both")])
    q = bo.groupby(["pipe", "cond", "subset"]).agg(
        boot_lo=("mean", lambda v: v.quantile(.025)), boot_hi=("mean", lambda v: v.quantile(.975)),
        rboot_lo=("ratio", lambda v: v.quantile(.025)), rboot_hi=("ratio", lambda v: v.quantile(.975))).reset_index()
    recov = recov.merge(q, on=["pipe", "cond", "subset"], how="left")
    k = recov.outcome == "O1"
    recov.loc[k, "ci_lo"], recov.loc[k, "ci_hi"] = recov.loc[k, "boot_lo"], recov.loc[k, "boot_hi"]
    recov.loc[k, "ratio_lo"], recov.loc[k, "ratio_hi"] = recov.loc[k, "rboot_lo"], recov.loc[k, "rboot_hi"]
    recov["ci_method"] = np.where(k, "refit participant bootstrap (1000)", "normal, recordings")
    recov = recov.drop(columns=["boot_lo", "boot_hi", "rboot_lo", "rboot_hi"])

    # (3) matched-event recovery. Universe = REF25 base-condition primary events in the eligible range
    #     whose saccade was actually modified; events not primary in both base and injected condition of
    #     the pipeline count as unmatched.
    mrec = []
    ref0 = d[(d.cond == "base") & (d["pipe"] == "REF25") & (d.cls == "primary")].set_index(["rec", "jump"])
    for cond, (o, nom) in nominal.items():
        if cond == "zero":
            continue
        ref0e = ref0[(ref0.amp >= 5) & (ref0.amp <= 25)] if o == "O1" else ref0[(ref0.step >= 5) & (ref0.step <= 25)]
        injset = d[(d.cond == cond) & d.inj.astype(bool)].set_index(["rec", "jump"]).index.unique()
        refu = ref0e.index[ref0e.index.isin(injset)]
        for p in sorted(d["pipe"].unique()):
            b0 = d[(d.cond == "base") & (d["pipe"] == p) & (d.cls == "primary")].set_index(["rec", "jump"])
            b1 = d[(d.cond == cond) & (d["pipe"] == p) & (d.cls == "primary")].set_index(["rec", "jump"])
            j = b0.join(b1, lsuffix="_0", rsuffix="_1", how="inner")
            j = j[j.index.isin(refu)]
            v = np.log(j.vpeak_1 / j.vpeak_0) if o == "O1" else j.lat_ms_1 - j.lat_ms_0
            per = v.groupby(level=0).mean()
            m, lo, hi, sd, n = ci(per)
            mrec.append(dict(cond=cond, outcome=o, pipe=p, nominal=nom, ref_eligible=len(ref0e),
                             injected_share=float(len(refu) / max(len(ref0e), 1)), injected_events=len(refu),
                             matched_events=len(j), match_rate=len(j) / max(len(refu), 1),
                             recovered=m, ci_lo=lo, ci_hi=hi, ratio_nominal=m / nom))
    mrec = pd.DataFrame(mrec)
    for n_, t in [("B5_agreement", agree), ("B5_recovery", recov), ("B5_matched", mrec),
                  ("B5_recording_outcomes", rec), ("B5_lines", lines.reset_index()), ("B5_precision", calib)]:
        t.to_csv(os.path.join(out_dir, f"{n_}.csv"), index=False)
    return agree, recov, mrec, lines, calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gb_dir")
    ap.add_argument("vr_dir")
    ap.add_argument("vr_primary")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=322)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--npur", type=int, default=100000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    prim = pd.read_parquet(a.vr_primary)
    vr_prec = prim[(prim["round"] == 1) & prim.ran_prec.notna()].ran_prec.to_numpy()
    files = sorted(glob.glob(os.path.join(a.gb_dir, "S_1*_S1_RAN.csv")))[:a.n]
    res = Parallel(n_jobs=a.jobs)(delayed(one_recording)(f, vr_prec) for f in files)
    d = pd.concat(res, ignore_index=True)
    d.to_parquet(os.path.join(a.out_dir, "B5_trials.parquet"))
    agree, recov, mrec, lines, calib = summarise(d, a.out_dir)

    pfiles = sorted(glob.glob(os.path.join(a.vr_dir, "S_1*_S1_*_PUR.csv")))[:a.npur]
    pr = pd.DataFrame([r for rows in Parallel(n_jobs=a.jobs)(delayed(pur_one)(f) for f in pfiles) for r in rows])
    pr.to_csv(os.path.join(a.out_dir, "B5_pur.csv"), index=False)
    pb = pr[pr.dg == 0].set_index("rec")
    prow = []
    for dg in sorted(pr.dg.unique()):
        di = pr[pr.dg == dg].set_index("rec")
        for c in ("pur_gain", "pur_gain_5", "pur_gain_10", "pur_gain_20"):
            m, lo, hi, sd, n = ci(di[c] - pb[c].reindex(di.index))
            prow.append(dict(dg=dg, measure=c, recovered=m, ci_lo=lo, ci_hi=hi, sd_rec=sd, n=n))
    prow = pd.DataFrame(prow)
    prow.to_csv(os.path.join(a.out_dir, "B5_pur_summary.csv"), index=False)

    with open(os.path.join(a.out_dir, "B5_summary.md"), "w") as fh:
        fh.write(f"# B5 conditional measurement sensitivity (GazeBase RAN S1, n={len(files)} recordings)\n\n")
        fh.write("VR precision target drawn per recording from GazeBaseVR round-1 RAN precision "
                 f"(median {np.median(vr_prec):.3f} deg).\n\n")
        fh.write("Draws at or below a recording's own 250 Hz precision are unattainable (no noise added, flagged); "
                 "noise is otherwise calibrated so the complete pipeline's achieved precision matches the draw.\n\n")
        fh.write("## Precision: target vs achieved (base condition)\n\n" + calib.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Main-sequence lines (base condition, per pipeline)\n\n" + lines.round(4).to_markdown() + "\n\n")
        fh.write("## Agreement with reference pipelines (base, jump-matched)\n\n" + agree.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Injected-effect recovery, complete pipeline (recording outcomes)\n\n"
                 + recov.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Injected-effect recovery, matched events\n\n" + mrec.round(4).to_markdown(index=False) + "\n\n")
        fh.write(f"## PUR gain injection (GazeBaseVR round 1 session 1, n={len(pfiles)})\n\n"
                 + prow.round(4).to_markdown(index=False) + "\n")
    print(open(os.path.join(a.out_dir, "B5_summary.md")).read())


if __name__ == "__main__":
    main()
