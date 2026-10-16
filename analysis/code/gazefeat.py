"""Automatic per-recording feature extraction for GazeBaseVR (and GazeBase for B5).

Frozen measurement protocol: refine-logs/FINAL_PROPOSAL.md, section "Measurement protocol".
No step uses manual input. All thresholds are fixed constants below or fields of Config.
"""
import os
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FNAME_RE = re.compile(r"S_(\d)(\d+)_S(\d)_(\d)_([A-Z]{3})\.csv")


@dataclass(frozen=True)
class Config:
    fs: float = 250.0
    sg_win: int = 7               # samples; 7 @250 Hz = 24 ms
    sg_ord: int = 2
    max_gap_ms: float = 12.0      # gaps longer than this are not interpolated
    pos_limit: float = 60.0       # |x| or |y| beyond this (deg) is invalid
    detector: str = "adaptive"    # "adaptive" | "fixed"
    pt_start: float = 100.0       # adaptive: initial and maximum peak threshold (deg/s)
    pt_tol: float = 1.0
    pt_maxit: int = 100
    fixed_onset: float = 30.0     # fixed I-VT variant
    fixed_peak: float = 75.0
    min_sac_ms: float = 12.0
    min_isi_ms: float = 60.0
    vmax: float = 1000.0
    geometry: str = "planar"      # "planar" | "sphere"
    ran_win: tuple = (0.080, 0.600)
    ran_dir_deg: float = 45.0
    ran_min_amp: float = 1.0      # noise floor for a response saccade
    pur_settle: float = 0.400     # steady-state starts 400 ms after motion onset
    pur_tail: float = 0.100       # and stops 100 ms before motion offset
    pur_min_cell_s: float = 1.0
    sac_margin_ms: float = 20.0

    @property
    def dt(self):
        return 1.0 / self.fs


DEFAULT = Config()


def parse_name(path):
    m = FNAME_RE.search(os.path.basename(path))
    rnd, sub, sess, order, task = m.groups()
    return dict(round=int(rnd), pid=int(sub), session=int(sess), task=task)


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------
def _interp_valid(t, v, tg, valid, max_gap):
    """Linear interpolation onto grid tg. A grid point is NaN unless both bracketing
    original samples are valid and no more than max_gap seconds apart."""
    out = np.interp(tg, t, np.where(valid, v, 0.0))
    i = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 2)
    ok = valid[i] & valid[i + 1] & ((t[i + 1] - t[i]) <= max_gap + 1e-9)
    exact = (np.abs(t[i] - tg) < 1e-9) & valid[i]
    out[~(ok | exact)] = np.nan
    return out


def read_raw(path, fmt="vr"):
    d = pd.read_csv(path)
    # duplicate timestamps: keep the first row (deterministic)
    d = d.drop_duplicates("n", keep="first").sort_values("n", kind="stable").reset_index(drop=True)
    if fmt == "gb":  # GazeBase (EyeLink 1000): n, x, y, val, xT, yT ; val==0 valid
        d.loc[d["val"] != 0, ["x", "y"]] = np.nan
    return d


def load(path, cfg=DEFAULT, fmt="vr", raw=None):
    d = read_raw(path, fmt) if raw is None else raw
    t = d["n"].to_numpy(float) / 1000.0
    tg = np.arange(t[0], t[-1] + 1e-9, cfg.dt)   # grid anchored at recording start
    xy = d[["x", "y"]].to_numpy(float)
    valid = np.isfinite(xy).all(1) & (np.abs(xy) < cfg.pos_limit).all(1)
    gap = cfg.max_gap_ms / 1000.0
    out = {"t": tg - tg[0], "t0": tg[0], "raw_t": t - tg[0], "raw_x": xy[:, 0], "raw_y": xy[:, 1],
           "raw_valid": valid}
    for c in ["x", "y", "lx", "ly", "rx", "ry"]:
        if c not in d:
            out[c] = None
            continue
        v = d[c].to_numpy(float)
        out[c] = _interp_valid(t, v, tg, valid & np.isfinite(v), gap)
    # target channels: zero-order hold (never interpolated across a jump)
    idx = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)
    for c in ["xT", "yT", "zT"]:
        if c in d and np.isfinite(d[c].to_numpy(float)).any():
            out[c] = d[c].to_numpy(float)[idx]
            out[c + "_raw"] = d[c].to_numpy(float)
        else:
            out[c] = None
    return out


def _sg(v, cfg):
    return savgol_filter(v, cfg.sg_win, cfg.sg_ord, deriv=1, delta=cfg.dt, mode="interp")


def velocity(x, y, cfg=DEFAULT):
    """Direct first-derivative Savitzky-Golay filter (order 2). Samples whose filter
    support touches missing data are set to NaN. Returns (vx, vy, speed) in deg/s."""
    nan = ~(np.isfinite(x) & np.isfinite(y))
    if nan.all():
        z = np.full_like(x, np.nan)
        return z, z, z
    idx = np.arange(len(x))
    xf = np.interp(idx, idx[~nan], x[~nan])
    yf = np.interp(idx, idx[~nan], y[~nan])
    if cfg.geometry == "sphere":
        # unit gaze vectors from (horizontal, vertical) angles; angular speed
        u = np.vstack([np.tan(np.deg2rad(xf)), np.tan(np.deg2rad(yf)), np.ones_like(xf)])
        u /= np.linalg.norm(u, axis=0)
        du = np.vstack([_sg(c, cfg) for c in u])
        speed = np.rad2deg(np.linalg.norm(du, axis=0))
        vx, vy = _sg(xf, cfg), _sg(yf, cfg)
    else:
        vx, vy = _sg(xf, cfg), _sg(yf, cfg)
        speed = np.hypot(vx, vy)
    bad = np.convolve(nan.astype(float), np.ones(cfg.sg_win), mode="same") > 0
    # mode="interp": the first/last half-window are fitted on the whole first/last window
    h = cfg.sg_win // 2
    if nan[:cfg.sg_win].any():
        bad[:h] = True
    if nan[-cfg.sg_win:].any():
        bad[-h:] = True
    for a in (vx, vy, speed):
        a[bad] = np.nan
    return vx, vy, speed


def angdist(x1, y1, x2, y2, geometry="planar"):
    if geometry == "sphere":
        def u(x, y):
            v = np.array([np.tan(np.deg2rad(x)), np.tan(np.deg2rad(y)), 1.0])
            return v / np.linalg.norm(v)
        return float(np.rad2deg(np.arccos(np.clip(u(x1, y1) @ u(x2, y2), -1, 1))))
    return float(np.hypot(x2 - x1, y2 - y1))


# ----------------------------------------------------------------------------
# event detection
# ----------------------------------------------------------------------------
def adaptive_threshold(speed, cfg=DEFAULT):
    """Nystrom & Holmqvist (2010)-style data-driven threshold over the whole recording:
    PT <- min(mean + 6 SD of samples below PT, pt_start), iterated to |dPT| < pt_tol.
    Onset/offset threshold = mean + 3 SD of the final sub-threshold samples."""
    s = speed[np.isfinite(speed)]
    pt = cfg.pt_start
    for _ in range(cfg.pt_maxit):
        sub = s[s < pt]
        if len(sub) < 10:
            break
        new = min(sub.mean() + 6 * sub.std(), cfg.pt_start)
        done = abs(new - pt) < cfg.pt_tol
        pt = new
        if done:
            break
    sub = s[s < pt]
    return float(pt), float(sub.mean() + 3 * sub.std())


def _runs(mask):
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    return np.where(d == 1)[0], np.where(d == -1)[0]  # start, end (exclusive)


def detect_saccades(x, y, speed, cfg=DEFAULT):
    """Saccade = run of samples above the peak threshold, extended backwards/forwards
    while speed stays above the onset threshold and keeps falling away from the peak
    (stops at a local minimum, so post-saccadic oscillations are not absorbed).
    Overlapping candidates are merged. Events shorter than min_sac_ms, touching
    missing data, or with peak speed > vmax are discarded. Glissades are not
    classified separately."""
    if cfg.detector == "adaptive":
        pt, on = adaptive_threshold(speed, cfg)
    else:
        pt, on = cfg.fixed_peak, cfg.fixed_onset
    sp = np.where(np.isfinite(speed), speed, 0.0)
    starts, ends = _runs(sp > pt)
    sacs = []
    for s0, e0 in zip(starts, ends):
        s = s0
        while s > 0 and sp[s - 1] > on and sp[s - 1] <= sp[s]:
            s -= 1
        e = e0
        while e < len(sp) and sp[e] > on and sp[e] <= sp[e - 1]:
            e += 1
        if sacs and s <= sacs[-1][1]:
            s = sacs.pop()[0]
        sacs.append((s, e))
    min_n = int(round(cfg.min_sac_ms / 1000 * cfg.fs))
    rows = []
    for s, e in sacs:
        if e - s < min_n:
            continue
        lo, hi = max(s - 1, 0), min(e + 1, len(speed))
        if not (np.isfinite(speed[lo:hi]).all() and np.isfinite(x[[s, e - 1]]).all()):
            continue
        vpk = float(np.max(speed[s:e]))
        if vpk > cfg.vmax:
            continue
        dx, dy = x[e - 1] - x[s], y[e - 1] - y[s]
        rows.append((s, e, (e - s) / cfg.fs * 1000.0,
                     angdist(x[s], y[s], x[e - 1], y[e - 1], cfg.geometry), vpk, float(dx), float(dy)))
    sac = pd.DataFrame(rows, columns=["on", "off", "dur_ms", "amp", "vpeak", "dx", "dy"])
    return sac, pt, on


def intervals(x, y, sac, n, cfg=DEFAULT):
    """Inter-saccadic intervals (ISI) without missing data, >= min_isi_ms. In VRG/PUR/VID
    these contain pursuit and are therefore not called fixations."""
    insac = np.zeros(n, bool)
    for s, e in zip(sac["on"], sac["off"]):
        insac[s:e] = True
    good = ~insac & np.isfinite(x) & np.isfinite(y)
    starts, ends = _runs(good)
    keep = (ends - starts) >= int(round(cfg.min_isi_ms / 1000 * cfg.fs))
    fx = pd.DataFrame({"on": starts[keep], "off": ends[keep]})
    fx["dur_ms"] = (fx["off"] - fx["on"]) / cfg.fs * 1000.0
    return fx, insac


def margin_mask(sac, n, cfg):
    m = np.zeros(n, bool)
    k = int(round(cfg.sac_margin_ms / 1000 * cfg.fs))
    for s, e in zip(sac["on"], sac["off"]):
        m[max(0, s - k):e + k] = True
    return m


def precision_rms(sig, fx, stationary=None, cfg=DEFAULT):
    """RMS sample-to-sample displacement on ORIGINAL consecutive valid samples
    (nominal spacing) lying inside detected inter-saccadic intervals (and, if
    given, inside target-stationary periods)."""
    rt, rx, ry, rv = sig["raw_t"], sig["raw_x"], sig["raw_y"], sig["raw_valid"]
    inside = np.zeros(len(sig["t"]), bool)
    for s, e in zip(fx["on"], fx["off"]):
        inside[s:e] = True
    if stationary is not None:
        inside &= stationary
    gi = np.clip(np.round(rt * cfg.fs).astype(int), 0, len(inside) - 1)
    ok = rv & inside[gi]
    pair = ok[:-1] & ok[1:] & (np.abs(np.diff(rt) - cfg.dt) < 0.5 * cfg.dt)
    if pair.sum() < 50:
        return np.nan
    d2 = np.diff(rx)[pair] ** 2 + np.diff(ry)[pair] ** 2
    return float(np.sqrt(np.mean(d2)))


def binocular(sig, fx):
    """Vertical binocular discrepancy ly - ry within ISIs, both eyes valid:
    offset = |median| over all ISI samples; variability = median within-ISI SD."""
    if sig.get("ly") is None:
        return np.nan, np.nan
    v = sig["ly"] - sig["ry"]
    allv, sds = [], []
    for s, e in zip(fx["on"], fx["off"]):
        w = v[s:e]
        w = w[np.isfinite(w)]
        if len(w) >= 10:
            allv.append(w)
            sds.append(w.std())
    if not allv:
        return np.nan, np.nan
    return float(abs(np.median(np.concatenate(allv)))), float(np.median(sds))


# ----------------------------------------------------------------------------
# task features
# ----------------------------------------------------------------------------
def _ran_classify(t, x, y, sac, t_on, tj, j, t_cap, tx, ty, win, cfg, precedence=True):
    """Classify one jump with response window win = (lo, hi) seconds after the logged jump
    time tj (exact time comparisons; window capped 50 ms before the next jump).
    Returns (cls, saccade row or None, anticipation flag). With precedence=True a saccade
    >= the noise floor starting before win[0] makes the trial anticipatory (protocol default);
    with precedence=False the first candidate in the window is evaluated regardless."""
    eps = 1e-6                                     # 1 us tolerance for float time arithmetic
    rel = t_on - tj
    ant = bool(((rel >= -eps) & (rel < win[0] - eps) & (sac.amp.to_numpy() >= cfg.ran_min_amp)).any())
    t_hi = min(tj + win[1], t_cap) - eps
    pre_ok = np.isfinite(x[max(0, j - int(round(0.1 * cfg.fs))):j]).all()
    if ant and precedence:
        return "anticipatory", None, ant
    cand = sac[(rel >= win[0] - eps) & (t_on < t_hi)]
    if len(cand):
        r = cand.iloc[0]                           # first detected saccade in the window
        s, e = int(r.on), int(r.off)
        if not (pre_ok and np.isfinite(x[j:e]).all()):
            return "missing", None, ant
        gx, gy = tx - x[s], ty - y[s]              # gaze-to-new-target vector
        cosang = (r.dx * gx + r.dy * gy) / (np.hypot(r.dx, r.dy) * np.hypot(gx, gy) + 1e-12)
        ok = (r.amp >= cfg.ran_min_amp) and cosang >= np.cos(np.deg2rad(cfg.ran_dir_deg))
        return ("primary" if ok else "misdirected"), r, ant
    # no candidate: absence is only asserted if the whole response window was observed
    hi_idx = int(np.searchsorted(t, t_hi, side="left"))
    if not (pre_ok and np.isfinite(x[j:hi_idx]).all()):
        return "missing", None, ant
    late = (t_on >= t_hi) & (t_on < t_cap)
    return ("late" if late.any() else "none"), None, ant


def ran_trials(sig, sac, insac, cfg=DEFAULT):
    """One row per logged target jump. Jumps are detected from the RAW target log
    (change > 0.5 deg); latency is measured from the logged jump time, and the jump is
    mapped to the first grid sample at/after that time only for indexing."""
    t, x, y, xT, yT = sig["t"], sig["x"], sig["y"], sig["xT"], sig["yT"]
    # target log may be supplied separately (B5: full-rate log for decimated gaze)
    rxT, ryT, rt = sig["xT_raw"], sig["yT_raw"], sig.get("tgt_t", sig["raw_t"])
    jr = np.where(np.hypot(np.diff(rxT), np.diff(ryT)) > 0.5)[0] + 1
    tjs = rt[jr]
    jumps = np.clip(np.ceil(tjs * cfg.fs - 1e-6).astype(int), 0, len(t) - 1)
    t_on = t[sac["on"].to_numpy()] if len(sac) else np.array([])
    fs = cfg.fs
    rows = []
    for k, (j, tj) in enumerate(zip(jumps, tjs)):
        if j < 1:
            continue
        t_cap = (tjs[k + 1] if k + 1 < len(tjs) else t[-1] + 1.0 / fs) - 0.050
        tx, ty = rxT[jr[k]], ryT[jr[k]]
        px, py = rxT[jr[k] - 1], ryT[jr[k] - 1]
        step = angdist(px, py, tx, ty, cfg.geometry)
        rec = dict(jump=k, t_jump=tj, step=step, sdx=tx - px, sdy=ty - py, tx=tx, ty=ty)
        cls, r, ant = _ran_classify(t, x, y, sac, t_on, tj, j, t_cap, tx, ty, cfg.ran_win, cfg)
        rec.update(cls=cls, ant=ant)
        if cls == "primary":
            s, e = int(r.on), int(r.off)
            rec.update(lat_ms=(t[s] - tj) * 1000.0, amp=r.amp, vpeak=r.vpeak,
                       dur_ms=r.dur_ms, gain=r.amp / step, sx=x[s], sy=y[s],
                       ang=float(np.degrees(np.arctan2(r.dy, r.dx))),
                       land_err=angdist(x[e - 1], y[e - 1], tx, ty, cfg.geometry))
        # sensitivity: no anticipatory precedence (first candidate in 80-600 ms evaluated)
        cls_np, r_np, _ = _ran_classify(t, x, y, sac, t_on, tj, j, t_cap, tx, ty, cfg.ran_win, cfg,
                                        precedence=False)
        rec["cls_np"] = cls_np
        if cls_np == "primary":
            rec.update(lat_np_ms=(t[int(r_np.on)] - tj) * 1000.0, amp_np=r_np.amp, vpeak_np=r_np.vpeak)
        # wide-window latency sensitivity (50-800 ms), same selection and qualification rules
        cls_w, r_w, _ = _ran_classify(t, x, y, sac, t_on, tj, j, t_cap, tx, ty, (0.050, 0.800), cfg)
        if cls_w == "primary":
            rec["lat_wide_ms"] = (t[int(r_w.on)] - tj) * 1000.0
        cap = int(np.searchsorted(t, t_cap, side="left"))
        # target-tracking error proxy: 700-950 ms after jump, non-saccade, no missing margins
        a0, a1 = j + int(0.700 * fs), min(j + int(0.950 * fs), cap)
        if a1 > a0:
            m = ~insac[a0:a1] & np.isfinite(x[a0:a1])
            if m.sum() >= int(0.1 * fs):
                rec["track_err"] = float(np.median(np.hypot(x[a0:a1][m] - tx, y[a0:a1][m] - ty)))
        rows.append(rec)
    return pd.DataFrame(rows)


def ran_summary(tr, cfg=DEFAULT):
    out = {"ran_n_jumps": len(tr)}
    for c in ("primary", "anticipatory", "misdirected", "late", "none", "missing"):
        out[f"ran_p_{c}"] = float((tr.cls == c).mean()) if len(tr) else np.nan
    p = tr[tr.cls == "primary"]
    elig = p[(p.step >= 5) & (p.step <= 25)]            # target-defined eligibility (O2)
    out["ran_lat_med"] = float(elig.lat_ms.median()) if len(elig) else np.nan
    out["ran_n_lat"] = len(elig)
    out["ran_lat_wide_med"] = float(tr[(tr.step >= 5) & (tr.step <= 25)].lat_wide_ms.median()) \
        if "lat_wide_ms" in tr else np.nan
    pn = tr[(tr.cls_np == "primary") & (tr.step >= 5) & (tr.step <= 25)] if "cls_np" in tr else tr.iloc[:0]
    out["ran_lat_np_med"] = float(pn.lat_np_ms.median()) if len(pn) else np.nan
    out["ran_p_ant_flag"] = float(tr.ant.mean()) if "ant" in tr and len(tr) else np.nan
    out["ran_gain_med"] = float(elig.gain.median()) if len(elig) else np.nan
    out["ran_track_err"] = float(tr.track_err.median()) if "track_err" in tr else np.nan
    ms = p[(p.amp >= 5) & (p.amp <= 25)]
    out["ran_n_ms"] = len(ms)
    if len(ms) >= 10:                                   # per-recording main sequence (label-free)
        b, a = np.polyfit(np.log(ms.amp), np.log(ms.vpeak), 1)
        out.update(ran_ms_v10=float(np.exp(a + b * np.log(10))), ran_ms_slope=float(b))
    else:
        out.update(ran_ms_v10=np.nan, ran_ms_slope=np.nan)
    return out


def pur_motion_segments(sig, cfg=DEFAULT):
    """Motion segments from the RAW logged target trajectory: consecutive position changes
    (target updates ~every 12 ms) with no pause > 100 ms and no direction reversal.
    Onset = first observed change, end = last observed change (both conservative for the
    steady-state window). Returned as grid indices [s, e)."""
    rt, rx = sig["raw_t"], sig["xT_raw"]
    d = np.diff(rx)
    ch = np.where(np.abs(d) > 1e-6)[0] + 1
    if len(ch) == 0:
        return np.array([], int), np.array([], int)
    sg = np.sign(d[ch - 1])
    brk = np.where((np.diff(rt[ch]) > 0.100) | (np.diff(sg) != 0))[0] + 1
    n = len(sig["t"])
    starts, ends = [], []
    for q in np.split(np.arange(len(ch)), brk):
        if len(q) < 2:
            continue
        starts.append(int(np.ceil(rt[ch[q[0]]] * cfg.fs - 1e-6)))
        ends.append(min(int(np.floor(rt[ch[q[-1]]] * cfg.fs + 1e-6)) + 1, n))
    return np.array(starts, int), np.array(ends, int)


def pur_features(sig, sac, vx, cfg=DEFAULT):
    """Signed horizontal steady-state pursuit gain per (speed stage x direction) cell,
    combined with equal weights. Motion boundaries come from the logged target trajectory;
    target velocity (gain denominator) from the logged target position differentiated
    over 100 ms (target updates at ~83 Hz)."""
    fs = cfg.fs
    xT = sig["xT"]
    w = int(round(0.100 * fs)) | 1
    vT = savgol_filter(xT, w, 1, deriv=1, delta=cfg.dt)
    starts, ends = pur_motion_segments(sig, cfg)
    moving = np.zeros(len(xT), bool)
    for s, e in zip(starts, ends):
        moving[s:e] = True
    near_sac = margin_mask(sac, len(xT), cfg)
    out = {}
    cells = {}
    total_smooth = 0
    for s, e in zip(starts, ends):
        if e - s < int(0.3 * fs):
            continue
        spd = np.median(np.abs(vT[s:e]))
        stage = min((5, 10, 20), key=lambda z: abs(z - spd))
        dirn = "R" if np.median(vT[s:e]) > 0 else "L"
        a0, a1 = s + int(cfg.pur_settle * fs), e - int(cfg.pur_tail * fs)
        if a1 <= a0:
            continue
        idx = np.arange(a0, a1)
        idx = idx[~near_sac[idx] & np.isfinite(vx[idx])]
        total_smooth += len(idx)
        cells.setdefault((stage, dirn), []).append(vx[idx] / vT[idx])
    gains = {}
    for (stage, dirn), parts in cells.items():
        g = np.concatenate(parts)
        out[f"pur_n_{stage}{dirn}"] = len(g) / fs
        gains[(stage, dirn)] = float(np.median(g)) if len(g) >= cfg.pur_min_cell_s * fs else np.nan
        out[f"pur_gain_{stage}{dirn}"] = gains[(stage, dirn)]
    full = [(s_, d_) for s_ in (5, 10, 20) for d_ in "LR"]
    vals = [gains.get(c, np.nan) for c in full]
    out["pur_gain"] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    for s_ in (5, 10, 20):
        v2 = [gains.get((s_, d_), np.nan) for d_ in "LR"]
        out[f"pur_gain_{s_}"] = float(np.mean(v2)) if np.all(np.isfinite(v2)) else np.nan
    out["pur_smooth_s"] = total_smooth / fs
    mov_s = moving.sum() / fs
    on = sac["on"].to_numpy()
    out["pur_catchup_rate"] = float(moving[on].sum() / mov_s) if mov_s > 0 else np.nan
    mv = moving & np.isfinite(sig["x"])
    out["pur_track_err"] = float(np.median(np.abs(sig["x"][mv] - xT[mv]))) if mv.sum() else np.nan
    return out, ~moving


def vrg_features(sig, cfg=DEFAULT, ipd_m=0.063):
    """Vergence angle = lx - rx (deg). Demand = 2 atan(IPD/2 / zT) with a nominal 63 mm
    IPD (individual IPD unknown). For each constant-depth period, steady-state response =
    median vergence after the first 1.0 s. vrg_gain = slope of response on demand."""
    if sig.get("zT") is None or sig.get("lx") is None:
        return {}
    zT = sig["zT"]
    verg = sig["lx"] - sig["rx"]
    demand = np.rad2deg(2 * np.arctan(ipd_m / 2 / zT))
    ch = np.where(np.abs(np.diff(zT)) > 1e-4)[0] + 1
    bounds = np.concatenate([[0], ch, [len(zT)]])
    resp, dem, sds = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s0 = a + int(1.0 * cfg.fs)
        if b - s0 < int(0.5 * cfg.fs):
            continue
        v = verg[s0:b]
        v = v[np.isfinite(v)]
        if len(v) < int(0.2 * cfg.fs):
            continue
        resp.append(np.median(v))
        dem.append(demand[a])
        sds.append(np.std(v))
    out = {"vrg_n_periods": len(resp)}
    if len(resp) >= 6 and np.ptp(dem) > 0:
        slope, icpt = np.polyfit(dem, resp, 1)
        out.update(vrg_gain=float(slope), vrg_offset=float(icpt), vrg_steady_sd=float(np.median(sds)))
    else:
        out.update(vrg_gain=np.nan, vrg_offset=np.nan, vrg_steady_sd=np.nan)
    return out


def tex_features(sac, fx):
    """Reading (exploratory): saccades >= 0.5 deg classified by horizontal dominance.
    forward: dx > 0; regression: -15 < dx < 0; return sweep: dx <= -15 (panel ~51 deg).
    regression rate = regressions / (forward + regressions)."""
    out = {}
    s = sac[sac.amp >= 0.5]
    horiz = np.abs(s.dy) < np.abs(s.dx)
    fwd = s[horiz & (s.dx > 0)]
    back = s[horiz & (s.dx < 0)]
    rsw = back[back.dx <= -15]
    reg = back[back.dx > -15]
    within = len(fwd) + len(reg)
    out["tex_fwd_amp"] = float(fwd.amp.median()) if len(fwd) else np.nan
    out["tex_regr_rate"] = float(len(reg) / within) if within else np.nan
    out["tex_n_return_sweeps"] = int(len(rsw))
    out["tex_fix_dur"] = float(fx.dur_ms.median()) if len(fx) else np.nan
    return out


# ----------------------------------------------------------------------------
def process(path, cfg=DEFAULT, fmt="vr", task=None):
    meta = parse_name(path) if fmt == "vr" else {}
    task = task or meta.get("task")
    rec_id = os.path.basename(path)[:-4]
    sig = load(path, cfg, fmt)
    n = len(sig["t"])
    x, y = sig["x"], sig["y"]
    vx, vy, speed = velocity(x, y, cfg)
    sac, pt, on = detect_saccades(x, y, speed, cfg)
    fx, insac = intervals(x, y, sac, n, cfg)
    dur = n / cfg.fs
    f = dict(rec=rec_id, **meta, dur_s=dur, loss=float(np.mean(~np.isfinite(x))),
             vel_thr=pt, onset_thr=on, sac_rate=len(sac) / dur,
             sac_amp_med=float(sac.amp.median()) if len(sac) else np.nan,
             isi_dur_med=float(fx.dur_ms.median()) if len(fx) else np.nan, isi_n=len(fx))
    f["binoc_v_off"], f["binoc_v_sd"] = binocular(sig, fx)
    trials = None
    stationary = None
    if task == "RAN":
        trials = ran_trials(sig, sac, insac, cfg)
        f.update(ran_summary(trials, cfg))
        trials.insert(0, "rec", rec_id)
    elif task == "PUR":
        pf, stationary = pur_features(sig, sac, vx, cfg)
        f.update(pf)
    elif task == "VRG":
        f.update(vrg_features(sig, cfg))
    elif task == "TEX":
        f.update(tex_features(sac, fx))
    f["precision_rms"] = precision_rms(sig, fx, stationary, cfg)
    return f, trials
