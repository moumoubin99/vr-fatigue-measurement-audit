"""Compare a reproduction run with the frozen outputs, file by file.

For every .csv, .parquet and .npz file under FROZEN_DIR, finds the file at the same relative path under NEW_DIR and
records: missing, identical, numeric match within tolerance, or different (with the largest absolute and relative
difference and the column where it occurs). Rows are compared in file order; if the shapes match but values differ,
the comparison is repeated after sorting both tables on their non-float columns.

Usage: python repro_compare.py FROZEN_DIR NEW_DIR OUT_CSV [--rtol 1e-6] [--atol 1e-9] [--skip PATTERN ...]
"""
import argparse
import fnmatch
import os

import numpy as np
import pandas as pd


def load(p):
    if p.endswith(".csv"):
        return pd.read_csv(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def cmp_arrays(a, b, rtol, atol):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return "shape", np.nan, np.nan
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        a, b = a.astype(float), b.astype(float)
        na, nb = np.isnan(a), np.isnan(b)
        if (na != nb).any():
            return "nan_pattern", np.nan, np.nan
        m = ~na
        if not m.any():
            return "identical", 0.0, 0.0
        d = np.abs(a[m] - b[m])
        rel = d / np.maximum(np.abs(a[m]), 1e-12)
        mad, mrd = float(d.max()), float(rel[d > atol].max()) if (d > atol).any() else 0.0
        if mad == 0:
            return "identical", 0.0, 0.0
        return ("close" if np.allclose(a[m], b[m], rtol=rtol, atol=atol) else "differs"), mad, mrd
    sa, sb = pd.Series(a.ravel(), dtype=object), pd.Series(b.ravel(), dtype=object)
    eq = sa.where(sa.notna(), "<NA>").astype(str).values == sb.where(sb.notna(), "<NA>").astype(str).values
    return ("identical" if eq.all() else "differs"), np.nan, np.nan


def cmp_frames(f, n, rtol, atol):
    cols_missing = sorted(set(f.columns) - set(n.columns))
    cols_extra = sorted(set(n.columns) - set(f.columns))
    common = [c for c in f.columns if c in n.columns]
    if len(f) != len(n):
        return dict(status="rows", cols_missing=";".join(cols_missing), cols_extra=";".join(cols_extra))

    def run(f, n):
        worst, out = "identical", dict(max_abs=0.0, max_rel=0.0, worst_col="")
        rank = {"identical": 0, "close": 1, "differs": 2, "nan_pattern": 3, "shape": 3}
        for c in common:
            s, mad, mrd = cmp_arrays(f[c].values, n[c].values, rtol, atol)
            if rank[s] > rank[worst]:
                worst = s
            if not np.isnan(mad) and mad > out["max_abs"]:
                out.update(max_abs=mad, max_rel=mrd, worst_col=c)
            elif s in ("differs", "nan_pattern") and not out["worst_col"]:
                out["worst_col"] = c
        return worst, out

    st, out = run(f.reset_index(drop=True), n.reset_index(drop=True))
    if st in ("differs", "nan_pattern"):
        keys = [c for c in common if f[c].dtype.kind not in "f"]
        if keys:
            try:
                st2, out2 = run(f.sort_values(keys).reset_index(drop=True), n.sort_values(keys).reset_index(drop=True))
                if st2 in ("identical", "close"):
                    st, out = st2 + "_sorted", out2
            except TypeError:
                pass
    if cols_missing or cols_extra:
        st += "+cols"
    return dict(status=st, cols_missing=";".join(cols_missing), cols_extra=";".join(cols_extra), **out)


def main(frozen, new, out_csv, rtol, atol, skip):
    rows = []
    for root, _, files in os.walk(frozen):
        for fn in sorted(files):
            if not fn.endswith((".csv", ".parquet", ".npz")):
                continue
            rel = os.path.relpath(os.path.join(root, fn), frozen)
            if any(fnmatch.fnmatch(rel, p) for p in skip):
                continue
            q = os.path.join(new, rel)
            r = dict(file=rel)
            if not os.path.exists(q):
                rows.append(dict(r, status="missing"))
                continue
            try:
                a, b = load(os.path.join(root, fn)), load(q)
                if isinstance(a, dict):
                    res = {}
                    for k in a:
                        s, mad, mrd = cmp_arrays(a[k], b[k], rtol, atol) if k in b else ("missing_key", np.nan, np.nan)
                        res[k] = s
                    st = "identical" if all(v == "identical" for v in res.values()) else \
                        "close" if all(v in ("identical", "close") for v in res.values()) else \
                        "differs:" + ",".join(k for k, v in res.items() if v not in ("identical", "close"))
                    rows.append(dict(r, status=st, n_rows_frozen=len(next(iter(a.values()))),
                                     n_rows_new=len(next(iter(b.values())))))
                else:
                    rows.append(dict(r, n_rows_frozen=len(a), n_rows_new=len(b), **cmp_frames(a, b, rtol, atol)))
            except Exception as e:  # unreadable or incompatible file
                rows.append(dict(r, status="error", note=str(e)[:200]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.status.str.split(":").str[0].value_counts().to_string())
    bad = df[~df.status.isin(["identical", "close", "identical_sorted", "close_sorted"])]
    if len(bad):
        print(bad.to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen")
    ap.add_argument("new")
    ap.add_argument("out_csv")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    main(a.frozen, a.new, a.out_csv, a.rtol, a.atol, a.skip)
