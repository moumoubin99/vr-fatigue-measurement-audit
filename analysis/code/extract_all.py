"""Run gazefeat.process over all GazeBaseVR recordings in parallel.

Usage: python extract_all.py DATA_DIR OUT_DIR [--limit N] [--jobs J] [--detector fixed] [--geometry sphere] [--tasks RAN,PUR]
Writes OUT_DIR/features.parquet, OUT_DIR/ran_trials.parquet, OUT_DIR/errors.csv
"""
import argparse
import glob
import os
import traceback

import pandas as pd
from joblib import Parallel, delayed

import gazefeat


def safe(path, cfg):
    try:
        f, ps = gazefeat.process(path, cfg)
        return f, ps, None
    except Exception:
        return None, None, (os.path.basename(path), traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--detector", default="adaptive", help="adaptive | fixed (I-VT sensitivity)")
    ap.add_argument("--geometry", default="planar", help="planar | sphere")
    ap.add_argument("--tasks", default="", help="comma list, e.g. RAN,PUR")
    a = ap.parse_args()
    cfg = gazefeat.replace(gazefeat.DEFAULT, detector=a.detector, geometry=a.geometry)
    files = sorted(glob.glob(os.path.join(a.data_dir, "S_*.csv")))
    if a.tasks:
        files = [f for f in files if f[-7:-4] in a.tasks.split(",")]
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(safe)(p, cfg) for p in files)
    feats = [r[0] for r in res if r[0] is not None]
    sacs = [r[1] for r in res if r[1] is not None and len(r[1])]
    errs = [r[2] for r in res if r[2] is not None]
    pd.DataFrame(feats).to_parquet(os.path.join(a.out_dir, "features.parquet"))
    if sacs:
        pd.concat(sacs, ignore_index=True).to_parquet(os.path.join(a.out_dir, "ran_trials.parquet"))
    pd.DataFrame(errs, columns=["file", "trace"]).to_csv(os.path.join(a.out_dir, "errors.csv"), index=False)
    print(f"files={len(files)} ok={len(feats)} errors={len(errs)}")


if __name__ == "__main__":
    main()
