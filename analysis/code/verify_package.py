from pathlib import Path
import ast, csv, hashlib, json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text())
for item in manifest["files"]:
    p = root / item["path"]
    assert p.is_file() and p.stat().st_size == item["bytes"], item["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"], item["path"]
    if p.suffix == ".py": ast.parse(p.read_text())
    elif p.suffix == ".csv":
        with p.open(newline="") as f: list(csv.reader(f))
    elif p.suffix == ".json": json.loads(p.read_text())
    elif p.suffix == ".parquet": pd.read_parquet(p)
    elif p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            for key in z.files: z[key]
print(f"PASS: {len(manifest['files'])} manifest entries and applicable file-format/syntax checks. No study analysis or training was run.")
