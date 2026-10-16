"""Table 2 source: attenuation bound from the participant-bootstrap upper limits in A4_v4/A4_std.csv
(the limits given in the text) and the split-half R_x values in REV1/A4_atten_bound.csv."""
import numpy as np, pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
std = pd.read_csv(RES / "A4_v4/A4_std.csv")
r_hi = std[std.rating == "eye"].set_index("feature").boot_hi
rx = pd.read_csv(RES / "REV1/A4_atten_bound.csv")
pick = {("O1", "SB point"), ("O1", "SB lower CI"), ("O2", "SB point"), ("O2", "SB lower CI"),
        ("O3", "SB point (balanced cell split)"), ("O3", "SB lower CI (balanced cell split)")}
rows = []
for _, r in rx.iterrows():
    if (r.feature, r.Rx_source) not in pick:
        continue
    row = dict(feature=r.feature, Rx_source=r.Rx_source, Rx=r.Rx, r_hi_boot=r_hi[r.feature])
    for ry in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        row[f"Ry={ry}"] = min(1.0, r_hi[r.feature] / np.sqrt(r.Rx * ry))
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(RES / "REV1/A4_table2.csv", index=False)
print(out.round(2).to_string(index=False))
