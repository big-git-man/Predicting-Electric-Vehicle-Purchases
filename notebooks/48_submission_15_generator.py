import json
from pathlib import Path

root = Path(r"C:\Users\aakif\Documents\DataCompetition")
submission14 = root / "submissions" / "submission_14.csv"
external_root = root / "external_data" / "s6e9_zoom_zoom_baseline"
output = root / "submissions" / "submission_15.csv"

# Exact public 0.94657 prediction from the archive
external_submission = external_root / "submission_latest_best.csv"

import pandas as pd

s14 = pd.read_csv(submission14)
ext = pd.read_csv(external_submission)

if len(s14) != len(ext):
    raise ValueError(f"Row count mismatch: S14={len(s14)}, external={len(ext)}")

if not s14["id"].equals(ext["id"]):
    ext = ext.set_index("id").loc[s14["id"]].reset_index()

# Submission 15:
# 85% of our own Submission 14
# 15% of the archived public 0.94657 prediction
blend = (
    0.85 * s14["Will_Buy_EV"].to_numpy()
    + 0.15 * ext["Will_Buy_EV"].to_numpy()
)

out = pd.DataFrame({
    "id": s14["id"],
    "Will_Buy_EV": blend
})

if out["Will_Buy_EV"].isna().any():
    raise ValueError("NaNs detected in submission.")

out.to_csv(output, index=False)

print(f"Saved: {output}")
print(f"Rows: {len(out):,}")
print(f"Prediction min: {out['Will_Buy_EV'].min():.10f}")
print(f"Prediction max: {out['Will_Buy_EV'].max():.10f}")
print("Blend: 85% Submission 14 + 15% archived 0.94657 prediction")
