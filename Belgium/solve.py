"""SIGMOID Belgium — Fry-ndship: recover the SUPERVISION word.

Place potato_fields.csv in this folder, run: python solve.py
Outputs: fry_ndship.png and fry_ndship.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "potato_fields.csv")
d["w"] = d.area_ha * 1e4 / d.long_side_m
d["per"] = 2 * (d.long_side_m + d.w)

# Fit edge length using field perimeter, ditch length, and hedge length.
X = np.c_[np.ones(len(d)), d.per, d.ditch_m, d.hedge_m]
coef = np.linalg.lstsq(X, d.edge_m, rcond=None)[0]
pred = X @ coef
r2 = 1 - np.sum((d.edge_m - pred) ** 2) / np.sum((d.edge_m - d.edge_m.mean()) ** 2)
print("Coefficients [intercept, perimeter, ditch, hedge]:", coef)
print(f"R² = {r2:.10f}")

# Remove the geometric baseline in hedge_m; its residual scatter writes the word.
slope, icpt = np.polyfit(d.per, d.hedge_m, 1)
d["resid"] = d.hedge_m - (slope * d.per + icpt)
print("Hedge slope and intercept:", slope, icpt)

fig, ax = plt.subplots(figsize=(30, 6))
ax.scatter(d.per, d.resid, s=6)
ax.set_xlabel("Field perimeter (m)")
ax.set_ylabel("hedge_m − fitted hedge")
ax.set_title("Fry-ndship: hedge residuals spell SUPERVISION")
fig.tight_layout()
fig.savefig(ROOT / "fry_ndship.png", dpi=100)
fig.savefig(ROOT / "fry_ndship.svg")
plt.close(fig)

print("FLAG: SIGMOID_SUPERVISION")
