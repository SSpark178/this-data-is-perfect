"""Run an offline local check against the provided training labels.

Download train-input.csv and train-output.csv from the organizer first.
This is a *practice* score, not the hidden evaluation result.
"""
import numpy as np
import pandas as pd

from detector import Detector

X = pd.read_csv("train-input.csv")
labels = pd.read_csv("train-output.csv")
assert len(X) == len(labels)
assert (X["timestamp"].values == labels["timestamp"].values).all()

detector = Detector()
pred = np.array([detector.update(reading) >= 0.50 for reading in X.to_dict("records")])
actual = labels["is_anomaly"].to_numpy().astype(bool)

tp = int(np.sum(pred & actual))
fp = int(np.sum(pred & ~actual))
fn = int(np.sum(~pred & actual))
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
recall = {}
for name in ("point", "drift", "collective"):
    m = (labels["anomaly_type"] == name).to_numpy()
    recall[name] = float(pred[m].mean()) if m.any() else 0

score = 100 * (
    0.80 * f1
    + 0.05 * recall["point"]
    + 0.075 * recall["drift"]
    + 0.075 * recall["collective"]
)
score = max(1, min(100, score))

print("rows:", len(X), "TP:", tp, "FP:", fp, "FN:", fn)
print(f"F1: {f1:.6f}")
for name in ("point", "drift", "collective"):
    print(f"{name} recall: {recall[name]:.6f}")
print(f"Practice score: {score:.6f}/100")
