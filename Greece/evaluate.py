"""Evaluate the three SIGMOID Greece classifiers at decision threshold 0.50.

Run: python evaluate.py
Requires this folder's test_dataset.csv, test_labels.csv, model.py, models/*.pth.
Outputs metrics.csv (summary) and predictions.csv (per-encounter evidence).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from model import FEATURE_COLUMNS, MythicThreatClassifier

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.50
NAMES = ("ares", "athena", "apollo")


def main():
    encounters = pd.read_csv(ROOT / "test_dataset.csv")
    labels = pd.read_csv(ROOT / "test_labels.csv")

    assert encounters["encounter_id"].is_unique, "Duplicate encounter IDs in features"
    assert labels["encounter_id"].is_unique, "Duplicate encounter IDs in labels"
    assert set(encounters["encounter_id"]) == set(labels["encounter_id"]), "Encounter IDs differ"
    assert encounters[FEATURE_COLUMNS].notna().all().all(), "Missing numeric values"

    # Align ground truth by encounter_id, NOT by possibly different CSV row order.
    target = labels.set_index("encounter_id").loc[
        encounters["encounter_id"], "is_catastrophic_threat"
    ].to_numpy(dtype=np.int64)
    features = torch.from_numpy(encounters[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    output = pd.DataFrame({
        "encounter_id": encounters["encounter_id"],
        "true_label": target,
    })
    metrics = []
    for name in NAMES:
        model = MythicThreatClassifier()
        state = torch.load(
            ROOT / "models" / f"model_{name}.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        model.eval()

        with torch.no_grad():
            probs = torch.sigmoid(model(features)).cpu().numpy()
        pred = (probs >= THRESHOLD).astype(np.int64)

        tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
        metrics.append({
            "model": name,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "accuracy": accuracy_score(target, pred),
            "precision": precision_score(target, pred, zero_division=0),
            "recall": recall_score(target, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "F1": f1_score(target, pred, zero_division=0),
            "balanced_accuracy": 0.5 * (
                tp / (tp + fn) + tn / (tn + fp)
            ),
        })
        output[f"{name}_probability"] = probs
        output[f"{name}_predicted_label"] = pred

    summary = pd.DataFrame(metrics)
    summary.to_csv(ROOT / "metrics.csv", index=False, float_format="%.10g")
    output.to_csv(ROOT / "predictions.csv", index=False, float_format="%.9g")

    print(f"Evaluated {len(target)} rows; {int(target.sum())} catastrophic threats.")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    chosen = summary.sort_values(
        by=["FN", "recall"], ascending=[True, False]
    ).iloc[0]["model"]
    print(f"Flag: SIGMOID_{chosen.upper()}")

    # Sanity checks for the supplied test inputs; fail if files differ.
    assert len(target) == 1000 and int(target.sum()) == 100
    assert summary["accuracy"].round(10).eq(0.95).all()
    assert dict(zip(summary.model, summary.FN)) == {
        "ares": 5, "athena": 25, "apollo": 45
    }


if __name__ == "__main__":
    main()
