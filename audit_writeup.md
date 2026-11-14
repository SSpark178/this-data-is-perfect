# This Data Is Perfect! — Fraud-detection pipeline audit

## Objective

Audit the reported **93.42% accuracy** of the Estonian e-Residency fraud classifier. Preserve the specified model: `StandardScaler()` followed by `KNeighborsClassifier(n_neighbors=1)` on six numeric features. Compare each questionable choice separately, then evaluate with no identity appearing in both training and evaluation.

## Three questionable choices and isolated experiments

| Choice | Test performed | Observed accuracy on supplied practice file | Interpretation |
|---|---|---:|---|
| Original pipeline | Scaler fitted to all rows; random 80/20 row split | **93.42%** | Reported score |
| Decoy 1: scaler fitted before splitting | Move the scaler inside a `Pipeline` and fit it only on training rows | **93.67%** | Small change; not the source of the exaggerated score |
| Decoy 2: random non-stratified split | Add `stratify=y` to the original row split | **94.18%** | Small change; not the source of the exaggerated score |
| Real issue: correlated records from the same identity cross the split | Split by identity via `GroupShuffleSplit` | **80.21%** | Large drop when same-identity records cannot leak across train/test |

## Why the original evaluation is misleading

The practice file has **1,971 application records**, representing **900 distinct `(date_of_birth, citizenship)` combinations**. In this dataset, multiple records from one such combination share a target label. A random row split can put records from the same group in both train and test, which is a problem when the objective is to estimate performance on new, unseen identities.

We use the combined `date_of_birth` and `citizenship` fields **only as an audit grouping key**, not as training features. This grouping is a proxy for identity based on the available dataset, not a claim that this combination uniquely identifies people in real life.

## Corrected full-data evaluation

The group-based method makes one out-of-fold prediction per original row, but in each fold leaves **every record from the held-out identity** out of training:

```python
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("applications.csv")
X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
y = df["label"]
groups = df["date_of_birth"].astype(str) + "|" + df["citizenship"].astype(str)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
predictions = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=groups)
accuracy = accuracy_score(y, predictions)
print(f"{accuracy:.6f}")
```

Observed on practice data: **1511 / 1971 = 76.6616%**, or **77%** rounded to a whole percent. Under the group-based interpretation the proposed flag is **`SIGMOID_77`**.

### Important interpretation note

The task statement expressly mentions `LeaveOneOut()` on the entire dataset. **Ungrouped row-wise `LeaveOneOut()` is not the same evaluation**: other rows sharing the left-out identity can remain in training; in the previous local analysis, it yielded approximately **95.94%**. This difference should be acknowledged in the submission rather than silently equating `LeaveOneGroupOut()` with literal row-wise LOO. Only the challenge evaluator can confirm whether its intended flag uses the identity-aware interpretation.

## How to reproduce

1. Place `applications.csv` next to `audit.py`.
2. Install the pinned dependencies from the challenge requirements.
3. Run `python audit.py`.
4. Compare the displayed results for each of the two decoys, the identity-aware holdout, row-wise LOO diagnostic, and grouped full-data evaluation.

The original `project_pipeline.py` is included separately for comparison.
