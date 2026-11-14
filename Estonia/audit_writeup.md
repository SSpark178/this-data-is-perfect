# Estonia — This Data Is Perfect! | Reproducible audit

## Verdict

The reported **93.42%** is an optimistic accuracy estimate caused primarily by **splitting correlated application records by row**. The data contains **1,971 applications** but only **900 groups** defined by the available combination `(date_of_birth, citizenship)`. Multiple records from a group have the same fraud label and can fall into both training and test data under the original random row split.

The two other questionable decisions do **not** explain this large inflation: fitting the scaler before splitting, and failing to stratify the random row split.

The model is **unchanged** throughout: `StandardScaler()` + `KNeighborsClassifier(n_neighbors=1)`, with the six numeric application features. The identity fields are never modeling inputs; they are used only to define groups for the audit.

## Evidence: three questionable choices tested separately

| Experiment | What changed relative to original | Accuracy | Interpretation |
|---|---|---:|---|
| Original `project_pipeline.py` | None: full-data scaling + nonstratified random 80/20 row split, seed 1137 | **93.42%** | Shipped result |
| Decoy 1: preprocessing leakage | Fit `StandardScaler` only on training data, then transform test via a pipeline; keep original row split | **93.67%** | Only 0.25 percentage points higher; does not explain the inflation |
| Decoy 2: no stratification | Add `stratify=y` to the random row split; retain the other original steps | **94.18%** | Only 0.76 percentage points higher; does not explain the inflation |
| Actual bug: overlapping identity groups | `GroupShuffleSplit` so the identity groups cannot appear in both train and test; model unchanged | **80.21%** | A large drop; reveals the issue |

These are separate experiments, not cumulative changes. The checks for the decoys remain above 93%, which isolates the real failure of the evaluation.

## Fully out-of-fold corrected evaluation

One out-of-fold prediction is produced for **every row** of the original dataset. In the corrected evaluation, **all** rows belonging to the left-out identity group are excluded from training, so the model cannot recognize the held-out applicant via related entries.

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

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
pred = cross_val_predict(pipeline, X, y, cv=LeaveOneGroupOut(), groups=groups)
accuracy = accuracy_score(y, pred)
print(f"{accuracy:.6f}")              # 0.766616
print(f"SIGMOID_{round(accuracy*100)}") # SIGMOID_77
```

**Result: 1,511 correct out of 1,971 rows = 76.6616%, rounded to 77%.**

### Why not ordinary `LeaveOneOut()`?

The challenge text explicitly names `LeaveOneOut()`. We verified its literal, ungrouped result too: **1,891 / 1,971 = 95.9411%, rounded to 96%**. This does **not** prevent a group from appearing in both train and test: only one *row* is left out.

We therefore present both, explicitly and without claiming they are identical:

| Full-data method | Correct / total | Accuracy | Interpretation |
|---|---|---:|---|
| `LeaveOneOut()`, one row held out | 1891 / 1971 | **95.94%** | Diagnostic: correlated rows still in training |
| `LeaveOneGroupOut()`, all rows of identity held out | 1511 / 1971 | **76.66%** | Leakage-free generalization estimate; one prediction per row |

The corrected, identity-aware method is the basis of the **proposed flag `SIGMOID_77`**. This is our interpretation of "or its equivalent for your fix": the evaluator's actual acceptance criteria determine the accepted flag. The grouping key is a **proxy for identity based on the available columns**, not a claim that date of birth and citizenship uniquely identify real people.

## Reproduce everything

```bash
python -m pip install -r requirements.txt
python audit.py
```

`audit.py` prints the original score, both decoy tests, the grouped holdout, literal row-level LeaveOneOut, grouped LeaveOneOut, and the proposed flag. The original `project_pipeline.py` is included unchanged to inspect the reported 93.42% and `applications.csv` is the source data.

## Files

- [Original pipeline](project_pipeline.py)
- [Input dataset](applications.csv)
- [Executable audit](audit.py)
- [Dependencies](requirements.txt)
- [Repository overview](README.md)
