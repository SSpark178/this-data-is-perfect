# Estonia — This Data Is Perfect! (SIGMOID)

Reproducible audit of the **93.42%** accuracy reported by the e-Residency fraud-screening pipeline.

## Submission

**Flag from the identity-aware corrected evaluation: `SIGMOID_77`.**

**Accuracy:** 1511 correct predictions out of 1971 applications = **76.6616%**, rounded to **77%**.

**Important cross-validation distinction:** The task calls for leave-one-out evaluation; to fix the actual problem of repeat applications from the same identity appearing in train and test, the corrected code uses **`LeaveOneGroupOut()`**. It still produces **one out-of-fold prediction for every original row**. A literal, ungrouped `LeaveOneOut()` gives **1891/1971 = 95.9411%** because another application from the held-out identity can remain in training. Both are shown in `audit.py` and explained in `audit_writeup.md`. The grouped result is our proposed flag based on the *leakage-free* interpretation, not a claim that the two CV methods are identical.

## Files

| File | What it contains |
|---|---|
| [`audit.py`](audit.py) | Executable, reproducible tests for the original pipeline, both decoys, grouped holdout, row LOO and corrected group LOO |
| [`audit_writeup.md`](audit_writeup.md) | Three questionable choices, isolated scores, why one causes over-optimism, and explanation of the corrected evaluation |
| [`project_pipeline.py`](project_pipeline.py) | The original organizer-provided pipeline |
| [`applications.csv`](applications.csv) | The organizer-provided practice dataset |
| [`requirements.txt`](requirements.txt) | Pinned dependencies for reproducibility |

## How to reproduce

Using Python 3.10–3.12 from the root of this repository:

```bash
python -m pip install -r requirements.txt
python audit.py
```

The script trains **`StandardScaler()` + `KNeighborsClassifier(n_neighbors=1)`** using only the six numeric application features. It reports every test and its accuracy and prints the proposed flag. The identity fields are used only as grouping keys for evaluating generalization, never as model inputs.

## Quick evidence

| Isolated experiment | Accuracy |
|---|---:|
| Original 80/20 random-row split | **93.42%** |
| Decoy: scaler fitted only on training rows | **93.67%** |
| Decoy: stratified random-row split | **94.18%** |
| Fix: no identity shared across split (`GroupShuffleSplit`) | **80.21%** |
| Diagnostic: literal row-wise `LeaveOneOut` | **95.94%** |
| Corrected: `LeaveOneGroupOut` on all original rows | **76.66%** |

The practice data has **1971 rows** and **900 distinct (date_of_birth, citizenship) groups**. Since records for a group share labels, row-wise evaluation inflates the estimate for new identities. See the [full explanation](audit_writeup.md).
