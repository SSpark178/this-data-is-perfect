# 🇷🇴 Romania — Shaken, Still Standing

**Accepted flag (reported by the contestant): `SIGMOID_CONVOLUTION`.**

## What the task asks

Classify each of the **5,000** sealed Romanian sensor records as a deep Vrancea earthquake (`vrancea=1`) or another disturbance (`vrancea=0`) using **7,500** labeled training records, each with **48 spectral-amplitude channels** (`c01`–`c48`).

The supplied solution follows a **mixture-of-experts** strategy. According to the contestant, the official evaluator printed **`SIGMOID_CONVOLUTION`** when run with its predictions. This README records the delivered method and reported validation scores; the organizer's hidden ground truth is not published.

## Why a single classifier is insufficient

The training and sealed data contain records from three different hidden acquisition networks. A single global decision rule mixes several incompatible label-generating relationships.

Reported **five-fold accuracy on the complete labeled training set**:

| Model | Reported accuracy |
|---|---:|
| Logistic regression | 0.600 |
| SVM, RBF kernel | 0.675 |
| Histogram-based gradient boosting (HGB) | 0.847 |

These numbers are supplied in the original solution account and are **not rerun in this repository update**.

## Network gate

The routing procedure in [`solution.py`](solution.py) uses both the training and unlabeled sealed features to estimate the global channel statistics; it does not use sealed labels.

- **Network C** is identified by **36 attenuated channels**, whose spread is about one quarter of the other networks. The code computes a standardized mean-square spread on these channels and selects C where the statistic is below **0.25**.
- Among other records, **`c13 < -0.6`** is assigned to network A; the rest to network B.
- Reported composition of the sealed set: **597 A, 2,403 B, 2,000 C**.

## Three specialized experts

| Network | Expert | Features or rule | Reported score |
|---|---|---|---:|
| **A** | Three-way parity / XOR | Thresholds: `c43 > -0.61`, `c35 > -0.23`, `c38 > 0.73`; predict `1 - (sum of booleans % 2)` | **0.972**, *training-subset accuracy* |
| **B** | StandardScaler + LogisticRegression (`C=0.3`) | 47 channels; excludes the gate channel `c13` | **0.923**, 5-fold CV |
| **C** | StandardScaler + PolynomialFeatures(degree=2) + LogisticRegression (`C=0.1`) | 12 full-scale channels; polynomial squares and interactions | **0.933**, 5-fold CV |

The 12 network-C signal channels are **`c03, c09, c12, c14, c15, c19, c27, c28, c30, c36, c39, c40`**; the other 36 are treated as low-amplitude noise.

**Important metric distinction:** The solution prints the A expert's in-sample agreement on the network-A training subset; it is not an out-of-fold cross-validation estimate. The B and C scores are five-fold cross-validation results. These scores come from the original solver's report.

## Generating the submission

[`solution.py`](solution.py) loads the CSVs, discovers network assignments, trains each specialized expert and routes each sealed record to the corresponding one. It writes **`predictions.csv`** containing exactly `record_id,vrancea` and invokes the organizer's [`evaluation_script.py`](evaluation_script.py).

```bash
pip install -r requirements.txt
python solution.py
```

Repository layout:

```text
Romania/
  README.md
  solution.py
  baseline_solution.py
  evaluation_script.py
  requirements.txt
  dataset/
    train.csv
    sealed.csv
    vault.npz  # original binary evaluator archive
```

The original `dataset/vault.npz` is committed in binary form and is used only by the evaluator, not by the training procedure; it does not store the true labels in plaintext. The dataset, code, and local evaluator supplied with the task are all included.

## How the flag is obtained

The organizer's evaluator applies an error-tolerant decoder to the full 5,000-record prediction sequence; it only releases a flag when enough labels are correct. The original solver reported that this prediction sequence was accepted and the decoder printed:

**`SIGMOID_CONVOLUTION`**

This GitHub upload preserves the provided implementation, its input data and the reported comparison of several attempted models. The user reported that the flag had already been accepted on the hackathon platform. No new evaluation was run as part of organizing the repository.
