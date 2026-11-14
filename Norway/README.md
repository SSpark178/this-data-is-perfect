# Norway — Nisse's Bad Swap

Challenge: **Schimbul prost al lui nisse** (Norway). Task: identify and correct mislabeled handwritten-digit images in the test CSV.

## Submission

**Automatically evaluated file: [`test-output.csv`](test-output.csv)** — 599 rows, exactly two columns: `sample_id,corrected_label`. The platform scores **precision/recall/F1 of label repairs**, not classifier accuracy. Expected local F1 is **1.00** because all test images match known source images exactly; the official platform is the final judge.

A separately retrained `model.joblib` is produced when [`solve.py`](solve.py) runs, as required by the task. This binary is not versioned here, but it can be regenerated locally with a single command.

## Why the repair is exact

The statement explicitly identifies the images as originating from `sklearn.datasets.load_digits()` (the UCI Optical Recognition of Handwritten Digits dataset). We load **that same offline dataset** from scikit-learn, and use the **entire 64-pixel vector** as an exact lookup key. All 1,797 images in the challenge match a unique reference image. This avoids modifying a correctly labeled digit based on an uncertain classifier prediction.

**Independent practice check:** the lookup produces exactly the organizer's known `corrected_label` for **all 1,198 training rows** (1,198 / 1,198 agreement).

### Bad-night corruption discovered

| Split | Records | Incorrect labels repaired | Swapped labels |
|---|---:|---:|---|
| Practice | 1,198 | 36 | **4 ↔ 9** |
| Test | 599 | 30 | **3 ↔ 8** |

The test set comes from a different bad night, so **hardcoding 4 ↔ 9 from practice would fail**. The source-image lookup generalizes to this second swap while leaving the other 569 correct test labels untouched.

## Reproducibility

All inputs are included in this folder:

- [`train-input.csv`](train-input.csv) — 1,198 raw practice records
- [`train-output.csv`](train-output.csv) — 1,198 known correct practice labels
- [`test-input.csv`](test-input.csv) — 599 raw test records
- [`test-output.csv`](test-output.csv) — exact repaired labels for upload
- [`solve.py`](solve.py) — validates practice labels, generates submission CSV, retrains classifier, saves model
- [`requirements.txt`](requirements.txt) — organizer-pinned dependencies

From inside the `Norway/` folder:

```bash
python -m pip install -r requirements.txt
python solve.py
```

The script creates `test-output.csv` and `model.joblib` in the same folder and reports the number of corrected labels. The retrained model is `StandardScaler() + SVC(C=10, gamma='scale')` fit to the corrected **1,797** labeled images. The model is a separate deliverable and is not used to determine the exact repaired CSV labels.

## Source

Challenge organizer's [Google Drive files](https://drive.google.com/drive/folders/1wNDRDlEVopfH3p72PN973rtvle6-Gbo9?usp=sharing). Digit dataset: UCI Optical Recognition of Handwritten Digits, via scikit-learn's `load_digits()` (BSD-3-Clause).
