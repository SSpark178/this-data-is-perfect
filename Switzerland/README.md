# Switzerland — The Weak Layer (SIGMOID)

**Challenge:** Decode a hidden word from the **structure** of a fitted decision tree, not from the predictions.

## Submission

Submit [`main.py`](main.py) to the organizer's evaluator. It reads `test-input.csv` and creates `test-output.txt` in the process working directory. It derives the result from an unseen input; the answer is **not hardcoded**.

## Method

1. Read the observations and select the six-dimensional? No: **all numeric feature columns** except the target `slab_released`.
2. Fit `DecisionTreeClassifier(random_state=42)` with all other hyperparameters at scikit-learn defaults.
3. Traverse the fitted tree in **preorder**: node, left subtree, right subtree.
4. At every internal node, take the uppercase first letter of its splitting feature's name.
5. Prefix `SIGMOID_` and write the resulting line to the output file.

## Practice evidence

| Check | Observed result |
|---|---|
| Practice records | 6,000 |
| Training accuracy | 1.0000 |
| Tree depth | 4 |
| Tree leaves | 5 |
| Preorder internal splitting features | `depth_hoar_pct` → `elevation_m` → `mean_grain_roundness` → `overburden_load_kpa` |
| Recovered practice word | `DEMO` |
| Expected practice output | `SIGMOID_DEMO` |
| Match | **Yes** |

**Important:** `SIGMOID_DEMO` is only the organizer's practice example, not the final hidden flag. The evaluation file is not public.

## Reproduce locally

Python 3.12; pandas 2.3.3, numpy 2.2.6 and scikit-learn 1.7.2 as specified by the task.

1. Download the organizer's `train-input.csv` and `train-output.txt` from the [official source](https://drive.google.com/drive/folders/1zyj2sDRo44C6zbY5jSeP0oXJbyYIWhe-?usp=sharing).
2. Place `main.py` beside them and copy `train-input.csv` to `test-input.csv`.
3. Run `python main.py` in this folder.
4. Check that `test-output.txt` contains exactly `SIGMOID_DEMO`.

The full practice input is available from the organizer's link above, not redistributed in this folder.
