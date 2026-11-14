# 🇩🇪 Germany — Cutia neagră (Recover the Black Box)

**Official platform result: PASS — 100.0/100 points, 136/150 oracle queries.**  
**Released flag: `SIGMOID_FACTORIAL`.**

The result above was reported by the SIGMOID platform after uploading [`main.py`](main.py). The screenshot of the accepted submission was provided by the participant on 14 November 2026.

## Submission

Submit **only [`main.py`](main.py)** to the challenge upload form. The organizer executes its `solve(oracle, points)` function once against a hidden deterministic black-box function. The script is self-contained apart from the permitted `numpy==2.2.4` dependency; it does not need to read local data files during evaluation.

## Method

The solution spends no more than 150 queries to infer the structure rather than fitting only a naive additive approximation:

1. Probe controlled, multivariable settings and estimate a smooth low-order polynomial model.
2. Detect the active pairwise product interaction, including its coefficient.
3. Search for a sharp switching surface, infer its gate variables and linear boundary, and estimate the jump.
4. Fit smooth and discontinuous terms jointly; generate 300 predictions and provide the structured result required by the evaluator.
5. Enforce the oracle budget, using previously cached queries when possible.

The solution does **not** hard-code the publicly disclosed practice function or hidden function.

## Reproduce the practice evaluation

Requires Python 3.10+ and the organizer's `harness.py`, `practice_function.py`, `evaluation_script.py`, and `challenge/challenge_set.json`. Download these separately from the challenge materials and arrange them according to the organizer's folder structure.

```bash
python -m pip install -r requirements.txt
python evaluation_script.py main.py --eval-type train
```

**Local practice verification:** 100.0/100 with 127/150 queries; perfect scored prediction accuracy on the public practice data. The official hidden test result is separately stated above.

**Challenge materials:** [official Google Drive folder](https://drive.google.com/drive/folders/1XRPCzOo2Jz1ymh80bvbmTG8OXH-EI_A_?usp=sharing).

No organizer-owned datasets, private test function, or evaluator internals are committed here.
