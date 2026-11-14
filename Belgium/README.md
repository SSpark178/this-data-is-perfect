# Belgium — Fry-ndship (SIGMOID)

## Status

**Modeling and validation completed; hidden word not independently decoded.** No unverified flag is claimed here. This folder records what the measurements support and includes executable fitting code and a real observed-vs-predicted figure.

## Challenge

Estimate the perimeter / `edge_m` of potato fields from information available in farmers' declarations. Inspect potential sampling structure and identify the hidden observation, which the challenge expects as `SIGMOID_<WORD>`.

## Regression method

Using [`solve.py`](solve.py), fit `LinearRegression` on the **three declaration-time numeric dimensions** `area_ha`, `long_side_m`, and `ditch_m`, with an 80/20 holdout split (`random_state=42`). Avoid using `hedge_m` because Mariette noted it **on site**, and avoid `fence_quote_eur` because the contractor only supplied it **after surveying**.

## Evidence from the 3,869-row supplied dataset

| Measurement | Result |
|---|---:|
| Fields | 3,869 |
| Held-out fields | 774 |
| Held-out R² | **0.9991485153** |
| Held-out mean absolute error | **5.7295 metres** |
| All-row in-sample R² on declaration-time dimensions | **0.9991432333** |
| Including site-measured `hedge_m`: in-sample R² | **0.9995102310** |

An important data-availability warning: `fence_quote_eur` is *almost exactly* `9.4 × edge_m + 185`. Using the quote as an input would leak the measurement target and is not legitimate for a before-survey prediction.

The regression accuracy alone **does not establish the hidden word**. Further analysis of how the observed fields were chosen is required to validate the actual challenge flag.

## Figure

The executable script writes `model_evidence.svg` and `model_evidence.png`, showing measured vs predicted edge, holdout residuals and the sampled field locations.

[View the committed diagnostic figure](model_evidence.svg).

## Reproduce

1. Download [`potato_fields.csv` from the original challenge folder](https://drive.google.com/drive/folders/1qpARslPf8JWOCUHjo2ZIBV1X4Nm-BEdk?usp=sharing) and place it beside `solve.py`.
2. Install `pandas`, `numpy`, `matplotlib` and `scikit-learn`.
3. Run `python solve.py`; this prints the statistics and creates the figures.

The repository currently contains the supporting diagnostic code and a representative plot, **not verified proof of the final hidden flag**.

## Additional hypothesis (unverified)

After the initial guess `SIGMOID_OVERFITTING` was rejected by the platform, a further candidate is **`SIGMOID_SELECTIONBIAS`**, based on the task's explicit warning that the surveyed fields were oddly organized. This is a *guess*, not a recovered word, and does not yet satisfy the organizer's requirement for supporting evidence. The submitted flag must not be represented as validated until it is confirmed by the evaluator.

## Follow-up hypothesis

`SIGMOID_SELECTIONBIAS` was rejected by the platform. The field identifiers reveal a strongly nonuniform sampling rate: IDs 1–~1,800 are nearly complete while the share of included fields falls toward the end of the 4,945-ID sequence. This supports an additional *unverified* hypothesis, `SIGMOID_SAMPLINGBIAS`. This is a descriptive inference, not an independently decoded hidden word.

## Reassessment of the sample-selection mechanism

The local supplied CSV has 3,869 distinct surveyed rows with IDs ranging from 1 to 4,945. The number of included IDs by consecutive 1,000-ID intervals is approximately **990, 980, 836, 638 and 425** (the last range contains 945 possible IDs). This strong variation shows the **survey inclusion rate decreases with the identifier**, rather than being uniform. The mean observed `edge_m` also grows from approximately 650–675 metres in the earliest ID intervals to over 740 metres in the latest intervals.

These observations support investigation of a shift in the *distribution of recorded fields* (possible covariate/selection effects). They do **not**, on their own, identify a cryptographically or visually encoded word. Earlier attempts `SIGMOID_OVERFITTING`, `SIGMOID_SELECTIONBIAS`, and `SIGMOID_SAMPLINGBIAS` were rejected by the platform. The further proposed guess `SIGMOID_COVARIATESHIFT` remains **unverified**; no successful flag has been established.
