# Belgium — Fry-ndship (SIGMOID)

## Correct flag

**`SIGMOID_SUPERVISION`**

This flag was confirmed by the solver after previous guesses failed. The hidden word is not inferred from a conventional model accuracy score; it is **drawn by the residuals of the surveyed hedge measurements**.

## Method and evidence

1. Read the **3,869** potato fields from `potato_fields.csv`.
2. Calculate the inferred short side in metres: `w = area_ha * 10000 / long_side_m`.
3. Calculate the rectangular perimeter: `per = 2 * (long_side_m + w)`.
4. Fit the edge length using `edge_m ~ 1 + per + ditch_m + hedge_m`, with least squares.
5. Fit a linear trend `hedge_m ~ per` and plot the **hedge residual** against `per`.
6. Read the point-cloud text: **SUPERVISION**.

### Verified numerical results (supplied dataset)

| Measure | Value |
|---|---:|
| Data rows | 3,869 |
| Edge-fit intercept | 0.0000921 |
| `per` coefficient | 0.4100022 |
| `ditch_m` coefficient | 1.0000047 |
| `hedge_m` coefficient | 0.9998849 |
| Edge-fit R² | **0.9999999435** |
| Hedge trend slope | **0.0400000598** |
| Hedge trend intercept | **−0.00113825** |
| Word visible in hedge residuals | **SUPERVISION** |

This figure is generated directly by [`solve.py`](solve.py), which saves both `fry_ndship.png` and `fry_ndship.svg`.

**Important distinction:** The earlier diagnostic `model_evidence.svg` was based on a different analysis and does not show the hidden word. The correct evidence is the **hedge residual figure** produced by this script.

## Reproduce

1. Download the data as `potato_fields.csv` from the [official organizer folder](https://drive.google.com/drive/folders/1qpARslPf8JWOCUHjo2ZIBV1X4Nm-BEdk?usp=sharing).
2. Place it beside the linked `solve.py` in this directory.
3. Install `pandas`, `numpy`, `matplotlib`.
4. Run `python solve.py`. Open the resulting **`fry_ndship.png`**: the scatterplot visibly spells **SUPERVISION** across the perimeter axis.

## Submission

- Flag: **`SIGMOID_SUPERVISION`**
- Code URL: [`solve.py`](solve.py)

The residual plot is the intended supporting figure, not the regression prediction scatterplot. The earlier incorrect hypotheses (`OVERFITTING`, `SELECTIONBIAS`, `SAMPLINGBIAS`, `COVARIATESHIFT`) have been superseded.
