# Albania — Looks like an anomaly! (SIGMOID)

**Challenge:** Streaming anomaly detection for six sensors, including point spikes, slow drifts and collective changes.

## Submission

Submit only [`detector.py`](detector.py) to the organizer's evaluator. The harness imports `Detector`, calls `update(reading: dict)` once per timestamp and expects a score in `[0, 1]`. The implementation returns **0.0** or **1.0** and uses only the current reading and earlier observations. No future rows or labels are accessed during prediction.

## Method

For each `bunker_zone`, maintain a bounded history of the previous 241 sensor observations. Use a robust historical median baseline and six sensor-specific scaling factors to compare the current reading with typical values.

- **Point anomalies:** very large single-channel standardized deviation.
- **Collective anomalies:** moderate-to-large joint deviation in vibration/acoustic activity.
- **Gradual drift:** recent 12- or 45-reading mean differs strongly from an older historical reference.
- **Temporal isolation:** history and counters are kept independently for each zone; the detector does not inspect future samples.

## Practice validation evidence

Tested with the organizer's 3,600-row labeled practice stream, processing the rows in chronological order one by one, thresholded at **0.50**:

| Metric | Practice value |
|---|---:|
| True positives | 272 |
| False positives | 108 |
| False negatives | 15 |
| Overall F1 | **0.815592** |
| Point anomaly recall | **1.0000** |
| Drift recall | **0.922222** |
| Collective recall | **0.9900** |
| Weighted practice score | **84.5890 / 100** |
| Acceptance threshold | 70.00 / 100 |

The score is computed using the official weighting: `100 × (0.80 × F1 + 0.05 × point recall + 0.075 × drift recall + 0.075 × collective recall)`.

**Note:** This is an in-sample *practice* evaluation for the supplied stream. The flag is released by the platform only if the unseen evaluation achieves at least 70/100; that result has not been independently confirmed here.

## Reproduce

Download the practice sensor dataset from the [organizer](https://drive.google.com/drive/folders/14AMHetQATgeB-wSm0-Da10NffyxOGQVo?usp=sharing) and place its feature stream and labels in this folder as `train-input.csv` and `train-output.csv`. Run `python evaluate_practice.py`.

Required: Python 3.11, numpy 2.2.4, pandas 2.2.3 (for the offline evaluator only). The submitted detector uses numpy and the standard library.

## Files

- [`detector.py`](detector.py): online causal detector
- [`evaluate_practice.py`](evaluate_practice.py): independent reproduction of the score components
