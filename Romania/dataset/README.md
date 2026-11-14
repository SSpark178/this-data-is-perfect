# Romania dataset

This directory contains the organizer-provided input CSVs:

- `train.csv`: 7,500 labeled records (48 channels and `vrancea`)
- `sealed.csv`: 5,000 unlabeled records

The organizer also supplied `vault.npz`, used **only by the evaluator**, not to train the mixture-of-experts. The original binary archive is **also committed** as [`vault.npz`](vault.npz). You can run the organizer's evaluator directly without fetching additional inputs.

The prediction script reads the two CSVs already committed in this folder and generates `predictions.csv` one record at a time, in sealed-file order.
