# Scratched in Bone, Stamped in Heart — China

**Flag:** `SIGMOID_FENGSHUI`

## Files

- `solve.py` — end-to-end Python pipeline for target glyph detection, counting, and five-bit decoding.
- `counts.csv` — 40 rows (`crate,set,glyph_id,count`), one per crate/set, generated with a 0.75 calibrated detection-confidence threshold.
- `requirements.txt` — Python dependencies.

## Approach

1. Read `train_labels.csv` and `crates.csv` to identify the eight target `glyph_id`s and their annotated training regions. The JPEG filenames do not encode the answer.
2. Learn a HOG+linear SVM candidate detector, and a compact nine-class CNN (eight target glyphs plus other/background) from labelled glyph patches. Validation images are held out from both detector and CNN training.
3. Run the detector at three scales across each unlabelled rubbing, with non-maximum suppression to avoid duplicate bounding boxes.
4. Score candidate patches using CNN checkpoints at epochs 10, 15 and 20. Calibrate target/non-target confidence per glyph using the held-out labelled images and gradient-boosted trees.
5. Count detections with calibrated probability `>= 0.75` in every set and write `counts.csv`.
6. Each crate's five counts has a lower-frequency (bit `0`) and higher-frequency (bit `1`) group. Divide counts into these two groups using within-group squared error (with a small penalty for single-element high groups on only five samples). Interpret the resulting 5-bit binary integers as `A=1` through `Z=26`, in crate order.

The counts are **algorithmic estimates** from a learned detector, not manually verified exact inscriptions. Retraining can shift individual counts due to model randomness and visual ambiguity. The reported snapshot decodes:

| Crate | Counts across sets 1–5 | Bits | Letter |
|---|---|---|---|
| 01 | 3, 2, 11, 9, 6 | 00110 | F |
| 02 | 2, 1, 7, 3, 7 | 00101 | E |
| 03 | 4, 7, 7, 8, 2 | 01110 | N |
| 04 | 3, 4, 9, 12, 11 | 00111 | G |
| 05 | 16, 6, 3, 16, 15 | 10011 | S |
| 06 | 1, 14, 2, 4, 6 | 01000 | H |
| 07 | 10, 4, 13, 2, 10 | 10101 | U |
| 08 | 3, 16, 3, 7, 9 | 01001 | I |

## Running

Extract the original `train/` and `crates/` ZIP archives into a `dataset/` directory, next to `train_labels.csv` and `crates.csv`:

```text
dataset/
  train_labels.csv
  crates.csv
  train/images/*.jpg
  crates/crate_01/set_1/*.jpg
  ...
  crates/crate_08/set_5/*.jpg
```

```bash
pip install -r requirements.txt
python solve.py --data dataset --out counts.csv
```

The script trains and calibrates its models, then prints the inferred flag. It uses CPU by default and can require several minutes.

**Dataset:** [Oracle-bone challenge Google Drive folder](https://drive.google.com/drive/folders/1TE_-nwZzBUTDztXivSbNqeyCFFaN1WDa?usp=sharing). The original images are not duplicated in this repository.