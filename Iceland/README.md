# Iceland — What The Data? (SIGMOID)

**Task:** Write a generic cleaner that converts a damaged UTF-8, semicolon-delimited sensor export into the exact CSV schema required by the hidden evaluator.

## Submission contract

Submit [`main.py`](main.py). It reads **`test-input.csv`** using that bare filename from the working directory and creates **`test-output.csv`**, with the columns:

```csv
reading_id,station_name,temperature_c,reading_date
```

## Cleaning pipeline

1. Ignore blank lines and all comment/preamble lines beginning with `#`, regardless of preamble length.
2. Parse the `;`-separated input and convert comma decimal separators to periods.
3. Repair mojibake by round-tripping incorrectly decoded UTF-8 bytes through legacy encodings (`latin-1` / `cp1252`), rather than using a lookup table of station names.
4. Convert values marked `F` into Celsius: `(F - 32) * 5 / 9`; keep values marked `C`.
5. Format temperatures to exactly **one decimal place** using decimal arithmetic.
6. Keep one record per `reading_id`, sort IDs ascending, and write UTF-8 comma-delimited output with no index.

## Practice evidence

- Practice input: **30 distinct reading IDs** plus retransmitted duplicates.
- Practice reference: **30 output rows**.
- Running the cleaner on the practice input and comparing with the organizer's reference produces **an exact match after normalizing CRLF vs LF line endings**, which the evaluator permits.
- Example restorations: `MÃ½vatn` → `Mývatn`, `HveragerÃ°i` → `Hveragerði`.
- Example conversion: `168,8 F` → `76.0 C`.

## Reproduction

Download `train-input.csv` and `train-output.csv` from the [organizer's folder](https://drive.google.com/drive/folders/1gzrKFtJKnJXCp470DLU1NWAZpbfe6Cz6?usp=sharing), put them here, and run `python verify_practice.py`.

Alternatively, rename/copy `train-input.csv` to `test-input.csv`, then run `python main.py` and compare the generated file with `train-output.csv`.

The code requires only Python 3.10–3.12's standard library, with no network access.
