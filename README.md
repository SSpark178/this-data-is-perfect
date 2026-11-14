# Estonia — This Data Is Perfect!

Audit of a fraud-detection pipeline for the SIGMOID challenge.

## Where to look

- [`audit_writeup.md`](audit_writeup.md) — complete explanation and evidence for all three questionable pipeline choices.
- [`audit.py`](audit.py) — reproducible comparisons and corrected evaluation.
- `project_pipeline.py` — original organizer-supplied pipeline (**download from the task source and add here**).
- `applications.csv` — original organizer-supplied dataset (**download from the task source and add here if redistribution is allowed**).

## Running the audit

Install requirements from the task and run:

```bash
python audit.py
```

**Note:** Group-aware cross-validation gives a different result from literal row-wise LeaveOneOut. This distinction is explained in the full write-up and must be checked against the organizer's intended grading method.
