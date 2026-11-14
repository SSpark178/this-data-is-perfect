# Rwanda — The Drying Floor at Karongi (SIGMOID)

## Confirmed flag

**`SIGMOID_CONVERGENCE`** — the solver reports this flag was accepted.

## The single-function correction

In the organizer-supplied `graph.py`, replace **only** `inspect_batch` with the function in [`inspect_batch_fix.py`](inspect_batch_fix.py). It sets the `approved` state field to the outcome of the moisture-range check:

```python
def inspect_batch(state: BatchState) -> dict:
    """Measure the lot and decide whether it is ready to leave the floor."""
    ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
    return {"inspections": state["inspections"] + 1, "approved": ready}
```

The drying loop now terminates whenever `10.5 <= moisture <= 12.5`; already-approved lots require no drying passes.

## Evaluation evidence

The uploaded [`run_log.txt`](run_log.txt) is the user's reported **actual evaluation output**, not a log independently generated here. It includes eight `PASS` lines and the final `SIGMOID_CONVERGENCE` flag.

| Batch | Drying passes | Final moisture | Grade |
|---|---:|---:|---|
| nyamasheke | 0 | 12.0 | A |
| rutsiro | 0 | 11.0 | A |
| huye | 1 | 12.5 | B |
| nyabihu | 2 | 12.3 | B |
| muhanga | 3 | 12.5 | B |
| karongi | 4 | 11.8 | A |
| gicumbi | 5 | 12.5 | B |
| ngororero | 6 | 12.5 | B |

**Reported result:** all eight batches finish within `recursion_limit=18`.

## Reproduce and complete the official submission

1. Obtain the **original** `graph.py`, `graph_reference.py`, `evaluation_script.py`, and `dataset/batches.json` from the [organizer's source](https://drive.google.com/drive/folders/1c1tXsOOX7kX3H881Qv8l6VXv7t6xRnp8?usp=sharing).
2. Replace **only** the body of `inspect_batch` in the original `graph.py`, leaving every other part untouched.
3. Run `python evaluation_script.py | tee run_log.txt` with `langgraph==1.2.16` and `langchain-core==1.6.12`.
4. Publish that **complete modified `graph.py`** in this folder and submit its URL with the evaluation log.

**Current limitation:** the original `graph.py` was not attached, so this folder includes the *exact function replacement* and evaluation log, but **not yet the complete corrected `graph.py`**. The organizer explicitly requires both full `graph.py` and `run_log.txt` for a complete submission.
