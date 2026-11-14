# Moldova — Feather Stories (SIGMOID)

## Confirmed flag

**`SIGMOID_ANSEMO`** — reported by the solver as the accepted/correct flag.

## Provided analysis scripts

The two Python scripts below were supplied as exploratory approaches. They are preserved as written, including their original comments and outputs. **Neither one derives the confirmed flag `ANSEMO`.** They should not be submitted as proof that the correct flag was calculated.

| File | Analysis | Intermediate / incorrect result |
|---|---|---|
| [`exploration_locality.py`](exploration_locality.py) | Filters white stork (`Ciconia ciconia`) ring readings, ranks localities by number of readings and distance to registered ringing sites, reconstructs a ring using per-position majority vote | `SIGMOID_RHGLWQ` |
| [`exploration_confusion.py`](exploration_confusion.py) | Defines visually similar letter classes, checks compatibility with known rings, isolates unrecognized readings, majority votes and applies an inverse substitution mapping | `SIGMOID_XWBPHD` |

### Key observation

Both explorations focus on locality **`L5819951` (Beleu)**, where the provided analysis identifies 24 white-stork readings; the nearest registered white-stork ringing site is described as approximately 74 km away. These are findings reported in the supplied code comments and have **not** been independently verified against the original CSV files here.

### Reproduce the exploratory calculations

Download the original task inputs as `ring_register.csv` and `observations.csv`, place both in this folder, and run the scripts with Python, pandas and numpy installed:

```bash
python exploration_locality.py
python exploration_confusion.py
```

**Evidence gap:** Additional code or explanation is needed to show how the original data lead to the confirmed word **ANSEMO**. The scripts document explorations, not a verified derivation of that word.
