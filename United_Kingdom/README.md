# United Kingdom — Call My BLEUff

**Status: analysis/evidence prepared; honest-seat sequence and flag NOT yet verified.** Do not submit the producers' baseline word as the flag.

## Goal

Pick the sole honest panelist (seat 1, 2 or 3) for each of the 24 rounds. Each seat gives a base-3 digit (seat minus one), and each three-round block decodes one of eight letters. The judging metric is the organizer's **corpus BLEU-4**: lowercased word/punctuation tokenization, clipped 1–4-gram precision, uniform geometric mean, zero smoothing and a single corpus-wide brevity penalty.

## Source data and code

- [`references.csv`](references.csv): 5 ground-truth descriptions per card (4,800 rows)
- [`panel.csv`](panel.csv): 3 candidates for each of 40 cards and 24 rounds (2,880 rows)
- [`bleu.py`](bleu.py): exact house scorer, from organizer
- [`baseline_solution.py`](baseline_solution.py): organizer's baseline
- [`analyze.py`](analyze.py): reproduces BLEU and tests mechanical reference-copy and text-damage signals
- [`metrics_per_deck.csv`](metrics_per_deck.csv): scores for all 72 seat/round decks
- [`evidence.svg`](evidence.svg): BLEU and direct reference copying compared per seat/round

## Verified findings

The producers' highest-BLEU seat per round decodes to **`SKELDRAP`**, which the task explicitly warns is **not the flag**. High corpus BLEU can be gamed without good factual communication.

Examples of telltale behavior in the provided data:

| Deck | Observable manipulation | Recorded evidence |
|---|---|---|
| Round 23, seat 3 | Copying long stretches directly from the references | **31 of 40** cards contain a verbatim 10-token reference sequence |
| Round 5, seat 2 | Repeated 4-grams / multiple overlapping descriptions in one answer | **18** surplus repeated 4-grams in the deck |
| Round 9, seat 1 | Drastically abbreviated replies; corpus BLEU can conceal per-card failures | **13 of 40** descriptions have at most 10 tokens |

**Why this matters:** BLEU assesses overlapping word sequences, not truth conditions, missing attributes, contradictory facts, excessive copying or naturalness. Neither choosing maximum corpus BLEU nor choosing minimum BLEU demonstrates which of three decks is honest.

## Run

```bash
pip install pandas matplotlib
python analyze.py
```

The script produces a diagnostics table and plot. **It is not yet a validated 24-round honest-seat classifier**. The full challenge additionally requires a demonstrated semantic/robustness check that unambiguously separates the 72 decks into honest and bluffing. We will document the final decoded word only after that check succeeds.

The underlying restaurant-domain data is related to the [E2E NLG Challenge dataset](https://github.com/tuetschek/e2e-dataset) (CC BY-SA 4.0), and the challenge-specific files are provided by the SIGMOID organizers.
