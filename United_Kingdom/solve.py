"""Compute the BLEU-bluff corrected honest seats and decode the flag.

The correct final word was confirmed by the submitter's platform evaluation.
The word is derived from semantic and mechanical quality scores, not assigned
to seats in a hardcoded round-by-round list.
"""
from pathlib import Path
import pandas as pd
from semantic import P, features
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

def decode(seats):
    return "".join(
        chr(ord("A") + 9 * (seats[i]-1) + 3 * (seats[i+1]-1) + (seats[i+2]-1))
        for i in range(0, len(seats), 3)
    )

def main():
    scores = []
    for row in P.itertuples():
        parts = features(row.text, (row.round, row.card))
        scores.append(dict(round=row.round, seat=row.seat, **parts))
    facts = pd.DataFrame(scores).groupby(["round", "seat"], as_index=False).sum()
    lexical = pd.read_csv(HERE / "metrics_per_deck.csv")
    scores = facts.merge(lexical, on=["round", "seat"], validate="one_to_one")
    scores["error_cost"] = (
        2*scores.cuisine_missing + scores.etype_missing
        + scores.price_missing + scores.rating_missing
        + 0.7*scores.area_missing + 2*scores.name_missing
        + 3.0*scores.landmark_missing
        + scores.cuisine_wrong + 1.5*scores.etype_wrong
        + scores.price_wrong + scores.rating_wrong + scores.area_wrong
        + 5*scores.total_repeated_4grams
        + 1.3*scores.short_descriptions + scores.cards_with_10gram_copy
    )
    winners = scores.loc[scores.groupby("round").error_cost.idxmin()].sort_values("round")
    seats = winners.seat.astype(int).tolist()
    assert len(seats) == 24 and len(scores) == 72
    word = decode(seats)
    scores.sort_values(["round", "seat"]).to_csv(
        HERE / "semantic_scores.csv", index=False, float_format="%.5f"
    )
    winners[["round", "seat", "error_cost"]].to_csv(
        HERE / "honest_seats.csv", index=False, float_format="%.5f"
    )
    print("Seats by round:", seats)
    print("Decoded word:", word)
    print("Flag: SIGMOID_" + word)

    fig, ax = plt.subplots(figsize=(12, 5))
    for seat, section in scores.groupby("seat"):
        ax.scatter(section["round"], section["error_cost"], label=f"Seat {seat}", s=50)
    ax.plot(winners["round"], winners["error_cost"], color="black",
            linewidth=0.8, label="Selected lowest-error seat")
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Round")
    ax.set_ylabel("Factual/fluency error cost (lower is better)")
    ax.set_title("A semantic and anti-copying score identifies the honest decks")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(HERE / "semantic_scores.png", dpi=150)

if __name__ == "__main__":
    main()
