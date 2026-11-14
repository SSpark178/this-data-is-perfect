"""Audit why whole-deck BLEU can reward poor descriptions.

Usage: python analyze.py
Files required in this folder: references.csv, panel.csv, bleu.py.
Outputs: metrics_per_deck.csv, evidence.png, baseline_word.txt.

Note: this is a diagnostics script, not a verified honest-seat classifier.
"""
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bleu import corpus_bleu, tokenize, ngrams

HERE = Path(__file__).resolve().parent


def decode(seats):
    return "".join(
        chr(65 + 9 * (seats[i]-1) + 3 * (seats[i+1]-1) + (seats[i+2]-1))
        for i in range(0, 24, 3)
    )


def copied_10grams(text, references):
    reference_grams = {gram for r in references for gram in ngrams(tokenize(r), 10)}
    return sum(gram in reference_grams for gram in ngrams(tokenize(text), 10))


def repeated_4grams(text):
    return sum(k-1 for k in ngrams(tokenize(text), 4).values() if k > 1)


def audit():
    panel = pd.read_csv(HERE / "panel.csv")
    references = pd.read_csv(HERE / "references.csv")
    refmap = references.groupby(["round", "card"]).reference.apply(list).to_dict()
    assert panel.groupby(["round", "seat"]).size().eq(40).all()

    records = []
    for (rnd, seat), deck in panel.groupby(["round", "seat"]):
        texts = deck.text.tolist()
        refs = [refmap[(rnd, int(card))] for card in deck.card]
        score = corpus_bleu(texts, refs)
        lengths = [len(tokenize(t)) for t in texts]
        records.append({
            "round": int(rnd),
            "seat": int(seat),
            "bleu": score["bleu"],
            "brevity_penalty": score["brevity_penalty"],
            "unigram_precision": score["precisions"][0],
            "fourgram_precision": score["precisions"][3],
            "short_descriptions": sum(n <= 10 for n in lengths),
            "mean_length": sum(lengths) / len(lengths),
            "cards_with_10gram_copy": sum(
                copied_10grams(text, ref) > 0 for text, ref in zip(texts, refs)
            ),
            "total_repeated_4grams": sum(repeated_4grams(t) for t in texts),
        })

    result = pd.DataFrame(records).sort_values(["round", "seat"])
    result.to_csv(HERE / "metrics_per_deck.csv", index=False, float_format="%.6f")
    winners = result.loc[result.groupby("round").bleu.idxmax()].sort_values("round")
    producers_word = decode(winners.seat.tolist())
    (HERE / "baseline_word.txt").write_text(
        "SIGMOID_" + producers_word + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for seat, subset in result.groupby("seat"):
        axes[0].plot(subset["round"], subset["bleu"], marker="o", label=f"Seat {seat}")
        axes[1].plot(
            subset["round"], subset["cards_with_10gram_copy"],
            marker="o", label=f"Seat {seat}"
        )
    axes[0].set_ylabel("Corpus BLEU-4 (house judge)")
    axes[1].set_ylabel("Cards containing exact reference 10-gram")
    axes[1].set_xlabel("Round")
    axes[1].set_xticks(range(1, 25))
    axes[0].set_title("Why BLEU needs semantic checking")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "evidence.png", dpi=160)

    print("Panels:", len(panel), "references:", len(references), "decks:", len(result))
    print("Producers' baseline word (NOT THE FLAG):", producers_word)
    print(result.to_string(index=False))


if __name__ == "__main__":
    audit()
