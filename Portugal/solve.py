"""SIGMOID Portugal — The Douro Paradox.

Reproduce the complete Simpson's paradox reversal in douro_vineyard_data.csv.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "douro_vineyard_data.csv")
target = "distance_from_porto_km"
cutoff = 105.0

overall = df.groupby("treatment")["harvest_success"].agg(["sum", "count", "mean"])
df["distance_group"] = pd.cut(
    df[target], [-float("inf"), cutoff, float("inf")],
    labels=["at_most_105_km", "over_105_km"]
)
by_group = (
    df.groupby(["distance_group", "treatment"], observed=True)["harvest_success"]
      .agg(["sum", "count", "mean"])
)
print("OVERALL:")
print(overall.to_string())
print("\nSPLIT BY", target, f"(threshold {cutoff} km):")
print(by_group.to_string())

assert overall.loc["treatment", "mean"] > overall.loc["control", "mean"]
for group in ["at_most_105_km", "over_105_km"]:
    rates = by_group.loc[group, "mean"]
    assert rates.loc["treatment"] < rates.loc["control"], group

figure_data = [
    ("Overall", overall.loc["control", "mean"], overall.loc["treatment", "mean"]),
    ("≤ 105 km", by_group.loc[("at_most_105_km", "control"), "mean"],
     by_group.loc[("at_most_105_km", "treatment"), "mean"]),
    ("> 105 km", by_group.loc[("over_105_km", "control"), "mean"],
     by_group.loc[("over_105_km", "treatment"), "mean"])
]
fig, ax = plt.subplots(figsize=(9, 5))
positions = list(range(len(figure_data)))
ax.bar([i - 0.18 for i in positions], [x[1] * 100 for x in figure_data],
       width=.36, label="Control")
ax.bar([i + 0.18 for i in positions], [x[2] * 100 for x in figure_data],
       width=.36, label="Treatment")
ax.set_xticks(positions, [x[0] for x in figure_data])
ax.set_ylabel("Successful harvests (%)")
ax.set_title("The Douro Paradox — pooled advantage reverses in both groups")
ax.legend()
ax.set_ylim(0, 40)
for i, row in enumerate(figure_data):
    for x, val in ((i - 0.18, row[1]), (i + 0.18, row[2])):
        ax.text(x, val * 100 + .5, f"{val:.2%}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "simpsons_paradox.png", dpi=170)
fig.savefig(HERE / "simpsons_paradox.svg")
plt.close(fig)
print("\nFLAG: SIGMOID_DISTANCE_FROM_PORTO_KM")
