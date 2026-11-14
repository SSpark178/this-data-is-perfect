"""SIGMOID Spain: reconstruct the fair-price Christmas souvenir board."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

receipts = pd.read_csv("purchases.csv")
guide = pd.read_csv("office_price_guide.csv")
data = receipts.merge(guide, on="Product", validate="many_to_one")
unit_price = data["Total_paid_eur"] / data["Quantity"]
valid = data.loc[
    (data["City"] == data["Home_city"])
    & unit_price.between(data["Fair_min_eur"], data["Fair_max_eur"], inclusive="both")
]
board = (
    pd.crosstab(valid["List_no"], valid["Pilot_no"])
    .reindex(index=range(1, 7), columns=range(1, 22), fill_value=0)
    .gt(0)
)

print(f"Valid receipts: {len(valid)}")
print("Board (# = fair-price home-city purchase; . = absent):")
for line in board.to_numpy():
    print("".join("#" if bit else "." for bit in line))

fig, ax = plt.subplots(figsize=(16, 5))
ax.imshow(board.to_numpy(), cmap="Greys", vmin=0, vmax=1,
          interpolation="nearest", aspect="equal")
ax.set_xticks(range(21), labels=range(1, 22))
ax.set_yticks(range(6), labels=guide.sort_values("List_no")["Product"])
ax.set_xlabel("Pilot number")
ax.set_ylabel("Souvenir (office list order)")
ax.set_title("Christmas board — fair-price purchases in home cities")
ax.set_xticks([i - 0.5 for i in range(22)], minor=True)
ax.set_yticks([i - 0.5 for i in range(7)], minor=True)
ax.grid(which="minor", color="#cccccc", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)
fig.tight_layout()
fig.savefig("christmas_board.png", dpi=200, bbox_inches="tight")
