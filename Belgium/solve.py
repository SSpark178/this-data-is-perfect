"""Belgium — Fry-ndship: field-edge regression and diagnostic figures.

Run in this folder after downloading potato_fields.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    directory = Path(__file__).resolve().parent
    data = pd.read_csv(directory / "potato_fields.csv")
    # Use ONLY variables available in the farmers' declarations.
    predictors = ["area_ha", "long_side_m", "ditch_m"]
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    reg = LinearRegression().fit(train[predictors], train["edge_m"])
    y_true = test["edge_m"].to_numpy()
    y_pred = reg.predict(test[predictors])
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Inspect the measurement context without letting site measurements or
    # post-survey contractor quotes leak into the training input.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(y_true, y_pred, s=9, alpha=.55)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set(xlabel="Measured edge (m)", ylabel="Predicted edge (m)",
                title=f"Holdout predictions: R²={r2:.5f}, MAE={mae:.2f} m")
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, s=9, alpha=.55)
    axes[1].axhline(0, color="red", ls="--")
    axes[1].set(xlabel="Predicted edge (m)", ylabel="Residual (m)",
                title="Out-of-sample residuals")
    axes[2].scatter(data["lon"], data["lat"], s=3, alpha=.5)
    axes[2].set(xlabel="Longitude", ylabel="Latitude",
                title="Survey locations (all 3,869 fields)")
    fig.suptitle("Fry-ndship — field-edge modeling diagnostics")
    fig.tight_layout()
    fig.savefig(directory / "model_evidence.png", dpi=180)
    fig.savefig(directory / "model_evidence.svg")

    # Why the quote must not be an input to a pre-survey estimator:
    quote = LinearRegression().fit(data[["edge_m"]], data["fence_quote_eur"])
    print("Fields:", len(data))
    print("Regression features:", predictors)
    print(f"Holdout R²: {r2:.9f}")
    print(f"Holdout MAE (m): {mae:.6f}")
    print(f"Quote from measured edge: EUR {quote.coef_[0]:.5f} * edge_m"
          f" + {quote.intercept_:.5f}")
    print(f"Quote relation R²: {quote.score(data[['edge_m']], data['fence_quote_eur']):.12f}")
    print("No puzzle word has been demonstrated from these diagnostics.")


if __name__ == "__main__":
    main()
