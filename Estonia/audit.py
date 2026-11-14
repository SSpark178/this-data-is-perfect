"""Reproducible audit for SIGMOID: This Data Is Perfect!

Run in this directory with applications.csv and project_pipeline.py downloaded
from the challenge source. Requires pandas, numpy, scikit-learn.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def estimator():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))


def main():
    df = pd.read_csv("applications.csv")
    X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
    y = df["label"]
    identities = df["date_of_birth"].astype(str) + "|" + df["citizenship"].astype(str)

    # Shipped evaluation: leak information from the full data through scaling.
    X_scaled = StandardScaler().fit_transform(X)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1137
    )
    original = accuracy_score(y_te, KNeighborsClassifier(n_neighbors=1).fit(x_tr, y_tr).predict(x_te))

    # Decoy 1: fit scaler only to training partition.
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1137)
    train_only_scale = accuracy_score(y_te, estimator().fit(x_tr, y_tr).predict(x_te))

    # Decoy 2: stratify random row split by label.
    x_tr, x_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1137, stratify=y
    )
    stratified = accuracy_score(y_te, KNeighborsClassifier(n_neighbors=1).fit(x_tr, y_tr).predict(x_te))

    # Real fix: prevent any identity from appearing in both train and test.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1137).split(X, y, identities))
    held_out_identity = accuracy_score(
        y.iloc[te], estimator().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    )

    # Document the exact ambiguity in the task wording:
    # ungrouped LeaveOneOut leaks other rows for the same identity.
    # Kept as a diagnostic, NOT used to build the grouped flag.
    row_predictions = cross_val_predict(estimator(), X, y, cv=LeaveOneOut())
    row_loo = accuracy_score(y, row_predictions)

    # Every original row gets an out-of-fold prediction. All rows with its
    # identity are held out together, so there is no cross-identity leakage.
    predictions = cross_val_predict(
        estimator(), X, y, cv=LeaveOneGroupOut(), groups=identities
    )
    corrected = accuracy_score(y, predictions)

    print(f"Rows: {len(df)}; distinct identity groups: {identities.nunique()}")
    for name, score in [
        ("Original row split", original),
        ("Decoy 1 - fit scaler after split", train_only_scale),
        ("Decoy 2 - stratify row split", stratified),
        ("Identity-aware holdout", held_out_identity),
        ("Ungrouped row LeaveOneOut (diagnostic)", row_loo),
        ("Corrected LeaveOneGroupOut", corrected),
    ]:
        print(f"{name}: {score:.6f} ({score * 100:.2f}%)")
    print(f"Correct predictions: {(predictions == y).sum()} / {len(df)}")
    print(f"Proposed flag based on grouped evaluation: SIGMOID_{round(corrected * 100)}")


if __name__ == "__main__":
    main()
