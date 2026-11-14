"""SIGMOID - The Weak Layer. Submit this file as main.py.

Reads test-input.csv in the current working directory and writes test-output.txt.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def decode(df):
    label = "slab_released"
    if label not in df:
        raise ValueError(f"Missing target column: {label}")
    x = df.drop(columns=[label])
    y = df[label]
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x, y)

    nodes = tree.tree_
    letters = []

    def visit(node):
        feature_index = int(nodes.feature[node])
        if feature_index < 0:  # leaf
            return
        name = x.columns[feature_index]
        letters.append(name[0].upper())
        visit(int(nodes.children_left[node]))
        visit(int(nodes.children_right[node]))

    visit(0)
    return "SIGMOID_" + "".join(letters)


if __name__ == "__main__":
    data = pd.read_csv("test-input.csv")
    result = decode(data)
    with open("test-output.txt", "w", encoding="utf-8") as out:
        out.write(result + "\n")
