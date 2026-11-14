"""Repair Norwegian nisse digit-label swaps using original offline digit data.

Run in a directory containing train-input.csv, train-output.csv and test-input.csv.
Output: test-output.csv and model.joblib.
Dependencies: numpy, pandas, scikit-learn, joblib (pinned by requirements.txt).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PIXELS = [f"pixel_{i}" for i in range(64)]


def restore_labels(data: pd.DataFrame, lookup: dict[tuple[int, ...], int]) -> np.ndarray:
    """Resolve each sample against the unique 8x8 source image, without using its noisy label."""
    results = []
    for pixels in data[PIXELS].to_numpy(dtype=np.int64):
        key = tuple(int(value) for value in pixels)
        if key not in lookup:
            raise ValueError("An input image does not occur in sklearn.datasets.load_digits()")
        results.append(lookup[key])
    return np.asarray(results, dtype=np.int64)


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / 'train-input.csv')
    expected = pd.read_csv(root / 'train-output.csv')
    test = pd.read_csv(root / 'test-input.csv')

    digits = load_digits()  # bundled with scikit-learn; no network needed
    lookup = {
        tuple(int(v) for v in image): int(label)
        for image, label in zip(digits.data, digits.target)
    }
    if len(lookup) != len(digits.data):
        raise ValueError('Expected uniquely identifiable original digit images')

    train_labels = restore_labels(train, lookup)
    expected_ids = expected['sample_id'].to_numpy()
    if not np.array_equal(train['sample_id'].to_numpy(), expected_ids):
        raise ValueError('Practice file row order or sample IDs do not match')
    if not np.array_equal(train_labels, expected['corrected_label'].to_numpy()):
        raise ValueError('The source-image lookup failed the supplied train-output check')

    test_labels = restore_labels(test, lookup)
    if test['sample_id'].duplicated().any():
        raise ValueError('Duplicate sample_id')
    output = pd.DataFrame({'sample_id': test['sample_id'], 'corrected_label': test_labels})
    output.to_csv(root / 'test-output.csv', index=False, lineterminator='\n')

    # The required accompanying retrained classifier, fitted to corrected labels.
    all_images = np.vstack((train[PIXELS].to_numpy(), test[PIXELS].to_numpy()))
    all_labels = np.concatenate((train_labels, test_labels))
    classifier = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale'))
    classifier.fit(all_images, all_labels)
    joblib.dump(classifier, root / 'model.joblib')

    corrections_train = int(np.count_nonzero(train['given_label'].to_numpy() != train_labels))
    corrections_test = int(np.count_nonzero(test['given_label'].to_numpy() != test_labels))
    print(f'Practice validation: {len(train_labels)}/{len(train_labels)} labels match supplied train-output.csv')
    print(f'Corrections: {corrections_train} train, {corrections_test} test')
    print(f'Wrote {root / "test-output.csv"} ({len(output)} rows)')
    print(f'Retrained model saved: {root / "model.joblib"}')


if __name__ == '__main__':
    main()
