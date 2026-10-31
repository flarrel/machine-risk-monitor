import numpy as np
import pandas as pd

from machine_risk.modeling import classification_metrics, select_f2_threshold


def test_threshold_is_selected_from_probabilities() -> None:
    target = pd.Series([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.4, 0.55, 0.9])

    threshold, score = select_f2_threshold(target, probabilities)

    assert threshold == 0.55
    assert score == 1


def test_classification_metrics_include_requested_measures() -> None:
    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = classification_metrics(target, probabilities, threshold=0.5)

    assert metrics["average_precision"] == 0.833333
    assert metrics["precision"] == 0.666667
    assert metrics["recall"] == 1
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
