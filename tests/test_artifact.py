from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from machine_risk.artifact import (
    ARTIFACT_VERSION,
    ArtifactError,
    load_artifact,
    save_artifact,
)
from machine_risk.data import DATASET_SHA256, FEATURE_COLUMNS, ROOT, TARGET_COLUMN


def artifact() -> dict:
    model = DummyClassifier(strategy="prior").fit(np.zeros((4, 1)), [0, 0, 0, 1])
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": "dummy",
        "pipeline": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "dataset_sha256": DATASET_SHA256,
        "development_rows": 4,
        "holdout_rows": 1,
        "holdout_id_hash": "abc",
        "decision_threshold": None,
    }


def test_artifact_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "model.joblib"
    save_artifact(artifact(), destination)

    loaded = load_artifact(destination)

    assert loaded["artifact_version"] == ARTIFACT_VERSION
    assert loaded["feature_columns"] == FEATURE_COLUMNS


def test_missing_fields_are_rejected(tmp_path: Path) -> None:
    payload = artifact()
    del payload["holdout_id_hash"]

    with pytest.raises(ArtifactError, match="missing fields"):
        save_artifact(payload, tmp_path / "invalid.joblib")


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    payload = artifact()
    payload["decision_threshold"] = 1.2

    with pytest.raises(ArtifactError, match="threshold"):
        save_artifact(payload, tmp_path / "invalid.joblib")


def test_trained_artifact_returns_probability() -> None:
    trained = load_artifact(ROOT / "artifacts" / "model" / "model.joblib")
    sample = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 300.1,
                "Process temperature [K]": 310.4,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 42.0,
                "Tool wear [min]": 120,
            }
        ]
    )

    probability = trained["pipeline"].predict_proba(sample)[0, 1]

    assert 0 <= probability <= 1
