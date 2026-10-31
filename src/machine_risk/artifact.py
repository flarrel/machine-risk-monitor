from pathlib import Path

import joblib

from machine_risk.data import FEATURE_COLUMNS, TARGET_COLUMN

ARTIFACT_VERSION = 1
REQUIRED_KEYS = {
    "artifact_version",
    "model_name",
    "pipeline",
    "feature_columns",
    "target_column",
    "dataset_sha256",
    "development_rows",
    "holdout_rows",
    "holdout_id_hash",
    "decision_threshold",
}


class ArtifactError(ValueError):
    pass


def validate_artifact(artifact: dict) -> None:
    missing = REQUIRED_KEYS - artifact.keys()
    if missing:
        raise ArtifactError(f"artifact is missing fields: {sorted(missing)}")
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactError(f"unsupported artifact version: {artifact['artifact_version']}")
    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ArtifactError("artifact feature columns do not match the application schema")
    if artifact["target_column"] != TARGET_COLUMN:
        raise ArtifactError("artifact target does not match the application schema")
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ArtifactError("artifact pipeline must provide predict_proba")
    threshold = artifact["decision_threshold"]
    if threshold is not None and not 0 < threshold < 1:
        raise ArtifactError("decision threshold must be between zero and one")


def save_artifact(artifact: dict, destination: Path) -> None:
    validate_artifact(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)


def load_artifact(source: Path) -> dict:
    artifact = joblib.load(source)
    if not isinstance(artifact, dict):
        raise ArtifactError("artifact payload must be a dictionary")
    validate_artifact(artifact)
    return artifact
