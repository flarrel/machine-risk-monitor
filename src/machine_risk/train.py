from __future__ import annotations

import json

from machine_risk.artifact import ARTIFACT_VERSION, save_artifact
from machine_risk.data import (
    DATASET_SHA256,
    FEATURE_COLUMNS,
    ROOT,
    TARGET_COLUMN,
    load_dataset,
)
from machine_risk.modeling import candidates, compare_candidates, holdout_id_hash, split_dataset

MODEL_DIR = ROOT / "artifacts" / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"


def train() -> dict:
    frame = load_dataset()
    development, holdout = split_dataset(frame)
    results, selected_name = compare_candidates(development)

    pipeline = candidates()[selected_name]
    pipeline.fit(development[FEATURE_COLUMNS], development[TARGET_COLUMN])

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": selected_name,
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "dataset_sha256": DATASET_SHA256,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_id_hash": holdout_id_hash(holdout),
        "decision_threshold": None,
    }
    save_artifact(artifact, MODEL_PATH)

    report = {
        "selection_metric": "average_precision",
        "cross_validation_folds": 5,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_evaluated": False,
        "selected_model": selected_name,
        "candidates": results,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    report = train()
    selected = report["selected_model"]
    average_precision = report["candidates"][selected]["average_precision"]["mean"]
    print(f"selected {selected}; cross-validated average precision={average_precision:.3f}")
    print("holdout remains unevaluated")


if __name__ == "__main__":
    main()
