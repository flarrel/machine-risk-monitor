from __future__ import annotations

import json
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from machine_risk.data import FEATURE_COLUMNS, ROOT, TARGET_COLUMN, load_dataset

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from matplotlib.figure import Figure

from machine_risk.artifact import load_artifact, save_artifact
from machine_risk.modeling import (
    classification_metrics,
    holdout_id_hash,
    out_of_fold_probabilities,
    select_f2_threshold,
    split_dataset,
)
from machine_risk.train import MODEL_PATH

EVALUATION_DIR = ROOT / "artifacts" / "evaluation"
REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"


def save_figures(target, probabilities: np.ndarray, metrics: dict) -> None:
    precision, recall, _ = precision_recall_curve(target, probabilities)

    figure = Figure(figsize=(6.4, 4.2), layout="constrained")
    axis = figure.subplots()
    axis.plot(recall, precision, color="#315f52", linewidth=2)
    axis.axhline(target.mean(), color="#9b6a45", linestyle="--", label="prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Holdout precision-recall curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend()
    figure.savefig(EVALUATION_DIR / "precision_recall.png", dpi=160)

    matrix = metrics["confusion_matrix"]
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    figure = Figure(figsize=(5.2, 4.2), layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(values, cmap="Greens")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No failure", "Failure"],
        yticklabels=["No failure", "Failure"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Holdout confusion matrix",
    )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=160)


def evaluate() -> dict:
    frame = load_dataset()
    development, holdout = split_dataset(frame)
    artifact = load_artifact(MODEL_PATH)
    if artifact["holdout_id_hash"] != holdout_id_hash(holdout):
        raise ValueError("holdout partition does not match the training artifact")

    oof_probabilities = out_of_fold_probabilities(artifact["pipeline"], development)
    threshold, development_f2 = select_f2_threshold(development[TARGET_COLUMN], oof_probabilities)

    holdout_target = holdout[TARGET_COLUMN]
    holdout_probabilities = artifact["pipeline"].predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    selected_metrics = classification_metrics(holdout_target, holdout_probabilities, threshold)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(development[FEATURE_COLUMNS], development[TARGET_COLUMN])
    dummy_probabilities = dummy.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    baseline = {
        "average_precision": round(
            float(average_precision_score(holdout_target, dummy_probabilities)), 6
        ),
        "roc_auc": round(float(roc_auc_score(holdout_target, dummy_probabilities)), 6),
    }

    report = {
        "selected_model": artifact["model_name"],
        "threshold_selection": {
            "data": "five-fold out-of-fold development predictions",
            "objective": "maximum F2",
            "threshold": round(threshold, 6),
            "development_f2": round(development_f2, 6),
        },
        "holdout": selected_metrics,
        "dummy_holdout": baseline,
        "holdout_id_hash": artifact["holdout_id_hash"],
        "limitations": [
            "The dataset is synthetic and does not establish production performance.",
            "The threshold prioritizes recall through F2 and may create avoidable inspections.",
            "Probabilities are not a substitute for maintenance judgment or safety controls.",
        ],
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    save_figures(holdout_target, holdout_probabilities, selected_metrics)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    artifact["decision_threshold"] = threshold
    save_artifact(artifact, MODEL_PATH)

    training_report_path = MODEL_PATH.with_name("training_report.json")
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    training_report["holdout_evaluated"] = True
    training_report["decision_threshold"] = round(threshold, 6)
    training_report_path.write_text(
        json.dumps(training_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    report = evaluate()
    metrics = report["holdout"]
    print(
        f"holdout average precision={metrics['average_precision']:.3f}; "
        f"recall={metrics['recall']:.3f}; precision={metrics['precision']:.3f}"
    )


if __name__ == "__main__":
    main()
