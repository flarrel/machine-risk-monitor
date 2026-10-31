from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from machine_risk.data import FEATURE_COLUMNS, ROOT, TARGET_COLUMN, load_dataset

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from matplotlib.figure import Figure

OUTPUT_DIR = ROOT / "artifacts" / "eda"
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]


def build_summary(frame: pd.DataFrame) -> dict:
    target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    numeric = frame[NUMERIC_FEATURES].describe().round(3)

    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
        "product_type_counts": frame["Type"].value_counts().sort_index().to_dict(),
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "failure_prevalence": round(float(frame[TARGET_COLUMN].mean()), 6),
        "numeric_summary": numeric.to_dict(),
    }


def plot_class_balance(frame: pd.DataFrame, destination: Path) -> None:
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    labels = ["normal", "failure"]

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    bars = ax.bar(labels, counts.values, color=["#5f7c8a", "#ba5a45"])
    ax.set(title="Machine failure class balance", ylabel="records")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)


def plot_feature_distributions(frame: pd.DataFrame, destination: Path) -> None:
    fig = Figure(figsize=(12, 7))
    axes = fig.subplots(2, 3)

    for ax, column in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        for value, label, color in [(0, "normal", "#5f7c8a"), (1, "failure", "#ba5a45")]:
            frame.loc[frame[TARGET_COLUMN] == value, column].plot(
                kind="hist",
                bins=25,
                density=True,
                alpha=0.55,
                label=label,
                color=color,
                ax=ax,
            )
        ax.set_title(column.replace(" [", "\n["))
        ax.set_ylabel("density")

    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    frame = load_dataset()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(frame)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_class_balance(frame, output_dir / "class_balance.png")
    plot_feature_distributions(frame, output_dir / "feature_distributions.png")
    return summary


def main() -> None:
    summary = run()
    print(
        f"prepared {summary['rows']} rows; failure prevalence={summary['failure_prevalence']:.2%}"
    )


if __name__ == "__main__":
    main()
