from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from machine_risk.data import FEATURE_COLUMNS, TARGET_COLUMN

RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
CATEGORICAL_FEATURES = ["Type"]


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development, holdout = train_test_split(
        frame,
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET_COLUMN],
    )
    return development.sort_index(), holdout.sort_index()


def holdout_id_hash(holdout: pd.DataFrame) -> str:
    values = ",".join(str(value) for value in sorted(holdout["UDI"].astype(int)))
    return sha256(values.encode()).hexdigest()


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_step, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline(
            [
                ("preprocess", preprocessor(scale=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessor(scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def compare_candidates(development: pd.DataFrame) -> tuple[dict, str]:
    features = development[FEATURE_COLUMNS]
    target = development[TARGET_COLUMN]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    results = {}

    for name, pipeline in candidates().items():
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        results[name] = {
            metric: {
                "mean": round(float(np.mean(scores[f"test_{metric}"])), 6),
                "std": round(float(np.std(scores[f"test_{metric}"])), 6),
                "folds": [round(float(value), 6) for value in scores[f"test_{metric}"]],
            }
            for metric in scoring
        }

    eligible = {name: value for name, value in results.items() if name != "dummy"}
    selected = max(eligible, key=lambda name: eligible[name]["average_precision"]["mean"])
    return results, selected
