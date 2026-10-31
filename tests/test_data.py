import pandas as pd
import pytest

from machine_risk.data import (
    EXPECTED_COLUMNS,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    validate_dataset,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for product_type in ["L", "M", "H"]:
        row = {
            "UDI": len(rows) + 1,
            "Product ID": f"{product_type}00001",
            "Type": product_type,
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
            "Machine failure": 0,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_valid_dataset() -> None:
    validate_dataset(sample_frame(), expected_rows=3)


def test_missing_column_is_rejected() -> None:
    frame = sample_frame().drop(columns="Torque [Nm]")

    with pytest.raises(DataValidationError, match="unexpected schema"):
        validate_dataset(frame)


def test_missing_value_is_rejected() -> None:
    frame = sample_frame()
    frame.loc[0, "Torque [Nm]"] = None

    with pytest.raises(DataValidationError, match="missing values"):
        validate_dataset(frame)


def test_features_exclude_targets() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS).isdisjoint(FAILURE_MODE_COLUMNS)
