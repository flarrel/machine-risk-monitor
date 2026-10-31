from __future__ import annotations

import shutil
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
DATASET_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "ai4i2020.csv"

ID_COLUMNS = ["UDI", "Product ID"]
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EXPECTED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS


class DataValidationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(destination: Path = RAW_CSV) -> Path:
    if destination.exists():
        if file_sha256(destination) != DATASET_SHA256:
            raise DataValidationError(f"checksum mismatch for {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "ai4i2020.zip"

    with (
        urllib.request.urlopen(DATASET_URL, timeout=30) as response,
        archive.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)

    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise DataValidationError(f"expected one CSV in archive, found {len(members)}")
        with bundle.open(members[0]) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)

    archive.unlink()
    if file_sha256(destination) != DATASET_SHA256:
        destination.unlink()
        raise DataValidationError("downloaded dataset checksum does not match the expected source")
    return destination


def validate_dataset(frame: pd.DataFrame, expected_rows: int | None = None) -> None:
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        missing = [column for column in EXPECTED_COLUMNS if column not in columns]
        extra = [column for column in columns if column not in EXPECTED_COLUMNS]
        raise DataValidationError(f"unexpected schema; missing={missing}, extra={extra}")

    if expected_rows is not None and len(frame) != expected_rows:
        raise DataValidationError(f"expected {expected_rows} rows, found {len(frame)}")
    if frame.isna().any().any():
        raise DataValidationError("dataset contains missing values")
    if set(frame["Type"].unique()) != {"L", "M", "H"}:
        raise DataValidationError("Type must contain L, M, and H")

    binary_columns = [TARGET_COLUMN, *FAILURE_MODE_COLUMNS]
    for column in binary_columns:
        values = set(frame[column].unique())
        if not values <= {0, 1}:
            raise DataValidationError(f"{column} contains non-binary values: {values}")


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    source = path or download_dataset()
    frame = pd.read_csv(source)
    validate_dataset(frame, expected_rows=10_000 if path is None else None)
    return frame
