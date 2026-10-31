import pandas as pd

from machine_risk.data import FEATURE_COLUMNS, TARGET_COLUMN
from machine_risk.modeling import holdout_id_hash, preprocessor, split_dataset


def frame(rows: int = 100) -> pd.DataFrame:
    records = []
    for index in range(rows):
        failed = int(index % 10 == 0)
        records.append(
            {
                "UDI": index + 1,
                "Product ID": f"{['L', 'M', 'H'][index % 3]}{index:05d}",
                "Type": ["L", "M", "H"][index % 3],
                "Air temperature [K]": 298 + index / rows,
                "Process temperature [K]": 308 + index / rows,
                "Rotational speed [rpm]": 1400 + index,
                "Torque [Nm]": 35 + index / 10,
                "Tool wear [min]": index * 2,
                TARGET_COLUMN: failed,
            }
        )
    return pd.DataFrame(records)


def test_split_is_deterministic_and_disjoint() -> None:
    first_development, first_holdout = split_dataset(frame())
    second_development, second_holdout = split_dataset(frame())

    assert len(first_development) == 80
    assert len(first_holdout) == 20
    assert set(first_development["UDI"]).isdisjoint(first_holdout["UDI"])
    assert holdout_id_hash(first_holdout) == holdout_id_hash(second_holdout)
    assert first_development.index.equals(second_development.index)


def test_preprocessor_uses_only_declared_features() -> None:
    data = frame()
    transformer = preprocessor(scale=True).fit(data[FEATURE_COLUMNS])

    output_names = set(transformer.get_feature_names_out())

    assert "Type_L" in output_names
    assert TARGET_COLUMN not in output_names
    assert "UDI" not in output_names
