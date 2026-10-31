from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from machine_risk.app import create_app
from machine_risk.data import ROOT


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(database_url=f"sqlite:///{tmp_path / 'predictions.db'}")
    return TestClient(app)


def sample() -> dict:
    return {
        "product_type": "L",
        "air_temperature_k": 300.1,
        "process_temperature_k": 310.4,
        "rotational_speed_rpm": 1500,
        "torque_nm": 42.0,
        "tool_wear_min": 120,
    }


def test_home_and_health(client: TestClient) -> None:
    home = client.get("/")
    health = client.get("/health")

    assert home.status_code == 200
    assert "Machine Risk Monitor" in home.text
    assert health.json() == {
        "status": "ok",
        "model": "random_forest",
        "threshold_status": "provisional",
    }


def test_prediction_is_persisted(client: TestClient) -> None:
    response = client.post("/api/predict", json=sample())

    assert response.status_code == 201
    prediction = response.json()
    assert 0 <= prediction["probability"] <= 1
    assert prediction["threshold_status"] == "provisional"

    history = client.get("/api/predictions").json()
    assert len(history) == 1
    assert history[0]["id"] == prediction["id"]


def test_out_of_range_input_is_rejected(client: TestClient) -> None:
    payload = sample()
    payload["tool_wear_min"] = 999

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422
    assert client.get("/api/predictions").json() == []


def test_batch_returns_scored_csv_and_persists_rows(client: TestClient) -> None:
    csv = (
        "product_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,"
        "torque_nm,tool_wear_min\n"
        "L,300.1,310.4,1500,42.0,120\n"
        "H,302.0,311.2,1400,55.0,205\n"
    )

    response = client.post(
        "/api/predict/batch",
        files={"file": ("machines.csv", BytesIO(csv.encode()), "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "failure_probability" in response.text
    assert len(client.get("/api/predictions").json()) == 2


def test_batch_schema_is_enforced(client: TestClient) -> None:
    response = client.post(
        "/api/predict/batch",
        files={"file": ("machines.csv", BytesIO(b"product_type\nL\n"), "text/csv")},
    )

    assert response.status_code == 422
    assert "missing_columns" in response.text


def test_default_model_artifact_exists() -> None:
    assert (ROOT / "artifacts" / "model" / "model.joblib").is_file()
