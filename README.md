# Machine Risk Monitor

Local failure-risk screening for the AI4I 2020 predictive-maintenance dataset. The application
scores one machine state or a CSV batch, stores prediction history in SQLite, and shows recent
results in a small browser interface.

The output is decision support. It must not trigger maintenance work or machinery controls
without inspection by qualified personnel.

## Requirements

- Python 3.12
- Git
- Docker Desktop is optional

No API key, hosted service, GPU, or paid account is required.

## Native setup

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
prepare-data
```

`prepare-data` downloads the official UCI CSV, checks its SHA-256 checksum and schema, and
recreates the analysis files under `artifacts/eda/`. The fitted, evaluated model is already stored
under `artifacts/model/`.

Start the application:

```bash
run-app
```

Open `http://127.0.0.1:8000`. Prediction history is written to `data/predictions.db` and remains
local.

## Using the application

The single-reading form accepts:

- product type: `L`, `M`, or `H`
- air temperature: 295.3–304.5 K
- process temperature: 305.7–313.8 K
- rotational speed: 1,168–2,886 rpm
- torque: 3.8–76.6 Nm
- tool wear: 0–253 minutes

For batch scoring, upload a CSV with this exact header:

```csv
product_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min
```

An example is available at `examples/machines.csv`. Uploads are limited to 1,000 rows and 2 MB.
The scored file downloads as `scored_machines.csv`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime and model status |
| `POST` | `/api/predict` | Score one JSON record |
| `POST` | `/api/predict/batch` | Score an uploaded CSV |
| `GET` | `/api/predictions?limit=50` | Read recent local history |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "product_type": "L",
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.4,
    "rotational_speed_rpm": 1500,
    "torque_nm": 42.0,
    "tool_wear_min": 120
  }'
```

Invalid categories, out-of-range values, malformed files, and incorrect batch columns return HTTP
validation errors instead of being silently coerced.

## Tests and reproducibility

```bash
pytest
ruff check .
ruff format --check .
```

The complete modeling workflow is available through:

```bash
train-model
evaluate-model
```

`train-model` recreates the development/holdout split, cross-validates the candidates, and replaces
the saved artifact with an unevaluated model. `evaluate-model` selects the F2 threshold from
out-of-fold development predictions, evaluates the fixed model on the holdout, and restores the
validated artifact. Do not use repeated holdout runs to tune modeling decisions.

## Docker

```bash
docker build -t machine-risk-monitor:local .
docker run --rm -p 127.0.0.1:8000:8000 machine-risk-monitor:local
```

The container runs as an unprivileged user and includes a `/health` health check. Its SQLite history
is ephemeral unless `/app/data` is mounted to a local directory.

## Project files

- `src/machine_risk/`: data, modeling, evaluation, application, and persistence code
- `tests/`: data, artifact, metric, API, and persistence checks
- `artifacts/eda/`: dataset summaries and distributions
- `artifacts/model/`: fitted pipeline and candidate-selection results
- `artifacts/evaluation/`: holdout metrics and evaluation figures
- `REPORT.md`: method, results, discussion, and limitations
- `data/README.md`: dataset provenance and checksum

## Limitations

- The source is synthetic and cannot establish performance on a real production line.
- The validated cutoff favors recall, which increases the number of machines sent for inspection.
- The application does not explain failure causes or estimate remaining useful life.
- Inputs are checked against dataset ranges, but there is no live sensor integration or drift
  monitoring.
- There is no authentication because the application is intended for local, single-user use.
