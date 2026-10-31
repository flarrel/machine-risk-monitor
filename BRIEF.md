# Project 1 Brief

Status: approved; checkpoint 1 complete

## Working title

Machine Risk Monitor

## Problem

Build a local application that estimates milling-machine failure risk from current operating conditions. A user can score one machine state or upload a CSV batch, inspect the risk result, and review earlier predictions.

The application is advisory. It does not claim to replace maintenance inspection or control machinery automatically.

## Why this is project 1

This project forms a coherent classical machine-learning and software-development unit. It demonstrates Python, statistics, preprocessing, model training and evaluation, an application layer, persistence, testing, documentation, and reproducibility without requiring a GPU, paid API, subscription, or cloud service.

It intentionally leaves deep learning, PyTorch/TensorFlow, computer vision, NLP, LLM/RAG, data engineering, BI, cloud/Kubernetes, agents, and IoT for later mini-projects where they can be demonstrated naturally.

## Dataset

Use the UCI AI4I 2020 Predictive Maintenance Dataset (dataset ID 601):

- 10,000 records
- machine operating measurements and product type
- binary machine-failure target
- no missing values in the published dataset
- approximately 510 KB
- CC BY 4.0
- synthetic data designed to reflect an industrial predictive-maintenance setting

The model inputs will be product type, air temperature, process temperature, rotational speed, torque, and tool wear.

Exclude identifiers and all target-derived failure-mode columns from model inputs. In particular, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` must never enter the feature pipeline. This prevents direct target leakage.

## User-facing scope

- Single-record risk form
- Failure probability and configurable risk band
- Batch CSV scoring with downloadable results
- Prediction-history table and summary chart
- Basic health/status page
- Clear statement that the result is decision support, not an automatic maintenance command

Keep the interface plain and internal-looking. Do not mention portfolios, internship criteria, assessments, SDGs, or AI-generated development.

## ML pipeline

1. Fetch and cache the dataset from its official source.
2. Validate column names, types, allowed product categories, and numeric ranges.
3. Produce reproducible EDA and class-distribution summaries.
4. Split data with stratification while keeping the final test set untouched.
5. Build preprocessing and model steps as scikit-learn pipelines.
6. Compare a class-weighted logistic-regression baseline with a nonlinear tree ensemble.
7. Select the model using cross-validated average precision and supporting metrics.
8. Choose the operating threshold on validation results, not on the test set.
9. Evaluate once on the held-out test set.
10. Save the fitted pipeline, threshold, schema, and metrics as versioned artifacts.

Because failures are the minority class, plain accuracy is not the decision metric. Report average precision, precision, recall, F1/F2, ROC-AUC, confusion matrix, and class prevalence. The selected model must beat a dummy baseline on the untouched test set; any performance limitation remains visible.

## Application architecture

```text
UCI CSV -> validation/EDA -> sklearn training/evaluation -> model artifact
                                                        |
browser -> FastAPI + server-rendered UI -> predictor ----+-> SQLite history
                       |
                       +-> single and batch REST endpoints
```

## Stack

- Python 3.12
- pandas and NumPy
- scikit-learn and joblib
- FastAPI, Pydantic, Jinja2, and Uvicorn
- SQLAlchemy with SQLite
- Matplotlib for saved analysis/evaluation figures
- pytest and HTTPX
- Docker for a reproducible local runtime
- Git with short lowercase milestone commits

No React, Node.js, external database server, cloud platform, hosted model, telemetry, or paid service is needed.

## API outline

- `GET /health`
- `POST /api/predict`
- `POST /api/predict/batch`
- `GET /api/predictions`

The HTML interface uses the same application services as the REST endpoints. Predictions and relevant input metadata are stored locally in SQLite.

## Market signals demonstrated

| Signal | Evidence in this project |
| --- | --- |
| Python | Data, training, application, persistence, and tests |
| Machine Learning / AI | Reproducible supervised failure-risk model |
| Statistics | Distribution analysis, class prevalence, metric interpretation, and threshold tradeoffs |
| Analytical Thinking / Problem Solving | Leakage prevention, imbalance handling, baseline comparison, and explicit tradeoffs |
| Software Development / Programming | Modular application with defined boundaries and reproducible commands |
| Scikit-learn | Preprocessing/model pipelines, cross-validation, fitting, and evaluation |
| Data Analysis / EDA | Reproducible summaries and figures derived from the raw dataset |
| Data Processing / Preprocessing | Schema validation, encoding, feature selection, and train/evaluation separation |
| Model Evaluation | Baseline comparison, held-out testing, threshold selection, and multiple metrics |
| Data Visualization / Dashboards | Evaluation figures and local prediction-history summary |
| SQL / Databases | SQLite prediction history accessed through SQLAlchemy |
| API / REST API | Typed single and batch prediction endpoints |
| Testing / QA | Unit and integration tests for data, model contract, persistence, and HTTP endpoints |
| Docker / Containerization | Locally runnable application image |
| Documentation | Setup/manual, model report, limitations, and citations |
| Git / Version Control | Five coherent checkpoint commits |

These signals may recur in later projects when appropriate. Their presence here does not reserve them exclusively for project 1.

## Verification and acceptance

- Raw-data retrieval is reproducible and source attribution is retained.
- Leakage columns are rejected by automated checks.
- Training produces a loadable model artifact and machine-readable metrics.
- Evaluation uses an untouched test set and compares against a dummy baseline.
- The application supports valid single and batch predictions.
- Invalid categories, malformed CSVs, and out-of-range values return clear errors.
- Prediction history persists in SQLite.
- Tests cover core pipeline and API behavior.
- The local Docker image builds and runs if Docker is installed; native local execution remains available.
- The manual explains setup, training, running, testing, and limitations.
- The report follows the assessment's useful structure: background, problem definition, proposed solution, results, discussion, limitations/ethics, and conclusion, with at least five credible citations.

## Implementation checkpoints

| Checkpoint | Scope | Commit subject |
| --- | --- | --- |
| 1 | Repository setup, data retrieval, schema validation, initial EDA | `data` |
| 2 | Preprocessing, baselines, candidate models, artifact contract | `model` |
| 3 | FastAPI application, SQLite history, local interface, batch scoring | `app` |
| 4 | Evaluation, thresholding, tests, Docker verification, fixes | `eval` or `fix` |
| 5 | Manual, structured report, final end-to-end verification | `docs` |

The user triggers checkpoint 1 with `build` and each later checkpoint with `continue`. There is no fixed timing requirement.

## Out of scope

- Real-time sensor ingestion
- Automatic machine shutdown or maintenance scheduling
- Failure-cause diagnosis from target-derived columns
- Deep learning or GPU training
- Hosted deployment or public demo
- Authentication, multi-user administration, or enterprise infrastructure
- Claims that a synthetic benchmark proves production readiness

## Authorization gate

Do not implement this brief until the user explicitly says `build` or `go ahead`.
