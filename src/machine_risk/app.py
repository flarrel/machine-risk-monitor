from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from machine_risk.database import PredictionStore
from machine_risk.predictor import Predictor
from machine_risk.schemas import BATCH_COLUMNS, MachineInput
from machine_risk.train import MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
MAX_BATCH_ROWS = 1_000
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def create_app(
    model_path: Path = MODEL_PATH,
    database_url: str | None = None,
) -> FastAPI:
    predictor = Predictor(model_path)
    database_url = database_url or os.getenv(
        "DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'predictions.db'}"
    )
    store = PredictionStore(database_url)
    templates = Jinja2Templates(directory=ROOT / "templates")

    app = FastAPI(title="Machine Risk Monitor", version="0.1.0")
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"threshold_status": predictor.threshold_status},
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": predictor.artifact["model_name"],
            "threshold_status": predictor.threshold_status,
        }

    @app.post("/api/predict", status_code=201)
    def predict(sample: MachineInput) -> dict:
        result = predictor.predict(sample)
        return store.add(sample, result)

    @app.get("/api/predictions")
    def predictions(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
        return store.recent(limit)

    @app.post("/api/predict/batch")
    async def predict_batch(file: UploadFile) -> StreamingResponse:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="CSV exceeds the 2 MB upload limit")

        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as error:
            raise HTTPException(status_code=422, detail="unable to read CSV") from error

        missing = [column for column in BATCH_COLUMNS if column not in frame.columns]
        extra = [column for column in frame.columns if column not in BATCH_COLUMNS]
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={"missing_columns": missing, "extra_columns": extra},
            )
        if frame.empty:
            raise HTTPException(status_code=422, detail="CSV contains no rows")
        if len(frame) > MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=422, detail=f"CSV exceeds the {MAX_BATCH_ROWS}-row limit"
            )

        scored = []
        for index, row in frame.iterrows():
            try:
                sample = MachineInput.model_validate(row.to_dict())
            except ValidationError as error:
                raise HTTPException(
                    status_code=422,
                    detail={"row": int(index) + 2, "errors": error.errors()},
                ) from error
            scored.append((sample, predictor.predict(sample)))

        records = store.add_many(scored)
        output = frame.copy()
        output["failure_probability"] = [record["probability"] for record in records]
        output["risk_band"] = [record["risk_band"] for record in records]
        output["threshold_status"] = [record["threshold_status"] for record in records]
        csv = output.to_csv(index=False)

        return StreamingResponse(
            iter([csv]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="scored_machines.csv"'},
        )

    return app


def main() -> None:
    uvicorn.run("machine_risk.app:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
