from dataclasses import dataclass
from pathlib import Path

from machine_risk.artifact import load_artifact
from machine_risk.schemas import MachineInput


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    risk_band: str
    threshold: float
    threshold_status: str
    model_name: str


class Predictor:
    def __init__(self, model_path: Path):
        self.artifact = load_artifact(model_path)
        configured = self.artifact["decision_threshold"]
        self.threshold = float(configured) if configured is not None else 0.5
        self.threshold_status = "validated" if configured is not None else "provisional"

    def predict(self, sample: MachineInput) -> PredictionResult:
        probability = float(self.artifact["pipeline"].predict_proba(sample.as_frame())[0, 1])
        if probability >= self.threshold:
            risk_band = "high"
        elif probability >= self.threshold / 2:
            risk_band = "watch"
        else:
            risk_band = "low"

        return PredictionResult(
            probability=probability,
            risk_band=risk_band,
            threshold=self.threshold,
            threshold_status=self.threshold_status,
            model_name=self.artifact["model_name"],
        )
