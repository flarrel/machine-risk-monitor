from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from machine_risk.predictor import PredictionResult
from machine_risk.schemas import MachineInput


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    source: Mapped[str] = mapped_column(String(16))
    product_type: Mapped[str] = mapped_column(String(1))
    air_temperature_k: Mapped[float] = mapped_column(Float)
    process_temperature_k: Mapped[float] = mapped_column(Float)
    rotational_speed_rpm: Mapped[int] = mapped_column(Integer)
    torque_nm: Mapped[float] = mapped_column(Float)
    tool_wear_min: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    threshold: Mapped[float] = mapped_column(Float)
    threshold_status: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(64))

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "product_type": self.product_type,
            "air_temperature_k": self.air_temperature_k,
            "process_temperature_k": self.process_temperature_k,
            "rotational_speed_rpm": self.rotational_speed_rpm,
            "torque_nm": self.torque_nm,
            "tool_wear_min": self.tool_wear_min,
            "probability": self.probability,
            "risk_band": self.risk_band,
            "threshold": self.threshold,
            "threshold_status": self.threshold_status,
            "model_name": self.model_name,
        }


class PredictionStore:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def add(self, sample: MachineInput, result: PredictionResult, source: str = "single") -> dict:
        return self.add_many([(sample, result)], source)[0]

    def add_many(
        self, predictions: list[tuple[MachineInput, PredictionResult]], source: str = "batch"
    ) -> list[dict]:
        records = [self._record(sample, result, source) for sample, result in predictions]
        with Session(self.engine, expire_on_commit=False) as session:
            session.add_all(records)
            session.commit()
        return [record.as_dict() for record in records]

    def recent(self, limit: int = 50) -> list[dict]:
        with self.sessions() as session:
            records = session.scalars(
                select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)
            ).all()
        return [record.as_dict() for record in records]

    @staticmethod
    def _record(sample: MachineInput, result: PredictionResult, source: str) -> PredictionRecord:
        return PredictionRecord(
            source=source,
            **sample.model_dump(),
            probability=result.probability,
            risk_band=result.risk_band,
            threshold=result.threshold,
            threshold_status=result.threshold_status,
            model_name=result.model_name,
        )
