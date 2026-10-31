from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class MachineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(ge=295.3, le=304.5)
    process_temperature_k: float = Field(ge=305.7, le=313.8)
    rotational_speed_rpm: int = Field(ge=1168, le=2886)
    torque_nm: float = Field(ge=3.8, le=76.6)
    tool_wear_min: int = Field(ge=0, le=253)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Type": self.product_type,
                    "Air temperature [K]": self.air_temperature_k,
                    "Process temperature [K]": self.process_temperature_k,
                    "Rotational speed [rpm]": self.rotational_speed_rpm,
                    "Torque [Nm]": self.torque_nm,
                    "Tool wear [min]": self.tool_wear_min,
                }
            ]
        )


BATCH_COLUMNS = list(MachineInput.model_fields)
