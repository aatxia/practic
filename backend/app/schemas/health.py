from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' if the service is healthy")
    app_env: str
    model_type: str
    features: dict[str, bool] = Field(
        ..., description="Which CV feature groups are enabled (hands/pose/face)"
    )
    ml_pipeline_status: str = Field(
        ...,
        description=(
            "'not_implemented' until Phase 6-10 land real CV/ML pipeline; "
            "'demo_mode' while running against synthetic data; 'ready' once a "
            "trained checkpoint is loaded."
        ),
    )
