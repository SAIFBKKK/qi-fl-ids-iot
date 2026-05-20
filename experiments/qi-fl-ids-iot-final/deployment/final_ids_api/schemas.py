from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="12 selected scaled features or 28 original scaled features")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    rows: list[list[float]]
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    prediction: int
    probability_attack: float
    label: str
    model_id: str
    selected_features_count: int
    threshold: float


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_id: str
    count: int
