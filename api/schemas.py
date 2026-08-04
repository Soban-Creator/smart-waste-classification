"""
Pydantic response schemas for the prediction API.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    architecture: str
    class_names: List[str]
    image_size: List[int]
    saved_at: str | None = None
    framework: str | None = None
    pytorch_version: str | None = None
    torchvision_version: str | None = None


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    predicted_class_index: int
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    class_probabilities: Dict[str, float]
    gradcam_base64: str
    timestamp: str


class BatchPredictionItem(PredictionResponse):
    pass


class BatchPredictionResponse(BaseModel):
    total_files: int
    successful_predictions: int
    failed_predictions: int
    results: List[BatchPredictionItem]
    errors: List[dict]