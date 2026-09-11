"""Request and response shapes. These drive validation and the /docs page."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """JSON body for POST /predict."""

    image_url: str = Field(
        ...,
        description="Public http(s) URL of a chest X-ray image (JPEG or PNG).",
        examples=["https://example.com/chest-xray.jpeg"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"image_url": "https://example.com/chest-xray.jpeg"}]
        }
    }


class ClassProbabilities(BaseModel):
    NORMAL: float
    PNEUMONIA: float


class PredictResponse(BaseModel):
    """Result returned by both prediction endpoints."""

    prediction: str = Field(description="NORMAL or PNEUMONIA.")
    label_index: int = Field(description="0 for NORMAL, 1 for PNEUMONIA.")
    probability_pneumonia: float = Field(
        description="Sigmoid output of the model: P(PNEUMONIA), between 0 and 1."
    )
    confidence: float = Field(
        description="Probability assigned to the predicted class."
    )
    class_probabilities: ClassProbabilities
    threshold: float = Field(
        description="Decision threshold applied to probability_pneumonia."
    )
    model_version: str
    inference_ms: float

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "PNEUMONIA",
                    "label_index": 1,
                    "probability_pneumonia": 0.973214,
                    "confidence": 0.973214,
                    "class_probabilities": {"NORMAL": 0.026786, "PNEUMONIA": 0.973214},
                    "threshold": 0.5,
                    "model_version": "resnet18-finetuned-a1",
                    "inference_ms": 84.21,
                }
            ]
        },
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    threshold: float

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    detail: str
