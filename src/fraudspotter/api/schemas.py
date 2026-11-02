from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NewsTextRequest(BaseModel):
    title: Optional[str] = Field(default="", description="News article headline/title")
    text: str = Field(..., min_length=5, description="Full text body of the news article")


class PredictionResponse(BaseModel):
    label: str = Field(..., description="'FAKE' or 'REAL'")
    prediction_class: int = Field(..., description="0 for FAKE, 1 for REAL")
    probability_fake: float = Field(..., ge=0.0, le=1.0)
    probability_real: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str = Field(..., description="Model identifier used for prediction")


class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    method: str = Field(..., description="'LIME' or 'IntegratedGradients' or 'FeatureCoefficients'")
    attributions: List[Dict[str, Any]] = Field(..., description="List of token or feature weights")


class HealthResponse(BaseModel):
    status: str
    version: str
    active_model: str
