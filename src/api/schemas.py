"""Pydantic request/response models with range validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class SessionFeatures(BaseModel):
    """One browsing session (raw inputs exactly as in ``data_dictionary.csv``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 60.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 40,
                "ProductRelated_Duration": 1400.0,
                "BounceRates": 0.005,
                "ExitRates": 0.02,
                "PageValues": 35.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    )

    Administrative: int = Field(..., ge=0, le=500, description="Account/admin pages viewed")
    Administrative_Duration: float = Field(..., ge=0, description="Seconds on admin pages")
    Informational: int = Field(..., ge=0, le=500, description="Informational pages viewed")
    Informational_Duration: float = Field(..., ge=0, description="Seconds on informational pages")
    ProductRelated: int = Field(..., ge=0, le=5000, description="Product pages viewed")
    ProductRelated_Duration: float = Field(..., ge=0, description="Seconds on product pages")
    BounceRates: float = Field(..., ge=0, le=1, description="Avg bounce rate of pages visited")
    ExitRates: float = Field(..., ge=0, le=1, description="Avg exit rate of pages visited")
    PageValues: float = Field(..., ge=0, description="Avg page value of pages visited")
    SpecialDay: float = Field(..., ge=0, le=1, description="Closeness to a special day (0-1)")
    Month: Literal["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: int = Field(..., ge=1, description="OS code (categorical)")
    Browser: int = Field(..., ge=1, description="Browser code (categorical)")
    Region: int = Field(..., ge=1, description="Region code (categorical)")
    TrafficType: int = Field(..., ge=1, description="Traffic source code (categorical)")
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool


class Contributor(BaseModel):
    feature: str
    value: float | int | str | bool | None
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    purchase_probability: float
    prediction: int
    threshold: float
    confidence: Literal["high", "medium", "low"]
    top_contributors: list[Contributor] | None = None


class BatchRequest(BaseModel):
    sessions: list[SessionFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
