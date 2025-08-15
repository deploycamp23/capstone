from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    payload: Dict[str, str] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    prediction: str


class ErrorResponse(BaseModel):
    detail: str
