from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict


class InferenceRequest(BaseModel):
    payload: Dict[str, str] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    prediction: str


class ErrorResponse(BaseModel):
    detail: str
