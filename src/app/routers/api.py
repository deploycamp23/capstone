from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends

from app.schemas import InferenceRequest, InferenceResponse
from app.services.model_service import ModelService

router = APIRouter(prefix="/api")

def get_model_service() -> ModelService:
    from app.main import get_container
    return get_container()["model_service"]


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/model")
def model_info(svc: ModelService = Depends(get_model_service)) -> dict:
    return svc.info()


@router.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest, svc: ModelService = Depends(get_model_service)) -> InferenceResponse:
    try:
        pred = svc.predict(req.payload)
        return InferenceResponse(prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
