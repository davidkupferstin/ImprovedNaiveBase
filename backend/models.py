from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: str
    full_results: Dict[str, float]

class TrainResponse(BaseModel):
    message: str
    accuracy: float
    features: Dict[str, List[str]]
    target_column: str

class ModelStatusResponse(BaseModel):
    status: str
    accuracy: Optional[float]
    features: Optional[Dict[str, List[str]]]
    target_column: Optional[str]
