from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churnguard.models import make_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churnguard_gradient_boosting.joblib"

app = FastAPI(
    title="ChurnGuard API",
    version="1.0.0",
    description="FastAPI service for customer churn prediction (ChurnGuard).",
)


class PredictRequest(BaseModel):
    rows: Union[Dict[str, Any], List[Dict[str, Any]]]
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_path: str
    threshold: float
    churn_probability: List[float]
    churn_prediction: List[int]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train it first with: python -m churnguard.predict --train"
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.get("/schema")
def schema():
    X, _ = make_dataset()
    example_row = X.iloc[0].to_dict()

    return {
        "expected_columns": list(X.columns),
        "example_request": {
            "rows": example_row,
            "threshold": 0.5,
        },
    }


@app.get("/example")
def example():
    X, _ = make_dataset()
    return {
        "rows": X.iloc[0].to_dict(),
        "threshold": 0.5,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = load_model()

    rows = req.rows
    if isinstance(rows, dict):
        rows = [rows]

    df = pd.DataFrame(rows)

    try:
        probs = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = (probs >= req.threshold).astype(int)

    return PredictResponse(
        model_path=str(MODEL_PATH),
        threshold=req.threshold,
        churn_probability=[float(p) for p in probs],
        churn_prediction=[int(p) for p in preds],
    )
