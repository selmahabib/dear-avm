"""
CatBoost inference API for the bAVM classification interface.

The model is a tree ensemble, so it cannot run in the browser and TreeSHAP
cannot be computed there either. This serves both from the exported model.

Run Nested_CV.py first - it writes catboost_model.cbm and catboost_scaler.csv
into the results folder. Copy both next to this file, then:

    uvicorn app:app --reload --port 8000

The interface expects http://localhost:8000/predict.
"""
import os

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("BAVM_MODEL", "catboost_model.cbm")
SCALER_PATH = os.getenv("BAVM_SCALER", "catboost_scaler.csv")

# Order matters: the model was fitted on columns in this order and CatBoost
# does not check names at predict time.
FEATURES = [
    "Race_nonwhite", "Deep_Location", "Size", "Exclusive_Deep_Drainage",
    "Single_Feeding_Artery", "Feeder_Count_Unknown", "Age",
    "Intranidal_Aneurysm", "HTN",
]

DISPLAY = {
    "Size": "Nidus size",
    "Age": "Age at diagnosis",
    "Deep_Location": "Deep location",
    "Race_nonwhite": "Non-white race",
    "Exclusive_Deep_Drainage": "Exclusive deep drainage",
    "Single_Feeding_Artery": "Single feeding artery",
    "Intranidal_Aneurysm": "Intranidal aneurysm",
    "HTN": "Hypertension",
    "Feeder_Count_Unknown": "Feeder count undocumented",
}

app = FastAPI(title="bAVM CatBoost API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_model = None
_explainer = None
_scaler = None


def _load():
    global _model, _explainer, _scaler
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run Nested_CV.py and copy the exported "
            f"model here.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"{SCALER_PATH} not found. The model expects scaled inputs, so "
            f"the scaler constants are required.")
    # Single-threaded: CatBoost's native thread pool can deadlock or crash
    # the worker when it runs inside an ASGI server's own threadpool.
    _model = CatBoostClassifier(thread_count=1)
    _model.load_model(MODEL_PATH)
    _scaler = pd.read_csv(SCALER_PATH).set_index("Feature")
    _explainer = shap.TreeExplainer(_model)


class Patient(BaseModel):
    """Tri-state fields accept null for undocumented findings.

    CatBoost routes missing values through its own learned default direction,
    which is why they are passed through as NaN rather than imputed here.
    """
    age: float = Field(..., ge=0, le=120)
    sizeCm: float = Field(..., ge=0, le=20)
    race: str
    deepLocation: bool | None = None
    hypertension: bool | None = None
    exclusiveDeepDrainage: bool | None = None
    intranidalAneurysm: bool | None = None
    singleFeedingArtery: bool | None = None


def _tri(v):
    return np.nan if v is None else float(v)


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "features": FEATURES}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(p: Patient):
    try:
        _load()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = {
        "Race_nonwhite": 1.0 if p.race == "nonwhite" else 0.0,
        "Deep_Location": _tri(p.deepLocation),
        "Size": float(p.sizeCm),
        "Exclusive_Deep_Drainage": _tri(p.exclusiveDeepDrainage),
        "Single_Feeding_Artery": _tri(p.singleFeedingArtery),
        # An undocumented feeder count is its own level in the model, so the
        # indicator is set rather than the value left missing.
        "Feeder_Count_Unknown": 1.0 if p.singleFeedingArtery is None else 0.0,
        "Age": float(p.age),
        "Intranidal_Aneurysm": _tri(p.intranidalAneurysm),
        "HTN": _tri(p.hypertension),
    }
    if p.singleFeedingArtery is None:
        raw["Single_Feeding_Artery"] = 0.0

    x = np.array([[raw[f] for f in FEATURES]], dtype=float)
    xs = (x - _scaler.loc[FEATURES, "mean"].to_numpy()) / \
         _scaler.loc[FEATURES, "scale"].to_numpy()

    prob = float(_model.predict_proba(xs)[0, 1])

    sv = np.asarray(_explainer.shap_values(xs), dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    if sv.shape[1] == len(FEATURES) + 1:
        sv = sv[:, :-1]
    sv = sv[0]

    base = float(np.asarray(_explainer.expected_value).ravel()[-1])

    contributions = sorted(
        [{"feature": DISPLAY[f], "value": round(float(v), 4),
          "missing": bool(np.isnan(raw[f]))}
         for f, v in zip(FEATURES, sv)],
        key=lambda d: abs(d["value"]), reverse=True)

    return {
        "probability": round(prob, 4),
        "baseLogOdds": round(base, 4),
        "contributions": contributions,
        "model": "CatBoost, full-cohort refit of the nested-CV configuration",
    }
