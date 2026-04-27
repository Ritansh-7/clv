import io
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"

app = FastAPI(title="CLV Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_scaler = None
_encoders = {}
_feature_cols = []
_leaderboard = pd.DataFrame()
_best_name = "Not trained"


def _load_artifacts():
    global _model, _scaler, _encoders, _feature_cols, _leaderboard, _best_name

    if not ARTIFACTS.exists():
        return

    try:
        model_path = ARTIFACTS / "best_model.pkl"
        scaler_path = ARTIFACTS / "scaler.pkl"
        encoders_path = ARTIFACTS / "encoders.pkl"
        features_path = ARTIFACTS / "feature_cols.json"
        leaderboard_path = ARTIFACTS / "leaderboard.csv"
        best_name_path = ARTIFACTS / "best_model_name.txt"

        if not all([
            model_path.exists(),
            scaler_path.exists(),
            encoders_path.exists(),
            features_path.exists(),
            leaderboard_path.exists(),
            best_name_path.exists(),
        ]):
            print("Artifacts missing. Train the model first.")
            return

        with open(model_path, "rb") as f:
            _model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)

        with open(encoders_path, "rb") as f:
            _encoders = pickle.load(f)

        with open(features_path, "r") as f:
            _feature_cols = json.load(f)

        _leaderboard = pd.read_csv(leaderboard_path)

        with open(best_name_path, "r") as f:
            _best_name = f.read().strip()

        print(f"Loaded model: {_best_name}")

    except Exception as e:
        _model = None
        _scaler = None
        _encoders = {}
        _feature_cols = []
        _leaderboard = pd.DataFrame()
        _best_name = "Not trained"
        print(f"Load error: {e}")


@app.on_event("startup")
async def startup():
    _load_artifacts()


def _ready():
    if _model is None or _scaler is None or not _feature_cols:
        raise HTTPException(
            status_code=503,
            detail="Models not trained or artifacts are missing. Run `python ml_training.py` first."
        )


class CustomerIn(BaseModel):
    Age: int = 35
    Annual_Income: int = 60000
    Household_Size: int = 3
    Quantity: int = 10
    Unit_Price: float = 150.0
    Discount_Applied: float = 0.10
    Revenue: int = 10000
    Profit_Margin: float = 0.30
    Loyalty_Score: float = 7.0
    Customer_Satisfaction: float = 4.0
    Purchase_Frequency: int = 12
    Days_Since_Last_Purchase: int = 30
    Newsletter_Subscribed: int = 1
    Support_Tickets: int = 2
    Email_Open_Rate: float = 0.35
    Social_Media_Engagement: float = 0.50
    Churn_Risk_Score: float = 0.20
    Return_Rate: float = 0.05
    Payment_Delay_Days: int = 5
    Customer_Acquisition_Cost: int = 500
    Net_Revenue: float = 9000.0
    CLV_to_CAC_Ratio: float = 20.0
    Engagement_Score: float = 6.0
    Profitability_Score: float = 0.70
    Seasonal_Factor: float = 1.0
    Cohort_Index: int = 0
    cost: float = 10.0
    conversion_rate: float = 0.12
    Customer_Segment: str = "Regular"
    Gender: str = "Male"
    Job_Category: str = "Professional"
    Education_Level: str = "Bachelor"
    Marital_Status: str = "Married"
    Payment_Method: str = "Credit Card"
    Marketing_Channel: str = "Email"
    First_Purchase_Channel: str = "Online"
    Season: str = "Fall"
    Age_Group: str = "30-45"
    Revenue_Tier: str = "Medium"
    Loyalty_Tier: str = "Gold"
    channel: str = "referral"


def _encode_row(row: dict) -> np.ndarray:
    encoded = row.copy()

    for col, le in _encoders.items():
        if col in encoded:
            v = str(encoded[col])
            try:
                encoded[col] = int(le.transform([v])[0])
            except Exception:
                encoded[col] = 0

    values = [encoded.get(f, 0) for f in _feature_cols]
    x = np.array([values], dtype=float)
    return _scaler.transform(x)


def _tier(v: float) -> str:
    if v < 5000:
        return "Low Value"
    if v < 15000:
        return "Medium Value"
    if v < 40000:
        return "High Value"
    return "Premium"


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": _best_name,
        "loaded": _model is not None
    }


@app.get("/leaderboard")
def leaderboard():
    _ready()
    if _leaderboard.empty:
        raise HTTPException(status_code=404, detail="No leaderboard available")
    return _leaderboard.to_dict(orient="records")


@app.get("/best-model")
def best_model():
    _ready()
    if _leaderboard.empty:
        raise HTTPException(status_code=404, detail="No leaderboard available")
    return {
        "best_model": _best_name,
        "metrics": _leaderboard.iloc[0].to_dict()
    }


@app.post("/predict")
def predict(c: CustomerIn):
    _ready()
    x = _encode_row(c.model_dump())
    clv = float(max(0, _model.predict(x)[0]))
    return {
        "predicted_clv": round(clv, 2),
        "clv_tier": _tier(clv),
        "model": _best_name
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    _ready()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.")

        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV has no rows.")

        df = df.copy()

        for col, le in _encoders.items():
            if col in df.columns:
                def safe_encode(v):
                    v = str(v)
                    try:
                        return int(le.transform([v])[0])
                    except Exception:
                        return 0
                df[col] = df[col].apply(safe_encode)

        for col in _feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[_feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = _scaler.transform(X)
        preds = np.clip(_model.predict(X_scaled), 0, None)

        results = [
            {"clv": round(float(p), 2), "tier": _tier(float(p))}
            for p in preds
        ]

        return {
            "rows": len(df),
            "model": _best_name,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/feature-importance")
def fi(top_n: int = 15):
    _ready()
    if not hasattr(_model, "feature_importances_"):
        return []

    s = pd.Series(_model.feature_importances_, index=_feature_cols).nlargest(top_n)
    return [
        {"feature": k, "importance": round(float(v), 6)}
        for k, v in s.items()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)