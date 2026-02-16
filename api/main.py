from fastapi import FastAPI
from api.schema import Transaction
from features.feature_utils import build_features
from api.inference import score_transaction, load_model
from explain.scm import causal_explanation

app = FastAPI(title="Real-Time Fraud Detection API")

@app.post("/score")
def score(tx: Transaction):

    # Convert Pydantic → dict
    tx_data = tx.model_dump()

    # Build features
    X = build_features(tx_data)

    # 🔑 Load latest model (updated by continual learning)
    model = load_model()

    # Score transaction
    result = score_transaction(model, X)

    # Causal explanation
    causal_info = causal_explanation(
        model=model,
        tx=tx_data,
        feature_builder=build_features
    )

    return {
        **result,
        "causal_explanation": causal_info
    }

import os
from datetime import datetime

MODEL_PATH = "model/fraud_model_v2.pkl"

@app.get("/model-status")
def model_status():
    last_updated = datetime.fromtimestamp(
        os.path.getmtime(MODEL_PATH)
    )

    return {
        "model_file": MODEL_PATH,
        "last_updated": last_updated.isoformat()
    }
