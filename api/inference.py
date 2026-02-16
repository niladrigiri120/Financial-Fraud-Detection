import joblib

MODEL_PATH = "model/fraud_model_v1.pkl"
THRESHOLD = "model/best_threshold.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def load_threshold():
    return joblib.load(THRESHOLD)

def score_transaction(model, X):
    prob = model.predict_proba(X)[0, 1]
    
    threshold = load_threshold()
    decision = "ALERT" if prob >= threshold else "APPROVE"

    return {
        "fraud_probability": float(prob),
        "decision": decision
    }
