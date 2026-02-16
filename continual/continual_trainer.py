import pandas as pd
import joblib
import os
import sys
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from features.feature_utils import build_features

MODEL_PATH = "model/fraud_model_v2.pkl"
OLD_SAMPLE_PATH = "data/train_sample.csv"

def retrain_model(new_data: pd.DataFrame):

    old_memory = pd.read_csv(OLD_SAMPLE_PATH)
    old_replay = old_memory.sample(3000, random_state=42)

    retrain_df = pd.concat([old_replay, new_data], ignore_index=True)

    X = build_features(retrain_df)
    y = retrain_df["isFraud"]

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=150,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

