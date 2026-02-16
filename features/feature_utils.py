import pandas as pd

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "balance_error",
    "tx_count_24h",
    "avg_amount_24h"
]

def build_features(tx_data):
    """
    tx_data can be dict or DataFrame
    """

    if isinstance(tx_data, dict):
        df = pd.DataFrame([tx_data])
    else:
        df = tx_data.copy()

    # Defaults
    if "tx_count_24h" not in df.columns or df["tx_count_24h"].isna().any():
        df["tx_count_24h"] = 0

    if "avg_amount_24h" not in df.columns or df["avg_amount_24h"].isna().any():
        df["avg_amount_24h"] = df["amount"]

    # balance error
    df["balance_error"] = (
        df["oldbalanceOrg"] - df["amount"]
    ).clip(lower=0)

    return df[FEATURES]
