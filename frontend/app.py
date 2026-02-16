import streamlit as st
import pandas as pd
import time
import requests
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


API_URL = "http://127.0.0.1:8000/score"

st.set_page_config(layout="wide")
st.title("🛡️ Real-Time Fraud Detection System")

# ----------------------------
# Manual Transaction Input
# ----------------------------
st.header("🔍 Manual Transaction Check")

with st.form("manual_tx"):
    tx_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    )

    amount = st.number_input("Transaction Amount", min_value=0.0)
    oldbalance = st.number_input("Old Balance", min_value=0.0)

    submit = st.form_submit_button("Check Transaction")

if submit:
    payload = {
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance,
        "newbalanceOrig": oldbalance - amount  # Assuming money is deducted
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        res = response.json()
        st.success(f"Fraud Probability: {round(res['fraud_probability'], 4)} | Decision: {res['decision']}")
        causal = res["causal_explanation"]

        st.subheader("🧠 Why this decision?")

        st.write(
            f"Original fraud risk: **{round(causal['original_risk'], 4)}**"
        )

        for item in causal["amount_interventions"]:
            st.write(
                f"If amount was **{round(item['counterfactual_amount'], 2)}** "
                f"({int(item['factor']*100)}% of original), "
                f"risk becomes **{round(item['risk_after'], 4)}** "
                f"(change: {round(item['risk_change'], 4)})"
            )


# ----------------------------
# Live Transaction Stream
# ----------------------------
st.header("📡 Live Transaction Stream")

table = st.empty()
rows = []

stream_df = pd.read_csv("data/continual_data.csv")

for _, row in stream_df.iterrows():

    payload = {
    "type": row.get("type", None),
    "amount": row["amount"],
    "oldbalanceOrg": row["oldbalanceOrg"],
    "newbalanceOrig": row.get("newbalanceOrig", None)
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        res = response.json()

        rows.append({
            "Amount": row["amount"],
            "Old Balance": row["oldbalanceOrg"],
            "Fraud Probability": res["fraud_probability"],
            "Decision": res["decision"],
            "Actual": "FRAUD" if row["isFraud"] == 1 else "NORMAL"
        })

        table.dataframe(
            pd.DataFrame(rows).tail(5),
            width="stretch"
        )

    time.sleep(0.5)

# ----------------------------
# Model Health Panel
# ----------------------------
st.header("📊 Model Health")

status = requests.get("http://127.0.0.1:8000/model-status")

if status.status_code == 200:
    info = status.json()
    st.info(f"🧠 Model last updated at: {info['last_updated']}")
else:
    st.warning("Model status unavailable")


