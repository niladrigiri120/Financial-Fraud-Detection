# 🛡️ Autonomous Real-Time Financial Fraud Detection  
### with Continual Learning and Causal Explainability

---

## 📌 Project Overview

This project implements an **autonomous real-time financial fraud detection system** that continuously adapts to new fraud patterns while providing **transparent, causal explanations** for its decisions.

The system is designed as a **production-style architecture**, separating:
- User Interface (Streamlit)
- Inference & Explainability (FastAPI)
- Continual Learning (Background retraining pipeline)

The model updates incrementally as new labeled transaction data becomes available, using **replay buffers**, **adaptive regularization**, and **drift-aware retraining**, while ensuring stable and interpretable predictions.

---

## 🎯 Objectives

- Detect fraudulent transactions in real time  
- Process streaming transaction data  
- Incrementally adapt to evolving fraud patterns  
- Preserve prior knowledge using replay buffers  
- Explain predictions using causal counterfactual reasoning  
- Provide a web-based monitoring and control dashboard  

---

## 🏗️ System Architecture

Streamlit Dashboard
├── Manual transaction input
├── Live transaction stream
├── Causal explanations
↓
FastAPI Backend
├── /score (inference + explanation)
├── /model-status (model health)
↓
Fraud Detection Model (pickle file)
↑
Continual Learning Pipeline
├── Replay buffer
├── Periodic retraining
├── Log-based monitoring


---

## 📊 Dataset

- **Primary Dataset**: PaySim (simulated mobile money transactions)
- **Features Used**:
  - `amount`
  - `oldbalanceOrg`
  - Derived behavioral features (engineered internally)
- **Target**:
  - `isFraud` (binary)

A representative subset of the original large dataset is stored as **long-term memory** for replay-based continual learning.

---

## 🧠 Fraud Detection Model

- **Model**: `HistGradientBoostingClassifier`
- **Why this model?**
  - Strong performance on tabular data
  - Robust to feature scaling
  - Efficient for real-time inference

The model outputs:
- Fraud probability
- Decision (`ALERT` / `APPROVE`)

---

## 🔁 Continual Learning Mechanism

### Incremental Updates
- New labeled transactions are buffered
- Retraining is triggered every **300 new rows**

### Replay Buffer
- A representative sample from historical training data is combined with new data
- Prevents catastrophic forgetting

### Adaptive Regularization
- L2 regularization is applied during retraining
- Stabilizes decision boundaries while adapting to new patterns

### Execution Model
- Continual learning runs as a **background process**
- Streamlit **does not train the model**
- The updated model is saved to disk and automatically reused by FastAPI

---

## 🧠 Causal Explainability

The system uses **counterfactual reasoning** to explain predictions:

- Structural causal logic is applied on transaction features
- Answers questions like:
  > “If the transaction amount were lower, how would the fraud risk change?”

### Output Includes:
- Original fraud risk
- Counterfactual risk under different interventions
- Risk change magnitude

This ensures **transparent and interpretable decisions**, not just black-box predictions.

---

## 🌐 Web Application

### Streamlit Dashboard
- Manual transaction testing
- Live streaming transaction feed
- Real-time fraud decisions
- Causal explanation panel
- Model health and retraining status
- Button to start continual learning

### FastAPI Backend
- `/score` → real-time fraud scoring + explanation
- `/model-status` → last retraining timestamp

Streamlit communicates with FastAPI via HTTP; it never loads or trains the model directly.

---

## 🚀 Deployment Workflow

1. Start FastAPI server  
2. Start Streamlit dashboard  
3. Start continual learning runner (background process)  
4. Model updates automatically reflected in predictions  

This architecture is **deployment-safe and non-blocking**.

---

## 📈 Evaluation Metrics

- Precision & Recall (fraud-focused)
- Alert rate control
- False positive stability
- Model update frequency
- Explanation consistency
- Inference latency

Thresholds are tuned to prioritize **fraud recall while controlling alert volume**.

---

## 🛠️ Tools & Technologies

- Python
- Scikit-learn
- Pandas & NumPy
- FastAPI
- Streamlit
- Joblib
- Docker (optional)
- Git

---

## ⚠️ Limitations

- PaySim is simulated data (not real banking data)
- True online learning is not supported by tree-based models
- Drift detection is statistical, not semantic
- Full retraining on all historical data is not performed continuously

---

## 🔮 Future Enhancements

- Advanced drift detection (PSI / KS tests)
- Multi-feature causal explanations
- Model versioning and rollback
- Role-based access for retraining controls
- Cloud-native deployment (Kubernetes)

---

## ✅ Conclusion

This project demonstrates a **realistic, production-aligned approach** to financial fraud detection by combining:

- Real-time inference
- Continual learning with replay buffers
- Adaptive regularization
- Causal explainability
- Web-based monitoring and control

The system balances **accuracy, adaptability, and transparency**, making it suitable for large-scale financial environments.

---

## 👤 Author

**Niladri Giri**  
Data Science & Machine Learning  

---
