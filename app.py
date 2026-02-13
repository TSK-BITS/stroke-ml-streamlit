import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.title("Bank Marketing Classification App")

st.write("Upload a CSV file to predict whether a customer will subscribe to a term deposit.")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    [
        "logistic",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model = joblib.load(f"models/{model_choice}.pkl")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=";")

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop("y", axis=1)
    y = df["y"]

    # Scale if needed
    if model_choice in ["logistic", "knn"]:
        X = scaler.transform(X)

    preds = model.predict(X)

    # ---------- REQUIRED METRICS ----------
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)

    # AUC (handle models safely)
    try:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
    except:
        auc = "Not available"

    # ---------- DISPLAY METRICS (PROFESSIONAL UI) ----------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("MCC Score", f"{mcc:.4f}")
    col6.metric("AUC Score", auc if isinstance(auc, str) else f"{auc:.4f}")

    # ---------- Classification Report ----------
    st.subheader("Classification Report")
    report = classification_report(y, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ---------- Confusion Matrix ----------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    st.dataframe(pd.DataFrame(cm))
