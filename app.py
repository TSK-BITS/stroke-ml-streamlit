import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
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
st.write("Download the dataset or upload a CSV to predict whether a customer will subscribe to a term deposit.")

# Load fixed dataset
fixed_dataset = pd.read_csv("bank-full.csv", sep=";")

# Download option
csv_data = fixed_dataset.to_csv(index=False, sep=";")
st.download_button(
    label="Download Bank Dataset",
    data=csv_data,
    file_name="bank-full.csv",
    mime="text/csv"
)

# Upload CSV option
uploaded_file = st.file_uploader("Or upload a CSV file to check predictions", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
else:
    df = fixed_dataset.copy()

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

# Preprocessing
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

X = df.drop("y", axis=1)
y = df["y"]

if model_choice in ["logistic", "knn"]:
    X = scaler.transform(X)

# Predictions
preds = model.predict(X)

# Evaluation Metrics
accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall = recall_score(y, preds)
f1 = f1_score(y, preds)
mcc = matthews_corrcoef(y, preds)

try:
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
except:
    auc = "Not available"

st.subheader("Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("MCC Score", f"{mcc:.4f}")
col6.metric("AUC Score", auc if isinstance(auc, str) else f"{auc:.4f}")

st.subheader("Classification Report")
report = classification_report(y, preds, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, preds)
st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
