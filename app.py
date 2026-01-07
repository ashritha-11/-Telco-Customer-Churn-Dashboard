import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide"
)

st.title("📊 Telco Customer Churn Dashboard")
st.markdown(
    "Predict customer churn, analyze model performance, and retain valuable customers!"
)

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# -----------------------
# DATA PREVIEW
# -----------------------
st.subheader("👀 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------
# FEATURES & TARGET
# -----------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -----------------------
# TRAIN-TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# PREPROCESSING
# -----------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# -----------------------
# MODEL TRAINING
# -----------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_p, y_train)

# -----------------------
# PREDICTIONS & METRICS
# -----------------------
y_pred = model.predict(X_test_p)
y_prob = model.predict_proba(X_test_p)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
cm = confusion_matrix(y_test, y_pred)

# -----------------------
# KPI CARDS with + / -
# -----------------------
st.subheader("🚀 Model Performance Metrics")
baseline_acc = 0.80
baseline_auc = 0.85

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    diff = acc - baseline_acc
    st.metric("Accuracy", f"{acc:.2%}", delta=f"{diff:+.2%}")

with kpi2:
    diff_auc = roc_auc - baseline_auc
    st.metric("ROC AUC", f"{roc_auc:.3f}", delta=f"{diff_auc:+.3f}")

with kpi3:
    churn_rate = y.mean()
    st.metric("Churn Rate", f"{churn_rate:.2%}")

# -----------------------
# CONFUSION MATRIX TABLE ABOVE GRAPHS
# -----------------------
st.subheader("🔎 Confusion Matrix Table & Graphs")
cm_df = pd.DataFrame(
    cm,
    columns=["Predicted Stay", "Predicted Churn"],
    index=["Actual Stay", "Actual Churn"]
)
st.dataframe(cm_df, use_container_width=True)

# -----------------------
# GRAPHS SIDE-BY-SIDE
# -----------------------
col1, col2 = st.columns(2)

# Confusion Matrix Heatmap
with col1:
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix Heatmap")
    st.pyplot(fig_cm)

# ROC Curve
with col2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1], [0,1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

# -----------------------
# CLASSIFICATION REPORT
# -----------------------
with st.expander("📄 Detailed Classification Report"):
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# -----------------------
# NEW CUSTOMER PREDICTION
# -----------------------
st.subheader("🧠 Predict Churn for New Customer")
user_data = {}
for col in X.columns:
    if col in categorical_cols:
        user_data[col] = st.selectbox(col, X[col].unique())
    else:
        user_data[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].median()))

if st.button("🔮 Predict Churn"):
    input_df = pd.DataFrame([user_data])
    input_p = preprocessor.transform(input_df)
    pred = model.predict(input_p)[0]
    prob = model.predict_proba(input_p)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely to Churn | Probability: {prob:.2%}")
    else:
        st.success(f"✅ Likely to Stay | Probability: {prob:.2%}")
