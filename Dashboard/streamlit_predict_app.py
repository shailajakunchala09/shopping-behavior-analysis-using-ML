"""
Companion live-prediction app for the Power BI dashboard.

Power BI is excellent for exploring historical data and model results, but it
can't run a Python model interactively per user input. This Streamlit app
provides live Decision Tree predictions using the exact model trained in the
notebook.

Run from the project root:
    streamlit run Dashboard/streamlit_predict_app.py
"""

import pickle
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Shopping Behaviour - Live Prediction",
    page_icon="🛍️",
    layout="centered"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# shopping-behaviour-analytics-project/
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Model" / "decision_tree_model.pkl"
ENCODERS_PATH = BASE_DIR / "Model" / "label_encoders.pkl"
METRICS_PATH = BASE_DIR / "Model" / "model_metrics.json"
DATA_PATH = BASE_DIR / "Dataset" / "shopping_behavior_cleaned.csv"


# ============================================================
# LOAD MODEL, ENCODERS, METRICS AND DATASET
# ============================================================

@st.cache_resource
def load_artifacts():

    # Check required files before loading
    required_files = {
        "Model": MODEL_PATH,
        "Encoders": ENCODERS_PATH,
        "Metrics": METRICS_PATH,
        "Dataset": DATA_PATH,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    # Load Decision Tree model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load Label Encoders
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    # Load model metrics
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    return model, encoders, metrics, df


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:
    model, encoders, metrics, df = load_artifacts()

except Exception as e:
    st.error("❌ Unable to load project files.")
    st.error(str(e))

    st.info(
        "Please make sure the following folders exist inside the project:\n\n"
        "Model/\n"
        "Dataset/\n"
        "Dashboard/"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🛍️ Shopping Behaviour — High Value Customer Predictor")

st.caption(
    f"Decision Tree Classifier · "
    f"Test accuracy {metrics['accuracy']:.1%} · "
    "Predicts whether a customer profile matches the "
    "'High Value Customer' segment."
)


# ============================================================
# PREDICTION FORM
# ============================================================

with st.form("predict_form"):

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            min_value=18,
            max_value=70,
            value=32
        )

        gender = st.selectbox(
            "Gender",
            encoders["Gender"].classes_
        )

        category = st.selectbox(
            "Category",
            encoders["Category"].classes_
        )

    with col2:

        item = st.selectbox(
            "Item Purchased",
            encoders["Item Purchased"].classes_
        )

        amount = st.number_input(
            "Purchase Amount (USD)",
            min_value=1,
            max_value=500,
            value=50
        )

    submitted = st.form_submit_button(
        "🔮 Predict Customer Value"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        row = pd.DataFrame([{
            "Age": age,
            "Gender Encoded": encoders["Gender"].transform([gender])[0],
            "Category Encoded": encoders["Category"].transform([category])[0],
            "Item Purchased Encoded": encoders["Item Purchased"].transform([item])[0],
            "Purchase Amount (USD)": amount,
        }])

        # Prediction
        pred = model.predict(row)[0]

        # Prediction probability
        proba = model.predict_proba(row)[0]

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        if pred == 1:

            st.success(
                f"✅ Predicted: **High Value Customer** "
                f"(confidence {proba[1]:.1%})"
            )

        else:

            st.info(
                f"Predicted: **Standard Customer** "
                f"(confidence {proba[0]:.1%})"
            )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.error(str(e))


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 Model Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.1%}"
)

c2.metric(
    "Rows Evaluated",
    sum(metrics["confusion_matrix"][0])
    + sum(metrics["confusion_matrix"][1])
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("📈 Feature Importance")

feature_importance = pd.Series(
    metrics["feature_importance"]
)

st.bar_chart(feature_importance)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.subheader("📁 Dataset Information")

d1, d2 = st.columns(2)

d1.metric(
    "Total Records",
    len(df)
)

d2.metric(
    "Total Features",
    len(df.columns)
)