import pickle
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Shopping Behaviour | Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,.08), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(14,165,233,.07), transparent 25%),
        #f7f9fc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg,#111827,#1e293b,#334155);
    border-radius: 24px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 16px 40px rgba(15,23,42,.14);
}

.badge {
    color: #e2e8f0;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 999px;
    padding: .35rem .8rem;
    font-size: .75rem;
    font-weight: 700;
}

.hero-title {
    color: white;
    font-size: 2.35rem;
    font-weight: 800;
    margin-top: 1rem;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #111827;
    margin-top: 1.4rem;
}

.section-subtitle {
    color: #64748b;
    margin-bottom: 1rem;
}

.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.2rem;
    min-height: 110px;
    box-shadow: 0 8px 24px rgba(15,23,42,.05);
}

.kpi-label {
    color: #64748b;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
}

.kpi-value {
    color: #111827;
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: .35rem;
}

.kpi-note {
    color: #94a3b8;
    font-size: .75rem;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(15,23,42,.06);
}

.result-card {
    background: white;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    padding: 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 10px 30px rgba(15,23,42,.06);
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: .78rem;
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Model" / "decision_tree_model.pkl"
ENCODERS_PATH = BASE_DIR / "Model" / "label_encoders.pkl"
METRICS_PATH = BASE_DIR / "Model" / "model_metrics.json"
DATA_PATH = BASE_DIR / "Dataset" / "shopping_behavior_cleaned.csv"


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_artifacts():

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

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(DATA_PATH)

    return model, encoders, metrics, df


try:

    model, encoders, metrics, df = load_artifacts()

except Exception as e:

    st.error("Unable to load project files.")
    st.error(str(e))
    st.stop()


# ============================================================
# KPI VALUES
# ============================================================

accuracy = float(metrics["accuracy"])

total_records = len(df)

total_features = len(df.columns)

avg_purchase = float(
    df["Purchase Amount (USD)"].mean()
)

total_purchase = float(
    df["Purchase Amount (USD)"].sum()
)

if "High Value Customer" in df.columns:

    high_value_rate = float(
        df["High Value Customer"].mean()
    )

else:

    high_value_rate = 0


# ============================================================
# HERO
# ============================================================

st.markdown(f"""
<div class="hero">

<span class="badge">
LIVE MACHINE LEARNING DASHBOARD
</span>

<div class="hero-title">
🛍️ Shopping Behaviour Intelligence
</div>

<p class="hero-subtitle">
Predict high-value customers, explore purchasing patterns,
and understand what drives customer value.
Decision Tree accuracy: <b>{accuracy:.1%}</b>
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-label">Model Accuracy</div>
    <div class="kpi-value">{accuracy:.1%}</div>
    <div class="kpi-note">Decision Tree accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-label">Customer Records</div>
    <div class="kpi-value">{total_records:,}</div>
    <div class="kpi-note">Dataset transactions</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-label">Average Purchase</div>
    <div class="kpi-value">${avg_purchase:,.2f}</div>
    <div class="kpi-note">Average transaction</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-label">High Value Rate</div>
    <div class="kpi-value">{high_value_rate:.1%}</div>
    <div class="kpi-note">High-value customers</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Live Customer Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter a customer profile and predict customer value.</div>',
    unsafe_allow_html=True
)


with st.form("predict_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            70,
            32
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
            1,
            500,
            50
        )

    submitted = st.form_submit_button(
        "🔮 Predict Customer Value",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RESULT
# ============================================================

if submitted:

    try:

        row = pd.DataFrame([{

            "Age": age,

            "Gender Encoded":
                encoders["Gender"].transform([gender])[0],

            "Category Encoded":
                encoders["Category"].transform([category])[0],

            "Item Purchased Encoded":
                encoders["Item Purchased"].transform([item])[0],

            "Purchase Amount (USD)": amount

        }])

        pred = model.predict(row)[0]

        proba = model.predict_proba(row)[0]

        if pred == 1:

            label = "High Value Customer"

            confidence = float(proba[1])

            icon = "🟢"

            message = (
                "This profile matches the high-value "
                "customer segment."
            )

        else:

            label = "Standard Customer"

            confidence = float(proba[0])

            icon = "🔵"

            message = (
                "This profile currently matches the "
                "standard customer segment."
            )


        st.markdown(f"""

        <div class="result-card">

        <h3>
        {icon} {label}
        </h3>

        <p style="color:#64748b;">
        {message}
        </p>

        </div>

        """, unsafe_allow_html=True)


        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Prediction",
            label
        )

        r2.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

        r3.metric(
            "Purchase Amount",
            f"${amount:,.0f}"
        )


        # Confidence donut

        confidence_fig = go.Figure(
            go.Pie(
                labels=["Confidence", "Remaining"],
                values=[
                    confidence,
                    1 - confidence
                ],
                hole=.72,
                textinfo="none",
                marker=dict(
                    colors=[
                        "#16a34a"
                        if pred == 1
                        else "#64748b",
                        "#e5e7eb"
                    ]
                )
            )
        )

        confidence_fig.update_layout(

            title="Prediction Confidence",

            height=280,

            margin=dict(
                t=55,
                b=10,
                l=10,
                r=10
            ),

            showlegend=False,

            annotations=[{
                "text": f"<b>{confidence:.0%}</b>",
                "x": .5,
                "y": .5,
                "font": {
                    "size": 28,
                    "color": "#111827"
                },
                "showarrow": False
            }]

        )

        st.plotly_chart(
            confidence_fig,
            use_container_width=True
        )


    except Exception as e:

        st.error("Prediction failed.")

        st.error(str(e))


# ============================================================
# ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Customer Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Interactive insights from the shopping behaviour dataset.</div>',
    unsafe_allow_html=True
)


# CATEGORY SALES

category_sales = (
    df.groupby(
        "Category",
        as_index=False
    )["Purchase Amount (USD)"]
    .sum()
    .sort_values(
        "Purchase Amount (USD)",
        ascending=False
    )
)


# GENDER

gender_counts = (
    df["Gender"]
    .value_counts()
    .rename_axis("Gender")
    .reset_index(name="Customers")
)


# AGE

age_order = [
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "56-70"
]

age_counts = (
    df["Age Group"]
    .value_counts()
    .reindex(
        age_order,
        fill_value=0
    )
    .rename_axis("Age Group")
    .reset_index(name="Customers")
)


# SPENDING SEGMENT

segment_counts = (
    df["Spending Segment"]
    .value_counts()
    .rename_axis("Spending Segment")
    .reset_index(name="Customers")
)


# ============================================================
# CHART 1 + 2
# ============================================================

chart1, chart2 = st.columns(2)


with chart1:

    fig = px.pie(

        category_sales,

        names="Category",

        values="Purchase Amount (USD)",

        hole=.58,

        title="Purchase Value by Category"

    )

    fig.update_traces(

        textposition="inside",

        textinfo="percent+label",

        hovertemplate=
        "%{label}<br>"
        "$%{value:,.0f}<br>"
        "%{percent}<extra></extra>"

    )

    fig.update_layout(

        height=390,

        margin=dict(
            t=60,
            b=10,
            l=10,
            r=10
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with chart2:

    fig = px.pie(

        gender_counts,

        names="Gender",

        values="Customers",

        hole=.58,

        title="Customer Gender Distribution"

    )

    fig.update_traces(

        textposition="inside",

        textinfo="percent+label",

        hovertemplate=
        "%{label}<br>"
        "%{value:,} customers<br>"
        "%{percent}<extra></extra>"

    )

    fig.update_layout(

        height=390,

        margin=dict(
            t=60,
            b=10,
            l=10,
            r=10
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CHART 3 + 4
# ============================================================

chart3, chart4 = st.columns(2)


with chart3:

    fig = px.bar(

        age_counts,

        x="Age Group",

        y="Customers",

        text="Customers",

        title="Customers by Age Group"

    )

    fig.update_traces(

        textposition="outside",

        hovertemplate=
        "%{x}<br>"
        "%{y:,} customers<extra></extra>"

    )

    fig.update_layout(

        height=390,

        margin=dict(
            t=60,
            b=10,
            l=10,
            r=10
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with chart4:

    fig = px.pie(

        segment_counts,

        names="Spending Segment",

        values="Customers",

        hole=.58,

        title="Spending Segment Mix"

    )

    fig.update_traces(

        textposition="inside",

        textinfo="percent+label",

        hovertemplate=
        "%{label}<br>"
        "%{value:,} customers<br>"
        "%{percent}<extra></extra>"

    )

    fig.update_layout(

        height=390,

        margin=dict(
            t=60,
            b=10,
            l=10,
            r=10
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 What Drives the Prediction?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Features contributing most to the Decision Tree model.</div>',
    unsafe_allow_html=True
)


feature_importance = (

    pd.Series(
        metrics["feature_importance"],
        name="Importance"
    )

    .sort_values(
        ascending=True
    )

    .reset_index()

    .rename(
        columns={
            "index": "Feature"
        }
    )

)


fig = px.bar(

    feature_importance,

    x="Importance",

    y="Feature",

    orientation="h",

    text="Importance",

    title="Decision Tree Feature Importance"

)


fig.update_traces(

    texttemplate="%{text:.1%}",

    textposition="outside",

    hovertemplate=
    "%{y}<br>"
    "Importance: %{x:.1%}<extra></extra>"

)


fig.update_layout(

    height=430,

    margin=dict(
        t=60,
        b=20,
        l=20,
        r=60
    ),

    xaxis_title="Relative importance",

    yaxis_title=""

)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📁 Dataset Overview</div>',
    unsafe_allow_html=True
)


d1, d2, d3, d4 = st.columns(4)


d1.metric(
    "Total Records",
    f"{total_records:,}"
)

d2.metric(
    "Total Features",
    f"{total_features:,}"
)

d3.metric(
    "Total Purchase Value",
    f"${total_purchase:,.0f}"
)

d4.metric(
    "Average Rating",
    f"{df['Review Rating'].mean():.2f} / 5"
)


with st.expander("View Dataset Columns"):

    st.write(
        ", ".join(
            df.columns.tolist()
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Shopping Behaviour Analytics •
Decision Tree Machine Learning •
Python • Pandas • Scikit-learn • Plotly • Streamlit

</div>
""", unsafe_allow_html=True)