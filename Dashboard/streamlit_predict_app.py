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
    page_title="Shopping Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 0%, rgba(99,102,241,.13), transparent 25%),
        radial-gradient(circle at 95% 5%, rgba(6,182,212,.10), transparent 25%),
        #f5f7fb;
}

.block-container {
    max-width: 1400px;
    padding: 2rem 3rem 4rem;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}


/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 85% 20%, rgba(56,189,248,.25), transparent 25%),
        radial-gradient(circle at 15% 80%, rgba(139,92,246,.25), transparent 25%),
        linear-gradient(135deg,#0f172a,#172554,#312e81);
    border-radius: 28px;
    padding: 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 25px 60px rgba(15,23,42,.20);
    border: 1px solid rgba(255,255,255,.08);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.18);
    color: #bae6fd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 18px 0 8px;
}

.hero-text {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
    max-width: 760px;
}

.hero-highlight {
    color: #67e8f9;
    font-weight: 800;
}


/* SECTION */

.section-title {
    color: #0f172a;
    font-size: 25px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 18px;
}


/* KPI */

.kpi {
    background: rgba(255,255,255,.90);
    backdrop-filter: blur(12px);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 22px;
    min-height: 140px;
    box-shadow: 0 12px 35px rgba(15,23,42,.06);
    transition: .2s ease;
}

.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 45px rgba(15,23,42,.10);
}

.kpi-icon {
    font-size: 25px;
}

.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-top: 8px;
}

.kpi-value {
    color: #0f172a;
    font-size: 30px;
    font-weight: 800;
    margin-top: 4px;
}

.kpi-note {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
}


/* CARDS */

.card {
    background: rgba(255,255,255,.94);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(15,23,42,.055);
}


/* FORM */

div[data-testid="stForm"] {
    background: rgba(255,255,255,.96);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 15px 40px rgba(15,23,42,.07);
}


/* BUTTON */

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
}


/* METRICS */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 16px;
}


/* RESULT */

.prediction-card {
    background:
        linear-gradient(135deg,#ecfeff,#eef2ff);
    border: 1px solid #c7d2fe;
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
}

.prediction-title {
    font-size: 25px;
    font-weight: 800;
    color: #0f172a;
}

.prediction-text {
    color: #64748b;
}


/* FOOTER */

.footer {
    margin-top: 50px;
    padding: 25px;
    text-align: center;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
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
# LOAD PROJECT FILES
# ============================================================

@st.cache_resource
def load_artifacts():

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
    st.code(str(e))
    st.stop()


# ============================================================
# DATA VALUES
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

avg_rating = float(
    df["Review Rating"].mean()
)

if "High Value Customer" in df.columns:
    high_value_rate = float(
        df["High Value Customer"].mean()
    )
else:
    high_value_rate = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛍️ Shopping AI")

    st.markdown("---")

    st.markdown("### Dashboard")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Customer Prediction",
            "Analytics",
            "Model Insights",
            "Dataset"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Model")

    st.success("Decision Tree • Online")

    st.caption(
        f"Accuracy: {accuracy:.1%}"
    )

    st.markdown("---")

    st.caption(
        "Shopping Behaviour Intelligence"
    )

    st.caption(
        "Python • Pandas • Scikit-learn • Plotly"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(f"""

<div class="hero">

<span class="hero-badge">
● LIVE MACHINE LEARNING PLATFORM
</span>

<div class="hero-title">
🛍️ Shopping Behaviour Intelligence
</div>

<div class="hero-text">
Transform shopping data into actionable customer intelligence.
Predict high-value customers, discover purchasing patterns,
and understand the factors influencing customer value.
<br><br>
Decision Tree model accuracy:
<span class="hero-highlight">
{accuracy:.1%}
</span>
</div>

</div>

""", unsafe_allow_html=True)


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{accuracy:.1%}</div>
        <div class="kpi-note">Decision Tree performance</div>
    </div>
    """, unsafe_allow_html=True)


with k2:

    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-icon">👥</div>
        <div class="kpi-label">Customer Records</div>
        <div class="kpi-value">{total_records:,}</div>
        <div class="kpi-note">Shopping transactions</div>
    </div>
    """, unsafe_allow_html=True)


with k3:

    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-icon">💰</div>
        <div class="kpi-label">Avg. Purchase</div>
        <div class="kpi-value">${avg_purchase:,.2f}</div>
        <div class="kpi-note">Average transaction value</div>
    </div>
    """, unsafe_allow_html=True)


with k4:

    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-icon">⭐</div>
        <div class="kpi-label">High Value Rate</div>
        <div class="kpi-value">{high_value_rate:.1%}</div>
        <div class="kpi-note">High-value customer segment</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">📊 Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">A high-level view of customer behaviour and business performance.</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CATEGORY SALES
    # --------------------------------------------------------

    category_sales = (
        df.groupby("Category", as_index=False)
        ["Purchase Amount (USD)"]
        .sum()
        .sort_values(
            "Purchase Amount (USD)",
            ascending=False
        )
    )


    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_counts = (
        df["Gender"]
        .value_counts()
        .rename_axis("Gender")
        .reset_index(name="Customers")
    )


    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if "Age Group" in df.columns:

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

    else:

        df["Age Group"] = pd.cut(
            df["Age"],
            bins=[17,25,35,45,55,70],
            labels=[
                "18-25",
                "26-35",
                "36-45",
                "46-55",
                "56-70"
            ]
        )

        age_counts = (
            df["Age Group"]
            .value_counts()
            .sort_index()
            .rename_axis("Age Group")
            .reset_index(name="Customers")
        )


    # --------------------------------------------------------
    # CHART 1
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        fig = px.pie(
            category_sales,
            names="Category",
            values="Purchase Amount (USD)",
            hole=.65,
            title="Revenue Distribution by Category"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=
            "<b>%{label}</b><br>"
            "Revenue: $%{value:,.0f}<br>"
            "Share: %{percent}<extra></extra>"
        )

        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=70,b=20,l=10,r=10),
            legend=dict(
                orientation="h",
                y=-0.08
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHART 2
    # --------------------------------------------------------

    with c2:

        fig = px.pie(
            gender_counts,
            names="Gender",
            values="Customers",
            hole=.65,
            title="Customer Demographics"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=
            "<b>%{label}</b><br>"
            "Customers: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        )

        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=70,b=20,l=10,r=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # AGE + CATEGORY BAR
    # --------------------------------------------------------

    c3, c4 = st.columns(2)


    with c3:

        fig = px.bar(
            age_counts,
            x="Age Group",
            y="Customers",
            text="Customers",
            title="Customer Distribution by Age"
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=
            "<b>%{x}</b><br>"
            "Customers: %{y:,}<extra></extra>"
        )

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Customers",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with c4:

        fig = px.bar(
            category_sales,
            x="Purchase Amount (USD)",
            y="Category",
            orientation="h",
            text="Purchase Amount (USD)",
            title="Revenue by Product Category"
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate=
            "<b>%{y}</b><br>"
            "Revenue: $%{x:,.0f}<extra></extra>"
        )

        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Revenue",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":

    st.markdown(
        '<div class="section-title">🎯 Customer Intelligence Engine</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Create a customer profile and generate a machine-learning prediction.</div>',
        unsafe_allow_html=True
    )


    with st.form("prediction_form"):

        col1, col2 = st.columns(2)


        with col1:

            age = st.slider(
                "Customer Age",
                18,
                70,
                32
            )

            gender = st.selectbox(
                "Gender",
                encoders["Gender"].classes_
            )

            category = st.selectbox(
                "Product Category",
                encoders["Category"].classes_
            )


        with col2:

            item = st.selectbox(
                "Product Purchased",
                encoders["Item Purchased"].classes_
            )

            amount = st.number_input(
                "Purchase Amount (USD)",
                min_value=1,
                max_value=500,
                value=50
            )


        submitted = st.form_submit_button(
            "🚀 Analyze Customer",
            type="primary",
            use_container_width=True
        )


    if submitted:

        try:

            row = pd.DataFrame([{

                "Age": age,

                "Gender Encoded":
                    encoders["Gender"]
                    .transform([gender])[0],

                "Category Encoded":
                    encoders["Category"]
                    .transform([category])[0],

                "Item Purchased Encoded":
                    encoders["Item Purchased"]
                    .transform([item])[0],

                "Purchase Amount (USD)": amount

            }])


            prediction = model.predict(row)[0]

            probabilities = model.predict_proba(row)[0]


            if prediction == 1:

                label = "High Value Customer"

                confidence = float(probabilities[1])

                icon = "🟢"

                color = "#10b981"

            else:

                label = "Standard Customer"

                confidence = float(probabilities[0])

                icon = "🔵"

                color = "#6366f1"


            st.markdown(f"""

            <div class="prediction-card">

                <div class="prediction-title">
                    {icon} {label}
                </div>

                <div class="prediction-text">
                    Machine learning analysis completed successfully.
                    The model confidence is
                    <b>{confidence:.1%}</b>.
                </div>

            </div>

            """, unsafe_allow_html=True)


            r1, r2, r3 = st.columns(3)


            r1.metric(
                "Customer Segment",
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


            # CONFIDENCE DONUT

            fig = go.Figure(

                go.Pie(

                    labels=[
                        "Confidence",
                        "Remaining"
                    ],

                    values=[
                        confidence,
                        1-confidence
                    ],

                    hole=.75,

                    marker=dict(
                        colors=[
                            color,
                            "#e2e8f0"
                        ]
                    ),

                    textinfo="none"

                )

            )


            fig.update_layout(

                title="Prediction Confidence",

                height=350,

                showlegend=False,

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                margin=dict(
                    t=70,
                    b=10,
                    l=10,
                    r=10
                ),

                annotations=[

                    {

                        "text":
                        f"<b>{confidence:.0%}</b>"
                        "<br><sup>Confidence</sup>",

                        "x": .5,

                        "y": .5,

                        "showarrow": False,

                        "font": {
                            "size": 24,
                            "color": "#0f172a"
                        }

                    }

                ]

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        except Exception as e:

            st.error("Prediction failed.")

            st.code(str(e))


# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="section-title">📊 Advanced Customer Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Explore customer behaviour, spending patterns and product performance.</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------
    # SPENDING DISTRIBUTION
    # -----------------------------------------------

    fig = px.histogram(
        df,
        x="Purchase Amount (USD)",
        nbins=30,
        marginal="box",
        title="Purchase Amount Distribution"
    )

    fig.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Purchase Amount (USD)",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------
    # RATING VS PURCHASE
    # -----------------------------------------------

    if "Review Rating" in df.columns:

        fig = px.scatter(
            df,
            x="Review Rating",
            y="Purchase Amount (USD)",
            color="Category",
            size="Purchase Amount (USD)",
            hover_data=[
                "Age",
                "Gender"
            ],
            title="Customer Rating vs Purchase Amount"
        )

        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------
    # TOP PRODUCTS
    # -----------------------------------------------

    product_sales = (
        df.groupby("Item Purchased")
        ["Purchase Amount (USD)"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .sort_values()
        .reset_index()
    )


    fig = px.bar(
        product_sales,
        x="Purchase Amount (USD)",
        y="Item Purchased",
        orientation="h",
        text="Purchase Amount (USD)",
        title="Top 10 Products by Revenue"
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="section-title">🧠 Model Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Understand which variables influence the Decision Tree prediction.</div>',
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
        "<b>%{y}</b><br>"
        "Importance: %{x:.1%}<extra></extra>"

    )


    fig.update_layout(

        height=520,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title="Relative Importance",

        yaxis_title=""

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # MODEL SUMMARY

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Accuracy",
        f"{accuracy:.1%}"
    )

    m2.metric(
        "Algorithm",
        "Decision Tree"
    )

    m3.metric(
        "Features",
        total_features
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">📁 Dataset Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Explore the underlying shopping behaviour dataset.</div>',
        unsafe_allow_html=True
    )


    d1, d2, d3, d4 = st.columns(4)


    d1.metric(
        "Records",
        f"{total_records:,}"
    )

    d2.metric(
        "Features",
        f"{total_features}"
    )

    d3.metric(
        "Total Revenue",
        f"${total_purchase:,.0f}"
    )

    d4.metric(
        "Average Rating",
        f"{avg_rating:.2f} / 5"
    )


    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=500
    )


    st.markdown("### Dataset Columns")

    st.write(
        df.columns.tolist()
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>Shopping Behaviour Intelligence</b><br>
Decision Tree Machine Learning • Python • Pandas • Scikit-learn • Plotly • Streamlit

</div>
""", unsafe_allow_html=True)