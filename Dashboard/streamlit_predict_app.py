import pickle
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Shopping Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main container */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    /* Forms */
    div[data-testid="stForm"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.05);
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    /* Section spacing */
    .section-space {
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Model" / "decision_tree_model.pkl"
ENCODERS_PATH = BASE_DIR / "Model" / "label_encoders.pkl"
METRICS_PATH = BASE_DIR / "Model" / "model_metrics.json"
DATA_PATH = BASE_DIR / "Dataset" / "shopping_behavior_cleaned.csv"


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_artifacts():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(ENCODERS_PATH, "rb") as file:
        encoders = pickle.load(file)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    df = pd.read_csv(DATA_PATH)

    return model, encoders, metrics, df


try:

    model, encoders, metrics, df = load_artifacts()

except Exception as error:

    st.error("Unable to load project files.")

    st.code(str(error))

    st.stop()


# ============================================================
# DATA CALCULATIONS
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

    high_value_rate = 0.0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛍️ Shopping AI")

    st.caption("Customer Intelligence Platform")

    st.divider()

    st.subheader("Dashboard")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Customer Prediction",
            "Analytics",
            "Model Insights",
            "Dataset",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Model")

    st.success("Decision Tree • Online")

    st.caption(
        f"Model Accuracy: {accuracy:.1%}"
    )

    st.divider()

    st.caption(
        "Shopping Behaviour Intelligence"
    )

    st.caption(
        "Python • Pandas • Scikit-learn • Plotly • Streamlit"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🛍️ Shopping Behaviour Intelligence")

st.write(
    "A machine-learning platform for understanding customer "
    "behaviour, purchasing patterns, and high-value customers."
)

st.info(
    f"🎯 Decision Tree Model Accuracy: **{accuracy:.1%}**"
)

st.divider()


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Business Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy:.1%}",
        help="Accuracy of the Decision Tree model.",
    )


with k2:

    st.metric(
        label="👥 Customer Records",
        value=f"{total_records:,}",
        help="Number of records in the dataset.",
    )


with k3:

    st.metric(
        label="💰 Average Purchase",
        value=f"${avg_purchase:,.2f}",
        help="Average customer purchase amount.",
    )


with k4:

    st.metric(
        label="⭐ High Value Rate",
        value=f"{high_value_rate:.1%}",
        help="Percentage of customers classified as high value.",
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.divider()

    st.header("📊 Executive Overview")

    st.caption(
        "Explore customer demographics, revenue distribution, "
        "and purchasing behaviour."
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
            ascending=False,
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
    # AGE GROUP
    # --------------------------------------------------------

    if "Age Group" in df.columns:

        age_order = [
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-70",
        ]

        age_counts = (
            df["Age Group"]
            .value_counts()
            .reindex(
                age_order,
                fill_value=0,
            )
            .rename_axis("Age Group")
            .reset_index(name="Customers")
        )

    else:

        age_group = pd.cut(
            df["Age"],
            bins=[17, 25, 35, 45, 55, 70],
            labels=[
                "18-25",
                "26-35",
                "36-45",
                "46-55",
                "56-70",
            ],
        )

        age_counts = (
            age_group
            .value_counts()
            .sort_index()
            .rename_axis("Age Group")
            .reset_index(name="Customers")
        )

    # --------------------------------------------------------
    # REVENUE BY CATEGORY
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            category_sales,
            names="Category",
            values="Purchase Amount (USD)",
            hole=0.55,
            title="Revenue Distribution by Category",
        )

        fig.update_layout(
            height=430,
            margin=dict(
                t=60,
                b=20,
                l=10,
                r=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CUSTOMER GENDER
    # --------------------------------------------------------

    with col2:

        fig = px.pie(
            gender_counts,
            names="Gender",
            values="Customers",
            hole=0.55,
            title="Customer Demographics",
        )

        fig.update_layout(
            height=430,
            margin=dict(
                t=60,
                b=20,
                l=10,
                r=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        fig = px.bar(
            age_counts,
            x="Age Group",
            y="Customers",
            text="Customers",
            title="Customer Distribution by Age",
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            margin=dict(
                t=60,
                b=20,
                l=10,
                r=10,
            ),
            xaxis_title="Age Group",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CATEGORY REVENUE
    # --------------------------------------------------------

    with col4:

        fig = px.bar(
            category_sales,
            x="Purchase Amount (USD)",
            y="Category",
            orientation="h",
            text="Purchase Amount (USD)",
            title="Revenue by Product Category",
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
        )

        fig.update_layout(
            height=430,
            margin=dict(
                t=60,
                b=20,
                l=10,
                r=10,
            ),
            xaxis_title="Revenue",
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":

    st.header("🎯 Customer Intelligence Engine")

    st.caption(
        "Enter customer details and let the Decision Tree "
        "model predict the customer segment."
    )

    st.divider()

    with st.form("prediction_form"):

        st.subheader("Customer Profile")

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider(
                "Customer Age",
                min_value=18,
                max_value=70,
                value=32,
            )

            gender = st.selectbox(
                "Gender",
                encoders["Gender"].classes_,
            )

            category = st.selectbox(
                "Product Category",
                encoders["Category"].classes_,
            )

        with col2:

            item = st.selectbox(
                "Product Purchased",
                encoders["Item Purchased"].classes_,
            )

            amount = st.number_input(
                "Purchase Amount (USD)",
                min_value=1,
                max_value=500,
                value=50,
            )

        submitted = st.form_submit_button(
            "🚀 Analyze Customer",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        try:

            row = pd.DataFrame(
                [
                    {
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

                        "Purchase Amount (USD)": amount,
                    }
                ]
            )

            prediction = model.predict(row)[0]

            probabilities = model.predict_proba(row)[0]

            if prediction == 1:

                label = "High Value Customer"

                confidence = float(
                    probabilities[1]
                )

                icon = "🟢"

            else:

                label = "Standard Customer"

                confidence = float(
                    probabilities[0]
                )

                icon = "🔵"

            st.divider()

            if prediction == 1:

                st.success(
                    f"{icon} Prediction: **{label}**"
                )

            else:

                st.info(
                    f"{icon} Prediction: **{label}**"
                )

            st.write(
                f"The model predicts this customer as "
                f"**{label}** with a confidence of "
                f"**{confidence:.1%}**."
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Customer Segment",
                    label,
                )

            with r2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.1%}",
                )

            with r3:

                st.metric(
                    "Purchase Amount",
                    f"${amount:,.0f}",
                )

            # ------------------------------------------------
            # CONFIDENCE CHART
            # ------------------------------------------------

            st.subheader("Prediction Confidence")

            fig = go.Figure()

            fig.add_trace(
                go.Pie(
                    labels=[
                        "Confidence",
                        "Remaining",
                    ],
                    values=[
                        confidence,
                        1 - confidence,
                    ],
                    hole=0.70,
                    textinfo="none",
                )
            )

            fig.update_layout(
                height=350,
                showlegend=False,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20,
                ),
                annotations=[
                    {
                        "text":
                            f"<b>{confidence:.0%}</b>"
                            "<br>Confidence",

                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {
                            "size": 24
                        },
                    }
                ],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(error)
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.header("📈 Advanced Customer Analytics")

    st.caption(
        "Explore spending behaviour, customer ratings, "
        "and product performance."
    )

    st.divider()

    # --------------------------------------------------------
    # PURCHASE DISTRIBUTION
    # --------------------------------------------------------

    fig = px.histogram(
        df,
        x="Purchase Amount (USD)",
        nbins=30,
        marginal="box",
        title="Purchase Amount Distribution",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Purchase Amount (USD)",
        yaxis_title="Customers",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # RATING VS PURCHASE
    # --------------------------------------------------------

    if "Review Rating" in df.columns:

        st.subheader(
            "⭐ Rating vs Purchase Amount"
        )

        fig = px.scatter(
            df,
            x="Review Rating",
            y="Purchase Amount (USD)",
            color="Category",
            size="Purchase Amount (USD)",
            hover_data=[
                "Age",
                "Gender",
            ],
            title="Customer Rating vs Purchase Amount",
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

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

    st.subheader(
        "🏆 Top 10 Products by Revenue"
    )

    fig = px.bar(
        product_sales,
        x="Purchase Amount (USD)",
        y="Item Purchased",
        orientation="h",
        text="Purchase Amount (USD)",
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
    )

    fig.update_layout(
        height=500,
        xaxis_title="Revenue",
        yaxis_title="Product",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.header("🧠 Model Intelligence")

    st.caption(
        "Understand the variables that influence "
        "the Decision Tree prediction."
    )

    st.divider()

    feature_importance = (
        pd.Series(
            metrics["feature_importance"],
            name="Importance",
        )
        .sort_values(
            ascending=True
        )
        .reset_index()
    )

    feature_importance.columns = [
        "Feature",
        "Importance",
    ]

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Decision Tree Feature Importance",
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    fig.update_layout(
        height=520,
        xaxis_title="Relative Importance",
        yaxis_title="Feature",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Model Summary")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Accuracy",
            f"{accuracy:.1%}",
        )

    with m2:

        st.metric(
            "Algorithm",
            "Decision Tree",
        )

    with m3:

        st.metric(
            "Features",
            total_features,
        )

    st.divider()

    st.write(
        "The feature importance chart shows how strongly "
        "each input variable contributes to the model's "
        "decision-making process."
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.header("📁 Dataset Intelligence")

    st.caption(
        "Explore the dataset used by the Shopping Behaviour "
        "Machine Learning platform."
    )

    st.divider()

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Records",
            f"{total_records:,}",
        )

    with d2:

        st.metric(
            "Features",
            f"{total_features}",
        )

    with d3:

        st.metric(
            "Total Revenue",
            f"${total_purchase:,.0f}",
        )

    with d4:

        st.metric(
            "Average Rating",
            f"{avg_rating:.2f} / 5",
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=500,
    )

    st.subheader("Dataset Columns")

    st.write(
        df.columns.tolist()
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛍️ Shopping Behaviour Intelligence • "
    "Decision Tree Machine Learning • "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)