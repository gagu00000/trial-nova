# NovaMart Marketing Analytics Dashboard
# Complete Streamlit App Skeleton (awaiting dataset integration)

import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------
# DATA LOADING
# --------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# Preload all datasets (files must be in ./data/ on GitHub)
campaign_df = load_csv("data/campaign_performance.csv")
customer_df = load_csv("data/customer_data.csv")
product_df = load_csv("data/product_sales.csv")
lead_df = load_csv("data/lead_scoring_results.csv")
feature_df = load_csv("data/feature_importance.csv")
learning_df = load_csv("data/learning_curve.csv")
geographic_df = load_csv("data/geographic_data.csv")
attribution_df = load_csv("data/channel_attribution.csv")
funnel_df = load_csv("data/funnel_data.csv")
journey_df = load_csv("data/customer_journey.csv")
corr_df = load_csv("data/correlation_matrix.csv")

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------
pages = [
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
    "Product Performance",
    "Geographic Analysis",
    "Attribution & Funnel",
    "ML Model Evaluation"
]

st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Go to:", pages)

# --------------------------
# PAGE FUNCTIONS
# --------------------------
def executive_overview():
    st.header("📌 Executive Overview")
    st.write("Key KPIs and high-level performance overview.")

    if campaign_df is not None:
        total_rev = campaign_df['revenue'].sum()
        total_conv = campaign_df['conversions'].sum()
        total_spend = campaign_df['spend'].sum()
        roas = total_rev / total_spend if total_spend > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"₹{total_rev:,.0f}")
        col2.metric("Conversions", f"{total_conv:,.0f}")
        col3.metric("Total Spend", f"₹{total_spend:,.0f}")
        col4.metric("ROAS", f"{roas:.2f}")

        st.subheader("Revenue Trend Over Time")
        fig = px.line(campaign_df, x='date', y='revenue', title='Revenue Trend')
        st.plotly_chart(fig, use_container_width=True)


def campaign_analytics():
    st.header("📢 Campaign Analytics")
    st.write("Time-series and comparison charts.")
    st.info("Full charts will populate once datasets are uploaded.")


def customer_insights():
    st.header("🧑‍🤝‍🧑 Customer Insights")
    st.write("Distribution and relationship visualizations.")
    st.info("Charts will display after dataset upload.")


def product_performance():
    st.header("🛒 Product Performance")
    st.write("Product hierarchy and category analytics.")


def geographic_analysis():
    st.header("🌍 Geographic Analysis")


def attribution_funnel():
    st.header("🔄 Attribution & Funnel Analysis")


def ml_model_evaluation():
    st.header("🤖 ML Model Evaluation")


# --------------------------
# PAGE ROUTER
# --------------------------
if page == "Executive Overview":
    executive_overview()
elif page == "Campaign Analytics":
    campaign_analytics()
elif page == "Customer Insights":
    customer_insights()
elif page == "Product Performance":
    product_performance()
elif page == "Geographic Analysis":
    geographic_analysis()
elif page == "Attribution & Funnel":
    attribution_funnel()
elif page == "ML Model Evaluation":
    ml_model_evaluation()
