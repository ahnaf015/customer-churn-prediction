"""Streamlit dashboard for churn prediction and AI retention."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from settings import (
    RAW_DATA_FILE,
    MODEL_FILE,
    FEATURE_COLS_FILE,
    FEATURE_IMPORTANCE_FILE,
    SHAP_SUMMARY_PLOT,
    REPORTS_DIR,
)
from preprocess import clean_data, encode_categoricals, get_feature_columns
from feature_engineering import engineer_features
from retention_agent import RetentionAgent

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    return model, feature_cols


@st.cache_resource
def load_feature_importance():
    return joblib.load(FEATURE_IMPORTANCE_FILE)


@st.cache_resource
def get_retention_agent():
    return RetentionAgent()


# ── Helper ───────────────────────────────────────────────────────────────────

def process_and_predict(df_raw: pd.DataFrame):
    """Run full pipeline: clean → features → encode → predict."""

    # Keep a copy with original columns for display
    df_display = df_raw.copy()
    df_display["TotalCharges"] = pd.to_numeric(df_display["TotalCharges"], errors="coerce")
    df_display["TotalCharges"] = df_display["TotalCharges"].fillna(df_display["TotalCharges"].median())

    # Pipeline
    df = clean_data(df_raw)
    df = engineer_features(df)

    # Stash display-friendly columns before encoding
    customer_ids = df["customerID"].values
    tenure_vals = df["tenure"].values
    monthly_vals = df["MonthlyCharges"].values
    clv_vals = df["CLV"].values

    df = encode_categoricals(df)

    model, feature_cols = load_model()
    X = df[feature_cols]

    churn_proba = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "customerID": customer_ids,
        "Churn_Probability": churn_proba,
        "tenure": tenure_vals,
        "MonthlyCharges": monthly_vals,
        "CLV": clv_vals,
    })
    results["Risk_Level"] = pd.cut(
        results["Churn_Probability"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        '<p class="main-header">🎯 Customer Churn Prediction + AI Retention</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Identify at-risk customers 30 days in advance and generate personalized retention campaigns</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📁 Data Upload")
    st.sidebar.markdown("Upload your customer data CSV file")

    uploaded_file = st.sidebar.file_uploader(
        "Choose file",
        type=["csv"],
        help="Upload CSV with customer data (Telco format)",
    )
    use_demo = st.sidebar.checkbox("Use Demo Data", value=False)

    if uploaded_file is None and not use_demo:
        st.info("👈 Upload customer data or enable demo mode to get started")
        st.markdown("""
        ### Features
        - Predict customer churn 30 days in advance
        - Identify top at-risk customers
        - Generate personalized retention emails with AI
        - Calculate optimal discount offers based on CLV
        - Model explainability with SHAP values

        ### Data Format
        Upload a CSV file matching the Telco Customer Churn dataset format, or use the built-in demo data.
        """)
        return

    # ── Load data ────────────────────────────────────────────────────────────
    if use_demo:
        df_raw = pd.read_csv(RAW_DATA_FILE)
        st.sidebar.success("Demo data loaded")
    else:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully")

    with st.spinner("Processing data and running predictions..."):
        results = process_and_predict(df_raw)

    # ── Top metrics ──────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    high_risk = (results["Risk_Level"] == "High").sum()
    avg_risk = results["Churn_Probability"].mean()
    clv_at_risk = results.loc[results["Risk_Level"] == "High", "CLV"].sum()

    c1.metric("Total Customers", f"{len(results):,}")
    c2.metric(
        "High Risk",
        f"{high_risk:,}",
        delta=f"{high_risk / len(results) * 100:.1f}%",
        delta_color="inverse",
    )
    c3.metric("Avg Churn Risk", f"{avg_risk:.1%}")
    c4.metric("CLV at Risk", f"${clv_at_risk:,.0f}")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "📧 Retention Emails",
        "🔍 Model Insights",
    ])

    # ──────────────────────────── TAB 1: Dashboard ───────────────────────────
    with tab1:
        st.subheader("Risk Distribution")

        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                results,
                x="Churn_Probability",
                nbins=30,
                title="Churn Probability Distribution",
                labels={"Churn_Probability": "Churn Probability"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            risk_counts = results["Risk_Level"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "Low": "#2ecc71",
                    "Medium": "#f39c12",
                    "High": "#e74c3c",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top at-risk table
        st.subheader("🚨 Top 20 At-Risk Customers")

        top_risk = results.nlargest(20, "Churn_Probability").copy()
        display = top_risk.copy()
        display["Churn_Probability"] = display["Churn_Probability"].apply(lambda x: f"{x:.1%}")
        display["CLV"] = display["CLV"].apply(lambda x: f"${x:,.0f}")
        display["MonthlyCharges"] = display["MonthlyCharges"].apply(lambda x: f"${x:.2f}")

        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    # ──────────────────────── TAB 2: Retention Emails ────────────────────────
    with tab2:
        st.subheader("📧 AI-Powered Retention Email Generator")
        st.markdown(
            "Select an at-risk customer to generate a personalized retention email "
            "using AI and customer interaction history."
        )

        high_risk_df = results[results["Risk_Level"] == "High"].nlargest(
            50, "Churn_Probability"
        )

        if high_risk_df.empty:
            st.warning("No high-risk customers found.")
        else:
            col_sel, col_btn = st.columns([2, 1])

            with col_sel:
                selected_id = st.selectbox(
                    "Select Customer",
                    high_risk_df["customerID"].tolist(),
                    help="Choose from top 50 at-risk customers",
                )

            with col_btn:
                generate_btn = st.button(
                    "🤖 Generate Email",
                    type="primary",
                    use_container_width=True,
                )

            if generate_btn:
                customer = results[results["customerID"] == selected_id].iloc[0]

                st.markdown("---")
                st.markdown("**Customer Profile:**")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Churn Risk", f"{customer['Churn_Probability']:.1%}")
                mc2.metric("Tenure", f"{int(customer['tenure'])} months")
                mc3.metric("Monthly Charges", f"${customer['MonthlyCharges']:.2f}")
                mc4.metric("CLV", f"${customer['CLV']:,.0f}")

                with st.spinner("🤖 AI is crafting a personalized email..."):
                    agent = get_retention_agent()
                    result = agent.generate_retention_email(
                        customer_id=selected_id,
                        churn_probability=customer["Churn_Probability"],
                        customer_data={
                            "tenure": int(customer["tenure"]),
                            "MonthlyCharges": customer["MonthlyCharges"],
                            "CLV": customer["CLV"],
                        },
                    )

                st.success(f"Email generated with {result['discount']}% discount offer")

                st.markdown("---")
                st.markdown("**Generated Retention Email:**")

                email_html = result["email"].replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px;
                                border-radius: 10px; border-left: 5px solid #1f77b4;
                                color: #1a1a1a;">
                    <p><strong style="color: #333;">To:</strong> {selected_id}</p>
                    <p><strong style="color: #333;">Subject:</strong> We Value Your Loyalty — Special Offer Inside</p>
                    <hr style="border-color: #ccc;">
                    <div style="color: #1a1a1a;">{email_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    label="📋 Download Email",
                    data=result["email"],
                    file_name=f"retention_email_{selected_id}.txt",
                    mime="text/plain",
                )

    # ──────────────────────── TAB 3: Model Insights ──────────────────────────
    with tab3:
        st.subheader("🔍 Model Performance & Insights")

        # Feature importance chart
        fi = load_feature_importance()

        fig = px.bar(
            fi.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Most Important Features",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model metrics
        st.markdown("---")
        st.markdown("**Model Performance Metrics:**")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("AUC Score", "0.848")
        mc2.metric("Precision (Churn)", "66%")
        mc3.metric("Recall (Churn)", "54%")

        # SHAP plot
        if st.checkbox("Show SHAP Explanation for Sample Customers"):
            st.markdown("**SHAP Values Explanation:**")
            st.info("SHAP values show which features contributed most to each prediction.")

            shap_path = SHAP_SUMMARY_PLOT
            if shap_path.exists():
                st.image(str(shap_path))
            else:
                st.warning("SHAP plot not available. Run `python src/explain_model.py` first.")


if __name__ == "__main__":
    main()
