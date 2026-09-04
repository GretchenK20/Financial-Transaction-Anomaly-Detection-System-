"""
Financial Transaction Anomaly Detection Dashboard
Streamlit app — calls a FastAPI endpoint for live scoring when reachable,
and falls back to pre-computed results in demo_data.json otherwise (e.g.
when deployed to Streamlit Community Cloud with no API to call).

No PyTorch / XGBoost / DuckDB / MLflow imports here — this file must run
with only requirements_streamlit.txt installed.
"""
import json
import os
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

API_BASE = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
DEMO_DATA_PATH = Path(__file__).parent / "demo_data.json"
ANOMALY_SCORES_PATH = Path(__file__).parent / "models" / "autoencoder" / "anomaly_scores.parquet"


@st.cache_data
def load_demo_data() -> dict:
    if not DEMO_DATA_PATH.exists():
        return {}
    with open(DEMO_DATA_PATH) as f:
        return json.load(f)


DEMO = load_demo_data()


def api_get(path: str, base_url: str, timeout: float = 5.0):
    """GET from the live API. Returns (json, True) on success, (None, False) on any failure."""
    try:
        r = requests.get(f"{base_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), True
    except Exception:
        return None, False


def score_transaction(transaction_id, base_url: str):
    """
    Try the live API first; fall back to demo_data.json if unreachable.
    Returns (data, source) where source is "live", "demo", or None (unavailable).
    """
    try:
        r = requests.get(f"{base_url}/transaction/{int(transaction_id)}", timeout=10)
        if r.status_code == 200:
            return r.json(), "live"
        if r.status_code == 404:
            return None, "not_found"
    except Exception:
        pass

    demo_record = DEMO.get("transactions", {}).get(str(int(transaction_id)))
    if demo_record:
        return demo_record, "demo"
    return None, None


st.set_page_config(
    page_title="Financial Transaction Anomaly Detection System",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Financial Transaction Anomaly Detection System")
st.caption("284K credit card transactions → dbt medallion → PyTorch autoencoder + XGBoost champion/challenger → LangChain agent")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    health, live = api_get("/health", API_BASE, timeout=3)

    if live:
        st.success(f"API: {health['status']} (live)")
        st.info(f"Champion: **{health['champion']}**")
        st.write(f"Autoencoder loaded: {'✅' if health['ae_loaded'] else '❌'}")
        st.write(f"XGBoost loaded: {'✅' if health['xgb_loaded'] else '❌'}")
    else:
        st.warning("🟡 Demo Mode — API unreachable, showing pre-computed results")
        if DEMO.get("champion"):
            st.info(f"Champion: **{DEMO['champion'].get('champion', 'N/A')}** (from demo data)")

    st.divider()
    st.header("Champion Model")
    champ = None
    if live:
        champ, _ = api_get("/champion", API_BASE, timeout=3)
    if champ is None:
        champ = DEMO.get("champion")

    if champ:
        history = champ.get("history", [])
        if history:
            latest = history[-1]
            metrics = latest.get("metrics", {})
            ae = metrics.get("autoencoder", {})
            xgb = metrics.get("xgboost", {})
            col1, col2 = st.columns(2)
            col1.metric("AE AUC", f"{ae.get('auc', 'N/A')}")
            col2.metric("XGB AUC", f"{xgb.get('auc', 'N/A')}")

    st.divider()
    api_url = st.text_input("API Base URL", value=API_BASE)
    if DEMO.get("transactions"):
        st.caption(
            "Demo transaction IDs available offline: "
            + ", ".join(sorted(DEMO["transactions"].keys(), key=int))
        )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Transaction Scoring", "Batch Analysis", "Fraud Statistics", "About"])

# ── Tab 1: Transaction Scoring ───────────────────────────────────────────────
with tab1:
    st.header("Transaction Fraud Assessment")

    col1, col2 = st.columns([2, 1])
    with col1:
        transaction_id = st.number_input(
            "Transaction ID",
            min_value=0,
            max_value=284806,
            value=0,
            step=1,
            help="Row index into the 284,807-transaction dataset (e.g. 0-284806)",
        )
    with col2:
        st.write("")
        st.write("")
        score_btn = st.button("Score Transaction", type="primary", use_container_width=True)

    if score_btn:
        with st.spinner("Scoring transaction..."):
            d, source = score_transaction(transaction_id, api_url)

            if source == "not_found":
                st.error(f"Transaction {transaction_id} not found")
            elif d is None:
                demo_ids = ", ".join(sorted(DEMO.get("transactions", {}).keys(), key=int)) or "none"
                st.error(
                    f"API unreachable and transaction {transaction_id} isn't in the offline demo "
                    f"set. Try one of these demo IDs instead: {demo_ids}"
                )
            else:
                if source == "demo":
                    st.info("🟡 Demo Mode — showing a pre-computed result (API unreachable)")

                # Risk score display
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Risk Score", f"{d['risk_score']:.3f}")
                c2.metric("Percentile", f"{d['risk_percentile']}th")
                c3.metric("Flagged Fraud", "🚨 YES" if d['is_high_risk'] else "✅ NO")
                c4.metric("Champion Model", (d['champion_model'] or "N/A").upper())

                st.divider()

                col_left, col_right = st.columns(2)

                # Risk gauge
                with col_left:
                    st.subheader("Risk Score Gauge")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=d['risk_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fraud Risk Score"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "#E63946" if d['is_high_risk'] else "#2A9D8F"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "#E9F5DB"},
                                {'range': [0.3, 0.6], 'color': "#FFE8A1"},
                                {'range': [0.6, 1.0], 'color': "#FFB3B3"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5,
                            },
                        },
                    ))
                    fig.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                # SHAP waterfall
                with col_right:
                    st.subheader("Top Fraud Drivers (SHAP)")
                    factors = d.get("top_risk_factors", [])
                    if factors:
                        features = [f["feature"].replace("_", " ") for f in factors]
                        values = [f["shap_value"] for f in factors]
                        colors = ["#E63946" if v > 0 else "#2A9D8F" for v in values]

                        fig2 = go.Figure(go.Bar(
                            x=values,
                            y=features,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:+.4f}" for v in values],
                            textposition='outside',
                        ))
                        fig2.update_layout(
                            height=300,
                            margin=dict(t=20, b=0, l=20, r=60),
                            xaxis_title="SHAP Value (impact on fraud risk)",
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                # Explanation
                st.info(f"**Fraud Explanation:** {d['explanation']}")

                # Raw scores
                with st.expander("Raw Model Scores"):
                    col_ae, col_xgb = st.columns(2)
                    col_ae.metric(
                        "Autoencoder Reconstruction Error",
                        f"{d['ae_reconstruction_error']:.4f}" if d['ae_reconstruction_error'] else "N/A"
                    )
                    col_xgb.metric(
                        "XGBoost Probability",
                        f"{d['xgb_probability']:.4f}" if d['xgb_probability'] else "N/A"
                    )
                    if "is_fraud" in d:
                        st.metric("Ground Truth (Class label)", "Fraud" if d["is_fraud"] else "Legitimate")

# ── Tab 2: Batch Analysis ────────────────────────────────────────────────────
with tab2:
    st.header("Batch Transaction Analysis")
    st.caption("Score multiple transactions and analyze risk distribution")

    default_ids = (
        sorted((int(i) for i in DEMO["transactions"]), key=int)
        if DEMO.get("transactions")
        else [0, 1, 2, 3]
    )

    ids_input = st.text_area(
        "Transaction IDs (one per line)",
        value="\n".join(str(i) for i in default_ids),
        height=120,
    )
    batch_btn = st.button("Run Batch Scoring", type="primary")

    if batch_btn:
        ids = [i.strip() for i in ids_input.strip().split("\n") if i.strip()]
        with st.spinner(f"Scoring {len(ids)} transactions..."):
            results = []
            errors = []
            any_demo = False
            for tid in ids:
                d, source = score_transaction(tid, api_url)
                if d is not None:
                    results.append(d)
                    any_demo = any_demo or (source == "demo")
                else:
                    errors.append({"transaction_id": tid, "error": source or "unavailable"})

            if any_demo:
                st.info("🟡 Some results shown are pre-computed demo data (API unreachable)")

            if results:
                df = pd.DataFrame([{
                    "transaction_id": r["transaction_id"],
                    "risk_score": r["risk_score"],
                    "percentile": r["risk_percentile"],
                    "flagged_fraud": r["is_high_risk"],
                    "top_driver": r["top_risk_factors"][0]["feature"].replace("_", " ") if r["top_risk_factors"] else "N/A",
                } for r in results])

                c1, c2, c3 = st.columns(3)
                c1.metric("Transactions Scored", len(results))
                c2.metric("Flagged Fraud", int(df["flagged_fraud"].sum()))
                c3.metric("Avg Risk Score", f"{df['risk_score'].mean():.3f}")

                col_l, col_r = st.columns(2)
                with col_l:
                    fig = px.bar(
                        df, x="transaction_id", y="risk_score",
                        color="flagged_fraud",
                        color_discrete_map={True: "#E63946", False: "#2A9D8F"},
                        title="Risk Scores by Transaction",
                        labels={"risk_score": "Risk Score", "transaction_id": "Transaction"},
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                    st.plotly_chart(fig, use_container_width=True)

                with col_r:
                    fig2 = px.histogram(
                        df, x="risk_score", nbins=10,
                        title="Risk Score Distribution",
                        color_discrete_sequence=["#2A9D8F"],
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.dataframe(df, use_container_width=True)

            if errors:
                st.warning(f"{len(errors)} transaction(s) could not be scored")
                st.dataframe(pd.DataFrame(errors), use_container_width=True)

# ── Tab 3: Fraud Statistics ──────────────────────────────────────────────────
with tab3:
    st.header("Dataset Fraud Statistics")
    st.caption("Computed from the real ULB credit card fraud dataset (284,807 transactions)")

    scores_df = None
    try:
        scores_df = pd.read_parquet(ANOMALY_SCORES_PATH)
    except Exception:
        scores_df = None  # file missing, or parquet support (pyarrow) not installed — e.g. on Streamlit Cloud

    if scores_df is not None:
        n_total = len(scores_df)
        n_fraud = int(scores_df["is_fraud"].sum())
        fraud_rate = n_fraud / n_total * 100
        n_flagged = int(scores_df["is_anomaly"].sum())
        flagged_rate = n_flagged / n_total * 100
        n_caught = int(((scores_df["is_fraud"] == 1) & (scores_df["is_anomaly"] == 1)).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{n_total:,}")
        c2.metric("Real Fraud Rate", f"{fraud_rate:.2f}%", help="Ground-truth Class=1 rate in the dataset")
        c3.metric("Flagged by Autoencoder", f"{n_flagged:,} ({flagged_rate:.1f}%)")
        c4.metric("Fraud Caught by AE", f"{n_caught} / {n_fraud}")

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.histogram(
                scores_df, x="reconstruction_error", color="is_fraud",
                nbins=60, title="Reconstruction Error: Fraud vs. Legitimate",
                color_discrete_map={0: "#2A9D8F", 1: "#E63946"},
                labels={"is_fraud": "Is Fraud"},
                log_y=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            counts = pd.DataFrame({
                "category": ["Legitimate", "Fraud"],
                "count": [n_total - n_fraud, n_fraud],
            })
            fig2 = px.bar(
                counts, x="category", y="count", log_y=True,
                title=f"Class Balance ({fraud_rate:.2f}% fraud)",
                color="category",
                color_discrete_map={"Legitimate": "#2A9D8F", "Fraud": "#E63946"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif DEMO.get("dataset_stats"):
        st.info("🟡 Demo Mode — showing pre-computed dataset statistics (local data files unavailable)")
        stats = DEMO["dataset_stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{stats['n_total']:,}")
        c2.metric("Real Fraud Rate", f"{stats['fraud_rate_pct']:.2f}%", help="Ground-truth Class=1 rate in the dataset")
        c3.metric("Flagged by Autoencoder", f"{stats['n_flagged_by_autoencoder']:,} ({stats['flagged_rate_pct']:.1f}%)")
        c4.metric("Fraud Caught by AE", f"{stats['n_fraud_caught_by_autoencoder']} / {stats['n_fraud']}")

        hist = DEMO.get("reconstruction_error_histogram")
        col_l, col_r = st.columns(2)
        with col_l:
            if hist:
                rows = []
                for label, key in [("Legitimate", "legitimate"), ("Fraud", "fraud")]:
                    edges = hist[key]["bin_edges"]
                    counts = hist[key]["counts"]
                    for i, c in enumerate(counts):
                        rows.append({"bin_center": (edges[i] + edges[i + 1]) / 2, "count": c, "class": label})
                hist_df = pd.DataFrame(rows)
                fig = px.bar(
                    hist_df, x="bin_center", y="count", color="class",
                    barmode="overlay", log_y=True,
                    title="Reconstruction Error: Fraud vs. Legitimate",
                    color_discrete_map={"Legitimate": "#2A9D8F", "Fraud": "#E63946"},
                    labels={"bin_center": "reconstruction_error"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            counts = pd.DataFrame({
                "category": ["Legitimate", "Fraud"],
                "count": [stats["n_total"] - stats["n_fraud"], stats["n_fraud"]],
            })
            fig2 = px.bar(
                counts, x="category", y="count", log_y=True,
                title=f"Class Balance ({stats['fraud_rate_pct']:.2f}% fraud)",
                color="category",
                color_discrete_map={"Legitimate": "#2A9D8F", "Fraud": "#E63946"},
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No local data and no demo_data.json found — run model training first.")

# ── Tab 4: About ─────────────────────────────────────────────────────────────
with tab4:
    st.header("About This System")
    st.markdown("""
    ## Financial Transaction Anomaly Detection System

    A production fraud-detection AI pipeline built on the real ULB "Credit Card Fraud
    Detection" dataset — 284,807 European card transactions from September 2013,
    with a 0.17% fraud rate (492 confirmed frauds).

    ### Architecture
    ```
    creditcard.csv → DuckDB Bronze → dbt Silver/Gold → ML Models → FastAPI → This Dashboard
    ```

    ### Pipeline Components
    | Layer | Technology | Description |
    |-------|-----------|-------------|
    | Ingestion | Python, DuckDB | Loads the raw transaction CSV into DuckDB bronze |
    | Transform | dbt, DuckDB | Bronze → Silver → Gold medallion architecture |
    | Models | PyTorch, XGBoost | Autoencoder anomaly detection + supervised fraud classifier |
    | Governance | SHAP, MLflow | Explainability and champion/challenger tracking |
    | API | FastAPI | REST endpoint with SHAP explanations |
    | Agent | LangChain, ChromaDB | Agentic AI with similar-fraud case retrieval |
    | Dashboard | Streamlit, Plotly | This interface |

    ### Data
    - **Source:** ULB Machine Learning Group's Credit Card Fraud Detection dataset (Kaggle: `mlg-ulb/creditcardfraud`)
    - **Features:** V1-V28 (PCA-anonymized transaction features) + Amount, Time, and engineered features
    - **Fraud rate:** 0.17% (492 of 284,807 transactions)

    ### Model Performance
    - **XGBoost (trained on real fraud labels):** ROC-AUC ~1.00, CV AUC ~0.98
    - **Autoencoder (unsupervised):** AUC ~0.93 vs. real fraud labels
    - **Champion:** XGBoost (promoted by champion/challenger framework, evaluated against ground-truth fraud labels)

    ### Deployment modes
    This dashboard runs in two modes:
    - **Live mode** — when the FastAPI backend (`api/main.py`) is reachable at the configured
      URL, every score is computed fresh from the trained models.
    - **Demo mode** — when deployed somewhere without that backend (e.g. Streamlit Community
      Cloud), the app automatically falls back to `demo_data.json`, a small set of real,
      pre-computed model outputs for a handful of transactions, plus real dataset-wide
      statistics — no live model inference, no fabricated numbers.

    Set the `STREAMLIT_API_URL` environment variable (or Streamlit secrets) to point this app
    at a real deployed API.
    """)
