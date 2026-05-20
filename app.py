"""app.py – Streamlit dashboard for TOTO Analytics & Ensemble Predictor."""
import streamlit as st
import pandas as pd
from toto_engine import (
    load_data, number_frequency_chart, overdue_analysis,
    pair_heatmap, hot_cold_table, weighted_lucky_pick, predict_ensemble
)

st.set_page_config(page_title="TOTO Predictor", layout="wide")
st.title("🎰 Singapore TOTO Analytics & Ensemble Predictor")
st.caption("Institutional‑grade lottery analytics pipeline – for educational purposes only.")

@st.cache_data(ttl=3600)
def get_data():
    return load_data()

df = get_data()

if df.empty:
    st.error("No data found. Run `python toto_engine.py scrape` or `backfill` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔮 Predictions", "📁 Raw Data"])

with tab1:
    st.header("Number Frequency")
    recent = st.slider("Recent draws to consider", 20, len(df), 100)
    fig = number_frequency_chart(df, recent_n=recent)
    st.plotly_chart(fig, use_container_width=True)

    st.header("Overdue Numbers")
    overdue = overdue_analysis(df)
    st.dataframe(overdue.head(15), use_container_width=True)

    st.header("Pair Co-occurrence")
    fig2 = pair_heatmap(df, recent_n=200)
    st.plotly_chart(fig2, use_container_width=True)

    st.header("Hot / Cold Numbers")
    hc = hot_cold_table(df, top_n=10)
    st.dataframe(hc, use_container_width=True)

with tab2:
    st.header("Next Draw Prediction (Ensemble: LSTM + Transformer)")
    st.warning("⚠️ Lottery draws are random. No model can predict winning numbers. Educational purposes only.")

    ensemble_pred = predict_ensemble()
    base_main, base_add = weighted_lucky_pick(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ensemble Model (LSTM+Transformer)")
        if ensemble_pred:
            main_str = ", ".join(map(str, ensemble_pred[0]))
            st.success(f"**Main numbers:** {main_str}")
            st.info(f"**Additional number:** {ensemble_pred[1]}")
        else:
            st.info("Ensemble model not trained yet. Run `python toto_engine.py train`.")
    with col2:
        st.subheader("Baseline (Weighted Random)")
        st.success(f"**Main numbers:** {', '.join(map(str, base_main))}")
        st.info(f"**Additional number:** {base_add}")

    # Overlap with last draw
    if len(df) >= 1:
        last_draw = df.iloc[-1]
        actual_main = set(last_draw[["n1","n2","n3","n4","n5","n6"]].values)
        actual_add = last_draw["additional"]

        ens_match_main = len(actual_main.intersection(ensemble_pred[0])) if ensemble_pred else 0
        ens_match_add = (ensemble_pred[1] == actual_add) if ensemble_pred else False
        base_match_main = len(actual_main.intersection(base_main))
        base_match_add = (base_add == actual_add)

        st.markdown("---")
        st.subheader("Overlap with Last Draw")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ensemble – Main matches", f"{ens_match_main}/6")
            st.metric("Ensemble – Add correct", "✅" if ens_match_add else "❌")
        with col2:
            st.metric("Baseline – Main matches", f"{base_match_main}/6")
            st.metric("Baseline – Add correct", "✅" if base_match_add else "❌")

with tab3:
    st.header("Historical Results")
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "toto_results.csv", "text/csv")
