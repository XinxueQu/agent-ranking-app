import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Alternate Ranking Method", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧪 Alternate Ranking Method</h1>", unsafe_allow_html=True)

# ---------------- Load shared data ----------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=csv"
    return pd.read_csv(url)

df = load_data()

# ---------------- New Filter Layout ----------------
st.sidebar.markdown("### 🔍 Filter Options (Alternate Page)")

zipcodes = st.sidebar.multiselect(
    "Zipcodes (multi-select)",
    sorted(df["PostalCode"].dropna().astype(str).unique())
)

price_range = st.sidebar.slider(
    "Price Range",
    min_value=int(df["ClosePrice"].min()),
    max_value=int(df["ClosePrice"].max()),
    value=(200000, 800000)
)

school = st.sidebar.text_input("Elementary School Contains")

min_sales = st.sidebar.number_input("Min Transactions", value=3)

st.sidebar.markdown("### ⚖️ Weighting")
w_sales  = st.sidebar.slider("Sales Weight",       0.0, 1.0, 0.4)
w_speed  = st.sidebar.slider("Speed Weight",       0.0, 1.0, 0.3)
w_close  = st.sidebar.slider("Close Rate Weight",  0.0, 1.0, 0.3)

run_alt = st.sidebar.button("Run Alternate Ranking")

# ---------------- Compute Ranking ----------------
if run_alt:

    df2 = df.copy()

    if zipcodes:
        df2 = df2[df2["PostalCode"].astype(str).isin(zipcodes)]

    df2 = df2[(df2["ClosePrice"] >= price_range[0]) &
              (df2["ClosePrice"] <= price_range[1])]

    if school:
        df2 = df2[df2["ElementarySchool"].astype(str).str.contains(school, case=False, na=False)]

    # ---- summarize by agent ----
    agent = df2.groupby("ListAgentFullName").agg(
        total_sales=("ClosePrice", "sum"),
        avg_dom=("DaysOnMarket", "mean"),
        close_rate=("is_closed", "mean"),
        n=("ListAgentFullName", "count")
    ).reset_index()

    agent = agent[agent["n"] >= min_sales]

    # ---- scores ----
    agent["sales_score"] = agent["total_sales"].rank(pct=True) * 100
    agent["speed_score"] = (1 - agent["avg_dom"].rank(pct=True)) * 100
    agent["close_score"] = agent["close_rate"].rank(pct=True) * 100

    agent["alt_score"] = (
        w_sales * agent["sales_score"] +
        w_speed * agent["speed_score"] +
        w_close * agent["close_score"]
    )

    agent = agent.sort_values("alt_score", ascending=False)

    st.subheader("📊 Alternate Ranked Agents")
    st.dataframe(agent, use_container_width=True)

    # ---- optional visualization ----
    fig = px.bar(agent.head(15), x="ListAgentFullName", y="alt_score",
                 title="Top Agents (Alternate Ranking)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Use the controls in the sidebar and click **Run Alternate Ranking**.")
