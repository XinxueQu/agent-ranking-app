import ast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Top 10 Agents", layout="wide")

st.title("🏅 Top 10 Agents")
st.write(
    "Select a geography and time window, then tune your price and quality filters to find the best-fit agents."
)


@st.cache_data
def load_data() -> pd.DataFrame:
    url = "https://www.dropbox.com/scl/fi/jg966zvvhdsdblmg9jhh8/transactions_2023.01.07_2026.01.06.xlsx?rlkey=gwk06io5pp4lhaa1v3d4f4oun&st=2f31dzw8&dl=1"
    usecols = [
        "ListAgentFullName",
        "is_closed",
        "DaysOnMarket",
        "pricing_accuracy",
        "City",
        "PostalCode",
        "ClosePrice",
        "ElementarySchool",
        "SubdivisionName",
        "CloseDate",
        "PropertyCondition",
        "ListingContractDate",
        "ListAgentDirectPhone",
    ]
    return pd.read_excel(url, usecols=usecols)


def is_resale(value) -> bool:
    if isinstance(value, list):
        return len(value) > 0 and value[0] == "Resale"
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return isinstance(parsed, list) and len(parsed) > 0 and parsed[0] == "Resale"
        except Exception:
            return False
    return False


def percentile_score(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(50.0, index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)


def pricing_accuracy_score(value: float) -> float:
    return (1 - abs(value - 1)) * 100


def score_days_on_market(series: pd.Series) -> pd.Series:
    return 100 - series.rank(pct=True) * 100


def to_top_percent_bucket(scores: pd.Series) -> pd.Series:
    pct_rank = scores.rank(pct=True, ascending=False, method="max") * 100

    def bucket(p: float) -> str:
        if p <= 1:
            return "Top 1%"
        if p <= 3:
            return "Top 3%"
        if p <= 5:
            return "Top 5%"
        if p <= 10:
            return "Top 10%"
        if p <= 20:
            return "Top 20%"
        if p <= 30:
            return "Top 30%"
        if p <= 60:
            return "Top 60%"
        if p <= 70:
            return "Top 70%"
        if p <= 80:
            return "Top 80%"
        return "Top 90%"

    return pct_rank.apply(bucket)

data = load_data().copy()

# Ensure numeric/date columns are properly typed
for num_col in ["ClosePrice", "DaysOnMarket", "pricing_accuracy", "is_closed"]:
    data[num_col] = pd.to_numeric(data[num_col], errors="coerce")

data["CloseDate"] = pd.to_datetime(data["CloseDate"], errors="coerce")
data["ListingContractDate"] = pd.to_datetime(data["ListingContractDate"], errors="coerce")
data["City"] = data["City"].astype(str).str.strip()
data["PostalCode"] = data["PostalCode"].astype(str).str.strip()
# Use listing date for active/unclosed records and close date for closed records.
data["ActivityDate"] = data["CloseDate"].fillna(data["ListingContractDate"])

st.subheader("📍 Geographic Selection")

cities = sorted([x for x in data["City"].dropna().unique() if x and x.lower() != "nan"])
selected_cities = st.multiselect("City (required, choose one or more)", options=cities)

if not selected_cities:
    st.info("Please choose at least one City to continue.")
    st.stop()

geo_filtered = data[data["City"].isin(selected_cities)].copy()

zip_options = sorted([x for x in geo_filtered["PostalCode"].dropna().unique() if x and x.lower() != "nan"])
selected_zips = st.multiselect("Zip Code (optional)", options=zip_options)
if selected_zips:
    geo_filtered = geo_filtered[geo_filtered["PostalCode"].isin(selected_zips)]

school_options = sorted(
    [x for x in geo_filtered["ElementarySchool"].dropna().unique() if str(x).strip() and str(x).lower() != "nan"]
)
selected_schools = st.multiselect("Elementary School (optional)", options=school_options)
if selected_schools:
    geo_filtered = geo_filtered[geo_filtered["ElementarySchool"].isin(selected_schools)]

if geo_filtered.empty:
    st.warning("No records found for the selected geographic filters.")
    st.stop()

st.subheader("⏳ Time Window")
selected_years = st.selectbox("Years to look back", options=[1, 2, 3], index=0)
latest_date = geo_filtered["ActivityDate"].max()

if pd.isna(latest_date):
    st.warning("No valid listing/close dates in selected geography.")
    st.stop()

cutoff_date = latest_date - pd.DateOffset(years=int(selected_years))
window_filtered = geo_filtered[geo_filtered["ActivityDate"] >= cutoff_date].copy()

if window_filtered.empty:
    st.warning("No records in this geography and lookback window.")
    st.stop()

# Keep resale-only consistency with the alternate ranking page
window_filtered = window_filtered[window_filtered["PropertyCondition"].apply(is_resale)].copy()
if window_filtered.empty:
    st.warning("No resale listings in this geography and lookback window.")
    st.stop()

# ---------------- Aggregate summary ----------------
st.subheader("📊 Regional Summary")
summary_total_agents = window_filtered["ListAgentFullName"].nunique()
summary_total_transactions = len(window_filtered)
summary_total_sales_m = window_filtered["ClosePrice"].sum() / 1_000_000
summary_avg_dom = window_filtered["DaysOnMarket"].mean()
summary_avg_close_rate = window_filtered["is_closed"].mean()
summary_avg_pricing_accuracy = window_filtered["pricing_accuracy"].mean()

m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)

m1.metric("Total Agents", f"{summary_total_agents:,}")
m2.metric(f"Transactions (Past {selected_years}y)", f"{summary_total_transactions:,}")
m3.metric("Total Sales (M$)", f"{summary_total_sales_m:,.2f}")
m4.metric("Avg Days on Market", f"{summary_avg_dom:,.1f}")
m5.metric("Avg Close Rate", f"{summary_avg_close_rate:.1%}")
m6.metric("Avg Pricing Accuracy", f"{summary_avg_pricing_accuracy:,.3f}")

# ---------------- Price distribution + range ----------------
st.subheader("💰 Price Distribution & Range")
fig_hist = px.histogram(
    window_filtered,
    x="ClosePrice",
    nbins=30,
    title="Close Price Distribution",
    labels={"ClosePrice": "Close Price"},
)
st.plotly_chart(fig_hist, use_container_width=True)

min_price = int(window_filtered["ClosePrice"].min())
max_price = int(window_filtered["ClosePrice"].max())
mean_price = int(window_filtered["ClosePrice"].mean())
std_price = float(window_filtered["ClosePrice"].std())

step_size = max(1000, min(5000, int((max_price - min_price) / 200) if max_price > min_price else 1000))

target_price = st.slider(
    "Target price",
    min_value=min_price,
    max_value=max_price,
    value=mean_price,
    step=step_size,
)

std_width = st.slider(
    "Price band width (standard deviations)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

lower_bound = target_price - std_width * std_price
upper_bound = target_price + std_width * std_price
st.write(f"Selected price range: **${lower_bound:,.0f} - ${upper_bound:,.0f}**")

in_price_range = window_filtered[
    (window_filtered["ClosePrice"] >= lower_bound) & (window_filtered["ClosePrice"] <= upper_bound)
].copy()

if in_price_range.empty:
    st.warning("No listings found in this price band.")
    st.stop()

# ---------------- Agent-level summary ----------------
agent_stats = (
    in_price_range.groupby("ListAgentFullName", dropna=False)
    .agg(
        total_transactions=("ListAgentFullName", "count"),
        total_sales=("ClosePrice", "sum"),
        closed_count=("is_closed", lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum()),
        mean_days_on_market=("DaysOnMarket", "mean"),
        median_days_on_market=("DaysOnMarket", "median"),
        avg_pricing_accuracy=("pricing_accuracy", "mean"),
        ListAgentDirectPhone=(
            "ListAgentDirectPhone",
            lambda x: x.dropna().astype(str).iloc[0] if x.notna().any() else "",
        ),
    )
    .reset_index()
)

agent_stats["close_rate"] = agent_stats["closed_count"] / agent_stats["total_transactions"]
agent_stats["pricing_accuracy_score"] = agent_stats["avg_pricing_accuracy"].apply(pricing_accuracy_score)
agent_stats["volume_score"] = percentile_score(agent_stats["total_transactions"])
agent_stats["sales_score"] = percentile_score(agent_stats["total_sales"])
agent_stats["close_rate_score"] = percentile_score(agent_stats["close_rate"])
agent_stats["days_on_market_median_score"] = score_days_on_market(agent_stats["median_days_on_market"])
agent_stats["days_on_market_mean_score"] = score_days_on_market(agent_stats["mean_days_on_market"])
agent_stats["total_sales_score"] = percentile_score(agent_stats["total_sales"])

# ---------------- Filters: min/max transaction count ----------------
st.subheader("🔍 Agent Filters")
min_tx = int(agent_stats["total_transactions"].min())
max_tx = int(agent_stats["total_transactions"].max())

f1, f2 = st.columns(2)
selected_min_tx = f1.number_input("Minimum transactions", min_value=0, value=min_tx, step=1)
selected_max_tx = f2.number_input(
    "Maximum transactions", min_value=selected_min_tx, value=max(max_tx, selected_min_tx), step=1
)

if selected_min_tx > selected_max_tx:
    st.error("Minimum transactions cannot exceed maximum transactions.")
    st.stop()

agent_stats = agent_stats[
    (agent_stats["total_transactions"] >= selected_min_tx)
    & (agent_stats["total_transactions"] <= selected_max_tx)
].copy()

if agent_stats.empty:
    st.warning("No agents match the selected transaction range.")
    st.stop()

st.subheader("👥 Filtered Agent Count")
st.metric("Agents matching current filters", f"{agent_stats['ListAgentFullName'].nunique():,}")

# ---------------- Weights ----------------
st.subheader("⚖️ Scoring Weights")
priority_options = {
    "Maximizing price (even if it takes longer)": {
        "Volume": 0.25,
        "Close Rate": 0.20,
        "Days on Market": 0.15,
        "Pricing Accuracy": 0.40,
    },
    "Selling efficiently at a fair market price": {
        "Volume": 0.20,
        "Close Rate": 0.25,
        "Days on Market": 0.30,
        "Pricing Accuracy": 0.25,
    },
    "A smooth, predictable closing": {
        "Volume": 0.20,
        "Close Rate": 0.40,
        "Days on Market": 0.20,
        "Pricing Accuracy": 0.20,
    },
    "A low-stress process with clear guidance": {
        "Volume": 0.25,
        "Close Rate": 0.35,
        "Days on Market": 0.15,
        "Pricing Accuracy": 0.25,
    },
}

selected_priority = st.selectbox("Choose seller priority", options=list(priority_options.keys()))
weights = priority_options[selected_priority]

wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("📦 Volume", f"{weights['Volume']:.2f}")
wc2.metric("🔒 Close Rate", f"{weights['Close Rate']:.2f}")
wc3.metric("⏳ Days on Market", f"{weights['Days on Market']:.2f}")
wc4.metric("🎯 Pricing Accuracy", f"{weights['Pricing Accuracy']:.2f}")

agent_stats["overall_score"] = (
    weights["Volume"] * agent_stats["volume_score"]
    + weights["Close Rate"] * agent_stats["close_rate_score"]
    + weights["Days on Market"] * agent_stats["days_on_market_median_score"]
    + weights["Pricing Accuracy"] * agent_stats["pricing_accuracy_score"]
)

agent_stats = agent_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["overall_score"]).copy()
agent_stats = agent_stats.sort_values(
    by=["overall_score", "total_transactions", "close_rate", "median_days_on_market"],
    ascending=[False, False, False, True],
)

agent_stats["Tier"] = to_top_percent_bucket(agent_stats["overall_score"])
# Reuse the same metric definitions as 1_Alternate_Ranking.py for performance tiers.
agent_stats["Volume Tier"] = to_top_percent_bucket(agent_stats["volume_score"])
agent_stats["Close Rate Tier"] = to_top_percent_bucket(agent_stats["close_rate_score"])
agent_stats["Median Days on Market Tier"] = to_top_percent_bucket(agent_stats["days_on_market_median_score"])
agent_stats["Mean Days on Market Tier"] = to_top_percent_bucket(agent_stats["days_on_market_mean_score"])
agent_stats["Total Sales Tier"] = to_top_percent_bucket(agent_stats["total_sales_score"])
agent_stats["Pricing Accuracy Tier"] = to_top_percent_bucket(-agent_stats["avg_pricing_accuracy"])

# Ranking/top10: DO tie breaking via sort keys
agent_stats = agent_stats.sort_values(
    by=["overall_score", "total_transactions", "total_sales", "close_rate", "median_days_on_market"],
    ascending=[False, False, False, False, True],
)

final_top10 = agent_stats.head(10).copy()
final_top10["Rank"] = (
    final_top10["overall_score"].rank(ascending=False, method="dense").astype("Int64")
)
final_top10 = final_top10.sort_values(["Rank", "overall_score"], ascending=[True, False])

st.subheader("🏆 Final Top 10 Agents")
st.caption("Top agents based on selected filters and weighted score. Tiers are computed across all filtered agents before selecting top 10.")

final_cols = [
    "Rank",
    "ListAgentFullName",
    "ListAgentDirectPhone",
    "overall_score",
    "Volume Tier",
    "Close Rate Tier",
    "Median Days on Market Tier",
    "Mean Days on Market Tier",
    "Total Sales Tier",
    "Pricing Accuracy Tier",
]

st.data_editor(
    final_top10[final_cols],
    use_container_width=True,
    hide_index=True,
    disabled=True,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank"),
        "ListAgentFullName": "Agent",
        "ListAgentDirectPhone": st.column_config.TextColumn("📞 Phone"),
        "overall_score": st.column_config.NumberColumn("Overall Score", format="%.1f"),
    },
)

st.subheader("📋 Selected Agent Performance Details")
detail_cols = [
    "ListAgentFullName",
    "total_transactions",
    "Volume Tier",
    "close_rate",
    "Close Rate Tier",
    "mean_days_on_market",
    "Mean Days on Market Tier",
    "median_days_on_market",
    "Median Days on Market Tier",
    "avg_pricing_accuracy",
    "Pricing Accuracy Tier",
    "total_sales",
    "Total Sales Tier",
]
st.dataframe(final_top10[detail_cols], use_container_width=True)
