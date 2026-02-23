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


def clean_subdivision_name(value: str) -> str:
    text = str(value or "")

    replacements = [
        (" 01", ""), (" 02", ""), (" 03", ""), (" 04", ""), (" 05", ""),
        (" 06", ""), (" 07", ""), (" 08", ""), (" 09", ""), (" 10", ""),
        (" 0", ""), (" 1", ""), (" 2", ""), (" 3", ""), (" 4", ""), (" 5", ""),
        (" 6", ""), (" 7", ""), (" 8", ""), (" 9", ""), ("0", ""), ("1", ""),
        ("2", ""), ("3", ""), ("4", ""), ("5", ""), ("6", ""), ("7", ""),
        ("8", ""), ("9", ""), ("&", ""),
        (" Ph", ""), (" Div ", ""), (" Resub", ""), (" Pud", ""), (" Inc", ""),
        (" Creeksec", " Creek"), (" Surv", ","), (" Annex", ","), (" Amd", ""),
        (" Add", ""), (" Sec", ""), ("-", " "), ("  ", ""), (" Blk", ""),
        (" Instl", " "), (" Phs", ""), (" Unit", ""), (" Subd", ""),
        (" Abc Mid Dec", ""), (" Tr D", ""), (" The", ""), (" aka ", ""),
        ("Town Center", "Towncenter,"), ("Condos", "Condo"), ("Enfield", "Enfield,"),
        ("Riviera Spgs", "Riviera Springs,"), ("Crk", "Creek,"), ("Brykerwoods", "Brykerwoods,"),
        ("Town-", "Town,"), ("Villagesec", "Village"), ("Sun City", "Sun City,"),
        ("Pemberton Heights", "Pemberton Heights,"), ("Rosedale", "Rosedale,"),
    ]

    for find_text, replacement_text in replacements:
        text = text.replace(find_text, replacement_text)

    # normalize spacing
    text = " ".join(text.split())
    return text.strip()


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
    s = pd.to_numeric(scores, errors="coerce")

    # Best scores get the smallest percentile number (e.g., 0.2%),
    # and ties take the BEST rank within the tie group.
    top_pct = s.rank(pct=True, ascending=False, method="min") * 100

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
        if p <= 50:
            return "Top 50%"
        return "Top 100%"

    return top_pct.apply(bucket)

data = load_data().copy()

# Ensure numeric/date columns are properly typed
for num_col in ["ClosePrice", "DaysOnMarket", "pricing_accuracy", "is_closed"]:
    data[num_col] = pd.to_numeric(data[num_col], errors="coerce")

data["CloseDate"] = pd.to_datetime(data["CloseDate"], errors="coerce")
data["ListingContractDate"] = pd.to_datetime(data["ListingContractDate"], errors="coerce")
data["City"] = data["City"].astype(str).str.strip()
data["PostalCode"] = data["PostalCode"].astype(str).str.strip()
data["CleanedSubdivision"] = data["SubdivisionName"].fillna("").astype(str).apply(clean_subdivision_name)
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

subdivision_options = sorted(
    [x for x in geo_filtered["CleanedSubdivision"].dropna().unique() if str(x).strip() and str(x).lower() != "nan"]
)
selected_subdivisions = st.multiselect("Subdivision (optional)", options=subdivision_options)
if selected_subdivisions:
    geo_filtered = geo_filtered[geo_filtered["CleanedSubdivision"].isin(selected_subdivisions)]

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

# Agent counts by selected geography scopes, all using the SAME scoped window (window_filtered)
city_agents_count = window_filtered["ListAgentFullName"].nunique()
zip_agents_count = (
    window_filtered[window_filtered["PostalCode"].isin(selected_zips)]["ListAgentFullName"].nunique()
    if selected_zips
    else None
)
school_agents_count = (
    window_filtered[window_filtered["ElementarySchool"].isin(selected_schools)]["ListAgentFullName"].nunique()
    if selected_schools
    else None
)
subdivision_agents_count = (
    window_filtered[window_filtered["CleanedSubdivision"].isin(selected_subdivisions)]["ListAgentFullName"].nunique()
    if selected_subdivisions
    else None
)

region_label_parts = [f"City={', '.join(selected_cities)}"]
if selected_zips:
    region_label_parts.append(f"Zip={', '.join(selected_zips)}")
if selected_schools:
    region_label_parts.append(f"School={', '.join(map(str, selected_schools))}")
if selected_subdivisions:
    region_label_parts.append(f"Subdivision={', '.join(map(str, selected_subdivisions))}")

st.caption(
    "Scope: "
    + " | ".join(region_label_parts)
    + f" | Time Window: Past {selected_years} year(s) from latest activity date | Property: Resale only"
)

m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)
m7, m8, m9 = st.columns(3)

m1.metric("Total Agents (Scoped)", f"{summary_total_agents:,}")
m2.metric(f"Transactions (Scoped, Past {selected_years}y)", f"{summary_total_transactions:,}")
m3.metric("Total Sales (Scoped, M$)", f"{summary_total_sales_m:,.2f}")
m4.metric("Avg Days on Market (Scoped)", f"{summary_avg_dom:,.1f}")
m5.metric("Avg Close Rate (Scoped)", f"{summary_avg_close_rate:.1%}")
m6.metric("Avg Pricing Accuracy (Scoped)", f"{summary_avg_pricing_accuracy:,.3f}")
m7.metric("Agents in Selected City (Scoped)", f"{city_agents_count:,}")
m8.metric(
    "Agents in Selected Zip (Scoped)",
    f"{zip_agents_count:,}" if zip_agents_count is not None else "—",
)
school_text = f"Schools: {school_agents_count:,}" if school_agents_count is not None else "Schools: —"
subdivision_text = (
    f"Subdivisions: {subdivision_agents_count:,}"
    if subdivision_agents_count is not None
    else "Subdivisions: —"
)
m9.metric(
    "Agents in Selected School/Subdivision (Scoped)",
    f"{school_text} | {subdivision_text}",
)

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

target_price_default = min(max(1_000_000, min_price), max_price)
target_price = st.slider(
    "Target price",
    min_value=min_price,
    max_value=max_price,
    value=target_price_default,
    step=step_size,
)

std_width = st.slider(
    "Price band width (standard deviations)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
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
        closed_count=("is_closed", "sum"),
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

# Derived metrics
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
final_top10["total_sales_m"] = (final_top10["total_sales"] / 1_000_000).round(2)

# Agent sales count context (within selected years)
years_filtered_all = data[data["ActivityDate"] >= cutoff_date].copy()
years_filtered_city = years_filtered_all[years_filtered_all["City"].isin(selected_cities)].copy()

sales_count_all_map = years_filtered_all.groupby("ListAgentFullName", dropna=False)["is_closed"].sum()
sales_count_city_map = years_filtered_city.groupby("ListAgentFullName", dropna=False)["is_closed"].sum()

if selected_zips:
    years_filtered_zip = years_filtered_city[years_filtered_city["PostalCode"].isin(selected_zips)].copy()
    sales_count_zip_map = years_filtered_zip.groupby("ListAgentFullName", dropna=False)["is_closed"].sum()
else:
    sales_count_zip_map = None

if selected_schools:
    years_filtered_school = years_filtered_city[years_filtered_city["ElementarySchool"].isin(selected_schools)].copy()
    sales_count_school_map = years_filtered_school.groupby("ListAgentFullName", dropna=False)["is_closed"].sum()
else:
    sales_count_school_map = None

final_top10["sales_count_all_years"] = final_top10["ListAgentFullName"].map(sales_count_all_map).fillna(0).astype(int)
final_top10["sales_count_selected_cities"] = final_top10["ListAgentFullName"].map(sales_count_city_map).fillna(0).astype(int)

if sales_count_zip_map is None:
    final_top10["sales_count_selected_zip"] = pd.NA
else:
    final_top10["sales_count_selected_zip"] = (
        final_top10["ListAgentFullName"].map(sales_count_zip_map).fillna(0).astype(int)
    )

if sales_count_school_map is None:
    final_top10["sales_count_selected_school"] = pd.NA
else:
    final_top10["sales_count_selected_school"] = (
        final_top10["ListAgentFullName"].map(sales_count_school_map).fillna(0).astype(int)
    )

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
        "Volume Tier": "Volume Tier",
        "Close Rate Tier": "Close Rate Tier",
        "Median Days on Market Tier": "Median DOM Tier",
        "Mean Days on Market Tier": "Mean DOM Tier",
        "Total Sales Tier": "Total Sales Tier",
        "Pricing Accuracy Tier": "Pricing Accuracy Tier",
    },
)

st.subheader("📋 Selected Agent Performance Details")
detail_cols = [
    "ListAgentFullName",
    "sales_count_all_years",
    "sales_count_selected_cities",
    "sales_count_selected_zip",
    "sales_count_selected_school",
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
    "total_sales_m",
    "Total Sales Tier",
]
st.dataframe(
    final_top10[detail_cols],
    use_container_width=True,
    column_config={
        "sales_count_all_years": "Sales Count (All Data, Selected Years)",
        "sales_count_selected_cities": "Sales Count (Selected Cities)",
        "sales_count_selected_zip": "Sales Count (Selected Zip)",
        "sales_count_selected_school": "Sales Count (Selected Elementary School)",
        "total_sales_m": st.column_config.NumberColumn("Total Sales (M$)", format="%.2f"),
        "Volume Tier": "Volume Tier",
        "Close Rate Tier": "Close Rate Tier",
        "Mean Days on Market Tier": "Mean DOM Tier",
        "Median Days on Market Tier": "Median DOM Tier",
        "Pricing Accuracy Tier": "Pricing Accuracy Tier",
        "Total Sales Tier": "Total Sales Tier",
    },
)
