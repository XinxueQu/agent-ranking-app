import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Alternate Ranking", layout="wide")

st.title("🧪 Alternate Ranking Page")
st.write("This page lets you explore price distributions by zipcode and target a price range using ±1 standard deviation.")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=csv"
    usecols = [
        "ListAgentFullName","is_closed","DaysOnMarket","pricing_accuracy",
        "PostalCode","ClosePrice","ElementarySchool","SubdivisionName",
        "CloseDate", "PropertyCondition"
    ]
    return pd.read_csv(url, usecols=usecols)

data = load_data()

# -------------------- Filters --------------------

# (a) ZIP CODE SELECTION
st.subheader("📍 Choose Zipcodes")

zip_options = sorted(data["PostalCode"].dropna().astype(str).unique())
zipcodes = st.multiselect("Select Zipcode(s)", options=zip_options)

if not zipcodes:
    st.info("Please select at least one zipcode to proceed.")
    st.stop()

# Filter data
filtered = data[data["PostalCode"].astype(str).isin(zipcodes)]

if filtered.empty:
    st.warning("No data found for selected zipcodes.")
    st.stop()

# (b) ELEMENTARY SCHOOL SELECTION
# Extract unique elementary schools for the chosen zipcode
if "ElementarySchool" in filtered.columns:
    school_list = sorted(filtered["ElementarySchool"].dropna().unique())
else:
    st.error("Column 'ElementarySchool' not found in dataset.")
    st.stop()

selected_schools = st.multiselect("🏫 Choose an Elementary School", options=school_list)

filtered = filtered[filtered["ElementarySchool"].isin(selected_schools)]

if filtered.empty:
    st.warning("No data available for this school district.")
    st.stop()

# (c) TIME WINDOW FILTER (based on CloseDate)
# Ensure CloseDate is treated as datetime
filtered["CloseDate"] = pd.to_datetime(filtered["CloseDate"], errors="coerce")

window_options = {
    "Past 1 Year": 1,
    "Past 2 Years": 2,
    "Past 3 Years": 3
}

selected_window_label = st.selectbox("⏳ Choose Time Window", list(window_options.keys()))
years_back = window_options[selected_window_label]

# Determine the cutoff date (max date in dataset gives more stable behavior)
latest_date = filtered["CloseDate"].max()
cutoff_date = latest_date - pd.DateOffset(years=years_back)

# Apply time filter to the ZIP+School filtered data
filtered = filtered[filtered["CloseDate"] >= cutoff_date]

if filtered.empty:
    st.warning(f"No records available for the selected time window ({selected_window_label}).")
    st.stop()

st.info(f"Showing results for CloseDate ≥ {cutoff_date.date()} (last {years_back} year(s))")

# (d) Only Look at Resale (not 'New Construction' or 'Updated/Remodeled')
import ast

def is_resale(x):
    # Case 1: real list
    if isinstance(x, list):
        return x[0] == "Resale"
    # Case 2: string like "['Resale','Good']"
    if isinstance(x, str):
        try:
            xx = ast.literal_eval(x)
            return isinstance(xx, list) and xx[0] == "Resale"
        except:
            return False
    return False

st.info(f"📊 Sample size BEFORE filtering for Resale properties: {len(filtered)}")

filtered = filtered[filtered["PropertyCondition"].apply(is_resale)]

st.info(f"📊 Sample size after filtering for Resale properties: {len(filtered)}")

# -------------------- Visualize Close Price Distribution --------------------
st.subheader("📊 Close Price Distribution")

fig_hist = px.histogram(
    filtered,
    x="ClosePrice",
    nbins=30,
    title="Distribution of Close Prices",
    labels={"ClosePrice": "Close Price"},
)
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------- Target Price + Std Dev --------------------
st.subheader("🎯 Choose a Target Price (via slider)")

min_price = int(filtered["ClosePrice"].min())
max_price = int(filtered["ClosePrice"].max())
mean_price = filtered["ClosePrice"].mean()
std_price = filtered["ClosePrice"].std()

# Choose a reasonable step (e.g., $1,000)
step_size = min(5000, int((max_price - min_price) / 200))

target_price = st.slider(
    "Select Target Close Price",
    min_value=min_price,
    max_value=max_price,
    value=int(mean_price),
    step=step_size,
    help="Select a target price. We will compute ±1 standard deviation around it."
)

# Compute close price distribution
sample_mean = filtered["ClosePrice"].mean()
sample_std = filtered["ClosePrice"].std()

# --- NEW: Allow user to adjust price range width ---
st.subheader("📏 Price Range Width")

std_width = st.slider(
    "Select width (multiples of standard deviation):",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Controls how wide the acceptable price band is."
)

# Compute dynamic price range
lower_bound = target_price - std_width * sample_std
upper_bound = target_price + std_width * sample_std

st.write(
    f"**Price Range:** ${lower_bound:,.0f} – ${upper_bound:,.0f} "
    f"(± {std_width} × std dev)"
)

# Highlight this range on histogram
fig_range = px.histogram(
    filtered,
    x="ClosePrice",
    nbins=30,
    title=f"Close Price Distribution with Target Range Highlighted",
)

fig_range.add_vrect(
    x0=lower_bound,
    x1=upper_bound,
    fillcolor="green",
    opacity=0.25,
    line_width=0,
)

st.plotly_chart(fig_range, use_container_width=True)

# -------------------- Show Listings Inside the Range --------------------
st.subheader("📄 Listings Within Target Range (Raw Data of Selected Sample)")

in_range = filtered[
    (filtered["ClosePrice"] >= lower_bound) &
    (filtered["ClosePrice"] <= upper_bound)
]

st.write(f"Found **{len(in_range)}** listings within ±1 SD of your target price.")

st.dataframe(in_range, use_container_width=True)



# ===============================================================
# 🚀 AGENT RANKING SECTION (Local Filtered Market)
# ===============================================================
st.header("🏆 Agent Rankings (Based on Filtered Listings--Local Market)")

if in_range.empty:
    st.warning("No listings in the selected price range → unable to compute agent rankings.")
    st.stop()

# ---------------------------------------------------------------
# Build Agent Summary from in_range dataset
# ---------------------------------------------------------------
rank_df = in_range.copy()

# Compute simple metrics
agent_stats = (
    rank_df.groupby("ListAgentFullName", dropna=False)
    .agg(
        total_records = ("ListAgentFullName", "count"),
        total_sales   = ("ClosePrice", "sum"),
        closed_count  = ("is_closed", "sum"),
        avg_days      = ("DaysOnMarket", "mean"),
        median_days   = ("DaysOnMarket", "median"),
        avg_pricing_accuracy = ("pricing_accuracy", "mean")
    )
    .reset_index()
)

# Derived metrics
agent_stats["close_rate"] = agent_stats["closed_count"] / agent_stats["total_records"]

def pricing_accuracy_score(x): return (1 - abs(x - 1)) * 100
def percentile_score(s): return s.rank(pct=True) * 100
def score_days_on_market(s): return 100 - s.rank(pct=True) * 100

agent_stats["pricing_accuracy_score"] = agent_stats["avg_pricing_accuracy"].apply(pricing_accuracy_score)
agent_stats["volume_score"]           = percentile_score(agent_stats["total_records"])
agent_stats["sales_score"]            = percentile_score(agent_stats["total_sales"])
agent_stats["close_rate_score"]       = percentile_score(agent_stats["close_rate"])
agent_stats["days_on_market_score"]   = score_days_on_market(agent_stats["median_days"])

# ---------------------------------------------------------------
# Weight inputs
# ---------------------------------------------------------------
st.subheader("⚖️ Scoring Weights")

col_w1, col_w2, col_w3, col_w4 = st.columns(4)
weight_volume = col_w1.number_input("📦 Volume", value=0.4)
weight_close  = col_w2.number_input("🔒 Close Rate", value=0.3)
weight_days   = col_w3.number_input("⏳ Days on Market", value=0.2)
weight_price  = col_w4.number_input("🎯 Pricing Accuracy", value=0.1)

total_weight = weight_volume + weight_close + weight_days + weight_price
if total_weight == 0:
    st.error("All weights are zero — cannot rank agents.")
    st.stop()

if total_weight != 1:
    weight_volume /= total_weight
    weight_close  /= total_weight
    weight_days   /= total_weight
    weight_price  /= total_weight
    st.info("Weights normalized to sum to 1.")

# ---------------------------------------------------------------
# Compute Overall Score
# ---------------------------------------------------------------
agent_stats["overall_score"] = (
    weight_volume * agent_stats["volume_score"] +
    weight_close  * agent_stats["close_rate_score"] +
    weight_days   * agent_stats["days_on_market_score"] +
    weight_price  * agent_stats["pricing_accuracy_score"]
)

# ---------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------
agent_stats["Rank"] = agent_stats["overall_score"].rank(ascending=False, method="dense").astype(int)
agent_stats = agent_stats.sort_values(["Rank", "overall_score"])

# Display final ranking
st.subheader("🏅 Ranked Agents (within selected filters & price window)")
st.dataframe(agent_stats, use_container_width=True)

# ---------------------------------------------------------------
# Optional: Agent Detail Selection + Radar Chart
# ---------------------------------------------------------------
st.subheader("📐 Compare an Agent Across Dimensions")

agent_list = agent_stats["ListAgentFullName"].tolist()
selected_agent = st.selectbox("Choose an agent to inspect", agent_list)

if selected_agent:
    row = agent_stats[agent_stats["ListAgentFullName"] == selected_agent].iloc[0]

    dim_labels = [
        "Volume Score", "Close Rate Score", "Days on Market Score", "Pricing Accuracy Score"
    ]
    dim_values = [
        row["volume_score"],
        row["close_rate_score"],
        row["days_on_market_score"],
        row["pricing_accuracy_score"]
    ]

    radar_df = pd.DataFrame({"Dimension": dim_labels, "Score": dim_values})

    fig_radar = px.line_polar(
        radar_df,
        r="Score",
        theta="Dimension",
        line_close=True,
        range_r=[0, 100],
        title=f"Performance Profile: {selected_agent}"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("📊 Raw Metrics")
    raw_cols = [
        "total_records", "closed_count", "close_rate",
        "avg_days", "median_days",
        "avg_pricing_accuracy", "total_sales"
    ]
    st.dataframe(row[["ListAgentFullName"] + raw_cols], use_container_width=True)

