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
        "CloseDate", "PropertyCondition", "ListingContractDate"
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

# (b) ELEMENTARY SCHOOL SELECTION (OPTIONAL)
st.subheader("🏫 Elementary School (Optional)")

if "ElementarySchool" not in filtered.columns:
    st.error("Column 'ElementarySchool' not found in dataset.")
    st.stop()

school_list = sorted(filtered["ElementarySchool"].dropna().unique())

selected_schools = st.multiselect(
    "Choose Elementary School(s) — leave empty to include all",
    options=school_list
)

# Apply filter ONLY if user selected something
if selected_schools:
    filtered = filtered[filtered["ElementarySchool"].isin(selected_schools)]
    if filtered.empty:
        st.warning("No data available for the selected school(s).")
        st.stop()

# (c) TIME WINDOW FILTER (CloseDate fallback to ListingContractDate)

# Ensure dates are datetime
filtered["CloseDate"] = pd.to_datetime(filtered.get("CloseDate"), errors="coerce")

# Fallback column (change name if needed)
if "ListingContractDate" in filtered.columns:
    filtered["ListingContractDate"] = pd.to_datetime(filtered["ListingContractDate"], errors="coerce")
else:
    filtered["ListingContractDate"] = pd.NaT

# Effective date: CloseDate if available, otherwise ListingContractDate
filtered["effective_date"] = filtered["CloseDate"].fillna(filtered["ListingContractDate"])

window_options = {
    "Past 1 Year": 1,
    "Past 2 Years": 2,
    "Past 3 Years": 3
}

selected_window_label = st.selectbox("⏳ Choose Time Window", list(window_options.keys()))
years_back = window_options[selected_window_label]

latest_date = filtered["effective_date"].max()
cutoff_date = latest_date - pd.DateOffset(years=years_back)

filtered = filtered[filtered["effective_date"] >= cutoff_date]

if filtered.empty:
    st.warning(f"No records available for the selected time window ({selected_window_label}).")
    st.stop()

st.info(
    f"Showing results where CloseDate (or ListingContractDate if missing) ≥ {cutoff_date.date()}"
)


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
agent_stats["close_rate"] = agent_stats["closed_count"] / agent_stats["total_records"] # could introduce NAs

def pricing_accuracy_score(x): return (1 - abs(x - 1)) * 100
#def percentile_score(s): return s.rank(pct=True) * 100
def percentile_score(s):
    s = s.astype(float)

    min_val = s.min()
    max_val = s.max()

    # Case 1: all values identical → give neutral score
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(50.0, index=s.index)

    # Case 2: normal min–max scaling
    return 100 * (s - min_val) / (max_val - min_val)

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
agent_stats = agent_stats.replace([np.inf, -np.inf], np.nan)
agent_stats = agent_stats.dropna(
    subset=["overall_score"],
    how="any"
)
#agent_stats["Rank"] = agent_stats["overall_score"].rank(ascending=False, method="dense").astype(int)
agent_stats["Rank"] = (
    agent_stats["overall_score"]
    .rank(ascending=False, method="dense")
    .astype("Int64")   # nullable integer type
)
agent_stats = agent_stats.sort_values(["Rank", "overall_score"])

# Display final ranking
#st.subheader("🏅 Ranked Agents (within selected filters & price window)")
#st.dataframe(agent_stats, use_container_width=True)

# ---------------------------------------------------------------
# 🔍 Agent Quality Filters
# ---------------------------------------------------------------
st.subheader("🔍 Agent Filters")

col_f1, col_f2 = st.columns(2)

min_records = col_f1.number_input(
    "Minimum number of listings",
    min_value=0,
    value=3,
    step=1,
    help="Exclude agents with fewer than this many listings"
)

min_sales = col_f2.number_input(
    "Minimum total sales ($)",
    min_value=0,
    value=0,
    step=50_000,
    help="Exclude agents with low total transaction volume"
)

filtered_agents = agent_stats[
    (agent_stats["total_records"] >= min_records) &
    (agent_stats["total_sales"] >= min_sales)
].copy()

if filtered_agents.empty:
    st.warning("No agents meet the selected minimum requirements.")
    st.stop()

filtered_agents["Rank"] = (
    filtered_agents["overall_score"]
    .rank(ascending=False, method="dense")
    .astype("Int64")
)

filtered_agents = filtered_agents.sort_values(
    ["Rank", "overall_score"],
    ascending=[True, False]
)

st.subheader("🏅 Ranked Agents (interactive)")

st.caption(
    f"Showing {len(filtered_agents)} agents "
    f"(min {min_records} listings, min ${min_sales:,.0f} sales)"
)

st.data_editor(
    filtered_agents,
    use_container_width=True,
    hide_index=True,
    disabled=True,
    column_config={
        "overall_score": st.column_config.NumberColumn(
            "Overall Score",
            format="%.1f"
        )
    }
)



if False:
    st.subheader("🏅 Ranked Agents (interactive)")
        
    st.data_editor(
        agent_stats,
        use_container_width=True,
        hide_index=True,
        disabled=True,   # read-only
        column_config={
            "overall_score": st.column_config.NumberColumn(
                "Overall Score",
                format="%.1f",
                help="Weighted composite score (0–100)"
            )
        }
    )

# ---------------------------------------------------------------
# 📐 Compare up to 3 Agents Across Dimensions
# ---------------------------------------------------------------
st.subheader("📐 Compare Agents Across Dimensions (up to 3)")

agent_list = agent_stats["ListAgentFullName"].tolist()

selected_agents = st.multiselect(
    "Choose up to 3 agents to compare",
    options=agent_list,
    max_selections=3
)

if not selected_agents:
    st.info("Select one or more agents to view comparison.")
    st.stop()

# ---------------------------------------------------------------
# Prepare data for radar chart
# ---------------------------------------------------------------
radar_rows = []

for agent in selected_agents:
    row = agent_stats.loc[
        agent_stats["ListAgentFullName"] == agent
    ].iloc[0]

    radar_rows.extend([
        {"Agent": agent, "Dimension": "Volume Score", "Score": row["volume_score"]},
        {"Agent": agent, "Dimension": "Close Rate Score", "Score": row["close_rate_score"]},
        {"Agent": agent, "Dimension": "Days on Market Score", "Score": row["days_on_market_score"]},
        {"Agent": agent, "Dimension": "Pricing Accuracy Score", "Score": row["pricing_accuracy_score"]},
    ])

radar_df = pd.DataFrame(radar_rows)

# ---------------------------------------------------------------
# Radar Chart (multi-agent)
# ---------------------------------------------------------------
fig_radar = px.line_polar(
    radar_df,
    r="Score",
    theta="Dimension",
    color="Agent",
    line_close=True,
    range_r=[0, 100],
    title="Performance Comparison Across Dimensions"
)

st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------
# Raw Metrics Table (side-by-side columns)
# ---------------------------------------------------------------
st.subheader("📊 Raw Metrics (Side-by-Side)")

raw_cols = [
    "total_records",
    "closed_count",
    "close_rate",
    "avg_days",
    "median_days",
    "avg_pricing_accuracy",
    "total_sales"
]

raw_table = (
    agent_stats
    .set_index("ListAgentFullName")
    .loc[selected_agents, raw_cols]
    .T
)

# Improve readability
raw_table.index.name = "Metric"
raw_table = raw_table.round(2)

st.dataframe(raw_table, use_container_width=True)

