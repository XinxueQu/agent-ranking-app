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
        "PostalCode","ClosePrice","ElementarySchool","SubdivisionName"
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
filtered["CloseDate"] = filtered.to_datetime(filtered["CloseDate"], errors="coerce")

# Let user choose a look-back window
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
step_size = max(1000, int((max_price - min_price) / 200))

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
    min_value=0.5,
    max_value=3.0,
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
st.subheader("📄 Listings Within Target Range")

in_range = filtered[
    (filtered["ClosePrice"] >= lower_bound) &
    (filtered["ClosePrice"] <= upper_bound)
]

st.write(f"Found **{len(in_range)}** listings within ±1 SD of your target price.")

st.dataframe(in_range, use_container_width=True)
