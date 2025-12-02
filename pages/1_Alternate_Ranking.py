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

# Optional boxplot for more insight
with st.expander("Show Boxplot"):
    fig_box = px.box(
        filtered,
        y="ClosePrice",
        title="Close Price Boxplot"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# -------------------- Target Price + Std Dev --------------------
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

# Compute range ± 1 standard deviation
lower_bound = target_price - std_price
upper_bound = target_price + std_price


st.markdown(f"""
### 📌 Price Range (± 1 Standard Deviation)
**Lower Bound:** ${lower_bound:,.0f}  
**Upper Bound:** ${upper_bound:,.0f}  
""")

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
