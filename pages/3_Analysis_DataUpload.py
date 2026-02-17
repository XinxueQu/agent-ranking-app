import ast

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="3_xxx", layout="wide")

st.title("🧪 Alternate Ranking Page (Upload Version)")
st.write("Upload a file and explore price distributions by zipcode with a target range based on standard deviation.")


@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


st.subheader("📤 Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if not uploaded_file:
    st.info("Please upload a file to continue.")
    st.stop()

try:
    data = load_uploaded_data(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read file: {exc}")
    st.stop()

required_columns = [
    "PostalCode",
    "ClosePrice",
    "CloseDate",
    "PropertyCondition",
]
missing = [c for c in required_columns if c not in data.columns]

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# -------------------- Filters --------------------
st.subheader("📍 Choose Zipcodes")
zip_options = sorted(data["PostalCode"].dropna().astype(str).unique())
zipcodes = st.multiselect("Select Zipcode(s)", options=zip_options)

if not zipcodes:
    st.info("Please select at least one zipcode to proceed.")
    st.stop()

filtered = data[data["PostalCode"].astype(str).isin(zipcodes)].copy()
if filtered.empty:
    st.warning("No data found for selected zipcodes.")
    st.stop()

# Optional school filter
if "ElementarySchool" in filtered.columns:
    st.subheader("🏫 Elementary School (Optional)")
    school_list = sorted(filtered["ElementarySchool"].dropna().astype(str).unique())
    selected_schools = st.multiselect(
        "Choose Elementary School(s) — leave empty to include all",
        options=school_list,
    )
    if selected_schools:
        filtered = filtered[filtered["ElementarySchool"].astype(str).isin(selected_schools)]
        if filtered.empty:
            st.warning("No data available for the selected school(s).")
            st.stop()

# CloseDate cleanup + window filter
filtered["CloseDate"] = (
    filtered["CloseDate"].astype(str).str.strip().replace(["", "None", "nan"], pd.NA)
)
filtered["CloseDate"] = pd.to_datetime(filtered["CloseDate"], errors="coerce")

window_options = {"Past 1 Year": 1, "Past 2 Years": 2, "Past 3 Years": 3}
selected_window_label = st.selectbox("⏳ Choose Time Window", list(window_options.keys()))
years_back = window_options[selected_window_label]

latest_date = filtered["CloseDate"].max()
if pd.isna(latest_date):
    st.warning("No valid CloseDate values found for selected filters.")
    st.stop()

cutoff_date = latest_date - pd.DateOffset(years=years_back)
filtered = filtered[(filtered["CloseDate"] >= cutoff_date) | (filtered["CloseDate"].isna())]

if filtered.empty:
    st.warning(f"No records available for the selected time window ({selected_window_label}).")
    st.stop()

st.info(f"Showing results for CloseDate ≥ {cutoff_date.date()} (last {years_back} year(s))")

# Resale-only filter (supports list/string-list/raw string)
def is_resale(value):
    if isinstance(value, list):
        return bool(value) and value[0] == "Resale"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "Resale":
            return True
        try:
            parsed = ast.literal_eval(stripped)
            return isinstance(parsed, list) and bool(parsed) and parsed[0] == "Resale"
        except Exception:
            return False
    return False

st.info(f"📊 Sample size BEFORE filtering for Resale properties: {len(filtered)}")
filtered = filtered[filtered["PropertyCondition"].apply(is_resale)]
st.info(f"📊 Sample size after filtering for Resale properties: {len(filtered)}")

if filtered.empty:
    st.warning("No resale records available after filtering.")
    st.stop()

# Numeric close price cleanup
filtered["ClosePrice"] = (
    filtered["ClosePrice"].astype(str).str.replace(r"[,$]", "", regex=True).str.strip()
)
filtered["ClosePrice"] = pd.to_numeric(filtered["ClosePrice"], errors="coerce")
filtered = filtered.dropna(subset=["ClosePrice"])

if filtered.empty:
    st.warning("No valid ClosePrice values available after cleaning.")
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

# -------------------- Target Price + Std Dev --------------------
st.subheader("🎯 Choose a Target Price (via slider)")

min_price = int(filtered["ClosePrice"].min())
max_price = int(filtered["ClosePrice"].max())
mean_price = filtered["ClosePrice"].mean()
sample_std = filtered["ClosePrice"].std()

if min_price == max_price:
    target_price = min_price
    st.info(f"Only one close price value is available: ${target_price:,.0f}")
else:
    step_size = max(1, min(5000, int((max_price - min_price) / 200)))
    target_price = st.slider(
        "Select Target Close Price",
        min_value=min_price,
        max_value=max_price,
        value=int(mean_price),
        step=step_size,
        help="Select a target price. We will compute ± standard deviation around it.",
    )

std_width = st.slider(
    "📏 Select width (multiples of standard deviation)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

if pd.isna(sample_std) or sample_std == 0:
    lower_bound = upper_bound = target_price
else:
    lower_bound = target_price - std_width * sample_std
    upper_bound = target_price + std_width * sample_std

st.write(f"**Price Range:** ${lower_bound:,.0f} – ${upper_bound:,.0f} (± {std_width} × std dev)")

fig_range = px.histogram(
    filtered,
    x="ClosePrice",
    nbins=30,
    title="Close Price Distribution with Target Range Highlighted",
)
fig_range.add_vrect(x0=lower_bound, x1=upper_bound, fillcolor="green", opacity=0.25, line_width=0)
st.plotly_chart(fig_range, use_container_width=True)

# -------------------- Show Listings Inside the Range --------------------
st.subheader("📄 Listings Within Target Range")
in_range = filtered[(filtered["ClosePrice"] >= lower_bound) & (filtered["ClosePrice"] <= upper_bound)]
st.write(f"Found **{len(in_range)}** listings within the selected range.")
st.dataframe(in_range, use_container_width=True)
