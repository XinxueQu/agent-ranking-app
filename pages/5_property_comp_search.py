import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Property Similar Sales Finder", layout="wide")

st.title("🏘️ Property Similar Sales Finder")
st.write(
    "Enter subject-property features, optionally lock exact-match fields, and find the top 10 similar sold properties from the past 3 years."
)


# -------------------- Load Data (same pattern as other pages) --------------------
@st.cache_data
def load_default_data():
    url = "https://www.dropbox.com/scl/fi/jg966zvvhdsdblmg9jhh8/transactions_2023.01.07_2026.01.06.xlsx?rlkey=gwk06io5pp4lhaa1v3d4f4oun&st=2f31dzw8&dl=1"
    usecols = [
        "PostalCode",
        "ClosePrice",
        "CloseDate",
        "ElementarySchool",
        "SubdivisionName",
        "Address",
        "SchoolDistrict",
        "LivingArea",
        "LotSizeSquareFeet",
        "View",
        "PoolYN",
        "PropertyCondition",
    ]
    return pd.read_excel(url, usecols=lambda c: c in usecols)


@st.cache_data
def read_uploaded_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


ALIASES = {
    "address": ["Address", "PropertyAddress", "StreetAddress", "FullAddress"],
    "zip": ["PostalCode", "Zip", "ZipCode", "ZIP", "ZIPCode"],
    "school_district": ["SchoolDistrict", "ElementarySchool", "School District", "District"],
    "size_sqft": ["LivingArea", "BuildingAreaTotal", "SquareFootage", "SizeSqFt", "SqFt"],
    "land_sqft": ["LotSizeSquareFeet", "LotSizeSqFt", "LandSqFt", "LotSize"],
    "view": ["View", "ViewYN", "ViewDescription"],
    "pool": ["Pool", "PoolYN", "PrivatePoolYN"],
    "close_price": ["ClosePrice", "SoldPrice", "SalePrice", "Close Price"],
    "close_date": ["CloseDate", "SoldDate", "SaleDate", "Close Date"],
}


def find_column(df: pd.DataFrame, logical_name: str):
    candidates = ALIASES.get(logical_name, [])
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    return None


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def parse_pool(value):
    txt = normalize_text(value)
    return txt in {"true", "yes", "y", "1", "pool"}


def price_similarity(row_price, subject_price, global_price_std):
    if pd.isna(row_price) or row_price <= 0 or subject_price is None or subject_price <= 0:
        return None

    # Use distribution-informed tolerance so fuzzy match is grounded in market price behavior.
    tolerance = max(subject_price * 0.10, global_price_std * 0.50, 1)
    diff = abs(row_price - subject_price)
    ratio = min(diff / tolerance, 1.0)
    return 1.0 - ratio


def score_similarity(row, subject, cols, global_price_std):
    score = 0.0
    weight = 0.0

    # Price is the primary fuzzy proxy.
    if cols["close_price"] and subject["close_price"]:
        row_price = pd.to_numeric(row[cols["close_price"]], errors="coerce")
        px_score = price_similarity(row_price, subject["close_price"], global_price_std)
        if px_score is not None:
            weight += 5.0
            score += 5.0 * px_score

    # Secondary fuzzy signals.
    if cols["size_sqft"] and subject["size_sqft"]:
        row_size = pd.to_numeric(row[cols["size_sqft"]], errors="coerce")
        if pd.notna(row_size) and row_size > 0:
            weight += 1.5
            pct_diff = abs(row_size - subject["size_sqft"]) / max(subject["size_sqft"], 1)
            score += 1.5 * max(0.0, 1 - min(pct_diff, 1.0))

    if cols["land_sqft"] and subject["land_sqft"]:
        row_land = pd.to_numeric(row[cols["land_sqft"]], errors="coerce")
        if pd.notna(row_land) and row_land > 0:
            weight += 1.0
            pct_diff = abs(row_land - subject["land_sqft"]) / max(subject["land_sqft"], 1)
            score += 1.0 * max(0.0, 1 - min(pct_diff, 1.0))

    if cols["view"] and subject["view"]:
        weight += 0.75
        if normalize_text(row[cols["view"]]) == normalize_text(subject["view"]):
            score += 0.75

    if cols["pool"] and subject["pool"] is not None:
        weight += 0.75
        if parse_pool(row[cols["pool"]]) == subject["pool"]:
            score += 0.75

    # Exact match contributes lightly unless explicitly locked.
    if cols["zip"] and subject["zip"]:
        weight += 0.5
        if normalize_text(row[cols["zip"]]) == normalize_text(subject["zip"]):
            score += 0.5

    if cols["school_district"] and subject["school_district"]:
        weight += 0.5
        if normalize_text(row[cols["school_district"]]) == normalize_text(subject["school_district"]):
            score += 0.5

    return 0.0 if weight == 0 else score / weight


def feature_price_separation(df: pd.DataFrame, feature_col: str, price_col: str):
    temp = df[[feature_col, price_col]].copy()
    temp[price_col] = pd.to_numeric(temp[price_col], errors="coerce")
    temp = temp.dropna(subset=[price_col])
    temp[feature_col] = temp[feature_col].astype(str).str.strip()
    temp = temp[temp[feature_col] != ""]

    if temp.empty or temp[feature_col].nunique() < 2:
        return None, None

    grouped = temp.groupby(feature_col)[price_col].agg(["count", "median"]).reset_index()
    grouped = grouped[grouped["count"] >= 5].copy()

    if grouped.empty or grouped.shape[0] < 2:
        return None, None

    total_std = temp[price_col].std()
    between_std = grouped["median"].std()
    if pd.isna(total_std) or total_std == 0:
        ratio = None
    else:
        ratio = between_std / total_std

    grouped = grouped.sort_values("median", ascending=False)
    return ratio, grouped


st.subheader("1) Data source")
source = st.radio(
    "Choose data source",
    options=["Use app default dataset", "Upload CSV/XLSX"],
    horizontal=True,
)

if source == "Use app default dataset":
    try:
        df = load_default_data()
    except Exception as exc:
        st.error(f"Could not load default dataset: {exc}")
        st.stop()
else:
    file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    if not file:
        st.info("Upload a dataset to start the comparable-sale search.")
        st.stop()
    try:
        df = read_uploaded_data(file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

cols = {k: find_column(df, k) for k in ALIASES}
required = [k for k in ["close_date", "close_price"] if not cols[k]]
if required:
    st.error(
        "Required columns not found: "
        + ", ".join(required)
        + ". Please ensure close date and close price columns exist."
    )
    st.stop()

# -------------------- Prep + 3-year window --------------------
working = df.copy()
working[cols["close_date"]] = pd.to_datetime(working[cols["close_date"]], errors="coerce")
working[cols["close_price"]] = pd.to_numeric(working[cols["close_price"]], errors="coerce")
working = working.dropna(subset=[cols["close_date"], cols["close_price"]])

if working.empty:
    st.warning("No valid sold records with close date and close price.")
    st.stop()

latest_close_date = working[cols["close_date"]].max()
cutoff = latest_close_date - pd.DateOffset(years=3)
working = working[working[cols["close_date"]] >= cutoff].copy()

if working.empty:
    st.warning("No sold properties found in the past 3 years.")
    st.stop()

st.caption(f"Using sold comps with close date on or after {cutoff.date()}.")

# -------------------- Feature impact analysis (price differentiation) --------------------
st.subheader("2) Why these features matter (price differentiation)")
feature_map = {
    "ZIP": cols["zip"],
    "School District": cols["school_district"],
    "View": cols["view"],
    "Pool": cols["pool"],
}

impact_rows = []
impact_tables = {}
for feature_name, feature_col in feature_map.items():
    if feature_col and feature_col in working.columns:
        ratio, grouped = feature_price_separation(working, feature_col, cols["close_price"])
        if ratio is not None:
            impact_rows.append({"Feature": feature_name, "Price Separation Score": ratio})
            impact_tables[feature_name] = grouped.head(10)

if impact_rows:
    impact_df = pd.DataFrame(impact_rows).sort_values("Price Separation Score", ascending=False)
    st.write(
        "Higher score means group medians for that feature are more spread out relative to overall market price variation."
    )
    st.dataframe(impact_df, use_container_width=True)

    fig = px.bar(
        impact_df,
        x="Feature",
        y="Price Separation Score",
        title="Feature Price Separation (higher = more differentiating)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See top median-price groups per feature"):
        for feature_name, table in impact_tables.items():
            st.markdown(f"**{feature_name}**")
            st.dataframe(table, use_container_width=True)
else:
    st.info("Not enough category depth in the current data to estimate feature-based price separation.")

# -------------------- Inputs --------------------
st.subheader("3) Enter subject property features")
left, right = st.columns(2)

zip_options = []
if cols["zip"]:
    zip_options = sorted(working[cols["zip"]].dropna().astype(str).str.strip().unique())

district_options = []
if cols["school_district"]:
    district_options = (
        sorted(working[cols["school_district"]].dropna().astype(str).str.strip().unique())
    )

with left:
    address = st.text_input("Address (optional)")
    zip_code = st.text_input("ZIP code (optional)")
    school_district = st.text_input("School district (optional)")
    view_type = st.text_input("View (optional)")

with right:
    size_sqft = st.number_input("Living size (sq ft, optional)", min_value=0, step=50, value=0)
    land_sqft = st.number_input("Land size (sq ft, optional)", min_value=0, step=100, value=0)
    close_price = st.number_input("Target price (optional)", min_value=0, step=10000, value=0)
    has_pool = st.selectbox("Pool (optional)", options=["Not specified", "Yes", "No"])

st.markdown("#### 🔒 Exact-match locks")
lock_zip = st.checkbox("Require same ZIP(s)", value=False)
lock_school = st.checkbox("Require same school district(s)", value=False)
lock_view = st.checkbox("Require same view", value=False)
lock_pool = st.checkbox("Require same pool value", value=False)

locked_zips = []
if lock_zip:
    locked_zips = st.multiselect("ZIP(s) to lock", options=zip_options, default=[zip_code] if zip_code else [])

locked_districts = []
if lock_school:
    locked_districts = st.multiselect(
        "School district(s) to lock",
        options=district_options,
        default=[school_district] if school_district else [],
    )

run = st.button("Find Top 10 Similar Sold Properties", type="primary")

if not run:
    st.stop()

subject = {
    "address": address.strip(),
    "zip": zip_code.strip(),
    "school_district": school_district.strip(),
    "size_sqft": int(size_sqft) if size_sqft > 0 else None,
    "land_sqft": int(land_sqft) if land_sqft > 0 else None,
    "view": view_type.strip(),
    "pool": None if has_pool == "Not specified" else has_pool == "Yes",
    "close_price": int(close_price) if close_price > 0 else None,
}

# -------------------- Apply exact locks --------------------
candidate = working.copy()

if lock_zip and cols["zip"]:
    if locked_zips:
        zset = {z.strip() for z in locked_zips}
        candidate = candidate[candidate[cols["zip"]].astype(str).str.strip().isin(zset)]
    elif subject["zip"]:
        candidate = candidate[candidate[cols["zip"]].astype(str).str.strip() == subject["zip"]]

if lock_school and cols["school_district"]:
    if locked_districts:
        dset = {d.strip().lower() for d in locked_districts}
        candidate = candidate[
            candidate[cols["school_district"]].astype(str).str.strip().str.lower().isin(dset)
        ]
    elif subject["school_district"]:
        candidate = candidate[
            candidate[cols["school_district"]].astype(str).str.strip().str.lower()
            == subject["school_district"].lower()
        ]

if lock_view and cols["view"] and subject["view"]:
    candidate = candidate[
        candidate[cols["view"]].astype(str).str.strip().str.lower() == subject["view"].lower()
    ]

if lock_pool and cols["pool"] and subject["pool"] is not None:
    candidate = candidate[candidate[cols["pool"]].apply(parse_pool) == subject["pool"]]

if candidate.empty:
    st.warning("No records left after applying exact-match locks. Try relaxing one or more locks.")
    st.stop()

# -------------------- Fuzzy ranking --------------------
global_price_std = working[cols["close_price"]].std()
candidate["similarity_score"] = candidate.apply(
    lambda row: score_similarity(row, subject, cols, global_price_std),
    axis=1,
)

candidate = candidate.sort_values("similarity_score", ascending=False)
result = candidate.head(10).copy()

if result.empty:
    st.warning("No comparable properties found.")
    st.stop()

st.success(f"Found {len(result)} comps. Showing top {len(result)} by similarity score.")

show_cols = [
    c
    for c in [
        cols["address"],
        cols["zip"],
        cols["school_district"],
        cols["size_sqft"],
        cols["land_sqft"],
        cols["view"],
        cols["pool"],
        cols["close_price"],
        cols["close_date"],
    ]
    if c and c in result.columns
]
show_cols.append("similarity_score")

st.subheader("Top 10 Similar Sold Properties")
st.dataframe(result[show_cols], use_container_width=True)
