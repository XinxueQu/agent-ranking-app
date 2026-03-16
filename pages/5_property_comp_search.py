from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Property Similar Sales Finder", layout="wide")

st.title("🏘️ Property Similar Sales Finder")
st.write(
    "Enter subject-property features, optionally lock exact-match fields, and find the top 10 similar sold properties from the past 3 years."
)


PRECOMPUTED_IMPORTANCE_PATH = Path("precomputed_knn_feature_importance.csv")


def resolve_default_cache_path() -> Path:
    data_path = Path("data")
    if data_path.exists() and data_path.is_file():
        cache_root = Path(".cache")
    else:
        cache_root = data_path
    return cache_root / "default_property_comp_cache.parquet"

@st.cache_data(ttl=3600)
def load_default_data():
    url = "https://www.dropbox.com/scl/fi/jg966zvvhdsdblmg9jhh8/transactions_2023.01.07_2026.01.06.xlsx?rlkey=gwk06io5pp4lhaa1v3d4f4oun&st=2f31dzw8&dl=1"
    usecols = [
        "Address",
        "PostalCode",
        "CountyOrParish",
        "City",
        "SubdivisionName",
        "SchoolDistrict",
        "ElementarySchool",
        "MiddleOrJuniorSchool",
        "HighSchool",
        "ClosePrice",
        "CloseDate",
        "LivingArea",
        "LotSizeSquareFeet",
        "Acres",
        "YearBuilt",
        "Levels",
        "GarageSpaces",
        "ParkingFeatures",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "FullBathrooms",
        "HalfBathrooms",
        "PoolYN",
        "WaterfrontYN",
        "AssociationYN",
        "View",
        "PropertyCondition",
        "ListPrice",
        "pricing_accuracy",
        "DaysOnMarket",
        "is_closed",
        "ListAgentFullName",
    ]
    cache_path = resolve_default_cache_path()
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=24):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    df = pd.read_excel(url, usecols=lambda c: c in usecols)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return df


@st.cache_data
def read_uploaded_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


ALIASES = {
    "address": ["Address", "PropertyAddress", "StreetAddress", "FullAddress"],
    "zip": ["PostalCode", "Zip", "ZipCode", "ZIP", "ZIPCode"],
    "county": ["County", "CountyOrParish"],
    "city": ["City"],
    "subdivision": ["SubdivisionName", "Subdivision"],
    "school_district": ["SchoolDistrict", "School District", "District"],
    "elementary": ["ElementarySchool", "Elementary"],
    "middle": ["MiddleOrJuniorSchool", "Middle School", "Middle"],
    "high": ["HighSchool", "High School"],
    "size_sqft": ["LivingArea", "BuildingAreaTotal", "SquareFootage", "TotalSqft", "SqFt"],
    "land_sqft": ["LotSizeSquareFeet", "LotSizeSqFt", "LandSqFt", "LotSize"],
    "acres": ["Acres", "LotSizeAcres"],
    "year_built": ["YearBuilt", "BuiltYear"],
    "levels": ["Levels", "StoriesTotal", "Stories"],
    "garage_spaces": ["GarageSpaces", "Garage", "GarageCars"],
    "parking_spaces": ["ParkingSpaces", "TotalParkingSpaces"],
    "beds": ["BedroomsTotal", "Beds", "Bedrooms"],
    "baths_total": ["BathroomsTotalInteger", "TotalBathrooms", "TtlBaths", "Total Baths"],
    "full_baths": ["FullBathrooms", "FullBaths"],
    "half_baths": ["HalfBathrooms", "HalfBaths"],
    "pool": ["PoolYN", "Pool", "PrivatePoolYN"],
    "waterfront": ["WaterfrontYN", "Waterfront", "WaterfrontFeatures"],
    "hoa": ["AssociationYN", "HOA"],
    "view": ["View", "ViewYN", "ViewDescription"],
    "condition": ["PropertyCondition", "Condition"],
    "close_price": ["ClosePrice", "SoldPrice", "SalePrice", "Close Price"],
    "list_price": ["ListPrice", "OriginalListPrice"],
    "close_date": ["CloseDate", "SoldDate", "SaleDate", "Close Date"],
    "agent_name": ["ListAgentFullName", "AgentName", "ListAgentName"],
    "is_closed": ["is_closed", "IsClosed", "ClosedYN"],
    "days_on_market": ["DaysOnMarket", "DOM"],
    "pricing_accuracy": ["pricing_accuracy", "PricingAccuracy"],
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


def parse_boolish(value):
    txt = normalize_text(value)
    return txt in {"true", "yes", "y", "1", "pool", "waterfront"}


def get_numeric(row, col):
    if not col:
        return None
    val = pd.to_numeric(row[col], errors="coerce")
    if pd.isna(val):
        return None
    return float(val)


def price_similarity(row_price, subject_price, global_price_std):
    if row_price is None or row_price <= 0 or subject_price is None or subject_price <= 0:
        return None
    tolerance = max(subject_price * 0.12, global_price_std * 0.50 if pd.notna(global_price_std) else 0, 1)
    diff = abs(row_price - subject_price)
    return 1.0 - min(diff / tolerance, 1.0)


def closeness_similarity(row_value, subject_value, tolerance_fraction):
    if row_value is None or subject_value is None:
        return None
    denominator = max(abs(subject_value) * tolerance_fraction, 1)
    diff = abs(row_value - subject_value)
    return 1.0 - min(diff / denominator, 1.0)




def percentile_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=series.index)
    return s.rank(pct=True) * 100


def pricing_accuracy_to_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (1 - (s - 1).abs()) * 100


def score_similarity(row, subject, cols, global_price_std):
    score = 0.0
    weight = 0.0

    score_items = [
        ("close_price", 6.0, price_similarity(get_numeric(row, cols["close_price"]), subject["close_price"], global_price_std)),
        ("size_sqft", 2.0, closeness_similarity(get_numeric(row, cols["size_sqft"]), subject["size_sqft"], 0.15)),
        ("year_built", 1.0, closeness_similarity(get_numeric(row, cols["year_built"]), subject["year_built"], 0.05)),
        ("acres", 1.3, closeness_similarity(get_numeric(row, cols["acres"]), subject["acres"], 0.35)),
        ("garage_spaces", 0.8, closeness_similarity(get_numeric(row, cols["garage_spaces"]), subject["garage_spaces"], 0.50)),
        ("beds", 0.8, closeness_similarity(get_numeric(row, cols["beds"]), subject["beds"], 0.40)),
        ("baths_total", 0.8, closeness_similarity(get_numeric(row, cols["baths_total"]), subject["baths_total"], 0.40)),
        ("land_sqft", 0.8, closeness_similarity(get_numeric(row, cols["land_sqft"]), subject["land_sqft"], 0.35)),
    ]

    for _, feature_weight, feature_score in score_items:
        if feature_score is not None:
            weight += feature_weight
            score += feature_weight * feature_score

    for categorical_key, cat_weight in [
        ("zip", 0.6),
        ("school_district", 0.6),
        ("subdivision", 0.4),
        ("city", 0.3),
        ("view", 0.4),
        ("levels", 0.4),
        ("condition", 0.3),
    ]:
        subject_value = subject.get(categorical_key)
        if cols.get(categorical_key) and subject_value:
            weight += cat_weight
            if normalize_text(row[cols[categorical_key]]) == normalize_text(subject_value):
                score += cat_weight

    for bool_key in ["pool", "waterfront", "hoa"]:
        if cols.get(bool_key) and subject.get(bool_key) is not None:
            weight += 0.35
            if parse_boolish(row[cols[bool_key]]) == subject[bool_key]:
                score += 0.35

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
    ratio = None if pd.isna(total_std) or total_std == 0 else between_std / total_std
    return ratio, grouped.sort_values("median", ascending=False)


def category_options(df: pd.DataFrame, col_name: str):
    if not col_name or col_name not in df.columns:
        return []
    values = (
        df[col_name]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values != ""]
    return sorted(values.unique().tolist())


def load_precomputed_importance(path: Path):
    if not path.exists():
        return None, "No precomputed KNN feature-importance file found in repo."
    try:
        pre = pd.read_csv(path)
    except Exception as exc:
        return None, f"Unable to read precomputed KNN file: {exc}"

    needed = {"Feature", "KNN Importance"}
    if not needed.issubset(pre.columns):
        return None, "Precomputed KNN file is missing required columns: Feature, KNN Importance."

    out = pre[["Feature", "KNN Importance"]].copy()
    out["KNN Importance"] = pd.to_numeric(out["KNN Importance"], errors="coerce")
    out = out.dropna(subset=["KNN Importance"]).sort_values("KNN Importance", ascending=False).head(10)
    if out.empty:
        return None, "Precomputed KNN file has no valid rows."
    return out, None


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
    st.error("Required columns not found: " + ", ".join(required) + ".")
    st.stop()

working = df.copy()
working[cols["close_date"]] = pd.to_datetime(working[cols["close_date"]], errors="coerce")
working[cols["close_price"]] = pd.to_numeric(working[cols["close_price"]], errors="coerce")
working = working.dropna(subset=[cols["close_date"], cols["close_price"]])

latest_close_date = working[cols["close_date"]].max()
if pd.isna(latest_close_date):
    st.warning("No valid sold records with close date and close price.")
    st.stop()

cutoff = latest_close_date - pd.DateOffset(years=3)
working = working[working[cols["close_date"]] >= cutoff].copy()
if working.empty:
    st.warning("No sold properties found in the past 3 years.")
    st.stop()

st.caption(f"Using sold comps with close date on or after {cutoff.date()}.")

@st.cache_data(ttl=1800)
def knn_feature_importance(df: pd.DataFrame, cols: dict):
    target_col = cols["close_price"]
    feature_keys = [
        "address", "zip", "county", "city", "subdivision", "school_district", "elementary", "middle", "high",
        "size_sqft", "land_sqft", "acres", "year_built", "levels", "garage_spaces", "parking_spaces",
        "beds", "baths_total", "full_baths", "half_baths", "pool", "waterfront", "hoa", "view", "condition", "list_price",
    ]
    available = [(k, cols.get(k)) for k in feature_keys if cols.get(k) and cols.get(k) in df.columns]
    if len(available) < 2:
        return None, "Not enough feature columns available for attribution."

    work = df[[target_col] + [c for _, c in available]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    numeric_cols = []
    categorical_cols = []
    for k, c in available:
        if k in {"size_sqft", "land_sqft", "acres", "year_built", "garage_spaces", "parking_spaces", "beds", "baths_total", "full_baths", "half_baths", "list_price"}:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            numeric_cols.append(c)
        elif k in {"pool", "waterfront", "hoa"}:
            work[c] = work[c].apply(parse_boolish).astype(str)
            categorical_cols.append(c)
        else:
            work[c] = work[c].astype(str).str.strip().replace("", pd.NA)
            categorical_cols.append(c)

    work = work.dropna(subset=[target_col])
    work = work.dropna(subset=numeric_cols, how="all") if numeric_cols else work
    work = work.dropna(subset=[c for c in categorical_cols], how="all") if categorical_cols else work
    if len(work) < 80:
        return None, "Need at least 80 valid records for stable KNN attribution."

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.inspection import permutation_importance
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as exc:
        return None, f"scikit-learn is required for KNN attribution ({exc})."

    X = work[[c for _, c in available]]
    y = work[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=4,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importances = pd.DataFrame({
        "source_col": X.columns,
        "importance": perm.importances_mean,
    })

    logical_map = {col_name: key for key, col_name in available}
    importances["feature"] = importances["source_col"].map(logical_map)
    out = (
        importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(10)
    )
    return out, None


st.subheader("2) Why these features matter (price differentiation)")
st.caption("Method used: **KNN-based feature attribution** via permutation importance on a KNN regressor trained to predict close price from all available subject-feature fields.")
st.caption("⚡ Fast load mode: page reads precomputed top-10 KNN importance from `precomputed_knn_feature_importance.csv`. Use refresh to recompute live only when needed.")

if "impact_df" not in st.session_state:
    pre_df, pre_err = load_precomputed_importance(PRECOMPUTED_IMPORTANCE_PATH)
    st.session_state["impact_df"] = pre_df
    st.session_state["impact_err"] = pre_err

col_a, col_b = st.columns([1, 1])
with col_a:
    refresh_live = st.button("Run Live KNN / Refresh", key="run_feature_importance")
with col_b:
    reload_pre = st.button("Reload Precomputed from Repo", key="reload_precomputed")

if refresh_live:
    with st.spinner("Computing KNN feature attribution..."):
        impact_df, impact_err = knn_feature_importance(working, cols)
        if impact_df is not None and not impact_df.empty:
            impact_df = impact_df.rename(columns={"feature": "Feature", "importance": "KNN Importance"})
            impact_df.to_csv(PRECOMPUTED_IMPORTANCE_PATH, index=False)
        st.session_state["impact_df"] = impact_df
        st.session_state["impact_err"] = impact_err

if reload_pre:
    pre_df, pre_err = load_precomputed_importance(PRECOMPUTED_IMPORTANCE_PATH)
    st.session_state["impact_df"] = pre_df
    st.session_state["impact_err"] = pre_err

impact_df = st.session_state.get("impact_df")
impact_err = st.session_state.get("impact_err")
if impact_err:
    st.info(impact_err)
elif impact_df is not None and not impact_df.empty:
    st.dataframe(impact_df, use_container_width=True)
    st.plotly_chart(
        px.bar(impact_df, x="Feature", y="KNN Importance", title="Top 10 Feature Importance (KNN permutation attribution)"),
        use_container_width=True,
    )
else:
    st.info("Feature importance unavailable. Run live KNN once, then results can be stored in-repo for fast loads.")

st.subheader("3) Subject property features")
st.caption("Expanded feature set includes SqFt, Year, Pool, Levels, Acres, Garage, Price, Beds, and Total Baths, plus more location/school filters.")

zip_options = category_options(working, cols["zip"])
county_options = category_options(working, cols["county"])
city_options = category_options(working, cols["city"])
subdivision_options = category_options(working, cols["subdivision"])
district_options = category_options(working, cols["school_district"])
elementary_options = category_options(working, cols["elementary"])
middle_options = category_options(working, cols["middle"])
high_options = category_options(working, cols["high"])
levels_options = category_options(working, cols["levels"])
view_options = category_options(working, cols["view"])


def pick_optional(label: str, options: list[str]):
    choices = ["Not specified", *options]
    return st.selectbox(label, choices)

loc1, loc2, prop1, prop2 = st.columns(4)
with loc1:
    address = st.text_input("Address")
    zip_code = pick_optional("Zip Code", zip_options)
    county = pick_optional("County", county_options)
    city = pick_optional("City", city_options)
with loc2:
    subdivision = pick_optional("Subdivision", subdivision_options)
    school_district = pick_optional("School District", district_options)
    elementary = pick_optional("Elementary", elementary_options)
    middle = pick_optional("Middle or Junior", middle_options)
    high = pick_optional("High School", high_options)
with prop1:
    target_price = st.number_input("Price", min_value=0, step=10000, value=0)
    size_sqft = st.number_input("Total Sqft", min_value=0, step=50, value=0)
    acres = st.number_input("Acres", min_value=0.0, step=0.01, value=0.0, format="%.3f")
    year_built = st.number_input("Year Built", min_value=0, step=1, value=0)
    levels = pick_optional("Levels", levels_options)
with prop2:
    garage_spaces = st.number_input("# Garage Spaces", min_value=0.0, step=1.0, value=0.0)
    parking_spaces = st.number_input("Total Parking Spaces", min_value=0.0, step=1.0, value=0.0)
    beds = st.number_input("Total Bedrooms", min_value=0.0, step=1.0, value=0.0)
    baths_total = st.number_input("Total Baths", min_value=0.0, step=0.5, value=0.0)
    view_type = pick_optional("View", view_options)

bool1, bool2, _ = st.columns(3)
with bool1:
    has_pool = st.selectbox("Private Pool?", ["Not specified", "Yes", "No"])
with bool2:
    is_waterfront = st.selectbox("Waterfront?", ["Not specified", "Yes", "No"])
    has_hoa = st.selectbox("HOA?", ["Not specified", "Yes", "No"])

st.markdown("#### 🔒 Exact-match locks")
lock_col1, lock_col2, lock_col3 = st.columns(3)
with lock_col1:
    lock_zip = st.checkbox("Require same ZIP(s)", value=False)
    lock_school = st.checkbox("Require same school district(s)", value=False)
    lock_subdivision = st.checkbox("Require same subdivision(s)", value=False)
with lock_col2:
    lock_city = st.checkbox("Require same city", value=False)
    lock_county = st.checkbox("Require same county", value=False)
    lock_levels = st.checkbox("Require same levels", value=False)
with lock_col3:
    lock_pool = st.checkbox("Require same pool", value=False)
    lock_waterfront = st.checkbox("Require same waterfront", value=False)
    lock_hoa = st.checkbox("Require same HOA", value=False)

locked_zips = st.multiselect("Locked ZIP(s)", options=zip_options) if lock_zip else []
locked_districts = st.multiselect("Locked School District(s)", options=district_options) if lock_school else []
locked_subdivisions = st.multiselect("Locked Subdivision(s)", options=subdivision_options) if lock_subdivision else []

run = st.button("Find Top 10 Similar Sold Properties", type="primary")
if not run:
    st.stop()

subject = {
    "address": address.strip(),
    "zip": "" if zip_code == "Not specified" else zip_code.strip(),
    "county": "" if county == "Not specified" else county.strip(),
    "city": "" if city == "Not specified" else city.strip(),
    "subdivision": "" if subdivision == "Not specified" else subdivision.strip(),
    "school_district": "" if school_district == "Not specified" else school_district.strip(),
    "elementary": "" if elementary == "Not specified" else elementary.strip(),
    "middle": "" if middle == "Not specified" else middle.strip(),
    "high": "" if high == "Not specified" else high.strip(),
    "close_price": float(target_price) if target_price > 0 else None,
    "size_sqft": float(size_sqft) if size_sqft > 0 else None,
    "acres": float(acres) if acres > 0 else None,
    "year_built": float(year_built) if year_built > 0 else None,
    "levels": "" if levels == "Not specified" else levels.strip(),
    "garage_spaces": float(garage_spaces) if garage_spaces > 0 else None,
    "parking_spaces": float(parking_spaces) if parking_spaces > 0 else None,
    "beds": float(beds) if beds > 0 else None,
    "baths_total": float(baths_total) if baths_total > 0 else None,
    "view": "" if view_type == "Not specified" else view_type.strip(),
    "pool": None if has_pool == "Not specified" else has_pool == "Yes",
    "waterfront": None if is_waterfront == "Not specified" else is_waterfront == "Yes",
    "hoa": None if has_hoa == "Not specified" else has_hoa == "Yes",
    "land_sqft": None,
    "condition": "",
}

# NEW ADDITION: similarity is computed for the entire 3-year working dataset first.
working_scored = working.copy()
price_std = working_scored[cols["close_price"]].std()
working_scored["similarity_score"] = working_scored.apply(
    lambda row: score_similarity(row, subject, cols, price_std), axis=1
)

# NEW ADDITION: exact-match locks define a subsample from the full scored dataset.
candidate = working_scored.copy()
if lock_zip and cols["zip"]:
    if locked_zips:
        zset = {z.strip() for z in locked_zips}
        candidate = candidate[candidate[cols["zip"]].astype(str).str.strip().isin(zset)]
    elif subject["zip"]:
        candidate = candidate[candidate[cols["zip"]].astype(str).str.strip() == subject["zip"]]
if lock_school and cols["school_district"]:
    if locked_districts:
        dset = {d.strip().lower() for d in locked_districts}
        candidate = candidate[candidate[cols["school_district"]].astype(str).str.strip().str.lower().isin(dset)]
    elif subject["school_district"]:
        candidate = candidate[candidate[cols["school_district"]].astype(str).str.strip().str.lower() == subject["school_district"].lower()]
if lock_subdivision and cols["subdivision"]:
    if locked_subdivisions:
        sset = {s.strip().lower() for s in locked_subdivisions}
        candidate = candidate[candidate[cols["subdivision"]].astype(str).str.strip().str.lower().isin(sset)]
    elif subject["subdivision"]:
        candidate = candidate[candidate[cols["subdivision"]].astype(str).str.strip().str.lower() == subject["subdivision"].lower()]
if lock_city and cols["city"] and subject["city"]:
    candidate = candidate[candidate[cols["city"]].astype(str).str.strip().str.lower() == subject["city"].lower()]
if lock_county and cols["county"] and subject["county"]:
    candidate = candidate[candidate[cols["county"]].astype(str).str.strip().str.lower() == subject["county"].lower()]
if lock_levels and cols["levels"] and subject["levels"]:
    candidate = candidate[candidate[cols["levels"]].astype(str).str.strip().str.lower() == subject["levels"].lower()]
if lock_pool and cols["pool"] and subject["pool"] is not None:
    candidate = candidate[candidate[cols["pool"]].apply(parse_boolish) == subject["pool"]]
if lock_waterfront and cols["waterfront"] and subject["waterfront"] is not None:
    candidate = candidate[candidate[cols["waterfront"]].apply(parse_boolish) == subject["waterfront"]]
if lock_hoa and cols["hoa"] and subject["hoa"] is not None:
    candidate = candidate[candidate[cols["hoa"]].apply(parse_boolish) == subject["hoa"]]

if candidate.empty:
    st.warning("No records left after applying exact-match locks. Relax one or more locks.")
    st.stop()

result = candidate.sort_values("similarity_score", ascending=False).head(10).copy()
st.success(f"Found {len(result)} comps. Showing top {len(result)} by similarity score.")

show_cols = [
    cols["address"],
    cols["zip"],
    cols["city"],
    cols["school_district"],
    cols["subdivision"],
    cols["size_sqft"],
    cols["year_built"],
    cols["pool"],
    cols["levels"],
    cols["acres"],
    cols["garage_spaces"],
    cols["beds"],
    cols["baths_total"],
    cols["close_price"],
    cols["close_date"],
]
show_cols = [c for c in show_cols if c and c in result.columns]
show_cols.append("similarity_score")

st.subheader("Top 10 Similar Sold Properties")
st.dataframe(result[show_cols], use_container_width=True)


# -------------------- NEW ADDITION: Agent-level ranking from subsample --------------------
st.subheader("🏅 Agent Ranking (from selected comp subsample)")
agent_col = cols.get("agent_name")
if not agent_col or agent_col not in candidate.columns:
    st.info("Agent ranking requires an agent name column (e.g., ListAgentFullName).")
else:
    agent_base = candidate.copy()
    agent_base[agent_col] = agent_base[agent_col].astype(str).str.strip()
    agent_base = agent_base[agent_base[agent_col] != ""]

    if agent_base.empty:
        st.info("No agent data available in the filtered subsample.")
    else:
        if cols.get("close_price"):
            agent_base[cols["close_price"]] = pd.to_numeric(agent_base[cols["close_price"]], errors="coerce")
        if cols.get("days_on_market") and cols["days_on_market"] in agent_base.columns:
            agent_base[cols["days_on_market"]] = pd.to_numeric(agent_base[cols["days_on_market"]], errors="coerce")
        if cols.get("pricing_accuracy") and cols["pricing_accuracy"] in agent_base.columns:
            agent_base[cols["pricing_accuracy"]] = pd.to_numeric(agent_base[cols["pricing_accuracy"]], errors="coerce")
        if cols.get("is_closed") and cols["is_closed"] in agent_base.columns:
            agent_base[cols["is_closed"]] = pd.to_numeric(agent_base[cols["is_closed"]], errors="coerce")

        agg_dict = {
            "transactions": (agent_col, "count"),
            "total_sales": (cols["close_price"], "sum"),
            "avg_similarity": ("similarity_score", "mean"),
            "median_similarity": ("similarity_score", "median"),
        }
        if cols.get("is_closed") and cols["is_closed"] in agent_base.columns:
            agg_dict["closed_count"] = (cols["is_closed"], "sum")
        if cols.get("days_on_market") and cols["days_on_market"] in agent_base.columns:
            agg_dict["avg_days_on_market"] = (cols["days_on_market"], "mean")
        if cols.get("pricing_accuracy") and cols["pricing_accuracy"] in agent_base.columns:
            agg_dict["avg_pricing_accuracy"] = (cols["pricing_accuracy"], "mean")

        agent_summary = agent_base.groupby(agent_col, dropna=False).agg(**agg_dict).reset_index()

        if "closed_count" in agent_summary.columns:
            agent_summary["close_rate"] = agent_summary["closed_count"] / agent_summary["transactions"].replace(0, pd.NA)
        else:
            agent_summary["close_rate"] = pd.NA

        # similar style scoring to other pages
        agent_summary["sales_score"] = percentile_score(agent_summary["total_sales"])
        agent_summary["volume_score"] = percentile_score(agent_summary["transactions"])
        agent_summary["similarity_score_norm"] = percentile_score(agent_summary["avg_similarity"])
        if "close_rate" in agent_summary.columns:
            agent_summary["close_rate_score"] = percentile_score(agent_summary["close_rate"])
        else:
            agent_summary["close_rate_score"] = 50.0
        if "avg_days_on_market" in agent_summary.columns:
            agent_summary["days_on_market_score"] = 100 - percentile_score(agent_summary["avg_days_on_market"])
        else:
            agent_summary["days_on_market_score"] = 50.0
        if "avg_pricing_accuracy" in agent_summary.columns:
            agent_summary["pricing_accuracy_score"] = pricing_accuracy_to_score(agent_summary["avg_pricing_accuracy"])
        else:
            agent_summary["pricing_accuracy_score"] = 50.0

        agent_summary["overall_agent_score"] = (
            0.40 * agent_summary["similarity_score_norm"]
            + 0.20 * agent_summary["sales_score"]
            + 0.15 * agent_summary["volume_score"]
            + 0.10 * agent_summary["close_rate_score"]
            + 0.10 * agent_summary["days_on_market_score"]
            + 0.05 * agent_summary["pricing_accuracy_score"]
        )

        top_agents = agent_summary.sort_values("overall_agent_score", ascending=False).head(10)

        st.caption(
            "Agent ranking is computed from the exact-lock subsample (if locks are set), after similarity scoring was done on the full 3-year dataset."
        )
        st.dataframe(top_agents, use_container_width=True)
