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
    required_cols = {
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
    }

    def include_col(col_name: str) -> bool:
        col = str(col_name)
        normalized = "".join(ch for ch in col.lower() if ch.isalnum())
        is_property_id_col = (
            normalized in {"propertyid", "propertyidentifier", "propertyidentifierid", "mlsnumber"}
            or ("property" in normalized and "id" in normalized)
            or normalized.startswith("acl")
        )
        return col in required_cols or is_property_id_col

    return pd.read_excel(url, usecols=include_col)


def find_property_id_column(df: pd.DataFrame) -> str | None:
    preferred_names = ["PropertyId", "PropertyID", "Property Id", "PropertyIdentifier", "ACL"]
    for col in preferred_names:
        if col in df.columns:
            return col

    for col in df.columns:
        normalized = "".join(ch for ch in str(col).lower() if ch.isalnum())
        if ("property" in normalized and "id" in normalized) or normalized.startswith("acl"):
            return col

    return None


def clean_property_id(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    if text.upper().startswith("ACL"):
        text = text[3:]

    return text.lstrip("-_:# ")


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
@@ -97,50 +133,51 @@ def to_top_percent_bucket(scores: pd.Series) -> pd.Series:

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
property_id_col = find_property_id_column(data)

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
@@ -310,50 +347,64 @@ in_price_range = window_filtered[
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

if property_id_col and property_id_col in in_price_range.columns:
    transaction_ids = (
        in_price_range.assign(
            _clean_property_id=in_price_range[property_id_col].apply(clean_property_id)
        )
        .groupby("ListAgentFullName", dropna=False)["_clean_property_id"]
        .apply(lambda values: "; ".join([v for v in pd.unique(values) if v]))
    )
    agent_stats["transaction_property_ids"] = (
        agent_stats["ListAgentFullName"].map(transaction_ids).fillna("")
    )
else:
    agent_stats["transaction_property_ids"] = ""

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

@@ -510,43 +561,45 @@ st.data_editor(
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
    "transaction_property_ids",
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
        "transaction_property_ids": st.column_config.TextColumn("Transaction Property IDs"),
    },
)
