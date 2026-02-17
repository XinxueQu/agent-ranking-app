import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Uploaded Dataset Ranking", layout="wide")
st.title("📁 Uploaded Dataset Ranking")
st.caption("Upload your own dataset, then run the same ranking analysis used in the main app.")

REQUIRED_COLUMNS = [
    "ListAgentFullName",
    "is_closed",
    "DaysOnMarket",
    "pricing_accuracy",
    "PostalCode",
    "ClosePrice",
    "ElementarySchool",
    "SubdivisionName",
]


def pricing_accuracy_score(x):
    return (1 - abs(x - 1)) * 100


def percentile_score(s):
    return s.rank(pct=True) * 100


def score_days_on_market(s):
    return 100 - s.rank(pct=True) * 100


def get_norm(df: pd.DataFrame, row: pd.Series, col: str, invert: bool = False) -> float:
    if col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors="coerce")
    x = pd.to_numeric(row.get(col, np.nan), errors="coerce")
    vmin, vmax = np.nanmin(s.values), np.nanmax(s.values)
    if np.isnan(x) or np.isnan(vmin) or np.isnan(vmax):
        return np.nan
    if vmax == vmin:
        return 50.0
    val = (x - vmin) / (vmax - vmin)
    if invert:
        val = 1.0 - val
    return float(np.clip(val * 100.0, 0, 100))


def build_agent_summary(df: pd.DataFrame) -> pd.DataFrame:
    agent_summary = (
        df.groupby("ListAgentFullName", dropna=False)
        .agg(
            total_records=("ListAgentFullName", "count"),
            total_sales=("ClosePrice", "sum"),
            closed_count=("is_closed", "sum"),
            closed_daysonmarket_mean=(
                "DaysOnMarket",
                lambda x: x[df.loc[x.index, "is_closed"]].mean(),
            ),
            closed_daysonmarket_median=(
                "DaysOnMarket",
                lambda x: x[df.loc[x.index, "is_closed"]].median(),
            ),
            avg_pricing_accuracy=("pricing_accuracy", "mean"),
        )
        .reset_index()
    )
    agent_summary["close_rate"] = agent_summary["closed_count"] / agent_summary["total_records"]
    agent_summary["pricing_accuracy_score"] = agent_summary["avg_pricing_accuracy"].apply(
        pricing_accuracy_score
    )
    agent_summary["sales_score"] = percentile_score(agent_summary["total_sales"])
    agent_summary["volume_score"] = percentile_score(agent_summary["total_records"])
    agent_summary["close_rate_score"] = percentile_score(agent_summary["close_rate"])
    agent_summary["avg_days_on_mkt_score"] = score_days_on_market(
        agent_summary["closed_daysonmarket_mean"]
    )
    agent_summary["median_days_on_mkt_score"] = score_days_on_market(
        agent_summary["closed_daysonmarket_median"]
    )
    return agent_summary


uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Dataset must contain the same columns as the default source.",
)

if not uploaded_file:
    st.info("Upload a file to begin.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read file: {exc}")
    st.stop()

missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing_columns:
    st.error(
        "Uploaded file is missing required columns: " + ", ".join(missing_columns)
    )
    st.stop()

data = data[REQUIRED_COLUMNS].copy()
agent_summary = build_agent_summary(data)

with st.form("upload_filters_and_weights"):
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📍 Filter Listings")
        zip_options = sorted(data["PostalCode"].dropna().astype(str).unique())
        zipcodes = st.multiselect("Zipcode(s)", options=zip_options)
        min_price = st.number_input("Minimum Price", value=0)
        max_price = st.number_input("Maximum Price", value=1_000_000)
        elementary = st.text_input("Elementary School")
        subdivision = st.text_input("Subdivision")
        min_volume = st.number_input("Minimum Total Transactions", value=0)
        min_median_close = st.number_input("Minimum Median Close Price", value=100_000)
        max_median_close = st.number_input("Maximum Median Close Price", value=1_000_000)

    with right_col:
        st.subheader("⚖️ Scoring Weights")
        weight_volume = st.number_input("Transaction Volume", value=0.5, key="up_w_vol")
        weight_close = st.number_input("Close Rate", value=0.3, key="up_w_close")
        weight_days = st.number_input("Days on Market", value=0.2, key="up_w_days")
        weight_price = st.number_input("Pricing Accuracy", value=0.0, key="up_w_price")

    submitted = st.form_submit_button("Run Rankings")

if not submitted:
    st.info("Configure filters and click Run Rankings.")
    st.stop()

total_weight = weight_volume + weight_close + weight_days + weight_price
if total_weight <= 0:
    st.error("All weights are zero. Please adjust.")
    st.stop()
if total_weight != 1:
    st.warning("Weights do not sum to 1. Normalizing automatically.")
    weight_volume /= total_weight
    weight_close /= total_weight
    weight_days /= total_weight
    weight_price /= total_weight

scored = agent_summary.copy()
scored["overall_score"] = (
    weight_volume * scored["volume_score"]
    + weight_close * scored["close_rate_score"]
    + weight_days * scored["median_days_on_mkt_score"]
    + weight_price * scored["pricing_accuracy_score"]
)

df_filtered = data.copy()
if zipcodes:
    z_set = {str(z).strip() for z in zipcodes}
    df_filtered = df_filtered[df_filtered["PostalCode"].astype(str).str.strip().isin(z_set)]

df_filtered = df_filtered[(df_filtered["ClosePrice"] >= min_price) & (df_filtered["ClosePrice"] <= max_price)]
if elementary:
    df_filtered = df_filtered[df_filtered["ElementarySchool"] == elementary]
if subdivision:
    df_filtered = df_filtered[df_filtered["SubdivisionName"] == subdivision]

median_close = (
    df_filtered.groupby("ListAgentFullName", dropna=False)["ClosePrice"]
    .median()
    .reset_index(name="Median Close Price")
)

valid_agents = median_close[
    (median_close["Median Close Price"] >= min_median_close)
    & (median_close["Median Close Price"] <= max_median_close)
]["ListAgentFullName"]

df_filtered = df_filtered[df_filtered["ListAgentFullName"].isin(valid_agents)]

filtered_agent_counts = (
    df_filtered.groupby("ListAgentFullName", dropna=False).size().reset_index(name="n")
)
filtered_agent_counts = filtered_agent_counts.merge(
    median_close, on="ListAgentFullName", how="left"
)
filtered_agent_counts_selected = filtered_agent_counts[filtered_agent_counts["n"] >= min_volume]

selected_agents = (
    scored[scored["ListAgentFullName"].isin(filtered_agent_counts_selected["ListAgentFullName"].unique())]
    .merge(
        filtered_agent_counts_selected[["ListAgentFullName", "Median Close Price"]],
        on="ListAgentFullName",
        how="left",
    )
    .sort_values(by="overall_score", ascending=False)
)

if selected_agents.empty:
    st.warning("No agents matched your filters.")
    st.stop()

rankings_tab, dimensions_tab = st.tabs(["🏆 Rankings", "📐 Multi-dimension view"])

with rankings_tab:
    tbl = selected_agents.copy()
    tbl["Rank"] = tbl["overall_score"].rank(ascending=False, method="dense").astype(int)
    tbl["Close Rate Rank"] = tbl["close_rate"].rank(ascending=False, method="dense").astype(int)
    tbl["Days on Market Rank"] = tbl["closed_daysonmarket_median"].rank(
        ascending=True, method="dense"
    ).astype(int)
    tbl["Pricing Accuracy Rank"] = tbl["avg_pricing_accuracy"].rank(
        ascending=False, method="dense"
    ).astype(int)

    final_cols = [
        "Rank",
        "ListAgentFullName",
        "overall_score",
        "Median Close Price",
        "total_sales",
        "closed_count",
        "close_rate",
        "closed_daysonmarket_median",
        "avg_pricing_accuracy",
        "Close Rate Rank",
        "Days on Market Rank",
        "Pricing Accuracy Rank",
    ]
    st.dataframe(tbl[final_cols], use_container_width=True)

with dimensions_tab:
    options = selected_agents["ListAgentFullName"].dropna().astype(str).sort_values().unique().tolist()
    agent_to_view = st.selectbox("Choose an agent", options=options)
    row = selected_agents.loc[selected_agents["ListAgentFullName"] == agent_to_view].iloc[0]

    dims = {
        "Volume": get_norm(selected_agents, row, "volume_score"),
        "Close Rate": get_norm(selected_agents, row, "close_rate_score"),
        "Days on Market": get_norm(selected_agents, row, "closed_daysonmarket_median", invert=True),
        "Total Sales": get_norm(selected_agents, row, "sales_score"),
    }
    dim_df = pd.DataFrame({"Dimension": list(dims.keys()), "Score": list(dims.values())}).dropna()

    if len(dim_df) >= 3:
        r = dim_df["Score"].tolist()
        theta = dim_df["Dimension"].tolist()
        fig_radar = go.Figure(data=go.Scatterpolar(r=r + [r[0]], theta=theta + [theta[0]], fill="toself"))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    fig_bar = px.bar(dim_df, x="Dimension", y="Score", range_y=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)
