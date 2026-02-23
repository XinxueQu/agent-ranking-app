@@ -37,133 +37,80 @@ def load_data() -> pd.DataFrame:
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
    pct_rank = scores.rank(pct=True, ascending=False, method="max") * 100

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
        if p <= 60:
            return "Top 60%"
        if p <= 70:
            return "Top 70%"
        if p <= 80:
            return "Top 80%"
        return "Top 90%"

    return pct_rank.apply(bucket)

def add_percentile_and_tier(
    df: pd.DataFrame,
    value_col: str,
    out_pct_col: str,
    out_tier_col: str,
    higher_is_better: bool = True,
    tie_break_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Adds:
      - out_pct_col: percentile score 0..100 (higher = better)
      - out_tier_col: "Top X%" bucket based on that percentile

    Tie-breaking:
      We add tiny deterministic noise based on tie_break_cols to avoid identical ranks.
    """
    s = pd.to_numeric(df[value_col], errors="coerce").fillna(np.nan)

    # Flip if lower is better (e.g., days on market)
    if not higher_is_better:
        s = -s

    # Deterministic tie-breaker noise
    if tie_break_cols:
        # Build a stable per-row numeric key from tie-break columns
        key = (
            df[tie_break_cols]
            .astype(str)
            .agg("|".join, axis=1)
            .apply(lambda x: abs(hash(x)) % 1_000_000)
            .astype(float)
        )
        noise = (key / 1_000_000) * 1e-6  # very tiny
        s_adj = s + noise
    else:
        s_adj = s

    # Percentile score: 0..100 where higher = better
    pct = s_adj.rank(pct=True, ascending=True, method="average") * 100
    df[out_pct_col] = pct

    # Convert percentile to "Top X%" where smaller top% is better
    top_pct = 100 - df[out_pct_col]

    def bucket(p: float) -> str:
        if p <= 1: return "Top 1%"
        if p <= 3: return "Top 3%"
        if p <= 5: return "Top 5%"
        if p <= 10: return "Top 10%"
        if p <= 20: return "Top 20%"
        if p <= 30: return "Top 30%"
        if p <= 50: return "Top 50%"
        return "Top 100%"

    df[out_tier_col] = top_pct.apply(bucket)
    return df
    
data = load_data().copy()

# Ensure numeric/date columns are properly typed
for num_col in ["ClosePrice", "DaysOnMarket", "pricing_accuracy", "is_closed"]:
    data[num_col] = pd.to_numeric(data[num_col], errors="coerce")

data["CloseDate"] = pd.to_datetime(data["CloseDate"], errors="coerce")
data["ListingContractDate"] = pd.to_datetime(data["ListingContractDate"], errors="coerce")
data["City"] = data["City"].astype(str).str.strip()
data["PostalCode"] = data["PostalCode"].astype(str).str.strip()
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
@@ -269,51 +216,51 @@ if in_price_range.empty:
    st.stop()

# ---------------- Agent-level summary ----------------
agent_stats = (
    in_price_range.groupby("ListAgentFullName", dropna=False)
    .agg(
        total_transactions=("ListAgentFullName", "count"),
        total_sales=("ClosePrice", "sum"),
        closed_count=("is_closed", lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum()),
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

agent_stats["close_rate"] = agent_stats["closed_count"] / agent_stats["total_transactions"]
agent_stats["pricing_accuracy_score"] = agent_stats["avg_pricing_accuracy"].apply(pricing_accuracy_score)
agent_stats["volume_score"] = percentile_score(agent_stats["total_transactions"])
agent_stats["sales_score"] = percentile_score(agent_stats["total_sales"])
agent_stats["close_rate_score"] = percentile_score(agent_stats["close_rate"])
agent_stats["days_on_market_score"] = 100 - agent_stats["median_days_on_market"].rank(pct=True) * 100
agent_stats["days_on_market_score"] = score_days_on_market(agent_stats["median_days_on_market"])

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

@@ -350,87 +297,84 @@ priority_options = {
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
    + weights["Days on Market"] * agent_stats["days_on_market_score"]
    + weights["Pricing Accuracy"] * agent_stats["pricing_accuracy_score"]
)

agent_stats = agent_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["overall_score"]).copy()
agent_stats = agent_stats.sort_values(
    by=["overall_score", "total_transactions", "close_rate", "median_days_on_market"],
    ascending=[False, False, False, True],
)

agent_stats["Tier"] = to_top_percent_bucket(agent_stats["overall_score"])
# Percentiles + tiers (ties broken deterministically)
tie_cols = ["ListAgentFullName", "total_transactions", "total_sales"]
# Tiers: NO tie break
agent_stats = add_percentile_and_tier(agent_stats, "total_transactions", "Volume Percentile", "Volume Tier", True, None)
agent_stats = add_percentile_and_tier(agent_stats, "close_rate", "Close Rate Percentile", "Close Rate Tier", True, None)
agent_stats = add_percentile_and_tier(agent_stats, "median_days_on_market", "Median DOM Percentile", "Median DOM Tier", False, None)
# Reuse the same metric definitions as 1_Alternate_Ranking.py for performance tiers.
agent_stats["Volume Tier"] = to_top_percent_bucket(agent_stats["volume_score"])
agent_stats["Close Rate Tier"] = to_top_percent_bucket(agent_stats["close_rate_score"])
agent_stats["Days on Market Tier"] = to_top_percent_bucket(agent_stats["days_on_market_score"])
agent_stats["Pricing Accuracy Tier"] = to_top_percent_bucket(agent_stats["pricing_accuracy_score"])

# Ranking/top10: DO tie breaking via sort keys
agent_stats = agent_stats.sort_values(
    by=["overall_score", "total_transactions", "total_sales", "close_rate", "median_days_on_market"],
    ascending=[False, False, False, False, True],
)

final_top10 = agent_stats.head(10).copy()
# Top 10 final table
final_top10 = agent_stats.head(10).copy()

st.subheader("🏆 Final Top 10 Agents")
st.caption("Top agents based on selected filters and weighted score. Tiers are computed across all filtered agents before selecting top 10.")

final_cols = [
    "ListAgentFullName",
    "ListAgentDirectPhone",
    "total_transactions",
    "total_sales",
    "Volume Tier",
    "Close Rate Tier",
    "Mean DOM Tier",
    "Median DOM Tier",
    "Days on Market Tier",
    "Pricing Accuracy Tier",
]

st.data_editor(
    final_top10[final_cols],
    use_container_width=True,
    hide_index=True,
    disabled=True,
    column_config={
        "ListAgentFullName": "Agent",
        "ListAgentDirectPhone": st.column_config.TextColumn("📞 Phone"),
        "total_transactions": st.column_config.NumberColumn("Transactions"),
        "total_sales": st.column_config.NumberColumn("Total Sales ($)", format="$%.0f"),
    },
)

st.subheader("📋 Selected Agent Performance Details")
detail_cols = [
    "ListAgentFullName",
    "Volume Tier",
    "Close Rate Tier",
    "Mean DOM Tier",
    "Median DOM Tier",
    "Days on Market Tier",
    "Pricing Accuracy Tier",
    "total_transactions",
    "closed_count",
    "close_rate",
    "mean_days_on_market",
    "median_days_on_market",
    "avg_pricing_accuracy",
    "total_sales",
]
st.dataframe(final_top10[detail_cols], use_container_width=True)
