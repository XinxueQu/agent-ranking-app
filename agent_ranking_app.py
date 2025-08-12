import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Agent Rankings", layout="wide")
st.markdown("<h1 style='text-align: center; color: darkblue;'>🏡 Top Real Estate Agent Rankings</h1>", unsafe_allow_html=True)

# -------------------- Load data (CSV is faster) --------------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=csv"
    usecols = [
        "ListAgentFullName","is_closed","DaysOnMarket","pricing_accuracy",
        "PostalCode","ClosePrice","ElementarySchool","SubdivisionName"
    ]
    return pd.read_csv(url, usecols=usecols)

data = load_data()

# -------------------- Helpers --------------------
def pricing_accuracy_score(x): return (1 - abs(x - 1)) * 100
def percentile_score(s): return s.rank(pct=True) * 100
def score_days_on_market(s): return 100 - s.rank(pct=True) * 100

@st.cache_data
def build_agent_summary(df: pd.DataFrame) -> pd.DataFrame:
    agent_summary = (
        df.groupby('ListAgentFullName', dropna=False)
        .agg(
            total_records=('ListAgentFullName', 'count'),
            closed_count=('is_closed', 'sum'),
            closed_daysonmarket_mean=('DaysOnMarket', lambda x: x[df.loc[x.index, 'is_closed']].mean()),
            closed_daysonmarket_median=('DaysOnMarket', lambda x: x[df.loc[x.index, 'is_closed']].median()),
            avg_pricing_accuracy=('pricing_accuracy', 'mean')
        )
        .reset_index()
    )
    agent_summary['close_rate']               = agent_summary['closed_count'] / agent_summary['total_records']
    agent_summary['pricing_accuracy_score']   = agent_summary['avg_pricing_accuracy'].apply(pricing_accuracy_score)
    agent_summary['volume_score']             = percentile_score(agent_summary['total_records'])
    agent_summary['close_rate_score']         = percentile_score(agent_summary['close_rate'])
    agent_summary['avg_days_on_mkt_score']    = score_days_on_market(agent_summary['closed_daysonmarket_mean'])
    agent_summary['median_days_on_mkt_score'] = score_days_on_market(agent_summary['closed_daysonmarket_median'])
    return agent_summary

agent_summary = build_agent_summary(data)

# -------------------- Form (inputs don't trigger reruns) --------------------
with st.form("filters_and_weights"):
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📍 Filter Listings")
        zipcode = st.text_input("Zipcode")
        min_price = st.number_input("Minimum Price", value=0)
        max_price = st.number_input("Maximum Price", value=1_000_000)
        elementary = st.text_input("Elementary School")
        subdivision = st.text_input("Subdivision")
        min_volume = st.number_input("Minimum Total Transactions", value=0)

    with right_col:
        st.subheader("⚖️ Scoring Weights")
        weight_volumne = st.number_input("Transaction Volume", value=0.4, key="w_vol")
        weight_close   = st.number_input("Close Rate",            value=0.3, key="w_close")
        weight_days    = st.number_input("Days on Market",        value=0.2, key="w_days")
        weight_price   = st.number_input("Pricing Accuracy",      value=0.1, key="w_price")

    submitted = st.form_submit_button("Run Rankings")

# -------------------- Compute + store results on submit --------------------
if submitted:
    total_weight = weight_volumne + weight_close + weight_days + weight_price
    if total_weight <= 0:
        st.error("All weights are zero. Please adjust.")
        st.stop()
    if total_weight != 1:
        st.warning("Weights do not sum to 1. Normalizing automatically.")
        weight_volumne /= total_weight
        weight_close   /= total_weight
        weight_days    /= total_weight
        weight_price   /= total_weight

    # Add overall score
    scored = agent_summary.copy()
    scored['overall_score'] = (
        weight_volumne * scored['volume_score'] +
        weight_close   * scored['close_rate_score'] +
        weight_days    * scored['median_days_on_mkt_score'] +
        weight_price   * scored['pricing_accuracy_score']
    )

    # --- Filtering ---
    df_filtered = data.copy()
    if zipcode and pd.notna(zipcode) and zipcode in df_filtered['PostalCode'].dropna().unique():
        df_filtered = df_filtered[df_filtered['PostalCode'] == zipcode]
    # Normalize to string for safe comparison
    if zipcode:
        z = str(zipcode).strip()
        df_filtered = df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z]
    df_filtered = df_filtered[df_filtered['ClosePrice'] >= min_price]
    df_filtered = df_filtered[df_filtered['ClosePrice'] <= max_price]
    if elementary and pd.notna(elementary) and elementary in df_filtered['ElementarySchool'].dropna().unique():
        df_filtered = df_filtered[df_filtered['ElementarySchool'] == elementary]
    if subdivision and pd.notna(subdivision) and subdivision in df_filtered['SubdivisionName'].dropna().unique():
        df_filtered = df_filtered[df_filtered['SubdivisionName'] == subdivision]

    filtered_agent_counts = (
        df_filtered.groupby('ListAgentFullName', dropna=False)
        .size()
        .reset_index(name='n')
    )
    filtered_agent_counts_selected = filtered_agent_counts[filtered_agent_counts['n'] >= min_volume]

    selected_agents = (
        scored[scored['ListAgentFullName'].isin(filtered_agent_counts_selected['ListAgentFullName'].unique())]
        .sort_values(by='overall_score', ascending=False)
    )
    first = ['ListAgentFullName', 'overall_score']
    rest = [c for c in selected_agents.columns if c not in first]
    selected_agents = selected_agents.loc[:, first + rest]

    # Save for paging/render after the form
    st.session_state.selected_agents = selected_agents
    st.session_state.df_filtered = df_filtered
    st.session_state.page_num = 1  # reset to first page on new results

# -------------------- Render (persist across Next/Previous) --------------------
# Reuse prior results if user is paging without resubmitting
if not submitted and "selected_agents" in st.session_state:
    selected_agents = st.session_state.selected_agents

if "selected_agents" in st.session_state:
    # --- Pagination state ---
    records_per_page = 10
    num_agents = len(selected_agents)
    total_pages = max((num_agents - 1) // records_per_page + 1, 1)

    if "page_num" not in st.session_state:
        st.session_state.page_num = 1

    start_idx = (st.session_state.page_num - 1) * records_per_page
    end_idx = start_idx + records_per_page
    paged_agents = selected_agents.iloc[start_idx:end_idx]

    st.subheader("🏆 Top Ranked Agents")
    st.dataframe(paged_agents, use_container_width=True, height=420)
    st.caption(f"Showing page {st.session_state.page_num} of {total_pages} ({num_agents} agents found)")

    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.page_num > 1:
            st.session_state.page_num -= 1
    with col3:
        if st.button("Next ➡️") and st.session_state.page_num < total_pages:
            st.session_state.page_num += 1

    # -------------------- Summary table (from selected_agents + filtered data) --------------------
    # -------------------- Summary table (from agent_summary + filtered data) --------------------
    df_filtered = st.session_state.df_filtered  # from earlier step
    
    # Denominator: total sales per agent within the CURRENT filtered slice
    totals = (
        df_filtered.groupby('ListAgentFullName', dropna=False)['ClosePrice']
        .sum()
        .rename('Total_Sales')
        .reset_index()
    )
    
    # Numerator: sales in the typed zipcode (normalize types for comparison)
    if zipcode:
        z_str = str(zipcode).strip()
        sales_in_zip = (
            df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z_str]
            .groupby('ListAgentFullName', dropna=False)['ClosePrice']
            .sum()
            .rename('Sales_In_Zip')
            .reset_index()
        )
    else:
        sales_in_zip = totals[['ListAgentFullName']].assign(Sales_In_Zip=np.nan)
    
    # Build table using PRE-COMPUTED metrics from agent_summary / selected_agents
    tbl = (
        selected_agents  # already derived from agent_summary and contains close_rate, med days, etc.
        .merge(totals, on='ListAgentFullName', how='left')
        .merge(sales_in_zip, on='ListAgentFullName', how='left')
    )
    
    tbl['%_Sales_in_Zip'] = (tbl['Sales_In_Zip'] / tbl['Total_Sales']).replace([np.inf, -np.inf], np.nan)
    
      # Ranks (descending, ties allowed)
    tbl['Rank'] = tbl['overall_score'].rank(ascending=False, method='dense').astype(int)
    tbl['Close Rate Rank'] = tbl['close_rate'].rank(ascending=False, method='dense').astype(int)
    tbl['Days on Market Rank'] = tbl['closed_daysonmarket_median'].rank(ascending=False, method='dense').astype(int)
    tbl['Pricing Accuracy Rank'] = tbl['avg_pricing_accuracy'].rank(ascending=False, method='dense').astype(int)

    final_cols = [
        'Rank', 'ListAgentFullName', 'overall_score',
        'Total_Sales', 'Closed_Transactions', '%_Sales_in_Zip',
        'close_rate', 'closed_daysonmarket_median', 'avg_pricing_accuracy',
        'Close Rate Rank', 'Days on Market Rank', 'Pricing Accuracy Rank'
    ]
    tbl = tbl[final_cols].sort_values(['Rank', 'overall_score'])
    
    st.subheader("📊 Summary by Agent (Filtered)")
    st.dataframe(tbl, use_container_width=True)

