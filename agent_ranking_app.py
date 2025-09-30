import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ---- near the top (after imports) ----
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🏆 Rankings"

def focus_dims():
    st.session_state.active_tab = "📐 Multi-dimension view"

def focus_rankings():
    st.session_state.active_tab = "🏆 Rankings"


# Put this helper above the code block (once in your app)
def get_norm(col: str, invert: bool = False) -> float:
    """Min–max normalize the selected agent's value to [0,100].
       If invert=True, lower raw values map to higher normalized scores."""
    if col not in selected_agents.columns:
        return np.nan
    s = pd.to_numeric(selected_agents[col], errors="coerce")
    x = pd.to_numeric(row.get(col, np.nan), errors="coerce")
    vmin, vmax = np.nanmin(s.values), np.nanmax(s.values)
    # Handle degenerate cases: all equal or missing
    if np.isnan(x) or np.isnan(vmin) or np.isnan(vmax):
        return np.nan
    if vmax == vmin:
        # Neutral score so charts still render (adjust if you prefer 100)
        return 50.0
    val = (x - vmin) / (vmax - vmin)
    if invert:
        val = 1.0 - val
    return float(np.clip(val * 100.0, 0, 100))



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
            total_sales = ('ClosePrice', 'sum'),
            closed_count=('is_closed', 'sum'),
            closed_daysonmarket_mean=('DaysOnMarket', lambda x: x[df.loc[x.index, 'is_closed']].mean()),
            closed_daysonmarket_median=('DaysOnMarket', lambda x: x[df.loc[x.index, 'is_closed']].median()),
            avg_pricing_accuracy=('pricing_accuracy', 'mean')
        )
        .reset_index()
    )
    agent_summary['close_rate']               = agent_summary['closed_count'] / agent_summary['total_records']
    agent_summary['pricing_accuracy_score']   = agent_summary['avg_pricing_accuracy'].apply(pricing_accuracy_score)
    agent_summary['sales_score']             = percentile_score(agent_summary['total_sales'])
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
        #zipcode = st.text_input("Zipcode")
        # NEW:
        zip_options = sorted(data["PostalCode"].dropna().astype(str).unique())
        zipcodes = st.multiselect("Zipcode(s)", options=zip_options)
        min_price = st.number_input("Minimum Price", value=0)
        max_price = st.number_input("Maximum Price", value=1_000_000)
        elementary = st.text_input("Elementary School")
        subdivision = st.text_input("Subdivision")
        min_volume = st.number_input("Minimum Total Transactions", value=0)
        
        #min_sales_pct = st.slider(
        #    "Min % Sales in Selected ZipCode",
        #    min_value=0,   max_value=100,  value=0, step=1,
        #    format="%d%%" )/100

        # Text input box (shown as percentage for user convenience)
        default_sales_pct = 10
        min_sales_pct = st.text_input(
            "Min % Sales in Selected ZipCode (0–100%)",
            value=str(default_sales_pct)
        )
        
        # Convert input to float safely
        try:
            min_sales_pct = float(min_sales_pct)
            # Validate range
            if not (0 <= min_sales_pct <= 100):
                st.warning("Value must be between 0 and 100%. Resetting to default (10%).")
                min_sales_pct = default_sales_pct
        except ValueError:
            st.warning("Invalid input. Resetting to default (0.1).")
            min_sales_pct = default_sales_pct
        min_sales_pct = min_sales_pct/100

    with right_col:
        st.subheader("⚖️ Scoring Weights")
        weight_volumne = st.number_input("Transaction Volume", value=0.5, key="w_vol")
        weight_close   = st.number_input("Close Rate",            value=0.3, key="w_close")
        weight_days    = st.number_input("Days on Market",        value=0.2, key="w_days")
        weight_price   = st.number_input("Pricing Accuracy",      value=0, key="w_price")

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
    # one zipcode
    #if zipcode and pd.notna(zipcode) and zipcode in df_filtered['PostalCode'].dropna().unique():
    #    df_filtered = df_filtered[df_filtered['PostalCode'] == zipcode]
    #if zipcode:
    #    z = str(zipcode).strip()
    #    df_filtered = df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z]
    #if zipcode:
    #    z = str(zipcode).strip()
    #    df_filtered = df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z]
    # NEW: multiple zipcodes
    if zipcodes:
        z_set = {str(z).strip() for z in zipcodes}
        df_filtered = df_filtered[df_filtered["PostalCode"].astype(str).str.strip().isin(z_set)]
    # Normalize to string for safe comparison
    
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
    
    median_close = (
        df_filtered.groupby('ListAgentFullName', dropna=False)['ClosePrice']
        .median()
        .reset_index(name='Median Close Price')
    )
    filtered_agent_counts = filtered_agent_counts.merge(median_close, on="ListAgentFullName", how="left")
    
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

# -- Always show the view selector so users can click the second view even before ranking --
active_tab = st.radio(
    "View",
    ["🏆 Rankings", "📐 Multi-dimension view"],
    horizontal=True,
    index=0 if st.session_state.active_tab == "🏆 Rankings" else 1,
    key="active_tab",
)

# Short-circuit if we don't have results yet
if "selected_agents" not in st.session_state:
    st.info("Run Rankings to see results.")
    st.stop()

selected_agents = st.session_state.selected_agents
df_filtered = st.session_state.df_filtered

# --- Pagination state (used by Rankings tab) ---
records_per_page = 10
num_agents = len(selected_agents)
total_pages = max((num_agents - 1) // records_per_page + 1, 1)
if "page_num" not in st.session_state:
    st.session_state.page_num = 1

start_idx = (st.session_state.page_num - 1) * records_per_page
end_idx = start_idx + records_per_page
paged_agents = selected_agents.iloc[start_idx:end_idx]

# ---------- Tab 1: Rankings ----------
if st.session_state.active_tab == "🏆 Rankings":
    st.subheader("🏆 Top Ranked Agents")

    if num_agents == 0:
        st.warning("No agents matched your filters. Adjust filters and re-run.")
    else:
        # ----- Build a summary-style table for the FIRST display (same schema as Table 2) -----
        # NEW:
        sales_in_zip_first = (
            df_filtered.groupby('ListAgentFullName', dropna=False)['ClosePrice']
            .sum()
            .rename('Sales_In_Zip')
            .reset_index()
        )
        # OLD Block
        #if 'zipcode' in locals() and zipcode:
        #    z_str = str(zipcode).strip()
        #    sales_in_zip_first = (
        #        df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z_str]
        #        .groupby('ListAgentFullName', dropna=False)['ClosePrice']
        #        .sum()
        #        .rename('Sales_In_Zip')
        #        .reset_index()
        #    )
        #else:
        #    sales_in_zip_first = selected_agents[['ListAgentFullName']].assign(Sales_In_Zip=np.nan)

        tbl_first = selected_agents.merge(sales_in_zip_first, on='ListAgentFullName', how='left')
        tbl_first['%_Sales_in_Zip'] = (tbl_first['Sales_In_Zip'] / tbl_first['total_sales']).replace([np.inf, -np.inf], np.nan)
        tbl_first = tbl_first[tbl_first['%_Sales_in_Zip']>=min_sales_pct]


        # Ranks (same definitions as second table)
        tbl_first['Rank']                  = tbl_first['overall_score'].rank(ascending=False, method='dense').astype(int)
        tbl_first['Close Rate Rank']       = tbl_first['close_rate'].rank(ascending=False, method='dense').astype(int)
        tbl_first['Days on Market Rank']   = tbl_first['closed_daysonmarket_median'].rank(ascending=True,  method='dense').astype(int)
        tbl_first['Pricing Accuracy Rank'] = tbl_first['avg_pricing_accuracy'].rank(ascending=False, method='dense').astype(int)
        tbl_first['Total Sales Rank']      = tbl_first['total_sales'].rank(ascending=False, method='dense').astype(int)
        tbl_first['Closed Count Rank']     = tbl_first['closed_count'].rank(ascending=False, method='dense').astype(int)

        final_cols_first = [
            'Rank', 'ListAgentFullName', 'overall_score',
            'total_sales', 'Total Sales Rank',
            'closed_count', 'Closed Count Rank',
            '%_Sales_in_Zip',
            'close_rate', 'Close Rate Rank',
            'closed_daysonmarket_median', 'Days on Market Rank',
            'avg_pricing_accuracy', 'Pricing Accuracy Rank'
        ]

        tbl_first = tbl_first[final_cols_first].sort_values(['Rank', 'overall_score'], ascending=[True, False]).reset_index(drop=True)

        st.subheader("📋 Top Ranked Agents (Summary View)")
        st.dataframe(tbl_first, use_container_width=True)

        # ----- Summary table -----
        #NEW: 
        # NEW:
        sales_in_zip = (
            df_filtered.groupby('ListAgentFullName', dropna=False)['ClosePrice']
            .sum()
            .rename('Sales_In_Zip')
            .reset_index()
        )
        # OLD Block
        #if 'zipcode' in locals() and zipcode:
        #    z_str = str(zipcode).strip()
        #    sales_in_zip = (
        #        df_filtered[df_filtered['PostalCode'].astype(str).str.strip() == z_str]
        #        .groupby('ListAgentFullName', dropna=False)['ClosePrice']
        #        .sum()
        #        .rename('Sales_In_Zip')
        #        .reset_index()
        #    )
        #else:
        #    sales_in_zip = selected_agents[['ListAgentFullName']].assign(Sales_In_Zip=np.nan)

        tbl = selected_agents.merge(sales_in_zip, on='ListAgentFullName', how='left')
        tbl['%_Sales_in_Zip'] = (tbl['Sales_In_Zip'] / tbl['total_sales']).replace([np.inf, -np.inf], np.nan)
        tbl = tbl[tbl['%_Sales_in_Zip']>=min_sales_pct]

        tbl['Rank']                = tbl['overall_score'].rank(ascending=False, method='dense').astype(int)
        tbl['Close Rate Rank']     = tbl['close_rate'].rank(ascending=False, method='dense').astype(int)
        tbl['Days on Market Rank'] = tbl['closed_daysonmarket_median'].rank(ascending=True, method='dense').astype(int)
        tbl['Pricing Accuracy Rank']= tbl['avg_pricing_accuracy'].rank(ascending=False, method='dense').astype(int)
        tbl['Total Sales Rank']    = tbl['total_sales'].rank(ascending=False, method='dense').astype(int)
        tbl['Closed Count Rank']   = tbl['closed_count'].rank(ascending=False, method='dense').astype(int)

        final_cols = [
            'Rank', 'ListAgentFullName', 'overall_score',
            'total_sales', 'Total Sales Rank',
            'closed_count', 'Closed Count Rank',
            '%_Sales_in_Zip',
            'close_rate', 'Close Rate Rank',
            'closed_daysonmarket_median', 'Days on Market Rank',
            'avg_pricing_accuracy', 'Pricing Accuracy Rank'
        ]
        tbl = tbl[final_cols].sort_values(['Rank', 'overall_score'])

        if "tbl" not in st.session_state:
            st.session_state.tbl = tbl.copy()
        else:
            st.session_state.tbl = tbl.copy()
            
        st.subheader("📊 Summary by Agent (Filtered)")
        st.dataframe(tbl, use_container_width=True)

# ---------- Tab 2: Multi-dimension view ----------
elif st.session_state.active_tab == "📐 Multi-dimension view":
    st.subheader("📐 Agent rating by dimension")
           
    if num_agents == 0:
        st.warning("No agents matched your filters. Adjust filters and re-run.")
        st.stop()

    # Pull tbl from session state (it was set in the Rankings tab)
    tb = st.session_state.get("tbl")
    if tb is None or tb.empty or "ListAgentFullName" not in tb.columns:
        st.warning("No summary table available yet. Run Rankings first.")
        st.stop()

    # Build stable, overlapping options (Series -> str -> dropna -> set -> sorted list)
    left = set(selected_agents["ListAgentFullName"].dropna().astype(str))
    right = set(tb["ListAgentFullName"].dropna().astype(str))
    options_custom = sorted(left & right)

    if not options_custom:
        st.warning("No overlapping agents between the current selection and the summary table.")
        st.stop()

    # Preserve prior selection if still valid
    default_index = 0
    prev = st.session_state.get("agent_to_view")
    if prev in options_custom:
        default_index = options_custom.index(prev)

    # Select agent; keep focus on this view after change
    agent_to_view = st.selectbox(
        "Choose an agent",
        options=options_custom,  #selected_agents["ListAgentFullName"].tolist(),
        key="agent_to_view",
        on_change=focus_dims,
    )

    if not agent_to_view:
        st.info("Pick an agent to render charts.")
        st.stop()

    row = selected_agents.loc[selected_agents["ListAgentFullName"] == agent_to_view].iloc[0]

    # Build normalized dimensions directly; don't rely on row[...] for *_norm columns
    dims = {
        "Volume":             get_norm("volume_score"),
        "Close Rate":         get_norm("close_rate_score"),
        "Days on Market":     get_norm("closed_daysonmarket_median", invert=True),
        #"Pricing Accuracy":   row.get("pricing_accuracy_score", np.nan), #get_norm("pricing_accuracy_score"),
        "Total Sales":        get_norm("sales_score"),
    }


    #if "sales_score" in row.index:
    #    dims["Total Sales"] = row["sales_score"]

    dim_df = pd.DataFrame({"Dimension": list(dims.keys()), "Score": list(dims.values())}).dropna()

    if dim_df.empty:
        st.warning("No dimension scores available for this agent. Re-run Rankings to refresh.")
    else:
        if len(dim_df) >= 3:
            r = dim_df["Score"].tolist()
            theta = dim_df["Dimension"].tolist()
            fig_radar = go.Figure(data=go.Scatterpolar(r=r + [r[0]], theta=theta + [theta[0]], fill="toself"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

        fig_bar = px.bar(dim_df, x="Dimension", y="Score", range_y=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

        cols_raw = [
            "total_records", "closed_count", "close_rate",
            "closed_daysonmarket_mean", "closed_daysonmarket_median",
            "avg_pricing_accuracy", "total_sales"
        ]
        cols_raw = [c for c in cols_raw if c in selected_agents.columns]
        st.caption("Underlying metrics")
        st.dataframe(
            selected_agents[selected_agents["ListAgentFullName"] == agent_to_view][["ListAgentFullName"] + cols_raw],
            use_container_width=True
        )


