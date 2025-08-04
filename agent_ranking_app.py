import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Agent Rankings", layout="wide")

# --- Style Enhancements ---
#st.title("Top Real Estate Agent Rankings")
st.markdown("<h1 style='text-align: center; color: darkblue;'>🏡 Top Real Estate Agent Rankings</h1>", unsafe_allow_html=True)


# --- Helper functions ---
def pricing_accuracy_score(x):
    return (1 - abs(x - 1)) * 100

def percentile_score(series):
    return series.rank(pct=True) * 100

def score_days_on_market(series):
    return 100 - series.rank(pct=True) * 100

# --- Load data ---
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=xlsx"
    df = pd.read_excel(url, engine='openpyxl')
    return df

data = load_data()

# --- Compute agent rankings ---
agent_summary = (
    data.groupby('ListAgentFullName')
    .agg(
        total_records=('ListAgentFullName', 'count'),
        closed_count=('is_closed', 'sum'),
        closed_daysonmarket_mean=('DaysOnMarket', lambda x: x[data.loc[x.index, 'is_closed']].mean()),
        closed_daysonmarket_median=('DaysOnMarket', lambda x: x[data.loc[x.index, 'is_closed']].median()),
        avg_pricing_accuracy=('pricing_accuracy', 'mean')
    )
    .reset_index()
)
agent_summary['close_rate'] = agent_summary['closed_count'] / agent_summary['total_records']
agent_summary['pricing_accuracy_score'] = agent_summary['avg_pricing_accuracy'].apply(pricing_accuracy_score)
agent_summary['volume_score'] = percentile_score(agent_summary['total_records']) 
agent_summary['close_rate_score'] = percentile_score(agent_summary['close_rate'])
agent_summary['avg_days_on_mkt_score'] = score_days_on_market(agent_summary['closed_daysonmarket_mean']) 
agent_summary['median_days_on_mkt_score'] = score_days_on_market(agent_summary['closed_daysonmarket_median'])
#agent_summary['overall_score'] = (
#    0.4 * agent_summary['volume_score'] +
#    0.3 * agent_summary['close_rate_score'] +
#    0.2 * agent_summary['avg_days_on_mkt_score'] + #['median_days_on_mkt_score']
#    0.1 * agent_summary['pricing_accuracy_score']
#)

# --- UI Inputs ---
#st.title("Top Real Estate Agent Rankings")

#zipcode = st.text_input("Zipcode")
#min_price = st.number_input("Minimum Price", value=0)
#max_price = st.number_input("Maximum Price", value=1_000_000)
#elementary = st.text_input("Elementary School")
#subdivision = st.text_input("Subdivision")
#min_volume = st.number_input("Minimum Total Transactions", value=0)

#weight_volumne = st.number_input("Weight on Transaction Volume", value=0.4)
#weight_close   = st.number_input("Weight on Close Rate", value=0.3)
#weight_days    = st.number_input("Weight on Median Days on Market", value=0.2)
#weight_price   = st.number_input("Weight on Pricing Accuracy", value=0.1)
#if weight_volumne + weight_close + weight_days + weight_price !=1:
#    weight_volumne = 0.4
#    weight_close = 0.3
#    weight_days = 0.4
#    weight_price = 0.1



# --- Inputs: Left for Filters, Right for Weights ---
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
    weight_vol = right_col.number_input("Transaction Volume", value=0.4)
    weight_close = right_col.number_input("Close Rate", value=0.3)
    weight_days = right_col.number_input("Days on Market", value=0.2)
    weight_price = right_col.number_input("Pricing Accuracy", value=0.1)
    
    total_weight = weight_vol + weight_close + weight_days + weight_price
    if total_weight != 1:
        st.warning("Weights do not sum to 1. Normalizing automatically.")
        weight_vol /= total_weight
        weight_close /= total_weight
        weight_days /= total_weight
        weight_price /= total_weight

# --- Apply Weights ---
agent_summary['overall_score'] = (
    weight_volumne * agent_summary['volume_score'] +
    weight_close * agent_summary['close_rate_score'] +
    weight_days * agent_summary['median_days_on_mkt_score'] + #avg_days_on_mkt_score
    weight_price * agent_summary['pricing_accuracy_score']
)


# --- Filtering ---
df_filtered = data.copy()
if zipcode and pd.notna(zipcode) and zipcode in df_filtered['PostalCode'].dropna().unique():
    df_filtered = df_filtered[df_filtered['PostalCode'] == zipcode]
df_filtered = df_filtered[df_filtered['ClosePrice'] >= min_price]
df_filtered = df_filtered[df_filtered['ClosePrice'] <= max_price]
if elementary and pd.notna(elementary) and elementary in df_filtered['ElementarySchool'].dropna().unique():
    df_filtered = df_filtered[df_filtered['ElementarySchool'] == elementary]
if subdivision and  pd.notna(subdivision) and subdivision in df_filtered['SubdivisionName'].dropna().unique():
    df_filtered = df_filtered[df_filtered['SubdivisionName'] == subdivision]

selected_agents = agent_summary[
    (agent_summary['ListAgentFullName'].isin(df_filtered['ListAgentFullName'].unique())) &
    (agent_summary['total_records'] >= min_volume)
].sort_values(by='overall_score', ascending=False)

# --- Display Results ---
#st.subheader("🏆 Top Ranked Agents")
#st.dataframe(selected_agents.head(10), use_container_width=True)


# --- Pagination ---
records_per_page = 10
num_agents = len(selected_agents)
total_pages = (num_agents - 1) // records_per_page + 1

if total_pages > 1:
    page_num = st.number_input(
        label="Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        format="%d"
    )
else:
    page_num = 1

start_idx = (page_num - 1) * records_per_page
end_idx = start_idx + records_per_page
paged_agents = selected_agents.iloc[start_idx:end_idx]

# --- Display Results ---
st.subheader("🏆 Top Ranked Agents")
st.caption(f"Showing page {page_num} of {total_pages} ({num_agents} agents found)")
st.dataframe(paged_agents, use_container_width=True)
