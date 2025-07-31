import pandas as pd 
import numpy as np 
import streamlit as st

st.set_page_config(page_title="Agent Rankings", layout="wide")
st.title("Top Real Estate Agent Rankings")

def pricing_accuracy_score(ratio):
    if pd.isna(ratio):
        return None  # or 0 if you want to penalize missing data
    thresholds = [
        (0.98, 1.02, 100),
        (0.96, 1.04, 95),
        (0.94, 1.06, 90),
        (0.92, 1.08, 85),
        (0.90, 1.10, 80),
        (0.88, 1.12, 75),
        (0.86, 1.14, 70),
        (0.84, 1.16, 65),
        (0.82, 1.18, 60)
    ]
    for low, high, score in thresholds:
        if low <= ratio <= high:
            return score
    return 0

def percentile_score(series):
    """
    Takes a pandas Series and returns a Series of continuous percentile scores (0–100).
    """
    #return (series.rank(pct=True) * 100).round(2) # this treats ties equally. So [1, 2, 3, 3, 3], 3 gets 73.33 percentile
    s = series.dropna()
    scaled = (s - s.min()) / (s.max() - s.min()) * 100
    return scaled.round(2)


def score_days_on_market(series):
    s = series.dropna()
    score = (s.max() - s) / (s.max() - s.min()) * 100
    return score.round(2)



transaction_data_path = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=xlsx"

# Find all CSV files in the folder
data_2022_2025May = pd.read_excel(transaction_data_path, engine='openpyxl')

# Group and summarize
agent_summary = (
    data_2022_2025May
    .groupby('ListAgentFullName')
    .agg(
        total_records=('ListAgentFullName', 'count'),
        closed_count=('is_closed', 'sum'),
        closed_daysonmarket_mean=('DaysOnMarket', lambda x: x[data_2022_2025May.loc[x.index, 'is_closed']].mean()),
        closed_daysonmarket_median=('DaysOnMarket', lambda x: x[data_2022_2025May.loc[x.index, 'is_closed']].median()),
        avg_pricing_accuracy=('pricing_accuracy', 'mean')
    )
    .reset_index()
)

agent_summary['close_rate'] = agent_summary['closed_count']/agent_summary['total_records']

agent_ranking = agent_summary.copy()

agent_ranking['pricing_accuracy_score'] = agent_ranking['avg_pricing_accuracy'].apply(pricing_accuracy_score)

agent_ranking['volume_score'] = percentile_score(agent_ranking['total_records']) 
agent_ranking['close_rate_score'] = percentile_score(agent_ranking['close_rate'])

agent_ranking['avg_days_on_mkt_score'] = score_days_on_market(agent_ranking['closed_daysonmarket_mean']) 
agent_ranking['median_days_on_mkt_score'] = score_days_on_market(agent_ranking['closed_daysonmarket_median'])

agent_ranking['overall_score']= (0.4 * agent_ranking['volume_score']+
                                0.3*agent_ranking['close_rate_score']+ 
                                0.2 *agent_ranking['avg_days_on_mkt_score']+ 
                                0.1*agent_ranking['pricing_accuracy_score']
                                )



# --- UI Inputs ---
st.title("Top Real Estate Agent Rankings")

selected_zipcode = st.text_input("Zipcode")
selected_min_price = st.number_input("Minimum Price", value=0)
selected_max_price = st.number_input("Maximum Price", value=1_000_000)
selected_elementary = st.text_input("Elementary School")
selected_subdivision = st.text_input("Subdivision")
selected_min_volume = st.number_input("Minimum Total Transactions", value=0)

#selected_zipcode =  78610 # PostalCode
#selected_min_price = 300000  #ClosePrice
#selected_max_price = 1000000
#selected_property_type = "" # currently, all data is residential
#selected_elementary = 'Lake Travis' #ElementarySchool
#selected_subdivision = '' # SubdivisionName

#selected_min_volume = 2

# --- Filtering ---
# Start with the full dataset
df_filtered = data_2022_2025May.copy()

# Apply filters conditionally
if selected_zipcode not in [None, '', np.nan]:
    df_filtered = df_filtered[df_filtered['PostalCode'] == selected_zipcode]

if selected_min_price not in [None, '', np.nan]:
    df_filtered = df_filtered[df_filtered['ClosePrice'] >= selected_min_price]

if selected_max_price not in [None, '', np.nan]:
    df_filtered = df_filtered[df_filtered['ClosePrice'] <= selected_max_price]

if (selected_elementary not in [None, '', np.nan] and 
    selected_elementary in df_filtered['ElementarySchool'].dropna().unique() ):
    df_filtered = df_filtered[df_filtered['ElementarySchool'] == selected_elementary]

if (selected_subdivision not in [None, '', np.nan] and
   selected_subdivision in df_filtered['SubdivisionName'].dropna().unique()):
    df_filtered = df_filtered[df_filtered['SubdivisionName'] == selected_subdivision]


# use filtered agent name to get their rankings
if pd.isna(selected_min_volume) or not isinstance(selected_min_volume, (int, float)):
    selected_min_volume = 0

selected_agent_ranking = agent_ranking.copy()

selected_agent_ranking = selected_agent_ranking[(selected_agent_ranking['ListAgentFullName'].isin( df_filtered['ListAgentFullName'].unique())) &
                                                  (selected_agent_ranking['total_records'] >=  selected_min_volume) ]

#print(selected_agent_ranking.sort_values(by='overall_score', ascending=False).head(10))
# --- Display results ---
st.subheader("Top Ranked Agents")
st.dataframe(selected_agents.head(10), use_container_width=True)
