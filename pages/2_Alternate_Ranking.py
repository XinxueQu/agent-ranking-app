import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Alternative Rankings", layout="wide")

st.title("🧪 Alternative Ranking Page")

st.write("This is a completely separate Streamlit page.")

# OPTIONAL — Load the same dataset as the main page
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1UktHniggnt5YMQ_UA8IG9uo_L9PXbcIQ/export?format=csv"
    return pd.read_csv(url)

df = load_data()

st.write("Sample of data:")
st.dataframe(df.head())
