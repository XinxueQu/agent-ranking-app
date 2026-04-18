"""
Shared MongoDB loader for the agent ranking Streamlit app.
Reads MONGO_URI from st.secrets (Streamlit Cloud) or environment variable (local).
Merges transactions with agent_contact to restore agent name/phone/email/office.
"""
import os
import streamlit as st
import pandas as pd
from pymongo import MongoClient

_TX_FIELDS = [
    "ListingId", "ListAgentMlsId",
    "is_closed", "DaysOnMarket", "pricing_accuracy",
    "PostalCode", "City", "CountyOrParish", "SubdivisionName",
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "CloseDate", "ListingContractDate",
    "PropertyCondition", "ElementarySchool", "MiddleOrJuniorSchool", "HighSchool",
    "LivingArea", "LotSizeSquareFeet", "LotSizeAcres",
    "YearBuilt", "PoolPrivateYN", "WaterfrontYN", "AssociationYN",
    "View", "StandardStatus",
    "UnparsedAddress",
]

_AGENT_FIELDS = [
    "AgentMlsId", "FullName", "DirectPhone", "Email",
    "Designation", "AOR", "OfficeName", "OfficeMlsId", "OfficePhone",
]


def _get_uri() -> str:
    try:
        return st.secrets["MONGO_URI"]
    except Exception:
        pass
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError(
            "MONGO_URI not set. Add it to .streamlit/secrets.toml or environment variables."
        )
    return uri


@st.cache_data(ttl=3600, show_spinner="Loading data from database...")
def load_transactions() -> pd.DataFrame:
    uri = _get_uri()
    client = MongoClient(uri)
    db = client["getrealistics"]

    # ── Fetch transactions ────────────────────────────────────────────
    tx_projection = {f: 1 for f in _TX_FIELDS}
    tx_projection["_id"] = 0
    records = list(
        db["transactions"].find(
            {"StandardStatus": {"$in": ["Closed", "Canceled", "Expired", "Withdrawn", "Delete"]}},
            tx_projection,
        )
    )

    # ── Fetch agents ──────────────────────────────────────────────────
    ag_projection = {f: 1 for f in _AGENT_FIELDS}
    ag_projection["_id"] = 0
    agents = list(db["agent_contact"].find({}, ag_projection))

    client.close()

    df = pd.DataFrame(records)
    agents_df = pd.DataFrame(agents).rename(columns={
        "AgentMlsId":  "ListAgentMlsId",
        "FullName":    "ListAgentFullName",
        "DirectPhone": "ListAgentDirectPhone",
        "Email":       "ListAgentEmail",
        "Designation": "ListAgentDesignation",
        "AOR":         "ListAgentAOR",
        "OfficeName":  "ListOfficeName",
        "OfficeMlsId": "ListOfficeMlsId",
        "OfficePhone": "ListOfficePhone",
    })

    # ── Merge agent details into transactions ─────────────────────────
    if "ListAgentMlsId" in df.columns and not agents_df.empty:
        df = df.merge(agents_df, on="ListAgentMlsId", how="left")

    # ── Column renames for backwards compatibility ────────────────────
    df = df.rename(columns={
        "UnparsedAddress": "Address",
        "LotSizeAcres":    "Acres",
        "PoolPrivateYN":   "PoolYN",
    })

    # ── Type coercions ────────────────────────────────────────────────
    for col in ["ClosePrice", "ListPrice", "OriginalListPrice", "DaysOnMarket",
                "LivingArea", "LotSizeSquareFeet", "Acres", "YearBuilt", "pricing_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CloseDate", "ListingContractDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "is_closed" in df.columns:
        df["is_closed"] = df["is_closed"].astype(bool)

    return df
