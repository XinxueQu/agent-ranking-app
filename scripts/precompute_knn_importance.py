"""Precompute top-10 KNN feature-importance results for the Streamlit page.

Usage:
  python scripts/precompute_knn_importance.py

This writes:
  data/precomputed_knn_feature_importance.csv
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

URL = "https://www.dropbox.com/scl/fi/jg966zvvhdsdblmg9jhh8/transactions_2023.01.07_2026.01.06.xlsx?rlkey=gwk06io5pp4lhaa1v3d4f4oun&st=2f31dzw8&dl=1"

USECOLS = [
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
]

FEATURES = [
    "Address",
    "PostalCode",
    "CountyOrParish",
    "City",
    "SubdivisionName",
    "SchoolDistrict",
    "ElementarySchool",
    "MiddleOrJuniorSchool",
    "HighSchool",
    "LivingArea",
    "LotSizeSquareFeet",
    "Acres",
    "YearBuilt",
    "Levels",
    "GarageSpaces",
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
]

NUM_COLS = [
    "LivingArea",
    "LotSizeSquareFeet",
    "Acres",
    "YearBuilt",
    "GarageSpaces",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "FullBathrooms",
    "HalfBathrooms",
    "ListPrice",
]


def main():
    df = pd.read_excel(URL, usecols=lambda c: c in USECOLS)
    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df = df.dropna(subset=["CloseDate", "ClosePrice"])

    latest = df["CloseDate"].max()
    df = df[df["CloseDate"] >= latest - pd.DateOffset(years=3)].copy()

    work = df[["ClosePrice"] + FEATURES].copy()
    cat_cols = [c for c in FEATURES if c not in NUM_COLS]

    for c in NUM_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in cat_cols:
        work[c] = work[c].astype(str).str.strip().replace("", pd.NA)

    work = work.dropna(subset=["ClosePrice"])
    work = work.dropna(subset=NUM_COLS, how="all")
    work = work.dropna(subset=cat_cols, how="all")

    X = work[FEATURES]
    y = work["ClosePrice"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[("prep", preprocessor), ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance"))]
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

    out = (
        pd.DataFrame({"Feature": FEATURES, "KNN Importance": perm.importances_mean})
        .sort_values("KNN Importance", ascending=False)
        .head(10)
    )
    out.to_csv("data/precomputed_knn_feature_importance.csv", index=False)
    print("Wrote data/precomputed_knn_feature_importance.csv")


if __name__ == "__main__":
    main()
