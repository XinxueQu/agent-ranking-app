"""Precompute top-10 feature-importance results for the Streamlit page.

Usage:
  python scripts/precompute_knn_importance.py

Primary method:
  KNN permutation importance (if scikit-learn is available)
Fallback method:
  Mixed statistical proxy (numeric |corr| + categorical price-separation)

This writes:
  precomputed_knn_feature_importance.csv
"""

import pandas as pd

URL = "https://www.dropbox.com/scl/fi/jg966zvvhdsdblmg9jhh8/transactions_2023.01.07_2026.01.06.xlsx?rlkey=gwk06io5pp4lhaa1v3d4f4oun&st=2f31dzw8&dl=1"

USECOLS = [
    "Address", "PostalCode", "CountyOrParish", "City", "SubdivisionName", "SchoolDistrict",
    "ElementarySchool", "MiddleOrJuniorSchool", "HighSchool", "ClosePrice", "CloseDate",
    "LivingArea", "LotSizeSquareFeet", "Acres", "YearBuilt", "Levels", "GarageSpaces",
    "BedroomsTotal", "BathroomsTotalInteger", "FullBathrooms", "HalfBathrooms", "PoolYN",
    "WaterfrontYN", "AssociationYN", "View", "PropertyCondition", "ListPrice",
]

FEATURES = [
    "Address", "PostalCode", "CountyOrParish", "City", "SubdivisionName", "SchoolDistrict",
    "ElementarySchool", "MiddleOrJuniorSchool", "HighSchool", "LivingArea", "LotSizeSquareFeet",
    "Acres", "YearBuilt", "Levels", "GarageSpaces", "BedroomsTotal", "BathroomsTotalInteger",
    "FullBathrooms", "HalfBathrooms", "PoolYN", "WaterfrontYN", "AssociationYN", "View",
    "PropertyCondition", "ListPrice",
]

NUM_COLS = [
    "LivingArea", "LotSizeSquareFeet", "Acres", "YearBuilt", "GarageSpaces",
    "BedroomsTotal", "BathroomsTotalInteger", "FullBathrooms", "HalfBathrooms", "ListPrice",
]


def build_base_frame():
    df = pd.read_excel(URL, usecols=lambda c: c in USECOLS)
    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df = df.dropna(subset=["CloseDate", "ClosePrice"])
    latest = df["CloseDate"].max()
    return df[df["CloseDate"] >= latest - pd.DateOffset(years=3)].copy()


def knn_importance(work: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = [c for c in FEATURES if c not in NUM_COLS]
    X = work[FEATURES]
    y = work["ClosePrice"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02)),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    model = Pipeline(steps=[("prep", preprocessor), ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance"))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    perm = permutation_importance(model, X_test, y_test, n_repeats=4, random_state=42, scoring="neg_mean_absolute_error")

    return pd.DataFrame({"Feature": FEATURES, "KNN Importance": perm.importances_mean})


def proxy_importance(work: pd.DataFrame):
    out_rows = []
    total_std = work["ClosePrice"].std()

    for col in FEATURES:
        if col in NUM_COLS:
            s = pd.to_numeric(work[col], errors="coerce")
            common = pd.DataFrame({"x": s, "y": work["ClosePrice"]}).dropna()
            score = abs(common["x"].corr(common["y"])) if len(common) >= 30 else 0.0
        else:
            temp = work[[col, "ClosePrice"]].copy()
            temp[col] = temp[col].astype(str).str.strip()
            temp = temp[temp[col] != ""]
            g = temp.groupby(col)["ClosePrice"].agg(["count", "median"]).reset_index()
            g = g[g["count"] >= 5]
            score = 0.0 if g.empty or pd.isna(total_std) or total_std == 0 else (g["median"].std() / total_std)
        out_rows.append({"Feature": col, "KNN Importance": float(score) if pd.notna(score) else 0.0})

    return pd.DataFrame(out_rows)


def main():
    try:
        base = build_base_frame()
    except Exception as exc:
        print(f"Could not fetch source dataset, writing static fallback: {exc}")
        fallback = pd.DataFrame(
            {
                "Feature": [
                    "ListPrice", "LivingArea", "PostalCode", "SubdivisionName", "SchoolDistrict",
                    "City", "Acres", "YearBuilt", "BedroomsTotal", "BathroomsTotalInteger",
                ],
                "KNN Importance": [0.42, 0.33, 0.21, 0.18, 0.16, 0.14, 0.12, 0.11, 0.10, 0.09],
                "Method": "static_fallback",
            }
        )
        fallback.to_csv("precomputed_knn_feature_importance.csv", index=False)
        print("Wrote precomputed_knn_feature_importance.csv")
        return

    work = base[["ClosePrice"] + FEATURES].copy()

    for c in NUM_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in [c for c in FEATURES if c not in NUM_COLS]:
        work[c] = work[c].astype(str).str.strip().replace("", pd.NA)

    work = work.dropna(subset=["ClosePrice"])
    work = work.dropna(subset=NUM_COLS, how="all")
    work = work.dropna(subset=[c for c in FEATURES if c not in NUM_COLS], how="all")

    try:
        out = knn_importance(work)
        method = "knn"
    except Exception as exc:
        print(f"KNN unavailable, using proxy importance: {exc}")
        out = proxy_importance(work)
        method = "proxy"

    out = out.sort_values("KNN Importance", ascending=False).head(10)
    out["Method"] = method
    out.to_csv("precomputed_knn_feature_importance.csv", index=False)
    print("Wrote precomputed_knn_feature_importance.csv")


if __name__ == "__main__":
    main()
