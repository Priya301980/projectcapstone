"""Build customer segments and a lapse-prediction model from cleaned data."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
PERIOD_END = pd.Timestamp("2025-09-30 23:59:59")


def build_segments(
    orders: pd.DataFrame, customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = orders.loc[orders["status"].eq("completed")]
    rfm = (
        completed.groupby("customer_id")
        .agg(
            recency=("order_ts", lambda dates: (SNAPSHOT - dates.max()).days),
            frequency=("order_id", "count"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )
    segment_data = customers.loc[
        customers["customer_id"].isin(completed["customer_id"]), ["customer_id"]
    ].merge(rfm, on="customer_id", how="left")
    segment_data[["recency", "frequency", "monetary"]] = segment_data[
        ["recency", "frequency", "monetary"]
    ].fillna({"recency": 183, "frequency": 0, "monetary": 0})

    scaler = StandardScaler()
    scaled = scaler.fit_transform(segment_data[["recency", "frequency", "monetary"]])
    model = KMeans(n_clusters=4, n_init=10, random_state=42)
    segment_data["cluster"] = model.fit_predict(scaled)

    profile = segment_data.groupby("cluster")[
        ["recency", "frequency", "monetary"]
    ].mean()
    clusters = list(profile.index)
    premium_cluster = profile["monetary"].idxmax()
    remaining = [cluster for cluster in clusters if cluster != premium_cluster]
    at_risk_cluster = profile.loc[remaining, "recency"].idxmax()
    remaining.remove(at_risk_cluster)
    frequent_cluster = profile.loc[remaining, "frequency"].idxmax()
    occasional_cluster = next(
        cluster for cluster in remaining if cluster != frequent_cluster
    )
    labels = {
        premium_cluster: "Premium",
        frequent_cluster: "Frequent",
        at_risk_cluster: "At-Risk",
        occasional_cluster: "Occasional",
    }

    segment_data["segment_name"] = segment_data["cluster"].map(labels)
    segment_data["is_premium"] = segment_data["cluster"].eq(premium_cluster)
    segment_data["segment_id"] = segment_data["cluster"]
    segment_output = segment_data[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ]

    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled)
    print("Cluster profiles:\n", profile)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    return segment_data, segment_output


def build_lapse_dataset(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    pre_orders = orders.loc[orders["order_ts"] < CUTOFF]
    cohort_ids = pre_orders.loc[
        pre_orders["status"].eq("completed"), "customer_id"
    ].unique()
    cohort = customers.loc[customers["customer_id"].isin(cohort_ids)].copy()

    pre_completed = pre_orders.loc[pre_orders["status"].eq("completed")]
    customer_features = (
        pre_completed.groupby("customer_id")
        .agg(
            recency=("order_ts", lambda dates: (CUTOFF - dates.max()).days),
            frequency=("order_id", "count"),
            monetary=("order_amount", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_delivery=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    pre_status = pre_orders.groupby("customer_id")["status"].agg(
        total=lambda status: status.isin(["completed", "cancelled"]).sum(),
        cancelled=lambda status: status.eq("cancelled").sum(),
    )
    cancel_rate = pre_status["cancelled"].div(pre_status["total"].replace(0, 1))
    cancel_rate = cancel_rate.rename("cancel_rate").reset_index()

    product_categories = products[["product_id", "category"]]
    pre_orders_with_category = pre_orders.merge(
        product_categories, on="product_id", how="left"
    )
    favorite_category = (
        pre_orders_with_category.groupby("customer_id")["category"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else pd.NA)
        .rename("fav_category")
        .reset_index()
    )

    cohort["tenure"] = (CUTOFF - cohort["signup_date"]).dt.days
    model_data = (
        cohort.merge(customer_features, on="customer_id", how="left")
        .merge(cancel_rate, on="customer_id", how="left")
        .merge(favorite_category, on="customer_id", how="left")
    )
    model_data = model_data.fillna(
        {
            "recency": (CUTOFF - pd.Timestamp("2025-04-01")).days,
            "frequency": 0,
            "monetary": 0,
            "cancel_rate": 0,
            "avg_delivery": 24,
            "avg_discount": 0,
            "fav_category": "Unknown",
        }
    )

    september_orders = orders.loc[
        orders["order_ts"].between(CUTOFF, PERIOD_END)
        & orders["status"].eq("completed")
    ]
    active_ids = set(september_orders["customer_id"])
    model_data["lapsed"] = (~model_data["customer_id"].isin(active_ids)).astype(int)
    return model_data


def run_part_b() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(
        CLEAN_DIR / "customers.csv", parse_dates=["signup_date"]
    )
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    segment_data, segment_output = build_segments(orders, customers)
    segment_output.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    model_data = build_lapse_dataset(orders, customers, products)
    features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancel_rate",
        "avg_delivery",
        "fav_category",
        "city",
        "age",
        "gender",
        "tenure",
    ]
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancel_rate",
        "avg_delivery",
        "age",
        "tenure",
    ]
    categorical_features = ["fav_category", "city", "gender"]
    x_train, x_test, y_train, y_test = train_test_split(
        model_data[features],
        model_data["lapsed"],
        test_size=0.2,
        stratify=model_data["lapsed"],
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    logistic_search = GridSearchCV(
        Pipeline(
            [
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
    )
    logistic_search.fit(x_train, y_train)
    print(
        "Logistic Regression best params:",
        logistic_search.best_params_,
        "F1:",
        logistic_search.best_score_,
    )

    forest_search = GridSearchCV(
        Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        ),
        {
            "clf__max_depth": [3, 5, 8],
            "clf__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
    )
    forest_search.fit(x_train, y_train)
    print(
        "Random Forest best params:",
        forest_search.best_params_,
        "F1:",
        forest_search.best_score_,
    )

    best_model = forest_search.best_estimator_
    joblib.dump(best_model, OUTPUT_DIR / "lapse_model.joblib")
    predictions = best_model.predict(x_test)
    print("Random Forest test F1:", f1_score(y_test, predictions))

    test_data = model_data.loc[x_test.index].copy()
    test_data["lapse_probability"] = best_model.predict_proba(x_test)[:, 1]
    premium_customers = segment_data.loc[
        segment_data["is_premium"], ["customer_id"]
    ]
    retention_list = (
        test_data.merge(premium_customers, on="customer_id", how="inner")
        .sort_values("lapse_probability", ascending=False)
        .head(10)
    )
    retention_list[["customer_id", "lapse_probability", "lapsed"]].rename(
        columns={"lapsed": "actually_lapsed"}
    ).to_csv(OUTPUT_DIR / "retention_list.csv", index=False)
    print(f"Part B completed. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_part_b()
