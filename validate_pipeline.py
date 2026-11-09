"""Validate cleaned CSV outputs and SQL query results against checkpoints."""

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DATABASE_PATH = CLEAN_DIR / "instamart.db"


def validate() -> None:
    raw_orders = pd.read_csv(RAW_DIR / "orders.csv")
    clean_orders = pd.read_csv(CLEAN_DIR / "orders.csv")
    clean_customers = pd.read_csv(CLEAN_DIR / "customers.csv")

    assert len(raw_orders) == 13084, f"Raw orders count mismatch: {len(raw_orders)}"
    assert raw_orders.duplicated().sum() == 119, "Exact duplicates count mismatch"
    assert len(clean_orders) == 12309, (
        f"Clean orders rows mismatch: {len(clean_orders)}"
    )
    assert (clean_orders["status"] == "completed").sum() == 11110, (
        "Completed orders count mismatch"
    )
    assert (clean_orders["status"] == "cancelled").sum() == 1199, (
        "Cancelled orders count mismatch"
    )
    assert clean_orders["order_amount"].isna().sum() == 0, (
        "Missing order amounts found"
    )

    completed_revenue = clean_orders.loc[
        clean_orders["status"] == "completed", "order_amount"
    ].sum()
    assert abs(completed_revenue - 2998668.80) < 0.01, (
        f"Completed revenue sum mismatch: {completed_revenue}"
    )

    city_counts = clean_customers["city"].value_counts().to_dict()
    assert city_counts.get("Pune") == 227, "Pune count mismatch"
    assert city_counts.get("Delhi") == 210, "Delhi count mismatch"
    assert city_counts.get("Bengaluru") == 197, "Bengaluru count mismatch"

    with sqlite3.connect(DATABASE_PATH) as connection:
        q1 = pd.read_sql_query(
            """
            SELECT customer_id
            FROM orders
            WHERE status = 'completed' AND order_amount > 500
            GROUP BY customer_id
            HAVING SUM(order_amount) > 8000
            """,
            connection,
        )
        assert len(q1) == 30, f"Q1 result count mismatch: {len(q1)}"

        q4 = pd.read_sql_query(
            """
            SELECT c.customer_id
            FROM customers c
            LEFT JOIN orders o
              ON c.customer_id = o.customer_id AND o.status = 'completed'
            WHERE o.order_id IS NULL
            """,
            connection,
        )
        assert len(q4) == 25, f"Q4 result count mismatch: {len(q4)}"

        q7 = pd.read_sql_query(
            """
            WITH daily AS (
                SELECT SUBSTR(order_ts, 1, 10) AS day,
                       SUM(order_amount) AS rev
                FROM orders
                WHERE status = 'completed'
                GROUP BY day
            )
            SELECT SUM(rev) OVER (ORDER BY day) AS cumulative_revenue
            FROM daily
            """,
            connection,
        )
        last_cumulative_revenue = q7.iloc[-1]["cumulative_revenue"]
        assert abs(last_cumulative_revenue - 2998668.80) < 0.01, (
            f"Q7 cumulative revenue mismatch: {last_cumulative_revenue}"
        )

        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        assert not foreign_key_violations, (
            f"Foreign key violations found: {foreign_key_violations}"
        )

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate()
