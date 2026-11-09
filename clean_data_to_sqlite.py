"""Clean the customer, order, and product CSV files and load them into SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DATABASE_PATH = ROOT / "data" / "clean" / "instamart.db"


def read_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RAW_DIR / name, dtype=str, keep_default_na=False)
    frame = frame.apply(lambda column: column.str.strip())
    return frame.replace("", pd.NA)


def clean_customers() -> pd.DataFrame:
    customers = read_csv("customers.csv")
    customers["customer_id"] = customers["customer_id"].str.upper()
    customers["city"] = customers["city"].str.title()
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    ages = pd.to_numeric(customers["age"], errors="coerce")
    customers["age"] = ages.where(ages.between(18, 100)).astype("Int64")
    genders = customers["gender"].str.lower()
    customers["gender"] = genders.map(
        {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    )

    customers = customers.dropna(subset=["customer_id", "city", "signup_date"])
    return customers.drop_duplicates(subset=["customer_id"], keep="first").reset_index(
        drop=True
    )


def clean_products() -> pd.DataFrame:
    products = read_csv("products.csv")
    products["product_id"] = products["product_id"].str.upper()
    products["category"] = products["category"].str.title()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")
    products = products.dropna(
        subset=["product_id", "product_name", "category", "price_inr"]
    )
    products = products[products["price_inr"] > 0]
    return products.drop_duplicates(subset=["product_id"], keep="first").reset_index(
        drop=True
    )


def clean_orders(products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    orders = read_csv("orders.csv")
    orders["order_id"] = orders["order_id"].str.upper()
    orders["customer_id"] = orders["customer_id"].str.upper()
    orders["product_id"] = orders["product_id"].str.upper()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["status"] = orders["status"].str.lower()
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    orders = orders.dropna(subset=["order_id", "customer_id", "product_id", "order_ts"])
    orders = orders.drop_duplicates()
    orders = orders.sort_values("order_ts", kind="stable").drop_duplicates(
        subset=["order_id"], keep="last"
    )
    orders = orders[
        orders["status"].isin(["completed", "cancelled", "pending"])
        & (orders["quantity"] > 0)
        & orders["quantity"].mod(1).eq(0)
        & orders["discount_pct"].between(0, 100)
        & orders["discount_pct"].mod(1).eq(0)
        & orders["customer_id"].isin(customers["customer_id"])
        & orders["product_id"].isin(products["product_id"])
    ].copy()

    prices = products.set_index("product_id")["price_inr"]
    expected_amount = (
        orders["product_id"].map(prices)
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
    invalid_amount = orders["order_amount"].isna() | (orders["order_amount"] < 0)
    orders.loc[invalid_amount, "order_amount"] = expected_amount[invalid_amount]

    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
    orders["quantity"] = orders["quantity"].astype("Int64")
    orders["discount_pct"] = orders["discount_pct"].astype("Int64")
    orders["delivery_minutes"] = orders["delivery_minutes"].astype("Int64")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return orders[
        [
            "order_id",
            "customer_id",
            "product_id",
            "quantity",
            "order_ts",
            "status",
            "discount_pct",
            "order_amount",
            "delivery_minutes",
        ]
    ].reset_index(drop=True)


def insert_frame(
    connection: sqlite3.Connection, table: str, frame: pd.DataFrame
) -> None:
    columns = list(frame.columns)
    names = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            tuple(
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
                for value in row
            )
        )
    connection.executemany(
        f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', rows
    )


def load_database(
    customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame
) -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DROP TABLE IF EXISTS orders")
        connection.execute("DROP TABLE IF EXISTS customers")
        connection.execute("DROP TABLE IF EXISTS products")
        connection.execute(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                signup_date TEXT NOT NULL,
                age INTEGER CHECK (age IS NULL OR age BETWEEN 18 AND 100),
                gender TEXT CHECK (gender IS NULL OR gender IN ('Male', 'Female'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                price_inr REAL NOT NULL CHECK (price_inr > 0)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                product_id TEXT NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                order_ts TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('completed', 'cancelled', 'pending')
                ),
                discount_pct INTEGER NOT NULL CHECK (discount_pct BETWEEN 0 AND 100),
                order_amount REAL NOT NULL CHECK (order_amount >= 0),
                delivery_minutes INTEGER CHECK (
                    delivery_minutes IS NULL OR delivery_minutes >= 0
                )
            )
            """
        )
        insert_frame(connection, "customers", customers)
        insert_frame(connection, "products", products)
        insert_frame(connection, "orders", orders)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(f"Foreign key violations: {violations}")


def main() -> None:
    customers = clean_customers()
    products = clean_products()
    orders = clean_orders(products, customers)
    load_database(customers, products, orders)
    print(f"SQLite database: {DATABASE_PATH}")
    print(
        "Loaded rows: "
        f"customers={len(customers)}, products={len(products)}, orders={len(orders)}"
    )


if __name__ == "__main__":
    main()
