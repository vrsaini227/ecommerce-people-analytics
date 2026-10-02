import pandas as pd
import numpy as np
import os


# ============================================================
# E-COMMERCE PATTERN ANALYSIS
# ============================================================

print("=" * 70)
print("E-COMMERCE PATTERN DISCOVERY")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = "data/processed/integrated_ecommerce_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nIntegrated Dataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. BASIC VALIDATION
# ------------------------------------------------------------

required_columns = [
    "user_id",
    "product_id",
    "quantity",
    "title",
    "category",
    "price",
    "discountPercentage",
    "rating",
    "item_value",
    "discount_amount",
    "discounted_item_value"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns validated successfully.")


# ------------------------------------------------------------
# 3. TOP PRODUCTS BY QUANTITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 PRODUCTS BY QUANTITY")
print("=" * 70)

top_products_quantity = (
    df.groupby(
        ["product_id", "title"],
        as_index=False
    )
    .agg(
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum")
    )
    .sort_values(
        "total_quantity",
        ascending=False
    )
    .head(10)
)

print(top_products_quantity.to_string(index=False))


# ------------------------------------------------------------
# 4. TOP PRODUCTS BY VALUE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 PRODUCTS BY CALCULATED VALUE")
print("=" * 70)

top_products_value = (
    df.groupby(
        ["product_id", "title"],
        as_index=False
    )
    .agg(
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum")
    )
    .sort_values(
        "total_value",
        ascending=False
    )
    .head(10)
)

print(top_products_value.to_string(index=False))


# ------------------------------------------------------------
# 5. CATEGORY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

category_analysis = (
    df.groupby("category", as_index=False)
    .agg(
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum"),
        average_price=("price", "mean"),
        average_discount=("discountPercentage", "mean"),
        average_rating=("rating", "mean"),
        unique_products=("product_id", "nunique"),
        unique_customers=("user_id", "nunique")
    )
)

category_analysis["value_per_quantity"] = (
    category_analysis["total_value"]
    / category_analysis["total_quantity"]
)

category_analysis = category_analysis.sort_values(
    "total_value",
    ascending=False
)

print(category_analysis.to_string(index=False))


# ------------------------------------------------------------
# 6. PRICE VS QUANTITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRICE VS QUANTITY ANALYSIS")
print("=" * 70)

price_quantity = (
    df.groupby(
        ["product_id", "title"],
        as_index=False
    )
    .agg(
        average_price=("price", "mean"),
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum"),
        average_rating=("rating", "mean"),
        average_discount=("discountPercentage", "mean")
    )
)

print(
    price_quantity[
        [
            "title",
            "average_price",
            "total_quantity",
            "total_value"
        ]
    ]
    .sort_values(
        "total_quantity",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 7. DISCOUNT ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISCOUNT ANALYSIS")
print("=" * 70)

discount_analysis = (
    df.groupby("category", as_index=False)
    .agg(
        average_discount=("discountPercentage", "mean"),
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum")
    )
    .sort_values(
        "average_discount",
        ascending=False
    )
)

print(discount_analysis.to_string(index=False))


# ------------------------------------------------------------
# 8. RATING VS VALUE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RATING VS VALUE")
print("=" * 70)

rating_analysis = (
    df.groupby("category", as_index=False)
    .agg(
        average_rating=("rating", "mean"),
        total_value=("discounted_item_value", "sum"),
        total_quantity=("quantity", "sum")
    )
    .sort_values(
        "total_value",
        ascending=False
    )
)

print(rating_analysis.to_string(index=False))


# ------------------------------------------------------------
# 9. HIGH-VALUE PRODUCT ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH-VALUE PRODUCTS")
print("=" * 70)

product_summary = (
    df.groupby(
        ["product_id", "title", "category"],
        as_index=False
    )
    .agg(
        total_quantity=("quantity", "sum"),
        total_value=("discounted_item_value", "sum"),
        average_price=("price", "mean"),
        average_discount=("discountPercentage", "mean"),
        average_rating=("rating", "mean")
    )
)

high_value_products = (
    product_summary
    .sort_values(
        "total_value",
        ascending=False
    )
    .head(20)
)

print(
    high_value_products.to_string(index=False)
)


# ------------------------------------------------------------
# 10. HIGH-QUANTITY PRODUCTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH-QUANTITY PRODUCTS")
print("=" * 70)

high_quantity_products = (
    product_summary
    .sort_values(
        "total_quantity",
        ascending=False
    )
    .head(20)
)

print(
    high_quantity_products.to_string(index=False)
)


# ------------------------------------------------------------
# 11. BUSINESS PATTERN CLASSIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRODUCT PATTERN CLASSIFICATION")
print("=" * 70)

quantity_median = product_summary["total_quantity"].median()
value_median = product_summary["total_value"].median()

def classify_product(row):

    high_quantity = row["total_quantity"] >= quantity_median
    high_value = row["total_value"] >= value_median

    if high_quantity and high_value:
        return "High Quantity - High Value"

    elif high_quantity and not high_value:
        return "High Quantity - Lower Value"

    elif not high_quantity and high_value:
        return "Low Quantity - High Value"

    else:
        return "Low Quantity - Lower Value"


product_summary["business_pattern"] = (
    product_summary.apply(
        classify_product,
        axis=1
    )
)

print(
    product_summary[
        [
            "title",
            "category",
            "total_quantity",
            "total_value",
            "business_pattern"
        ]
    ]
    .sort_values(
        "total_value",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------------------------

OUTPUT_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

top_products_quantity.to_csv(
    f"{OUTPUT_DIR}/top_products_quantity.csv",
    index=False
)

top_products_value.to_csv(
    f"{OUTPUT_DIR}/top_products_value.csv",
    index=False
)

category_analysis.to_csv(
    f"{OUTPUT_DIR}/pattern_category_analysis.csv",
    index=False
)

product_summary.to_csv(
    f"{OUTPUT_DIR}/product_pattern_analysis.csv",
    index=False
)

discount_analysis.to_csv(
    f"{OUTPUT_DIR}/discount_analysis.csv",
    index=False
)

rating_analysis.to_csv(
    f"{OUTPUT_DIR}/rating_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATTERN ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles saved inside:")
print("data/processed/")
print(" - top_products_quantity.csv")
print(" - top_products_value.csv")
print(" - pattern_category_analysis.csv")
print(" - product_pattern_analysis.csv")
print(" - discount_analysis.csv")
print(" - rating_analysis.csv")