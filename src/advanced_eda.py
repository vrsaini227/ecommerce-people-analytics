import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/processed/integrated_ecommerce_data.csv"
)

print("\n" + "=" * 70)
print("ADVANCED EDA & STATISTICAL VALIDATION")
print("=" * 70)

print("\nIntegrated Dataset:", df.shape)


# =========================================================
# 1. CART LEVEL DATASET
# =========================================================

cart_df = (
    df.groupby("cart_id")
    .agg(
        user_id=("user_id", "first"),
        cart_total=("cart_total", "first"),
        cart_discounted_total=(
            "cart_discounted_total",
            "first"
        ),
        total_products=(
            "cart_total_products",
            "first"
        ),
        total_quantity=(
            "cart_total_quantity",
            "first"
        ),
        item_lines=(
            "product_id",
            "count"
        )
    )
    .reset_index()
)


print("\n" + "=" * 70)
print("CART LEVEL DATASET")
print("=" * 70)

print(cart_df.head())

print("\nCart Dataset Shape:")
print(cart_df.shape)


# =========================================================
# 2. VERIFY CART TOTAL DUPLICATION
# =========================================================

duplicate_check = (
    df.groupby("cart_id")["cart_total"]
    .nunique()
)

print("\nMaximum unique cart_total values per cart:")
print(duplicate_check.max())


# =========================================================
# 3. CUSTOMER LEVEL DATASET
# =========================================================

customer_df = (
    df.groupby("user_id")
    .agg(
        age=("age", "first"),
        gender=("gender", "first"),

        total_spending=(
            "item_value",
            "sum"
        ),

        discounted_spending=(
            "discounted_item_value",
            "sum"
        ),

        total_quantity=(
            "quantity",
            "sum"
        ),

        number_of_carts=(
            "cart_id",
            "nunique"
        ),

        unique_products=(
            "product_id",
            "nunique"
        ),

        unique_categories=(
            "category",
            "nunique"
        ),

        average_rating=(
            "rating",
            "mean"
        ),

        average_discount=(
            "discountPercentage",
            "mean"
        )
    )
    .reset_index()
)


print("\n" + "=" * 70)
print("CUSTOMER LEVEL DATASET")
print("=" * 70)

print(customer_df.head())

print("\nCustomer Dataset Shape:")
print(customer_df.shape)


# =========================================================
# 4. CUSTOMER SPENDING STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("CUSTOMER SPENDING STATISTICS")
print("=" * 70)

print(
    customer_df[
        [
            "total_spending",
            "discounted_spending",
            "total_quantity",
            "number_of_carts"
        ]
    ].describe()
)


# =========================================================
# 5. GENDER ANALYSIS - CUSTOMER LEVEL
# =========================================================

print("\n" + "=" * 70)
print("CUSTOMER-LEVEL GENDER ANALYSIS")
print("=" * 70)


gender_customer = (
    customer_df
    .groupby("gender")
    .agg(
        customers=("user_id", "count"),
        average_spending=(
            "total_spending",
            "mean"
        ),
        median_spending=(
            "total_spending",
            "median"
        ),
        average_quantity=(
            "total_quantity",
            "mean"
        ),
        average_categories=(
            "unique_categories",
            "mean"
        )
    )
)


print(gender_customer)


# =========================================================
# 6. AGE GROUP ANALYSIS
# =========================================================

customer_df["age_group"] = pd.cut(
    customer_df["age"],
    bins=[0, 25, 30, 35, 40, 100],
    labels=[
        "≤25",
        "26-30",
        "31-35",
        "36-40",
        "41+"
    ]
)


age_group_analysis = (
    customer_df
    .groupby(
        "age_group",
        observed=True
    )
    .agg(
        customers=("user_id", "count"),
        average_spending=(
            "total_spending",
            "mean"
        ),
        median_spending=(
            "total_spending",
            "median"
        ),
        average_quantity=(
            "total_quantity",
            "mean"
        )
    )
)


print("\n" + "=" * 70)
print("AGE GROUP ANALYSIS")
print("=" * 70)

print(age_group_analysis)


# =========================================================
# 7. CATEGORY ANALYSIS - QUANTITY VS VALUE
# =========================================================

category_analysis = (
    df.groupby("category")
    .agg(
        quantity_sold=(
            "quantity",
            "sum"
        ),

        calculated_value=(
            "item_value",
            "sum"
        ),

        average_price=(
            "price",
            "mean"
        ),

        unique_products=(
            "product_id",
            "nunique"
        )
    )
    .reset_index()
)


category_analysis["value_per_quantity"] = (
    category_analysis["calculated_value"]
    / category_analysis["quantity_sold"]
)


category_analysis = category_analysis.sort_values(
    "quantity_sold",
    ascending=False
)


print("\n" + "=" * 70)
print("CATEGORY: QUANTITY VS VALUE")
print("=" * 70)

print(category_analysis)


# =========================================================
# 8. OUTLIER DETECTION - ITEM VALUE
# =========================================================

Q1 = df["item_value"].quantile(0.25)

Q3 = df["item_value"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


item_outliers = df[
    (df["item_value"] < lower_bound)
    |
    (df["item_value"] > upper_bound)
]


print("\n" + "=" * 70)
print("ITEM VALUE OUTLIER ANALYSIS")
print("=" * 70)

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Bound:", lower_bound)

print("Upper Bound:", upper_bound)

print(
    "Number of Outliers:",
    len(item_outliers)
)

print(
    "Outlier Percentage:",
    round(
        len(item_outliers) / len(df) * 100,
        2
    ),
    "%"
)


# =========================================================
# 9. PRICE OUTLIERS
# =========================================================

price_Q1 = df["price"].quantile(0.25)

price_Q3 = df["price"].quantile(0.75)

price_IQR = price_Q3 - price_Q1

price_upper = (
    price_Q3 + 1.5 * price_IQR
)


price_outliers = df[
    df["price"] > price_upper
]


print("\n" + "=" * 70)
print("PRICE OUTLIER ANALYSIS")
print("=" * 70)

print("Price Q1:", price_Q1)

print("Price Q3:", price_Q3)

print("Price IQR:", price_IQR)

print("Upper Price Bound:", price_upper)

print(
    "High Price Records:",
    len(price_outliers)
)


# =========================================================
# 10. NON-TRIVIAL CORRELATION ANALYSIS
# =========================================================

correlation_features = [
    "age",
    "weight",
    "quantity",
    "discountPercentage",
    "rating",
    "stock"
]


customer_correlations = customer_df[
    [
        "age",
        "total_spending",
        "total_quantity",
        "unique_products",
        "unique_categories",
        "average_rating",
        "average_discount"
    ]
].corr(
    method="spearman"
)


print("\n" + "=" * 70)
print("CUSTOMER LEVEL SPEARMAN CORRELATION")
print("=" * 70)

print(
    customer_correlations.round(3)
)


# =========================================================
# 11. SAVE DATASETS
# =========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


cart_df.to_csv(
    "data/processed/cart_level.csv",
    index=False
)


customer_df.to_csv(
    "data/processed/customer_level.csv",
    index=False
)


category_analysis.to_csv(
    "data/processed/category_analysis.csv",
    index=False
)


# =========================================================
# 12. VISUALIZATION
# =========================================================

os.makedirs(
    "reports/figures",
    exist_ok=True
)


# Customer spending distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    customer_df["total_spending"],
    bins=30,
    kde=True
)

plt.title(
    "Customer-Level Spending Distribution"
)

plt.xlabel(
    "Total Calculated Spending"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/customer_spending_distribution.png",
    dpi=300
)

plt.show()


# Boxplot

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=customer_df["total_spending"]
)

plt.title(
    "Customer-Level Spending Outliers"
)

plt.xlabel(
    "Total Calculated Spending"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/customer_spending_boxplot.png",
    dpi=300
)

plt.show()


# Quantity vs value

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=customer_df,
    x="total_quantity",
    y="total_spending"
)

plt.title(
    "Customer Quantity vs Spending"
)

plt.xlabel(
    "Total Quantity"
)

plt.ylabel(
    "Total Calculated Spending"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/quantity_vs_spending.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 70)
print("ADVANCED EDA COMPLETED")
print("=" * 70)

print(
    "\nFiles saved:"
)

print(
    "data/processed/cart_level.csv"
)

print(
    "data/processed/customer_level.csv"
)

print(
    "data/processed/category_analysis.csv"
)