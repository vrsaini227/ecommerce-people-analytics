import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =========================================================
# 1. Load Integrated Dataset
# =========================================================

df = pd.read_csv(
    "data/processed/integrated_ecommerce_data.csv"
)


print("\n" + "=" * 70)
print("E-COMMERCE EXPLORATORY DATA ANALYSIS")
print("=" * 70)


print("\nDataset Shape:")
print(df.shape)


# =========================================================
# 2. Basic Information
# =========================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# =========================================================
# 3. Descriptive Statistics
# =========================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    df.describe()
)


# =========================================================
# 4. Unique Values
# =========================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)


print(
    "Unique Users:",
    df["user_id"].nunique()
)

print(
    "Unique Products:",
    df["product_id"].nunique()
)

print(
    "Unique Categories:",
    df["category"].nunique()
)

print(
    "Unique Carts:",
    df["cart_id"].nunique()
)

print(
    "Unique Brands:",
    df["brand"].nunique()
)


# =========================================================
# 5. Category Analysis
# =========================================================

print("\n" + "=" * 70)
print("CATEGORY ANALYSIS")
print("=" * 70)


category_summary = (
    df.groupby("category")
    .agg(
        total_quantity=("quantity", "sum"),
        total_item_value=("item_value", "sum"),
        average_price=("price", "mean"),
        average_rating=("rating", "mean"),
        number_of_products=("product_id", "nunique")
    )
    .sort_values(
        "total_item_value",
        ascending=False
    )
)


print(category_summary)


# =========================================================
# 6. Gender Analysis
# =========================================================

print("\n" + "=" * 70)
print("GENDER ANALYSIS")
print("=" * 70)


gender_summary = (
    df.groupby("gender")
    .agg(
        total_quantity=("quantity", "sum"),
        total_item_value=("item_value", "sum"),
        average_item_value=("item_value", "mean"),
        average_cart_value=("cart_total", "mean")
    )
)


print(gender_summary)


# =========================================================
# 7. Age Analysis
# =========================================================

print("\n" + "=" * 70)
print("AGE ANALYSIS")
print("=" * 70)


age_summary = (
    df.groupby("age")
    .agg(
        total_quantity=("quantity", "sum"),
        total_item_value=("item_value", "sum"),
        average_item_value=("item_value", "mean")
    )
    .sort_values(
        "total_item_value",
        ascending=False
    )
)


print(age_summary.head(15))


# =========================================================
# 8. Customer-level Summary
# =========================================================

print("\n" + "=" * 70)
print("CUSTOMER SUMMARY")
print("=" * 70)


customer_summary = (
    df.groupby("user_id")
    .agg(
        age=("age", "first"),
        gender=("gender", "first"),
        total_spending=("item_value", "sum"),
        discounted_spending=(
            "discounted_item_value",
            "sum"
        ),
        total_quantity=("quantity", "sum"),
        number_of_carts=("cart_id", "nunique"),
        unique_products=(
            "product_id",
            "nunique"
        ),
        unique_categories=(
            "category",
            "nunique"
        ),
        average_rating=("rating", "mean")
    )
    .reset_index()
)


print(
    customer_summary.head(10)
)


print(
    "\nCustomer Summary Shape:",
    customer_summary.shape
)


# =========================================================
# 9. Save Customer Summary
# =========================================================

customer_summary.to_csv(
    "data/processed/customer_summary.csv",
    index=False
)


# =========================================================
# 10. Create Visualization Directory
# =========================================================

os.makedirs(
    "reports/figures",
    exist_ok=True
)


# =========================================================
# 11. Category Bar Chart
# =========================================================

plt.figure(
    figsize=(12, 6)
)

category_summary[
    "total_item_value"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Total Item Value by Product Category"
)

plt.xlabel(
    "Total Item Value"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/category_value.png",
    dpi=300
)

plt.show()


# =========================================================
# 12. Gender Spending
# =========================================================

plt.figure(
    figsize=(8, 5)
)

gender_summary[
    "total_item_value"
].plot(
    kind="bar"
)

plt.title(
    "Total Item Value by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Total Item Value"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "reports/figures/gender_spending.png",
    dpi=300
)

plt.show()


# =========================================================
# 13. Age Distribution
# =========================================================

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    df["age"],
    bins=15,
    kde=True
)

plt.title(
    "Customer Age Distribution"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/age_distribution.png",
    dpi=300
)

plt.show()


# =========================================================
# 14. Price Distribution
# =========================================================

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    df["price"],
    bins=30,
    kde=True
)

plt.title(
    "Product Price Distribution"
)

plt.xlabel(
    "Price"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/price_distribution.png",
    dpi=300
)

plt.show()


# =========================================================
# 15. Quantity Distribution
# =========================================================

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    df["quantity"],
    discrete=True
)

plt.title(
    "Product Quantity Distribution"
)

plt.xlabel(
    "Quantity"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/quantity_distribution.png",
    dpi=300
)

plt.show()


# =========================================================
# 16. Correlation Analysis
# =========================================================

numeric_columns = [
    "age",
    "height",
    "weight",
    "quantity",
    "cart_total",
    "cart_discounted_total",
    "cart_total_products",
    "cart_total_quantity",
    "price",
    "discountPercentage",
    "rating",
    "stock",
    "item_value",
    "discount_amount",
    "discounted_item_value"
]


correlation_matrix = df[
    numeric_columns
].corr()


print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(
    correlation_matrix.round(3)
)


# =========================================================
# 17. Correlation Heatmap
# =========================================================

plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix of Numerical Variables"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300
)

plt.show()


# =========================================================
# 18. Top Customers
# =========================================================

top_customers = (
    customer_summary
    .sort_values(
        "total_spending",
        ascending=False
    )
    .head(10)
)


print("\n" + "=" * 70)
print("TOP 10 CUSTOMERS BY CALCULATED ITEM VALUE")
print("=" * 70)

print(
    top_customers
)


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)