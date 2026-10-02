import pandas as pd
import numpy as np
import ast
import os


# ==========================================
# Load Raw Data
# ==========================================

users_df = pd.read_csv(
    "data/raw/users.csv"
)

products_df = pd.read_csv(
    "data/raw/products.csv"
)

carts_df = pd.read_csv(
    "data/raw/carts.csv"
)


print("\n" + "=" * 60)
print("RAW DATA LOADED")
print("=" * 60)

print("Users:", users_df.shape)
print("Products:", products_df.shape)
print("Carts:", carts_df.shape)


# ==========================================
# USERS CLEANING
# ==========================================

print("\nCleaning Users dataset...")


# Remove duplicate users
users_df = users_df.drop_duplicates(
    subset=["id"]
)


# Convert numeric columns
users_df["age"] = pd.to_numeric(
    users_df["age"],
    errors="coerce"
)

users_df["height"] = pd.to_numeric(
    users_df["height"],
    errors="coerce"
)

users_df["weight"] = pd.to_numeric(
    users_df["weight"],
    errors="coerce"
)


# ==========================================
# PRODUCTS CLEANING
# ==========================================

print("Cleaning Products dataset...")


# Remove duplicate products
products_df = products_df.drop_duplicates(
    subset=["id"]
)


# Numeric conversion
numeric_product_columns = [
    "price",
    "discountPercentage",
    "rating",
    "stock",
    "weight",
    "minimumOrderQuantity"
]


for column in numeric_product_columns:

    products_df[column] = pd.to_numeric(
        products_df[column],
        errors="coerce"
    )


# ==========================================
# CART CLEANING
# ==========================================

print("Cleaning Carts dataset...")


# Remove duplicate carts
carts_df = carts_df.drop_duplicates(
    subset=["id"]
)


# Numeric conversion
numeric_cart_columns = [
    "total",
    "discountedTotal",
    "userId",
    "totalProducts",
    "totalQuantity"
]


for column in numeric_cart_columns:

    carts_df[column] = pd.to_numeric(
        carts_df[column],
        errors="coerce"
    )


# ==========================================
# Missing Value Report
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUE REPORT")
print("=" * 60)


print("\nUsers:")
print(users_df.isnull().sum())


print("\nProducts:")
print(products_df.isnull().sum())


print("\nCarts:")
print(carts_df.isnull().sum())


# ==========================================
# Remove invalid key records
# ==========================================

users_df = users_df.dropna(
    subset=["id"]
)

products_df = products_df.dropna(
    subset=["id"]
)

carts_df = carts_df.dropna(
    subset=["id", "userId"]
)


# ==========================================
# Basic Range Validation
# ==========================================

# Age should be positive
users_df = users_df[
    users_df["age"] > 0
]


# Height should be positive
users_df = users_df[
    users_df["height"] > 0
]


# Weight should be positive
users_df = users_df[
    users_df["weight"] > 0
]


# Product price should not be negative
products_df = products_df[
    products_df["price"] >= 0
]


# Product stock should not be negative
products_df = products_df[
    products_df["stock"] >= 0
]


# Cart totals should not be negative
carts_df = carts_df[
    carts_df["total"] >= 0
]


carts_df = carts_df[
    carts_df["discountedTotal"] >= 0
]


# ==========================================
# Save Clean Basic Data
# ==========================================

users_df.to_csv(
    "data/processed/users_clean.csv",
    index=False
)

products_df.to_csv(
    "data/processed/products_clean.csv",
    index=False
)

carts_df.to_csv(
    "data/processed/carts_clean.csv",
    index=False
)


# ==========================================
# Final Summary
# ==========================================

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print("\nClean Users:", users_df.shape)

print("Clean Products:", products_df.shape)

print("Clean Carts:", carts_df.shape)


print("\nClean datasets saved inside:")
print("data/processed/")