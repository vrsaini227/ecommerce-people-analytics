import pandas as pd
import json


# =========================
# Load raw CSV files
# =========================

users_df = pd.read_csv("data/raw/users.csv")
products_df = pd.read_csv("data/raw/products.csv")
carts_df = pd.read_csv("data/raw/carts.csv")


# =========================
# USERS INSPECTION
# =========================

print("\n" + "=" * 60)
print("USERS DATASET")
print("=" * 60)

print("\nShape:")
print(users_df.shape)

print("\nColumns:")
print(users_df.columns.tolist())

print("\nData Types:")
print(users_df.dtypes)

print("\nFirst 5 records:")
print(users_df.head())

print("\nMissing Values:")
print(users_df.isnull().sum())


# =========================
# PRODUCTS INSPECTION
# =========================

print("\n" + "=" * 60)
print("PRODUCTS DATASET")
print("=" * 60)

print("\nShape:")
print(products_df.shape)

print("\nColumns:")
print(products_df.columns.tolist())

print("\nData Types:")
print(products_df.dtypes)

print("\nFirst 5 records:")
print(products_df.head())

print("\nMissing Values:")
print(products_df.isnull().sum())


# =========================
# CARTS INSPECTION
# =========================

print("\n" + "=" * 60)
print("CARTS DATASET")
print("=" * 60)

print("\nShape:")
print(carts_df.shape)

print("\nColumns:")
print(carts_df.columns.tolist())

print("\nData Types:")
print(carts_df.dtypes)

print("\nFirst 5 records:")
print(carts_df.head())

print("\nMissing Values:")
print(carts_df.isnull().sum())


# =========================
# CART PRODUCT STRUCTURE
# =========================

print("\n" + "=" * 60)
print("CART PRODUCT STRUCTURE")
print("=" * 60)

first_cart = carts_df.iloc[0]

print("\nFirst cart:")
print(first_cart)

print("\nProducts column type:")
print(type(first_cart["products"]))

print("\nRaw products value:")
print(first_cart["products"])