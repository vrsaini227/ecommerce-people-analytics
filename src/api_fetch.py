import requests
import pandas as pd
import json
import os


# ==========================================
# API URLs
# ==========================================

USERS_API = (
    "https://dummyjson.com/users"
    "?limit=0"
    "&select=id,age,gender,height,weight,eyeColor,hair,address"
)

PRODUCTS_API = (
    "https://dummyjson.com/products"
    "?limit=0"
)

CARTS_API = (
    "https://dummyjson.com/carts"
    "?limit=0"
)


# ==========================================
# Create directories
# ==========================================

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ==========================================
# Fetch API
# ==========================================

def fetch_api_data(url):

    try:
        response = requests.get(url, timeout=20)

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:

        print("API Error:", error)

        return None


# ==========================================
# USERS
# ==========================================

print("\nFetching complete Users dataset...")

users_data = fetch_api_data(USERS_API)

if users_data:

    print("Users API: SUCCESS")

    print(
        "Total users returned:",
        len(users_data.get("users", []))
    )


# ==========================================
# PRODUCTS
# ==========================================

print("\nFetching complete Products dataset...")

products_data = fetch_api_data(PRODUCTS_API)

if products_data:

    print("Products API: SUCCESS")

    print(
        "Total products returned:",
        len(products_data.get("products", []))
    )


# ==========================================
# CARTS
# ==========================================

print("\nFetching complete Carts dataset...")

carts_data = fetch_api_data(CARTS_API)

if carts_data:

    print("Carts API: SUCCESS")

    print(
        "Total carts returned:",
        len(carts_data.get("carts", []))
    )


# ==========================================
# Save RAW JSON
# ==========================================

if users_data:

    with open(
        "data/raw/users.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            users_data,
            file,
            indent=4
        )


if products_data:

    with open(
        "data/raw/products.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            products_data,
            file,
            indent=4
        )


if carts_data:

    with open(
        "data/raw/carts.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            carts_data,
            file,
            indent=4
        )


# ==========================================
# Convert to DataFrame
# ==========================================

users_df = pd.DataFrame(
    users_data["users"]
)

products_df = pd.DataFrame(
    products_data["products"]
)

carts_df = pd.DataFrame(
    carts_data["carts"]
)


# ==========================================
# Save CSV
# ==========================================

users_df.to_csv(
    "data/raw/users.csv",
    index=False
)

products_df.to_csv(
    "data/raw/products.csv",
    index=False
)

carts_df.to_csv(
    "data/raw/carts.csv",
    index=False
)


# ==========================================
# Dataset Summary
# ==========================================

print("\n" + "=" * 60)
print("COMPLETE DATA COLLECTION")
print("=" * 60)

print("\nUsers:", users_df.shape)

print("Products:", products_df.shape)

print("Carts:", carts_df.shape)


print("\nData collection completed successfully.")