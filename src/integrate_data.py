import pandas as pd
import ast
import os


# =========================================================
# 1. Load Clean Datasets
# =========================================================

print("\n" + "=" * 70)
print("LOADING CLEAN DATASETS")
print("=" * 70)


users_df = pd.read_csv(
    "data/processed/users_clean.csv"
)

products_df = pd.read_csv(
    "data/processed/products_clean.csv"
)

carts_df = pd.read_csv(
    "data/processed/carts_clean.csv"
)


print("\nUsers:", users_df.shape)
print("Products:", products_df.shape)
print("Carts:", carts_df.shape)


# =========================================================
# 2. Prepare Product Data
# =========================================================

print("\n" + "=" * 70)
print("PREPARING PRODUCT DATA")
print("=" * 70)


# Missing brands
products_df["brand"] = products_df["brand"].fillna(
    "Unknown"
)


# =========================================================
# 3. Flatten Cart Products
# =========================================================

print("\n" + "=" * 70)
print("FLATTENING CART PRODUCTS")
print("=" * 70)


cart_items = []


for _, cart in carts_df.iterrows():

    cart_id = cart["id"]

    user_id = cart["userId"]

    cart_total = cart["total"]

    cart_discounted_total = cart["discountedTotal"]

    total_products = cart["totalProducts"]

    total_quantity = cart["totalQuantity"]


    # Convert string representation of list
    # into Python list
    try:

        products_list = ast.literal_eval(
            cart["products"]
        )

    except (ValueError, SyntaxError) as error:

        print(
            f"Error parsing cart {cart_id}: {error}"
        )

        continue


    # Extract every product from cart

    for product in products_list:

        cart_items.append({

            "cart_id": cart_id,

            "user_id": user_id,

            "product_id": product.get("id"),

            "quantity": product.get("quantity"),

            "cart_total": cart_total,

            "cart_discounted_total":
                cart_discounted_total,

            "cart_total_products":
                total_products,

            "cart_total_quantity":
                total_quantity

        })


# Convert list to DataFrame

cart_items_df = pd.DataFrame(
    cart_items
)


print(
    "\nFlattened Cart Items:",
    cart_items_df.shape
)


print("\nFirst 10 Cart Items:")

print(
    cart_items_df.head(10)
)


# =========================================================
# 4. Validate Cart Product IDs
# =========================================================

print("\n" + "=" * 70)
print("PRODUCT ID VALIDATION")
print("=" * 70)


product_ids = set(
    products_df["id"]
)


cart_product_ids = set(
    cart_items_df["product_id"]
)


missing_product_ids = (
    cart_product_ids - product_ids
)


print(
    "\nUnique product IDs in carts:",
    len(cart_product_ids)
)


print(
    "Products available in master dataset:",
    len(product_ids)
)


print(
    "Product IDs missing from product dataset:",
    len(missing_product_ids)
)


if missing_product_ids:

    print(
        "\nMissing Product IDs:"
    )

    print(
        sorted(missing_product_ids)
    )

else:

    print(
        "\nAll cart product IDs matched successfully."
    )


# =========================================================
# 5. Validate User IDs
# =========================================================

print("\n" + "=" * 70)
print("USER ID VALIDATION")
print("=" * 70)


user_ids = set(
    users_df["id"]
)


cart_user_ids = set(
    cart_items_df["user_id"]
)


missing_user_ids = (
    cart_user_ids - user_ids
)


print(
    "\nUnique users in carts:",
    len(cart_user_ids)
)


print(
    "Users available in master dataset:",
    len(user_ids)
)


print(
    "User IDs missing:",
    len(missing_user_ids)
)


if missing_user_ids:

    print(
        "\nMissing User IDs:"
    )

    print(
        sorted(missing_user_ids)
    )

else:

    print(
        "\nAll cart user IDs matched successfully."
    )


# =========================================================
# 6. Prepare User Data
# =========================================================

print("\n" + "=" * 70)
print("PREPARING USER DATA")
print("=" * 70)


# We only need analytical fields.
# Names, emails, passwords, etc. are not included.

user_analysis = users_df[
    [
        "id",
        "age",
        "gender",
        "height",
        "weight",
        "eyeColor",
        "hair",
        "address"
    ]
].copy()


# Rename user ID

user_analysis = user_analysis.rename(
    columns={
        "id": "user_id"
    }
)


# =========================================================
# 7. Prepare Product Data
# =========================================================

product_analysis = products_df[
    [
        "id",
        "title",
        "category",
        "price",
        "discountPercentage",
        "rating",
        "stock",
        "brand"
    ]
].copy()


product_analysis = product_analysis.rename(
    columns={
        "id": "product_id"
    }
)


# =========================================================
# 8. Merge Cart Items + Products
# =========================================================

print("\n" + "=" * 70)
print("MERGING CART ITEMS WITH PRODUCTS")
print("=" * 70)


integrated_df = cart_items_df.merge(
    product_analysis,
    on="product_id",
    how="left",
    indicator="_product_match"
)


# =========================================================
# 9. Merge Users
# =========================================================

print("\nMerging Users...")


integrated_df = integrated_df.merge(
    user_analysis,
    on="user_id",
    how="left",
    indicator="_user_match"
)


# =========================================================
# 10. Match Validation
# =========================================================

print("\n" + "=" * 70)
print("MERGE VALIDATION")
print("=" * 70)


print(
    "\nProduct Match:"
)

print(
    integrated_df["_product_match"].value_counts()
)


print(
    "\nUser Match:"
)

print(
    integrated_df["_user_match"].value_counts()
)


# =========================================================
# 11. Remove Merge Helper Columns
# =========================================================

integrated_df = integrated_df.drop(
    columns=[
        "_product_match",
        "_user_match"
    ]
)


# =========================================================
# 12. Create Calculated Fields
# =========================================================

print("\n" + "=" * 70)
print("CREATING ANALYTICAL FEATURES")
print("=" * 70)


# Estimated item value

integrated_df["item_value"] = (
    integrated_df["price"]
    * integrated_df["quantity"]
)


# Estimated discount amount

integrated_df["discount_amount"] = (
    integrated_df["item_value"]
    * integrated_df["discountPercentage"]
    / 100
)


# Estimated discounted item value

integrated_df["discounted_item_value"] = (
    integrated_df["item_value"]
    - integrated_df["discount_amount"]
)


# =========================================================
# 13. Save Cart Items Dataset
# =========================================================

cart_items_df.to_csv(
    "data/processed/cart_items.csv",
    index=False
)


# =========================================================
# 14. Save Integrated Dataset
# =========================================================

integrated_df.to_csv(
    "data/processed/integrated_ecommerce_data.csv",
    index=False
)


# =========================================================
# 15. Final Summary
# =========================================================

print("\n" + "=" * 70)
print("INTEGRATION COMPLETED")
print("=" * 70)


print(
    "\nCart Items:",
    cart_items_df.shape
)


print(
    "Integrated Dataset:",
    integrated_df.shape
)


print(
    "\nIntegrated Dataset Columns:"
)


print(
    integrated_df.columns.tolist()
)


print(
    "\nFirst 10 Integrated Records:"
)


print(
    integrated_df.head(10)
)


print("\nSaved files:")

print(
    "data/processed/cart_items.csv"
)

print(
    "data/processed/integrated_ecommerce_data.csv"
)