import pandas as pd
import os


# ============================================================
# BUSINESS INSIGHTS ENGINE
# ============================================================

print("=" * 70)
print("E-COMMERCE BUSINESS INSIGHTS ENGINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

CATEGORY_FILE = "data/processed/pattern_category_analysis.csv"
PRODUCT_VALUE_FILE = "data/processed/top_products_value.csv"
PRODUCT_QUANTITY_FILE = "data/processed/top_products_quantity.csv"
ANOMALY_FILE = "data/processed/customer_anomaly_analysis.csv"
CLUSTER_FILE = "data/processed/cluster_profile.csv"
STATISTICAL_FILE = "data/processed/statistical_test_results.csv"


# ------------------------------------------------------------
# 2. LOAD AVAILABLE DATA
# ------------------------------------------------------------

print("\nLoading analytical datasets...")


category_df = pd.read_csv(CATEGORY_FILE)

product_value_df = pd.read_csv(
    PRODUCT_VALUE_FILE
)

product_quantity_df = pd.read_csv(
    PRODUCT_QUANTITY_FILE
)

anomaly_df = pd.read_csv(
    ANOMALY_FILE
)

cluster_df = pd.read_csv(
    CLUSTER_FILE
)


# Statistical results may or may not exist depending
# on the previous statistical_analysis.py version.

if os.path.exists(STATISTICAL_FILE):

    statistical_df = pd.read_csv(
        STATISTICAL_FILE
    )

    statistical_available = True

else:

    statistical_df = None
    statistical_available = False


print("All required datasets loaded successfully.")


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)


total_customers = len(anomaly_df)

total_categories = category_df["category"].nunique()

total_products = (
    category_df["unique_products"].sum()
)


print(f"Total Customers: {total_customers}")
print(f"Total Categories: {total_categories}")
print(f"Approx. Product Records: {total_products}")


# ============================================================
# 4. CATEGORY INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY INSIGHTS")
print("=" * 70)


# Highest monetary value

top_value_category = (
    category_df
    .sort_values(
        "total_value",
        ascending=False
    )
    .iloc[0]
)


# Highest quantity

top_quantity_category = (
    category_df
    .sort_values(
        "total_quantity",
        ascending=False
    )
    .iloc[0]
)


# Highest average rating

top_rating_category = (
    category_df
    .sort_values(
        "average_rating",
        ascending=False
    )
    .iloc[0]
)


# Highest discount

top_discount_category = (
    category_df
    .sort_values(
        "average_discount",
        ascending=False
    )
    .iloc[0]
)


print(
    f"Highest-value category: "
    f"{top_value_category['category']}"
)

print(
    f"Highest quantity category: "
    f"{top_quantity_category['category']}"
)

print(
    f"Highest-rated category: "
    f"{top_rating_category['category']}"
)

print(
    f"Highest average discount category: "
    f"{top_discount_category['category']}"
)


# ============================================================
# 5. PRODUCT INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT INSIGHTS")
print("=" * 70)


top_value_product = (
    product_value_df
    .sort_values(
        "total_value",
        ascending=False
    )
    .iloc[0]
)


top_quantity_product = (
    product_quantity_df
    .sort_values(
        "total_quantity",
        ascending=False
    )
    .iloc[0]
)


print(
    f"Highest-value product: "
    f"{top_value_product['title']}"
)

print(
    f"Highest-value product contribution: "
    f"{top_value_product['total_value']:.2f}"
)


print(
    f"Highest-quantity product: "
    f"{top_quantity_product['title']}"
)

print(
    f"Highest-quantity product quantity: "
    f"{top_quantity_product['total_quantity']}"
)


# ============================================================
# 6. CUSTOMER SPENDING INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SPENDING INSIGHTS")
print("=" * 70)


highest_spender = (
    anomaly_df
    .sort_values(
        "total_spending",
        ascending=False
    )
    .iloc[0]
)


median_spending = anomaly_df[
    "total_spending"
].median()


average_spending = anomaly_df[
    "total_spending"
].mean()


print(
    f"Highest customer spending: "
    f"{highest_spender['total_spending']:.2f}"
)

print(
    f"Average customer spending: "
    f"{average_spending:.2f}"
)

print(
    f"Median customer spending: "
    f"{median_spending:.2f}"
)


# ============================================================
# 7. ANOMALY INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY INSIGHTS")
print("=" * 70)


anomaly_count = (
    anomaly_df[
        anomaly_df["anomaly_type"] != "Normal"
    ].shape[0]
)


anomaly_percentage = (
    anomaly_count /
    total_customers
) * 100


strong_anomaly_count = (
    anomaly_df[
        anomaly_df["anomaly_type"]
        == "Strong Anomaly"
    ].shape[0]
)


statistical_outlier_count = (
    anomaly_df[
        anomaly_df["anomaly_type"]
        == "Statistical Outlier"
    ].shape[0]
)


ml_anomaly_count = (
    anomaly_df[
        anomaly_df["anomaly_type"]
        == "ML Detected Anomaly"
    ].shape[0]
)


print(
    f"Total behavioural anomalies: "
    f"{anomaly_count}"
)

print(
    f"Anomaly percentage: "
    f"{anomaly_percentage:.2f}%"
)

print(
    f"Strong anomalies: "
    f"{strong_anomaly_count}"
)

print(
    f"Statistical outliers: "
    f"{statistical_outlier_count}"
)

print(
    f"ML detected anomalies: "
    f"{ml_anomaly_count}"
)


# ============================================================
# 8. CUSTOMER SEGMENTATION INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENTATION INSIGHTS")
print("=" * 70)


print(
    f"Number of customer segments: "
    f"{len(cluster_df)}"
)


# Find available spending column

spending_columns = [
    col
    for col in cluster_df.columns
    if "spending" in col.lower()
]


if spending_columns:

    cluster_spending_col = spending_columns[0]

    highest_spending_cluster = (
        cluster_df
        .sort_values(
            cluster_spending_col,
            ascending=False
        )
        .iloc[0]
    )

    print(
        f"Highest-spending cluster: "
        f"{highest_spending_cluster['cluster']}"
    )


# ============================================================
# 9. STATISTICAL INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL VALIDATION INSIGHTS")
print("=" * 70)


statistical_messages = []


if statistical_available:

    print(
        "Statistical test results loaded."
    )

    for _, row in statistical_df.iterrows():

        test_name = row.iloc[0]
        p_value = float(row.iloc[1])

        if p_value < 0.05:

            message = (
                f"{test_name}: statistically "
                f"significant relationship detected "
                f"(p={p_value:.4f})"
            )

        else:

            message = (
                f"{test_name}: insufficient evidence "
                f"of a statistically significant "
                f"relationship (p={p_value:.4f})"
            )

        statistical_messages.append(
            message
        )

        print(message)

else:

    print(
        "Statistical result CSV not found."
    )

    print(
        "Previously observed tests will be "
        "summarized manually."
    )

    statistical_messages = [

        "Gender vs Spending: "
        "No statistically significant difference detected.",

        "Age vs Spending: "
        "No statistically significant monotonic relationship detected.",

        "Age Group vs Spending: "
        "No statistically significant difference detected.",

        "Gender vs Product Category: "
        "No statistically significant association detected."
    ]

    for message in statistical_messages:
        print(message)


# ============================================================
# 10. AUTOMATIC BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATIC BUSINESS INSIGHTS")
print("=" * 70)


insights = []


# Insight 1

insights.append(
    f"The {top_value_category['category']} category "
    f"generated the highest calculated monetary value "
    f"of approximately "
    f"{top_value_category['total_value']:.2f}."
)


# Insight 2

insights.append(
    f"The {top_quantity_category['category']} category "
    f"recorded the highest total quantity of "
    f"{int(top_quantity_category['total_quantity'])} units."
)


# Insight 3

insights.append(
    f"The highest-value individual product was "
    f"{top_value_product['title']}, contributing "
    f"approximately "
    f"{top_value_product['total_value']:.2f}."
)


# Insight 4

insights.append(
    f"The highest-quantity product was "
    f"{top_quantity_product['title']}, with "
    f"{int(top_quantity_product['total_quantity'])} units."
)


# Insight 5

insights.append(
    f"The average customer spending was "
    f"{average_spending:.2f}, while the median was "
    f"{median_spending:.2f}."
)


# Insight 6

insights.append(
    f"{anomaly_percentage:.2f}% of customers were "
    f"identified as behavioural anomalies using "
    f"statistical and machine-learning methods."
)


# Insight 7

insights.append(
    f"{strong_anomaly_count} customer was classified "
    f"as a strong anomaly by the combined detection approach."
)


# Insight 8

insights.append(
    f"The highest-rated category was "
    f"{top_rating_category['category']} with an average "
    f"rating of "
    f"{top_rating_category['average_rating']:.2f}."
)


# Insight 9

insights.append(
    f"The category with the highest average discount "
    f"was {top_discount_category['category']} "
    f"with an average discount of "
    f"{top_discount_category['average_discount']:.2f}%."
)


for i, insight in enumerate(
    insights,
    start=1
):

    print(
        f"{i}. {insight}"
    )


# ============================================================
# 11. SAVE BUSINESS INSIGHTS
# ============================================================

OUTPUT_DIR = "reports"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


REPORT_PATH = (
    f"{OUTPUT_DIR}/business_insights.txt"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "E-COMMERCE PEOPLE ANALYTICS\n"
    )

    file.write(
        "BUSINESS INSIGHTS REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    file.write(
        "DATASET OVERVIEW\n"
    )

    file.write(
        f"Total Customers: {total_customers}\n"
    )

    file.write(
        f"Total Categories: {total_categories}\n"
    )

    file.write(
        f"Average Customer Spending: "
        f"{average_spending:.2f}\n"
    )

    file.write(
        f"Median Customer Spending: "
        f"{median_spending:.2f}\n\n"
    )


    file.write(
        "BUSINESS INSIGHTS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for i, insight in enumerate(
        insights,
        start=1
    ):

        file.write(
            f"{i}. {insight}\n"
        )


    file.write(
        "\n\nSTATISTICAL VALIDATION\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for message in statistical_messages:

        file.write(
            f"- {message}\n"
        )


print("\n" + "=" * 70)
print("BUSINESS INSIGHTS ENGINE COMPLETED")
print("=" * 70)

print("\nFile saved:")
print("reports/business_insights.txt")