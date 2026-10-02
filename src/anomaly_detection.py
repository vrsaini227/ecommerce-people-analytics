import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CUSTOMER ANOMALY DETECTION
# ============================================================

print("=" * 70)
print("CUSTOMER ANOMALY DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD CUSTOMER DATA
# ------------------------------------------------------------

DATA_PATH = "data/processed/customer_level.csv"

df = pd.read_csv(DATA_PATH)

print("\nCustomer Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. REQUIRED FEATURES
# ------------------------------------------------------------

required_features = [
    "total_spending",
    "total_quantity",
    "unique_products",
    "unique_categories",
    "average_rating",
    "average_discount"
]

missing_features = [
    col for col in required_features
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("\nRequired features validated successfully.")


# ------------------------------------------------------------
# 3. MISSING VALUE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(df[required_features].isnull().sum())


# ------------------------------------------------------------
# 4. STATISTICAL OUTLIER DETECTION - IQR
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICAL OUTLIER DETECTION")
print("=" * 70)


def calculate_iqr_bounds(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return q1, q3, iqr, lower_bound, upper_bound


# Spending outliers

q1_spending, q3_spending, iqr_spending, \
lower_spending, upper_spending = calculate_iqr_bounds(
    df["total_spending"]
)

df["spending_outlier"] = (
    (df["total_spending"] < lower_spending) |
    (df["total_spending"] > upper_spending)
)


print("\nSpending Statistics:")

print(f"Q1: {q1_spending:.2f}")
print(f"Q3: {q3_spending:.2f}")
print(f"IQR: {iqr_spending:.2f}")
print(f"Lower Bound: {lower_spending:.2f}")
print(f"Upper Bound: {upper_spending:.2f}")

print(
    f"Spending Outliers: "
    f"{df['spending_outlier'].sum()}"
)


# Quantity outliers

q1_quantity, q3_quantity, iqr_quantity, \
lower_quantity, upper_quantity = calculate_iqr_bounds(
    df["total_quantity"]
)

df["quantity_outlier"] = (
    (df["total_quantity"] < lower_quantity) |
    (df["total_quantity"] > upper_quantity)
)


print("\nQuantity Statistics:")

print(f"Q1: {q1_quantity:.2f}")
print(f"Q3: {q3_quantity:.2f}")
print(f"IQR: {iqr_quantity:.2f}")
print(f"Lower Bound: {lower_quantity:.2f}")
print(f"Upper Bound: {upper_quantity:.2f}")

print(
    f"Quantity Outliers: "
    f"{df['quantity_outlier'].sum()}"
)


# ------------------------------------------------------------
# 5. PREPARE ML FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPARING MACHINE LEARNING FEATURES")
print("=" * 70)

X = df[required_features].copy()

print("\nFeatures used:")
print(required_features)


# ------------------------------------------------------------
# 6. LOG TRANSFORMATION
# ------------------------------------------------------------

# Spending and quantity are heavily right-skewed.
# log1p reduces the effect of extreme values.

for column in [
    "total_spending",
    "total_quantity",
    "unique_products",
    "unique_categories"
]:

    X[column] = np.log1p(X[column])


print("\nLog transformation completed.")


# ------------------------------------------------------------
# 7. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")


# ------------------------------------------------------------
# 8. ISOLATION FOREST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ISOLATION FOREST ANOMALY DETECTION")
print("=" * 70)


model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)


# Prediction:
#  1  = normal
# -1  = anomaly

df["ml_prediction"] = model.predict(X_scaled)

df["anomaly_score"] = model.decision_function(
    X_scaled
)


df["ml_anomaly"] = (
    df["ml_prediction"] == -1
)


print(
    f"\nML anomalies detected: "
    f"{df['ml_anomaly'].sum()}"
)


# ------------------------------------------------------------
# 9. COMBINED ANOMALY CLASSIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMBINED ANOMALY CLASSIFICATION")
print("=" * 70)


def classify_anomaly(row):

    statistical_count = (
        int(row["spending_outlier"]) +
        int(row["quantity_outlier"])
    )

    ml_anomaly = row["ml_anomaly"]

    if ml_anomaly and statistical_count >= 1:
        return "Strong Anomaly"

    elif ml_anomaly:
        return "ML Detected Anomaly"

    elif statistical_count >= 1:
        return "Statistical Outlier"

    else:
        return "Normal"


df["anomaly_type"] = df.apply(
    classify_anomaly,
    axis=1
)


print(
    df["anomaly_type"]
    .value_counts()
)


# ------------------------------------------------------------
# 10. ANOMALY REASON
# ------------------------------------------------------------

def determine_reason(row):

    reasons = []

    if row["spending_outlier"]:
        reasons.append("Unusually High Spending")

    if row["quantity_outlier"]:
        reasons.append("Unusually High Quantity")

    if row["unique_products"] >= df["unique_products"].quantile(0.90):
        reasons.append("High Product Diversity")

    if row["unique_categories"] >= df["unique_categories"].quantile(0.90):
        reasons.append("High Category Diversity")

    if not reasons:
        reasons.append("Multivariate Behaviour Pattern")

    return ", ".join(reasons)


df["anomaly_reason"] = df.apply(
    determine_reason,
    axis=1
)


# ------------------------------------------------------------
# 11. TOP ANOMALOUS CUSTOMERS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP ANOMALOUS CUSTOMERS")
print("=" * 70)


anomalies = (
    df[df["anomaly_type"] != "Normal"]
    .sort_values(
        "anomaly_score",
        ascending=True
    )
)


display_columns = [
    "user_id",
    "age",
    "total_spending",
    "total_quantity",
    "unique_products",
    "unique_categories",
    "average_rating",
    "average_discount",
    "anomaly_score",
    "anomaly_type",
    "anomaly_reason"
]


print(
    anomalies[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 12. HIGHEST SPENDING CUSTOMERS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 HIGHEST SPENDING CUSTOMERS")
print("=" * 70)


top_spenders = (
    df.sort_values(
        "total_spending",
        ascending=False
    )
    .head(10)
)


print(
    top_spenders[
        [
            "user_id",
            "age",
            "total_spending",
            "total_quantity",
            "unique_products",
            "unique_categories",
            "anomaly_type"
        ]
    ]
    .to_string(index=False)
)


# ------------------------------------------------------------
# 13. ANOMALY SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ANOMALY SUMMARY")
print("=" * 70)


total_customers = len(df)

total_anomalies = (
    (df["anomaly_type"] != "Normal")
    .sum()
)


anomaly_percentage = (
    total_anomalies /
    total_customers
) * 100


print(f"Total Customers: {total_customers}")
print(f"Total Anomalies: {total_anomalies}")
print(
    f"Anomaly Percentage: "
    f"{anomaly_percentage:.2f}%"
)


# ------------------------------------------------------------
# 14. SAVE RESULTS
# ------------------------------------------------------------

OUTPUT_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df.to_csv(
    f"{OUTPUT_DIR}/customer_anomaly_analysis.csv",
    index=False
)


anomalies.to_csv(
    f"{OUTPUT_DIR}/customer_anomalies.csv",
    index=False
)


top_spenders.to_csv(
    f"{OUTPUT_DIR}/top_spending_customers.csv",
    index=False
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ANOMALY DETECTION COMPLETED")
print("=" * 70)

print("\nFiles saved:")
print("data/processed/customer_anomaly_analysis.csv")
print("data/processed/customer_anomalies.csv")
print("data/processed/top_spending_customers.csv")