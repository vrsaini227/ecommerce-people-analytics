import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# 1. LOAD CUSTOMER DATA
# =========================================================

df = pd.read_csv(
    "data/processed/customer_level.csv"
)


print("\n" + "=" * 70)
print("CUSTOMER SEGMENTATION")
print("=" * 70)

print("\nCustomer Dataset Shape:")
print(df.shape)


# =========================================================
# 2. SELECT FEATURES
# =========================================================

features = [
    "total_spending",
    "total_quantity",
    "unique_products",
    "unique_categories",
    "average_rating",
    "average_discount"
]


X = df[features].copy()


print("\n" + "=" * 70)
print("SELECTED FEATURES")
print("=" * 70)

print(features)

print("\nFeature Statistics:")
print(X.describe())


# =========================================================
# 3. HANDLE MISSING VALUES
# =========================================================

print("\nMissing Values:")

print(
    X.isnull().sum()
)


X = X.fillna(
    X.median(numeric_only=True)
)


# =========================================================
# 4. LOG TRANSFORM SKEWED FEATURES
# =========================================================

X_transformed = X.copy()


skewed_features = [
    "total_spending",
    "total_quantity",
    "unique_products",
    "unique_categories"
]


for column in skewed_features:

    X_transformed[column] = np.log1p(
        X_transformed[column]
    )


print("\n" + "=" * 70)
print("LOG TRANSFORMATION COMPLETED")
print("=" * 70)


# =========================================================
# 5. STANDARDIZATION
# =========================================================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X_transformed
)


print("\n" + "=" * 70)
print("FEATURE SCALING COMPLETED")
print("=" * 70)


# =========================================================
# 6. ELBOW METHOD
# =========================================================

inertia_values = []

k_values = range(2, 8)


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )


# =========================================================
# 7. SILHOUETTE SCORES
# =========================================================

silhouette_values = []


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(
        score
    )


print("\n" + "=" * 70)
print("CLUSTER EVALUATION")
print("=" * 70)


for k, inertia, silhouette in zip(
    k_values,
    inertia_values,
    silhouette_values
):

    print(
        f"K={k} | "
        f"Inertia={inertia:.2f} | "
        f"Silhouette={silhouette:.4f}"
    )


# =========================================================
# 8. ELBOW PLOT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title(
    "Elbow Method for Customer Segmentation"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares (Inertia)"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "reports/figures/elbow_method.png",
    dpi=300
)

plt.show()


# =========================================================
# 9. SILHOUETTE PLOT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "reports/figures/silhouette_scores.png",
    dpi=300
)

plt.show()


# =========================================================
# 10. SELECT K
# =========================================================

best_k = list(
    k_values
)[
    np.argmax(
        silhouette_values
    )
]


print("\n" + "=" * 70)
print("SELECTED CLUSTER COUNT")
print("=" * 70)

print(
    "Best K according to silhouette score:",
    best_k
)


# =========================================================
# 11. FINAL K-MEANS MODEL
# =========================================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)


df["cluster"] = final_model.fit_predict(
    X_scaled
)


# =========================================================
# 12. CLUSTER SIZE
# =========================================================

print("\n" + "=" * 70)
print("CLUSTER SIZE")
print("=" * 70)

cluster_sizes = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print(
    cluster_sizes
)


# =========================================================
# 13. CLUSTER PROFILE
# =========================================================

cluster_profile = (
    df.groupby("cluster")
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

        average_unique_products=(
            "unique_products",
            "mean"
        ),

        average_categories=(
            "unique_categories",
            "mean"
        ),

        average_rating=(
            "average_rating",
            "mean"
        ),

        average_discount=(
            "average_discount",
            "mean"
        ),

        average_age=(
            "age",
            "mean"
        )
    )
    .round(2)
)


print("\n" + "=" * 70)
print("CUSTOMER CLUSTER PROFILE")
print("=" * 70)

print(
    cluster_profile
)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

df.to_csv(
    "data/processed/customer_segments.csv",
    index=False
)


cluster_profile.to_csv(
    "data/processed/cluster_profile.csv"
)


# =========================================================
# 15. 2D VISUALIZATION
# =========================================================

plt.figure(
    figsize=(10, 7)
)

sns.scatterplot(
    data=df,
    x="total_quantity",
    y="total_spending",
    hue="cluster",
    palette="tab10",
    s=80
)

plt.title(
    "Customer Segments: Quantity vs Spending"
)

plt.xlabel(
    "Total Quantity"
)

plt.ylabel(
    "Total Calculated Spending"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/customer_clusters.png",
    dpi=300
)

plt.show()


# =========================================================
# 16. COMPLETION
# =========================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENTATION COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(
    "data/processed/customer_segments.csv"
)

print(
    "data/processed/cluster_profile.csv"
)

print(
    "reports/figures/elbow_method.png"
)

print(
    "reports/figures/silhouette_scores.png"
)

print(
    "reports/figures/customer_clusters.png"
)