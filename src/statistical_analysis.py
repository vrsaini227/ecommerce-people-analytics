import pandas as pd
import numpy as np

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    kruskal,
    chi2_contingency
)


# =========================================================
# LOAD CUSTOMER DATA
# =========================================================

customer_df = pd.read_csv(
    "data/processed/customer_level.csv"
)

integrated_df = pd.read_csv(
    "data/processed/integrated_ecommerce_data.csv"
)


print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

print(
    "\nCustomers:",
    len(customer_df)
)


# =========================================================
# SIGNIFICANCE LEVEL
# =========================================================

alpha = 0.05


print(
    "\nSignificance Level:",
    alpha
)


# =========================================================
# TEST 1
# GENDER vs SPENDING
# Mann-Whitney U Test
# =========================================================

print("\n" + "=" * 70)
print("TEST 1: GENDER vs SPENDING")
print("=" * 70)


male_spending = customer_df.loc[
    customer_df["gender"] == "male",
    "total_spending"
]

female_spending = customer_df.loc[
    customer_df["gender"] == "female",
    "total_spending"
]


u_stat, p_value = mannwhitneyu(
    male_spending,
    female_spending,
    alternative="two-sided"
)


print(
    "\nMale customers:",
    len(male_spending)
)

print(
    "Female customers:",
    len(female_spending)
)

print(
    "\nMale median spending:",
    male_spending.median()
)

print(
    "Female median spending:",
    female_spending.median()
)

print(
    "\nMann-Whitney U statistic:",
    u_stat
)

print(
    "p-value:",
    p_value
)


if p_value < alpha:

    print(
        "\nResult: Statistically significant difference detected."
    )

else:

    print(
        "\nResult: No statistically significant difference detected."
    )


# =========================================================
# TEST 2
# AGE vs SPENDING
# Spearman Correlation
# =========================================================

print("\n" + "=" * 70)
print("TEST 2: AGE vs SPENDING")
print("=" * 70)


spearman_rho, spearman_p = spearmanr(
    customer_df["age"],
    customer_df["total_spending"]
)


print(
    "\nSpearman correlation:",
    spearman_rho
)

print(
    "p-value:",
    spearman_p
)


if spearman_p < alpha:

    print(
        "\nResult: Statistically significant monotonic relationship detected."
    )

else:

    print(
        "\nResult: No statistically significant monotonic relationship detected."
    )


# =========================================================
# TEST 3
# AGE GROUPS vs SPENDING
# Kruskal-Wallis Test
# =========================================================

print("\n" + "=" * 70)
print("TEST 3: AGE GROUP vs SPENDING")
print("=" * 70)


age_groups = [
    customer_df.loc[
        customer_df["age_group"] == group,
        "total_spending"
    ].values
    for group in customer_df["age_group"].dropna().unique()
]


group_names = (
    customer_df["age_group"]
    .dropna()
    .unique()
)


for name, values in zip(
    group_names,
    age_groups
):

    print(
        f"{name}: {len(values)} customers"
    )


kruskal_stat, kruskal_p = kruskal(
    *age_groups
)


print(
    "\nKruskal-Wallis statistic:",
    kruskal_stat
)

print(
    "p-value:",
    kruskal_p
)


if kruskal_p < alpha:

    print(
        "\nResult: At least one age group differs significantly."
    )

else:

    print(
        "\nResult: No statistically significant difference detected among age groups."
    )


# =========================================================
# TEST 4
# GENDER vs CATEGORY
# Chi-Square Test
# =========================================================

print("\n" + "=" * 70)
print("TEST 4: GENDER vs PRODUCT CATEGORY")
print("=" * 70)


gender_category_table = pd.crosstab(
    integrated_df["gender"],
    integrated_df["category"]
)


print(
    "\nContingency Table:"
)

print(
    gender_category_table
)


chi2_stat, chi2_p, degrees_freedom, expected = (
    chi2_contingency(
        gender_category_table
    )
)


print(
    "\nChi-square statistic:",
    chi2_stat
)

print(
    "Degrees of freedom:",
    degrees_freedom
)

print(
    "p-value:",
    chi2_p
)


if chi2_p < alpha:

    print(
        "\nResult: Gender and product-category distribution are statistically associated."
    )

else:

    print(
        "\nResult: No statistically significant association detected."
    )


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("STATISTICAL TESTING SUMMARY")
print("=" * 70)


print(
    "\n1. Gender vs Spending p-value:",
    p_value
)

print(
    "2. Age vs Spending p-value:",
    spearman_p
)

print(
    "3. Age Group vs Spending p-value:",
    kruskal_p
)

print(
    "4. Gender vs Category p-value:",
    chi2_p
)


print("\nSignificance rule:")

print(
    "p < 0.05 → statistically significant"
)

print(
    "p >= 0.05 → insufficient evidence of a statistically significant relationship"
)


print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 70)