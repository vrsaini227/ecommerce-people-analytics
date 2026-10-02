import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# PLOTLY CHART CONTROLS
# =========================================================
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "modeBarButtonsToAdd": [
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d"
    ]
}


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="E-Commerce People Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.kpi-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #374151;
    text-align: center;
}

.kpi-label {
    color: #9ca3af;
    font-size: 14px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 5px;
}

.info-box {
    background: #1f2937;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #60a5fa;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED = BASE_DIR / "data" / "processed"
REPORTS = BASE_DIR / "reports"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    integrated = pd.read_csv(
        PROCESSED / "integrated_ecommerce_public.csv"
    )

    customer = pd.read_csv(
        PROCESSED / "customer_level.csv"
    )

    category = pd.read_csv(
        PROCESSED / "pattern_category_analysis.csv"
    )

    product_pattern = pd.read_csv(
        PROCESSED / "product_pattern_analysis.csv"
    )

    customer_segments = pd.read_csv(
        PROCESSED / "customer_segments.csv"
    )

    cluster_profile = pd.read_csv(
        PROCESSED / "cluster_profile.csv"
    )

    anomalies = pd.read_csv(
        PROCESSED / "customer_anomaly_analysis.csv"
    )

    return (
        integrated,
        customer,
        category,
        product_pattern,
        customer_segments,
        cluster_profile,
        anomalies
    )


try:

    (
        integrated,
        customer,
        category,
        product_pattern,
        customer_segments,
        cluster_profile,
        anomalies
    ) = load_data()

except Exception as e:

    st.error("Data loading failed.")
    st.exception(e)
    st.stop()


# =========================================================
# SIDEBAR - FILTER CONTROLS
# =========================================================


def initialize_filter_state():
    """Create separate pending and applied filter states."""

    genders = sorted(
        integrated["gender"].dropna().unique().tolist()
    )

    categories_list = sorted(
        integrated["category"].dropna().unique().tolist()
    )

    min_age = int(integrated["age"].min())
    max_age = int(integrated["age"].max())

    # Pending values are what the user is currently selecting.
    if "pending_gender" not in st.session_state:
        st.session_state.pending_gender = genders.copy()

    if "pending_age" not in st.session_state:
        st.session_state.pending_age = (min_age, max_age)

    if "pending_category" not in st.session_state:
        st.session_state.pending_category = categories_list.copy()

    # Applied values are used by the dashboard until Apply Filters is pressed.
    if "applied_gender" not in st.session_state:
        st.session_state.applied_gender = genders.copy()

    if "applied_age" not in st.session_state:
        st.session_state.applied_age = (min_age, max_age)

    if "applied_category" not in st.session_state:
        st.session_state.applied_category = categories_list.copy()

    return genders, categories_list, min_age, max_age


def apply_filters():
    """Apply pending sidebar selections to the dashboard."""

    st.session_state.applied_gender = st.session_state.pending_gender.copy()
    st.session_state.applied_age = tuple(st.session_state.pending_age)
    st.session_state.applied_category = st.session_state.pending_category.copy()
    st.session_state.filter_message = "Filters applied successfully."


def reset_filters():
    """Reset both pending and applied filters to the full dataset."""

    genders = sorted(
        integrated["gender"].dropna().unique().tolist()
    )
    categories_list = sorted(
        integrated["category"].dropna().unique().tolist()
    )
    min_age = int(integrated["age"].min())
    max_age = int(integrated["age"].max())

    st.session_state.pending_gender = genders.copy()
    st.session_state.pending_age = (min_age, max_age)
    st.session_state.pending_category = categories_list.copy()

    st.session_state.applied_gender = genders.copy()
    st.session_state.applied_age = (min_age, max_age)
    st.session_state.applied_category = categories_list.copy()

    st.session_state.filter_message = "Filters reset successfully."


def refresh_dashboard_data():
    """Clear cached CSV data so the next run reads fresh files."""

    st.cache_data.clear()
    st.session_state.filter_message = "Dashboard data refreshed successfully."


# Initialize filter state after data has loaded.
genders, categories_list, min_age, max_age = initialize_filter_state()


with st.sidebar:

    st.markdown("## 🎛️ Dashboard Controls")

    st.markdown("### 🔎 Customer Filters")
    st.caption("Select filters below, then click Apply Filters.")

    # -----------------------------------------------------
    # PENDING FILTERS
    # -----------------------------------------------------

    st.multiselect(
        "Gender",
        options=genders,
        key="pending_gender"
    )

    st.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        key="pending_age"
    )

    st.multiselect(
        "Product Category",
        options=categories_list,
        key="pending_category"
    )

    st.divider()

    # -----------------------------------------------------
    # FILTER ACTIONS
    # -----------------------------------------------------

    apply_col, reset_col = st.columns(2)

    with apply_col:
        st.button(
            "✅ Apply Filters",
            use_container_width=True,
            type="primary",
            on_click=apply_filters
        )

    with reset_col:
        st.button(
            "↩️ Reset Filters",
            use_container_width=True,
            on_click=reset_filters
        )

    st.button(
        "♻️ Refresh Dashboard Data",
        use_container_width=True,
        on_click=refresh_dashboard_data
    )

    # Show a small status message after an action.
    if st.session_state.get("filter_message"):
        st.success(st.session_state.filter_message, icon="✅")
        # Clear it after displaying it for the current run.
        st.session_state.filter_message = ""

    st.divider()

    # -----------------------------------------------------
    # CURRENTLY APPLIED FILTER SUMMARY
    # -----------------------------------------------------

    st.markdown("### 📌 Applied Filters")

    applied_gender_display = ", ".join(st.session_state.applied_gender)
    if not applied_gender_display:
        applied_gender_display = "None"

    applied_category_count = len(st.session_state.applied_category)
    total_category_count = len(categories_list)

    st.caption(f"**Gender:** {applied_gender_display}")
    st.caption(
        f"**Age:** {st.session_state.applied_age[0]} - "
        f"{st.session_state.applied_age[1]}"
    )
    st.caption(
        f"**Categories:** {applied_category_count}/"
        f"{total_category_count} selected"
    )


# =========================================================
# FILTER DATA
# =========================================================

# Use ONLY the applied filters here. This means changing a widget does
# not alter the dashboard until the user clicks Apply Filters.
selected_gender = st.session_state.applied_gender
selected_age = st.session_state.applied_age
selected_categories = st.session_state.applied_category

filtered_integrated = integrated[
    (integrated["gender"].isin(selected_gender)) &
    (integrated["age"].between(
        selected_age[0],
        selected_age[1]
    )) &
    (integrated["category"].isin(selected_categories))
].copy()


# Customer IDs matching filters

filtered_user_ids = filtered_integrated[
    "user_id"
].unique()


filtered_customer = customer[
    customer["user_id"].isin(filtered_user_ids)
].copy()


filtered_anomalies = anomalies[
    anomalies["user_id"].isin(filtered_user_ids)
].copy()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 E-Commerce People Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data-driven customer behaviour, product patterns, '
    'segmentation and behavioural anomaly analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = filtered_customer["user_id"].nunique()

calculated_value = filtered_integrated[
    "item_value"
].sum()

avg_spending = (
    filtered_customer["total_spending"].mean()
    if len(filtered_customer) > 0
    else 0
)

median_spending = (
    filtered_customer["total_spending"].median()
    if len(filtered_customer) > 0
    else 0
)

total_quantity = filtered_integrated[
    "quantity"
].sum()

anomaly_count = len(
    filtered_anomalies[
        filtered_anomalies["anomaly_type"] != "Normal"
    ]
)


def format_currency(value):

    if value >= 10000000:
        return f"₹{value / 10000000:.2f} Cr"

    if value >= 100000:
        return f"₹{value / 100000:.2f} L"

    return f"₹{value:,.0f}"


# =========================================================
# KPI SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📌 Key Performance Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5, c6 = st.columns(6)


with c1:

    st.metric(
        "Customers",
        f"{total_customers:,}"
    )


with c2:

    st.metric(
        "Calculated Item Value",
        format_currency(calculated_value)
    )


with c3:

    st.metric(
        "Average Spending",
        format_currency(avg_spending)
    )


with c4:

    st.metric(
        "Median Spending",
        format_currency(median_spending)
    )


with c5:

    st.metric(
        "Quantity",
        f"{int(total_quantity):,}"
    )


with c6:

    st.metric(
        "Behavioural Anomalies",
        f"{anomaly_count:,}"
    )


st.caption(
    "Note: Calculated Item Value is derived from the available "
    "product/cart data and should not be interpreted as actual "
    "company revenue or profit."
)


# =========================================================
# NAVIGATION
# =========================================================

st.divider()

section = st.radio(
    "Navigate Dashboard",
    [
        "📈 Overview",
        "👥 Customer Analysis",
        "🎯 Customer Segmentation",
        "🛍️ Product Analysis",
        "⚠️ Anomaly Analysis",
        "💡 Business Insights",
        "🔎 Data Explorer"
    ],
    horizontal=True
)


# =========================================================
# OVERVIEW
# =========================================================

if section == "📈 Overview":

    st.markdown(
        '<div class="section-title">📈 Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    category_summary = (
        filtered_integrated
        .groupby("category")
        .agg(
            calculated_value=("item_value", "sum"),
            quantity=("quantity", "sum")
        )
        .reset_index()
    )

    with col1:

        fig = px.bar(
            category_summary.sort_values(
                "calculated_value",
                ascending=False
            ),
            x="category",
            y="calculated_value",
            title="Calculated Item Value by Category",
            labels={
                "calculated_value": "Calculated Value",
                "category": "Category"
            }
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    with col2:

        fig = px.bar(
            category_summary.sort_values(
                "quantity",
                ascending=False
            ),
            x="category",
            y="quantity",
            title="Quantity by Category",
            labels={
                "quantity": "Total Quantity",
                "category": "Category"
            }
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )


# =========================================================
# CUSTOMER ANALYSIS
# =========================================================

elif section == "👥 Customer Analysis":

    st.markdown(
        '<div class="section-title">👥 Customer Analysis</div>',
        unsafe_allow_html=True
    )

    if len(filtered_customer) == 0:

        st.warning("No customers match the selected filters.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            fig = px.scatter(
                filtered_customer,
                x="age",
                y="total_spending",
                color="gender",
                hover_data=[
                    "user_id",
                    "total_quantity",
                    "unique_products",
                    "unique_categories"
                ],
                title="Age vs Customer Spending"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                filtered_customer,
                x="total_spending",
                color="gender",
                nbins=30,
                title="Customer Spending Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Customer Summary")

        display_columns = [
            "user_id",
            "age",
            "gender",
            "total_spending",
            "total_quantity",
            "unique_products",
            "unique_categories"
        ]

        available_columns = [
            c for c in display_columns
            if c in filtered_customer.columns
        ]

        st.dataframe(
            filtered_customer[
                available_columns
            ].sort_values(
                "total_spending",
                ascending=False
            ),
            use_container_width=True,
            height=400
        )


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

elif section == "🎯 Customer Segmentation":

    st.markdown(
        '<div class="section-title">🎯 Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    segment_data = customer_segments[
        customer_segments["user_id"].isin(
            filtered_user_ids
        )
    ].copy()

    # Meaningful labels
    segment_data["segment_label"] = (
        segment_data["cluster"]
        .map({
            0: "Higher-Value / Higher-Engagement",
            1: "Lower-Value / Lower-Engagement"
        })
        .fillna(
            segment_data["cluster"].astype(str)
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        distribution = (
            segment_data["segment_label"]
            .value_counts()
            .reset_index()
        )

        distribution.columns = [
            "segment",
            "customers"
        ]

        fig = px.pie(
            distribution,
            names="segment",
            values="customers",
            title="Customer Segment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    with col2:

        fig = px.scatter(
            segment_data,
            x="total_quantity",
            y="total_spending",
            color="segment_label",
            hover_data=[
                "user_id",
                "unique_products",
                "unique_categories"
            ],
            title="Customer Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    st.subheader("Segment Profiles")

    profile = cluster_profile.copy()

    if "cluster" in profile.columns:

        profile["segment_label"] = (
            profile["cluster"]
            .map({
                0: "Higher-Value / Higher-Engagement",
                1: "Lower-Value / Lower-Engagement"
            })
            .fillna(
                profile["cluster"].astype(str)
            )
        )

    st.dataframe(
        profile,
        use_container_width=True
    )


# =========================================================
# PRODUCT ANALYSIS
# =========================================================

elif section == "🛍️ Product Analysis":

    st.markdown(
        '<div class="section-title">🛍️ Product Analysis</div>',
        unsafe_allow_html=True
    )

    product_summary = (
        filtered_integrated
        .groupby("title")
        .agg(
            total_value=("item_value", "sum"),
            total_quantity=("quantity", "sum")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        top_value = product_summary.nlargest(
            10,
            "total_value"
        )

        fig = px.bar(
            top_value.sort_values(
                "total_value"
            ),
            x="total_value",
            y="title",
            orientation="h",
            title="Top 10 Products by Calculated Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    with col2:

        top_quantity = product_summary.nlargest(
            10,
            "total_quantity"
        )

        fig = px.bar(
            top_quantity.sort_values(
                "total_quantity"
            ),
            x="total_quantity",
            y="title",
            orientation="h",
            title="Top 10 Products by Quantity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

elif section == "⚠️ Anomaly Analysis":

    st.markdown(
        '<div class="section-title">'
        '⚠️ Behavioural Anomaly Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    if len(filtered_anomalies) == 0:

        st.info(
            "No anomaly records match the selected filters."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            distribution = (
                filtered_anomalies[
                    "anomaly_type"
                ]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "anomaly_type",
                "count"
            ]

            fig = px.pie(
                distribution,
                names="anomaly_type",
                values="count",
                title="Anomaly Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.scatter(
                filtered_anomalies,
                x="total_quantity",
                y="total_spending",
                color="anomaly_type",
                hover_data=[
                    "user_id",
                    "age",
                    "unique_products",
                    "unique_categories"
                ],
                title="Customer Spending vs Quantity"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Detected Behavioural Anomalies"
        )

        anomaly_columns = [
            "user_id",
            "age",
            "total_spending",
            "total_quantity",
            "unique_products",
            "unique_categories",
            "anomaly_type",
            "anomaly_reason"
        ]

        available = [
            c for c in anomaly_columns
            if c in filtered_anomalies.columns
        ]

        st.dataframe(
            filtered_anomalies[
                available
            ].sort_values(
                "total_spending",
                ascending=False
            ),
            use_container_width=True,
            height=450
        )

        st.info(
            "Behavioural anomaly means an unusual pattern in the "
            "available analytical data. It does not by itself "
            "indicate fraud or malicious activity."
        )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif section == "💡 Business Insights":

    st.markdown(
        '<div class="section-title">💡 Business Insights</div>',
        unsafe_allow_html=True
    )

    report_path = REPORTS / "business_insights.txt"

    if report_path.exists():

        report_text = report_path.read_text(
            encoding="utf-8"
        )

        st.text_area(
            "Generated Analytical Report",
            report_text,
            height=500
        )

        st.download_button(
            "📥 Download Business Insights Report",
            data=report_text,
            file_name="business_insights.txt",
            mime="text/plain"
        )

    else:

        st.warning(
            "Business insights report not found."
        )


# =========================================================
# DATA EXPLORER
# =========================================================

elif section == "🔎 Data Explorer":

    st.markdown(
        '<div class="section-title">🔎 Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Showing {len(filtered_integrated):,} filtered records."
    )

    st.dataframe(
        filtered_integrated,
        use_container_width=True,
        height=550
    )

    csv_data = filtered_integrated.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_ecommerce_data.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "E-Commerce People Analytics | MCA Major Project | "
    "Python • Pandas • NumPy • Plotly • Scikit-learn • Streamlit"
)

st.caption(
    "Analytical system based on authorized/demo e-commerce data. "
    "Designed for business intelligence and behavioural analysis."
)
