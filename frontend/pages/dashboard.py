"""
InsightFlow AI - Power BI Style Dashboard Page

This page is intentionally thin:
- Reads the active dataset ID from Streamlit session state.
- Calls dashboard_generator for data fetching/processing.
- Renders slicers, KPI cards and interactive charts.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.utils.dashboard_generator import (
    apply_filters,
    build_dashboard,
    build_chart_figures,
)


st.set_page_config(
    page_title="InsightFlow Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .dashboard-title {
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .dashboard-subtitle {
        color: #8f98a8;
        margin-bottom: 1.25rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #151923, #1d2330);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 112px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.16);
    }

    .kpi-label {
        color: #9ca6b7;
        font-size: 0.82rem;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #f7f9fc;
        font-size: 1.75rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .filter-panel {
        background: #151923;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION / DATASET CHECK
# ============================================================

dataset_id = (
    st.session_state.get("dashboard_dataset_id")
    or st.session_state.get("dataset_id")
)

if not dataset_id:
    st.warning("No active dataset is available.")
    if st.button("← Back to InsightFlow", width="stretch"):
        st.switch_page("app.py")
    st.stop()


# ============================================================
# HEADER
# ============================================================

top_left, top_right = st.columns([5, 1])

with top_left:
    st.markdown(
        '<div class="dashboard-title">InsightFlow AI Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-subtitle">'
        "Interactive business dashboard generated from the prepared dataset"
        "</div>",
        unsafe_allow_html=True,
    )

with top_right:
    if st.button("← Back", width="stretch"):
        st.switch_page("app.py")


# ============================================================
# BUILD DASHBOARD MODEL
# ============================================================

try:
    with st.spinner("Building dashboard from prepared data..."):
        dashboard = build_dashboard(dataset_id)

except Exception as error:
    st.error(f"Could not build dashboard: {error}")
    st.stop()


df = dashboard["dataframe"]
filter_config = dashboard["filters"]


# ============================================================
# SIDEBAR SLICERS
# ============================================================

st.sidebar.header("Dashboard Filters")

categorical_selections = {}
numeric_selections = {}
date_selections = {}


for column in filter_config["categorical"]:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not values:
        continue

    selected = st.sidebar.multiselect(
        column.replace("_", " ").title(),
        options=["__ALL__"] + values,
        default=["__ALL__"],
        key=f"dashboard_filter_cat_{column}",
    )

    categorical_selections[column] = selected


for column in filter_config["numeric"]:
    numeric = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if numeric.empty:
        continue

    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if minimum == maximum:
        continue

    selected_range = st.sidebar.slider(
        column.replace("_", " ").title(),
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        key=f"dashboard_filter_num_{column}",
    )

    numeric_selections[column] = selected_range


for column in filter_config["date"]:
    dates = pd.to_datetime(
        df[column],
        errors="coerce",
    ).dropna()

    if dates.empty:
        continue

    minimum = dates.min().date()
    maximum = dates.max().date()

    if minimum == maximum:
        continue

    selected_range = st.sidebar.date_input(
        column.replace("_", " ").title(),
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
        key=f"dashboard_filter_date_{column}",
    )

    if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
        date_selections[column] = selected_range


selections = {
    "categorical": categorical_selections,
    "numeric": numeric_selections,
    "date": date_selections,
}

filtered_df = apply_filters(
    df,
    selections,
)


# ============================================================
# KPI CARDS
# ============================================================

# Rebuild KPI cards from the filtered data so slicers affect them.
filtered_dashboard = {
    **dashboard,
    "dataframe": filtered_df,
}

from frontend.utils.dashboard_generator import build_kpis

kpis = build_kpis(filtered_df)

kpi_columns = st.columns(4)

for index, card in enumerate(kpis):
    with kpi_columns[index]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{card["label"]}</div>
                <div class="kpi-value">{card["value"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="section-title">Business Overview</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} records."
)


# ============================================================
# CHARTS
# ============================================================

charts = build_chart_figures(filtered_df)

if not charts:
    st.info(
        "There are not enough suitable fields to generate dashboard charts."
    )
else:
    for index in range(0, len(charts), 2):
        left, right = st.columns(2)

        with left:
            st.markdown(
                f"**{charts[index]['title']}**"
            )
            st.plotly_chart(
                charts[index]["figure"],
                width="stretch",
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
            )

        if index + 1 < len(charts):
            with right:
                st.markdown(
                    f"**{charts[index + 1]['title']}**"
                )
                st.plotly_chart(
                    charts[index + 1]["figure"],
                    width="stretch",
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Dashboard Data Preview"):
    st.dataframe(
        filtered_df.head(500),
        width="stretch",
        hide_index=True,
    )
