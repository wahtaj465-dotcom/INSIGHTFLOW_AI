"""
InsightFlow AI - Power BI Style Dashboard Generator

This file is intentionally responsible for:
1. Fetching the prepared dataset from FastAPI.
2. Converting the response into a DataFrame.
3. Inferring useful KPI/filter/chart fields from the dataset.
4. Applying dashboard slicers.
5. Building Plotly figures for the dashboard page.

The Streamlit page should only call the functions in this file and
render the returned objects. Existing agent/report functionality is
not modified by this module.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import api


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILTER_VALUES = 30
MAX_CHART_CATEGORIES = 12
MAX_SCATTER_ROWS = 3000


# ============================================================
# TEXT / COLUMN HELPERS
# ============================================================

def _pretty_name(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.title()


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        series = df[column]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            unique_count = series.nunique(dropna=True)
            if 2 <= unique_count <= MAX_FILTER_VALUES:
                columns.append(str(column))
    return columns


def _date_columns(df: pd.DataFrame) -> list[str]:
    dates = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            dates.append(str(column))
            continue

        if not pd.api.types.is_object_dtype(series):
            continue

        sample = series.dropna().astype(str).head(200)

        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.80:
            dates.append(str(column))

    return dates


def _business_score(column: str) -> int:
    name = _normalise(column)

    score = 0

    semantic_groups = {
        "revenue sales amount": 10,
        "profit margin": 9,
        "rating score": 8,
        "vote review review count engagement": 8,
        "price cost revenue": 7,
        "quantity units orders": 7,
        "count total": 5,
    }

    for keywords, weight in semantic_groups.items():
        if any(token in name.split() for token in keywords.split()):
            score += weight

    # Technical/location identifiers should not dominate dashboards.
    if any(token in name.split() for token in (
        "id", "latitude", "longitude", "code", "zipcode"
    )):
        score -= 12

    return score


def _rank_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric = _safe_numeric_columns(df)

    ranked = sorted(
        numeric,
        key=lambda column: (
            _business_score(column),
            -df[column].nunique(dropna=True),
        ),
        reverse=True,
    )

    return ranked


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dashboard_dataframe(dataset_id: str) -> pd.DataFrame:
    """
    Fetch the already-prepared/cleaned dataset from FastAPI.

    The backend endpoint is deliberately separate from the agent
    endpoint. No LLM call is made here.
    """

    payload = api.get_dashboard_data(dataset_id)

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid dashboard-data response.")

    records = payload.get("records", [])

    if not isinstance(records, list):
        raise RuntimeError("Dashboard records are not a list.")

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("The prepared dataset is empty.")

    # Convert date-like columns when enough values parse successfully.
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.80:
                df[column] = parsed

    return df


# ============================================================
# KPI INFERENCE
# ============================================================

def _kpi(
    label: str,
    value: Any,
    suffix: str = "",
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "suffix": suffix,
    }


def build_kpis(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build four business-oriented KPI cards.

    The selection is dataset-aware rather than hard-coded to Zomato.
    """

    numeric = _rank_numeric_columns(df)
    kpis: list[dict[str, Any]] = []

    # KPI 1: records
    kpis.append(
        _kpi("Records", f"{len(df):,}")
    )

    # Prefer semantically meaningful numeric fields.
    selected: list[str] = []

    for column in numeric:
        if column not in selected:
            selected.append(column)
        if len(selected) >= 3:
            break

    for column in selected:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        name = _normalise(column)

        if any(token in name.split() for token in ("rating", "score")):
            value = round(float(series.mean()), 2)
            kpis.append(_kpi(f"Avg {_pretty_name(column)}", value))
        elif any(token in name.split() for token in ("vote", "review", "engagement")):
            value = round(float(series.mean()), 2)
            kpis.append(_kpi(f"Avg {_pretty_name(column)}", value))
        elif any(token in name.split() for token in ("revenue", "sales", "profit", "amount")):
            value = float(series.sum())
            kpis.append(_kpi(f"Total {_pretty_name(column)}", f"{value:,.0f}"))
        else:
            value = round(float(series.mean()), 2)
            kpis.append(_kpi(f"Avg {_pretty_name(column)}", value))

        if len(kpis) >= 4:
            break

    # Guarantee four cards for a consistent executive layout.
    if len(kpis) < 4:
        kpis.append(_kpi("Columns", f"{len(df.columns):,}"))

    return kpis[:4]


# ============================================================
# FILTER / SLICER INFERENCE
# ============================================================

def build_filter_config(df: pd.DataFrame) -> dict[str, Any]:
    categorical = _categorical_columns(df)
    dates = _date_columns(df)
    numeric = _rank_numeric_columns(df)

    filters = {
        "categorical": categorical[:6],
        "date": dates[:2],
        "numeric": numeric[:2],
    }

    return filters


def apply_filters(
    df: pd.DataFrame,
    selections: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the selections produced by the dashboard slicers.
    """

    filtered = df.copy()

    for column, selected_values in selections.get("categorical", {}).items():
        if (
            column in filtered.columns
            and selected_values
            and "__ALL__" not in selected_values
        ):
            filtered = filtered[
                filtered[column].astype(str).isin(
                    [str(v) for v in selected_values]
                )
            ]

    for column, bounds in selections.get("numeric", {}).items():
        if column not in filtered.columns:
            continue

        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue

        numeric = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[
            numeric.between(float(bounds[0]), float(bounds[1]), inclusive="both")
        ]

    for column, bounds in selections.get("date", {}).items():
        if column not in filtered.columns:
            continue

        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue

        dates = pd.to_datetime(filtered[column], errors="coerce")
        start = pd.Timestamp(bounds[0])
        end = pd.Timestamp(bounds[1])

        filtered = filtered[
            dates.between(start, end, inclusive="both")
        ]

    return filtered


# ============================================================
# CHART BUILDING
# ============================================================

def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=330)
    return figure


def _base_layout(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=55, b=35),
        legend_title_text="",
        hovermode="x unified",
    )
    return figure


def _best_category(df: pd.DataFrame) -> str | None:
    candidates = _categorical_columns(df)

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda c: (
            _business_score(c),
            -df[c].nunique(dropna=True),
        ),
        reverse=True,
    )[0]


def _best_numeric(df: pd.DataFrame) -> str | None:
    ranked = _rank_numeric_columns(df)
    return ranked[0] if ranked else None


def _second_numeric(df: pd.DataFrame, first: str | None) -> str | None:
    for column in _rank_numeric_columns(df):
        if column != first:
            return column
    return None


def build_chart_figures(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate a small set of high-value, interactive dashboard charts.
    """

    if df.empty:
        return []

    charts: list[dict[str, Any]] = []

    categorical = _best_category(df)
    numeric = _best_numeric(df)
    numeric2 = _second_numeric(df, numeric)
    dates = _date_columns(df)

    # --------------------------------------------------------
    # 1. Category volume
    # --------------------------------------------------------

    if categorical:
        grouped = (
            df[categorical]
            .astype(str)
            .value_counts()
            .head(MAX_CHART_CATEGORIES)
            .sort_values()
        )

        chart_df = grouped.reset_index()
        chart_df.columns = [categorical, "Count"]

        figure = px.bar(
            chart_df,
            x="Count",
            y=categorical,
            orientation="h",
            title=f"Records by {_pretty_name(categorical)}",
        )

        charts.append(
            {
                "title": f"Records by {_pretty_name(categorical)}",
                "figure": _base_layout(figure),
            }
        )

    # --------------------------------------------------------
    # 2. Average business metric by category
    # --------------------------------------------------------

    if categorical and numeric:
        grouped = (
            df.assign(
                __metric=pd.to_numeric(
                    df[numeric],
                    errors="coerce",
                )
            )
            .groupby(categorical, dropna=False)["__metric"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(MAX_CHART_CATEGORIES)
            .reset_index()
        )

        if not grouped.empty:
            grouped.columns = [categorical, "Value"]

            figure = px.bar(
                grouped,
                x=categorical,
                y="Value",
                title=(
                    f"Average {_pretty_name(numeric)} "
                    f"by {_pretty_name(categorical)}"
                ),
            )

            charts.append(
                {
                    "title": (
                        f"Average {_pretty_name(numeric)} "
                        f"by {_pretty_name(categorical)}"
                    ),
                    "figure": _base_layout(figure),
                }
            )

    # --------------------------------------------------------
    # 3. Distribution
    # --------------------------------------------------------

    if numeric:
        series = pd.to_numeric(
            df[numeric],
            errors="coerce",
        ).dropna()

        if not series.empty:
            figure = px.histogram(
                series,
                x=series,
                nbins=30,
                title=f"Distribution of {_pretty_name(numeric)}",
            )

            charts.append(
                {
                    "title": f"Distribution of {_pretty_name(numeric)}",
                    "figure": _base_layout(figure),
                }
            )

    # --------------------------------------------------------
    # 4. Numeric relationship
    # --------------------------------------------------------

    if numeric and numeric2:
        chart_df = df[[numeric, numeric2]].copy()
        chart_df[numeric] = pd.to_numeric(
            chart_df[numeric],
            errors="coerce",
        )
        chart_df[numeric2] = pd.to_numeric(
            chart_df[numeric2],
            errors="coerce",
        )
        chart_df = chart_df.dropna().head(MAX_SCATTER_ROWS)

        if len(chart_df) >= 5:
            figure = px.scatter(
                chart_df,
                x=numeric,
                y=numeric2,
                title=(
                    f"{_pretty_name(numeric2)} vs "
                    f"{_pretty_name(numeric)}"
                ),
                opacity=0.70,
            )

            charts.append(
                {
                    "title": (
                        f"{_pretty_name(numeric2)} vs "
                        f"{_pretty_name(numeric)}"
                    ),
                    "figure": _base_layout(figure),
                }
            )

    # --------------------------------------------------------
    # 5. Time trend
    # --------------------------------------------------------

    if dates and numeric:
        date_column = dates[0]

        chart_df = df[[date_column, numeric]].copy()
        chart_df[date_column] = pd.to_datetime(
            chart_df[date_column],
            errors="coerce",
        )
        chart_df[numeric] = pd.to_numeric(
            chart_df[numeric],
            errors="coerce",
        )
        chart_df = chart_df.dropna()

        if not chart_df.empty:
            chart_df["__period"] = (
                chart_df[date_column]
                .dt.to_period("M")
                .dt.to_timestamp()
            )

            trend = (
                chart_df
                .groupby("__period")[numeric]
                .mean()
                .reset_index()
            )

            figure = px.line(
                trend,
                x="__period",
                y=numeric,
                markers=True,
                title=(
                    f"Average {_pretty_name(numeric)} over time"
                ),
            )

            charts.append(
                {
                    "title": (
                        f"Average {_pretty_name(numeric)} over time"
                    ),
                    "figure": _base_layout(figure),
                }
            )

    return charts[:6]


# ============================================================
# DASHBOARD MODEL
# ============================================================

def build_dashboard(dataset_id: str) -> dict[str, Any]:
    """
    Main entry point used by dashboard.py.

    All dataset fetching and analytical preparation lives here.
    """

    df = fetch_dashboard_dataframe(dataset_id)

    return {
        "dataframe": df,
        "kpis": build_kpis(df),
        "filters": build_filter_config(df),
        "numeric_columns": _rank_numeric_columns(df),
        "categorical_columns": _categorical_columns(df),
        "date_columns": _date_columns(df),
    }


def refresh_dashboard_cache() -> None:
    """Clear cached dashboard data after a dataset is replaced."""
    fetch_dashboard_dataframe.clear()
