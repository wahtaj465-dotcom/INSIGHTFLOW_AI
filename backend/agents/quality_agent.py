"""
InsightFlow AI
Quality Intelligence Agent

Deterministic and explainable dataset quality assessment.

Quality dimensions:

1. Completeness
2. Uniqueness
3. Validity
4. Consistency
5. Anomaly Quality

Architecture:

Schema Intelligence
        ↓
Anomaly Intelligence
        ↓
Quality Intelligence

The Quality Agent does NOT independently detect anomalies.
It consumes the report produced by anomaly_agent.py.

No LLM is required.
"""

import math

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

QUALITY_WEIGHTS = {
    "completeness": 0.30,
    "uniqueness": 0.15,
    "validity": 0.20,
    "consistency": 0.15,
    "anomaly_quality": 0.20,
}


# ============================================================
# UTILITIES
# ============================================================

def _safe_round(value, digits=2):
    """
    Convert a numeric value safely to a rounded Python float.
    """

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return round(value, digits)

    except (TypeError, ValueError):
        return 0.0


def _clamp_score(value):
    """
    Keep quality scores between 0 and 100.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return _safe_round(
        max(0.0, min(100.0, value))
    )


def _get_semantic_type(schema, column):
    """
    Retrieve semantic type from Schema Intelligence.

    Supports both:

        semantic_type
        detected_type
    """

    if not isinstance(schema, dict):
        return None

    info = schema.get(column, {})

    if not isinstance(info, dict):
        return None

    semantic_type = (
        info.get("semantic_type")
        or info.get("detected_type")
    )

    if semantic_type is None:
        return None

    return str(semantic_type).strip().lower()


def _is_missing_like(series):
    """
    Detect real missing values and common string
    representations of missing data.
    """

    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):

        normalized = (
            series
            .astype("string")
            .str.strip()
            .str.lower()
        )

        missing_tokens = {
            "",
            "na",
            "n/a",
            "nan",
            "none",
            "null",
            "nil",
            "missing",
            "unknown",
            "?",
            "-",
        }

        mask = (
            mask
            | normalized.isin(missing_tokens)
        )

    return mask


def _is_identifier_type(semantic_type):
    return semantic_type in {
        "identifier",
        "id",
    }


def _get_anomaly_column_summary(
    anomaly_report,
    column,
):
    """
    Retrieve a column summary from Anomaly Intelligence.
    """

    if not isinstance(anomaly_report, dict):
        return {}

    summary = anomaly_report.get(
        "column_summary",
        {}
    )

    if not isinstance(summary, dict):
        return {}

    info = summary.get(
        column,
        {}
    )

    return (
        info
        if isinstance(info, dict)
        else {}
    )


# ============================================================
# 1. COMPLETENESS
# ============================================================

def calculate_completeness(df):
    """
    Measure the proportion of usable cells.

    Score:
        100 - missing cell percentage
    """

    total_cells = (
        df.shape[0]
        * df.shape[1]
    )

    if total_cells == 0:

        return {
            "score": 0.0,
            "missing_cells": 0,
            "total_cells": 0,
            "missing_percentage": 0.0,
        }

    missing_cells = 0

    column_results = {}

    for column in df.columns:

        missing_count = int(
            _is_missing_like(
                df[column]
            ).sum()
        )

        missing_percentage = (
            missing_count
            / len(df)
            * 100
            if len(df)
            else 0.0
        )

        column_results[column] = {
            "missing_values":
                missing_count,

            "missing_percentage":
                _safe_round(
                    missing_percentage
                ),

            "score":
                _clamp_score(
                    100
                    - missing_percentage
                ),
        }

        missing_cells += missing_count

    missing_percentage = (
        missing_cells
        / total_cells
        * 100
    )

    score = (
        100
        - missing_percentage
    )

    return {
        "score":
            _clamp_score(score),

        "missing_cells":
            int(missing_cells),

        "total_cells":
            int(total_cells),

        "missing_percentage":
            _safe_round(
                missing_percentage
            ),

        "columns":
            column_results,
    }


# ============================================================
# 2. UNIQUENESS
# ============================================================

def calculate_uniqueness(
    df,
    schema,
    anomaly_report=None,
):
    """
    Measure record uniqueness.

    Two concepts are considered:

    1. Complete duplicate rows
    2. Duplicate values in columns identified as IDs

    Repeated categorical values are NOT penalized.
    """

    total_rows = len(df)

    if total_rows == 0:

        return {
            "score": 0.0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
            "identifier_columns": {},
            "identifier_penalty_percentage": 0.0,
        }

    # --------------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_rows
        / total_rows
        * 100
    )

    # Prefer Anomaly Intelligence if available.

    if isinstance(anomaly_report, dict):

        duplicate_report = (
            anomaly_report.get(
                "duplicates",
                {}
            )
        )

        if isinstance(
            duplicate_report,
            dict
        ):

            duplicate_rows = int(
                duplicate_report.get(
                    "duplicate_rows",
                    duplicate_rows
                )
            )

            duplicate_percentage = float(
                duplicate_report.get(
                    "duplicate_percentage",
                    duplicate_percentage
                )
            )

    # --------------------------------------------------------
    # IDENTIFIER UNIQUENESS
    # --------------------------------------------------------

    identifier_results = {}

    identifier_duplicate_rates = []

    anomaly_identifier_columns = {}

    if isinstance(anomaly_report, dict):

        identifier_report = (
            anomaly_report.get(
                "identifier_anomalies",
                {}
            )
        )

        if isinstance(identifier_report, dict):

            anomaly_identifier_columns = (
                identifier_report.get(
                    "columns",
                    {}
                )
            )

    for column in df.columns:

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )

        if not _is_identifier_type(
            semantic_type
        ):
            continue

        missing_mask = (
            _is_missing_like(
                df[column]
            )
        )

        valid = (
            df.loc[
                ~missing_mask,
                column
            ]
        )

        valid_count = len(valid)

        duplicate_identifier_rows = 0

        anomaly_info = (
            anomaly_identifier_columns.get(
                column,
                {}
            )
            if isinstance(
                anomaly_identifier_columns,
                dict
            )
            else {}
        )

        if isinstance(
            anomaly_info,
            dict
        ) and anomaly_info:

            duplicate_identifier_rows = int(
                anomaly_info.get(
                    "duplicate_identifier_rows",
                    0
                )
            )

        elif valid_count:

            duplicate_identifier_rows = int(
                valid.duplicated(
                    keep=False
                ).sum()
            )

        duplicate_rate = (
            duplicate_identifier_rows
            / valid_count
            * 100
            if valid_count
            else 0.0
        )

        identifier_duplicate_rates.append(
            duplicate_rate
        )

        identifier_results[column] = {
            "values_checked":
                int(valid_count),

            "duplicate_identifier_rows":
                int(
                    duplicate_identifier_rows
                ),

            "duplicate_percentage":
                _safe_round(
                    duplicate_rate
                ),

            "score":
                _clamp_score(
                    100
                    - duplicate_rate
                ),
        }

    # --------------------------------------------------------
    # COMBINE ROW + ID UNIQUENESS
    # --------------------------------------------------------

    identifier_penalty = (
        sum(identifier_duplicate_rates)
        / len(identifier_duplicate_rates)
        if identifier_duplicate_rates
        else 0.0
    )

    row_uniqueness_score = (
        100
        - duplicate_percentage
    )

    if identifier_duplicate_rates:

        # Duplicate records and duplicate identifiers
        # both provide uniqueness evidence.

        score = (
            0.60
            * row_uniqueness_score
            +
            0.40
            * (
                100
                - identifier_penalty
            )
        )

    else:

        score = row_uniqueness_score

    return {
        "score":
            _clamp_score(score),

        "duplicate_rows":
            int(duplicate_rows),

        "duplicate_percentage":
            _safe_round(
                duplicate_percentage
            ),

        "identifier_penalty_percentage":
            _safe_round(
                identifier_penalty
            ),

        "identifier_columns":
            identifier_results,
    }


# ============================================================
# 3. VALIDITY
# ============================================================

def calculate_validity(
    df,
    schema,
):
    """
    Check whether non-missing values conform to their
    semantic type.
    """

    total_checked = 0
    invalid_count = 0

    column_results = {}

    boolean_values = {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "1",
        "0",
        "t",
        "f",
    }

    numeric_types = {
        "numerical",
        "numeric",
        "number",
        "integer",
        "float",
        "continuous",
        "discrete",
    }

    datetime_types = {
        "datetime",
        "date",
        "timestamp",
    }

    boolean_types = {
        "boolean",
        "bool",
    }

    identifier_types = {
        "identifier",
        "id",
    }

    for column in df.columns:

        series = df[column]

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )

        missing_mask = (
            _is_missing_like(series)
        )

        non_missing = (
            series[
                ~missing_mask
            ]
        )

        checked = int(
            len(non_missing)
        )

        invalid = 0

        if checked == 0:

            column_results[column] = {
                "semantic_type":
                    semantic_type,

                "checked_values":
                    0,

                "invalid_values":
                    0,

                "validity_score":
                    100.0,
            }

            continue

        # ----------------------------------------------------
        # NUMERIC
        # ----------------------------------------------------

        if semantic_type in numeric_types:

            converted = pd.to_numeric(
                non_missing,
                errors="coerce"
            )

            invalid = int(
                converted.isna().sum()
            )

        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        elif semantic_type in datetime_types:

            try:

                converted = pd.to_datetime(
                    non_missing,
                    errors="coerce",
                    format="mixed",
                )

            except (TypeError, ValueError):

                converted = pd.to_datetime(
                    non_missing,
                    errors="coerce",
                )

            invalid = int(
                converted.isna().sum()
            )

        # ----------------------------------------------------
        # BOOLEAN
        # ----------------------------------------------------

        elif semantic_type in boolean_types:

            normalized = (
                non_missing
                .astype(str)
                .str.strip()
                .str.lower()
            )

            invalid = int(
                (
                    ~normalized.isin(
                        boolean_values
                    )
                ).sum()
            )

        # ----------------------------------------------------
        # IDENTIFIER
        # ----------------------------------------------------

        elif semantic_type in identifier_types:

            normalized = (
                non_missing
                .astype(str)
                .str.strip()
            )

            invalid = int(
                (
                    normalized == ""
                ).sum()
            )

        # ----------------------------------------------------
        # TEXT / CATEGORY / UNKNOWN
        # ----------------------------------------------------

        else:

            normalized = (
                non_missing
                .astype(str)
                .str.strip()
            )

            invalid = int(
                (
                    normalized == ""
                ).sum()
            )

        valid_count = (
            checked
            - invalid
        )

        column_score = (
            valid_count
            / checked
            * 100
        )

        total_checked += checked
        invalid_count += invalid

        column_results[column] = {
            "semantic_type":
                semantic_type,

            "checked_values":
                checked,

            "invalid_values":
                invalid,

            "validity_score":
                _clamp_score(
                    column_score
                ),
        }

    if total_checked:

        overall_score = (
            (
                total_checked
                - invalid_count
            )
            / total_checked
            * 100
        )

    else:

        overall_score = 100.0

    return {
        "score":
            _clamp_score(
                overall_score
            ),

        "checked_values":
            int(total_checked),

        "invalid_values":
            int(invalid_count),

        "columns":
            column_results,
    }


# ============================================================
# 4. CONSISTENCY
# ============================================================

def calculate_consistency(
    df,
    schema,
):
    """
    Detect representation inconsistencies.

    Examples:

        Male
        male
        MALE
        " male "

    represent the same normalized category.
    """

    total_checked = 0
    inconsistent_values = 0

    column_results = {}

    categorical_types = {
        "categorical",
        "category",
        "boolean",
        "bool",
    }

    text_types = {
        "text",
        "string",
    }

    for column in df.columns:

        series = df[column]

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )

        missing_mask = (
            _is_missing_like(series)
        )

        non_missing = (
            series[
                ~missing_mask
            ]
        )

        checked = int(
            len(non_missing)
        )

        if checked == 0:
            continue

        inconsistencies = 0

        # ----------------------------------------------------
        # CATEGORY / BOOLEAN
        # ----------------------------------------------------

        if semantic_type in categorical_types:

            raw = (
                non_missing
                .astype(str)
            )

            normalized = (
                raw
                .str.strip()
                .str.lower()
            )

            comparison_df = pd.DataFrame({
                "raw": raw.values,
                "normalized":
                    normalized.values,
            })

            grouped = (
                comparison_df
                .groupby(
                    "normalized",
                    dropna=False
                )["raw"]
                .nunique()
            )

            inconsistent_groups = (
                grouped[
                    grouped > 1
                ].index
            )

            if len(inconsistent_groups):

                inconsistencies = int(
                    comparison_df[
                        "normalized"
                    ]
                    .isin(
                        inconsistent_groups
                    )
                    .sum()
                )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        elif semantic_type in text_types:

            raw = (
                non_missing
                .astype(str)
            )

            stripped = (
                raw.str.strip()
            )

            inconsistencies = int(
                (
                    raw != stripped
                ).sum()
            )

        total_checked += checked

        inconsistent_values += (
            inconsistencies
        )

        column_score = (
            (
                checked
                - inconsistencies
            )
            / checked
            * 100
        )

        column_results[column] = {
            "checked_values":
                checked,

            "inconsistent_values":
                int(inconsistencies),

            "consistency_score":
                _clamp_score(
                    column_score
                ),
        }

    if total_checked:

        overall_score = (
            (
                total_checked
                - inconsistent_values
            )
            / total_checked
            * 100
        )

    else:

        overall_score = 100.0

    return {
        "score":
            _clamp_score(
                overall_score
            ),

        "checked_values":
            int(total_checked),

        "inconsistent_values":
            int(inconsistent_values),

        "columns":
            column_results,
    }


# ============================================================
# 5. ANOMALY QUALITY
# ============================================================

def calculate_anomaly_quality(
    df,
    anomaly_report=None,
):
    """
    Convert Anomaly Intelligence output into a quality score.

    Important:

    This function does NOT detect anomalies.

    anomaly_agent.py owns anomaly detection.

    This function only evaluates how much anomaly evidence
    exists and converts it into an explainable quality
    component.
    """

    if not isinstance(
        anomaly_report,
        dict
    ):

        return {
            "score": 100.0,
            "source": "unavailable",
            "values_checked": 0,
            "outliers": 0,
            "extreme_outliers": 0,
            "outlier_percentage": 0.0,
            "suspicious_rows": 0,
            "high_risk_rows": 0,
            "columns": {},
        }

    numeric_report = (
        anomaly_report.get(
            "numeric_outliers",
            {}
        )
    )

    summary = (
        anomaly_report.get(
            "summary",
            {}
        )
    )

    column_summary = (
        anomaly_report.get(
            "column_summary",
            {}
        )
    )

    values_checked = int(
        numeric_report.get(
            "values_checked",
            0
        )
    )

    outliers = int(
        numeric_report.get(
            "outliers",
            0
        )
    )

    extreme_outliers = int(
        numeric_report.get(
            "extreme_outliers",
            0
        )
    )

    outlier_percentage = float(
        numeric_report.get(
            "outlier_percentage",
            0.0
        )
    )

    suspicious_rows = int(
        summary.get(
            "suspicious_rows",
            0
        )
    )

    high_risk_rows = int(
        summary.get(
            "high_risk_rows",
            0
        )
    )

    row_count = len(df)

    suspicious_row_percentage = (
        suspicious_rows
        / row_count
        * 100
        if row_count
        else 0.0
    )

    high_risk_percentage = (
        high_risk_rows
        / row_count
        * 100
        if row_count
        else 0.0
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------
    #
    # Ordinary outliers are weak evidence because legitimate
    # datasets can naturally contain them.
    #
    # Extreme outliers and high-risk rows receive stronger
    # penalties.
    # --------------------------------------------------------

    numeric_penalty = min(
        30.0,
        outlier_percentage * 0.60
    )

    extreme_percentage = (
        extreme_outliers
        / values_checked
        * 100
        if values_checked
        else 0.0
    )

    extreme_penalty = min(
        25.0,
        extreme_percentage * 1.50
    )

    suspicious_penalty = min(
        20.0,
        suspicious_row_percentage * 0.30
    )

    high_risk_penalty = min(
        25.0,
        high_risk_percentage * 1.00
    )

    total_penalty = (
        numeric_penalty
        + extreme_penalty
        + suspicious_penalty
        + high_risk_penalty
    )

    score = (
        100
        - total_penalty
    )

    columns = {}

    if isinstance(
        column_summary,
        dict
    ):

        for column, info in (
            column_summary.items()
        ):

            if not isinstance(
                info,
                dict
            ):
                continue

            outlier_rate = float(
                info.get(
                    "outlier_percentage",
                    0.0
                )
            )

            extreme_count = int(
                info.get(
                    "extreme_outliers",
                    0
                )
            )

            issue_count = int(
                info.get(
                    "issue_count",
                    0
                )
            )

            column_penalty = min(
                60.0,
                (
                    outlier_rate * 0.75
                    + extreme_count * 2.0
                    + issue_count * 2.0
                )
            )

            columns[column] = {
                "outliers":
                    int(
                        info.get(
                            "outliers",
                            0
                        )
                    ),

                "extreme_outliers":
                    extreme_count,

                "outlier_percentage":
                    _safe_round(
                        outlier_rate
                    ),

                "issue_count":
                    issue_count,

                "anomaly_quality_score":
                    _clamp_score(
                        100
                        - column_penalty
                    ),
            }

    return {
        "score":
            _clamp_score(score),

        "source":
            "anomaly_agent",

        "values_checked":
            values_checked,

        "outliers":
            outliers,

        "extreme_outliers":
            extreme_outliers,

        "outlier_percentage":
            _safe_round(
                outlier_percentage
            ),

        "suspicious_rows":
            suspicious_rows,

        "suspicious_row_percentage":
            _safe_round(
                suspicious_row_percentage
            ),

        "high_risk_rows":
            high_risk_rows,

        "high_risk_row_percentage":
            _safe_round(
                high_risk_percentage
            ),

        "penalties": {
            "numeric_outliers":
                _safe_round(
                    numeric_penalty
                ),

            "extreme_outliers":
                _safe_round(
                    extreme_penalty
                ),

            "suspicious_rows":
                _safe_round(
                    suspicious_penalty
                ),

            "high_risk_rows":
                _safe_round(
                    high_risk_penalty
                ),

            "total":
                _safe_round(
                    total_penalty
                ),
        },

        "columns":
            columns,
    }


# ============================================================
# COLUMN QUALITY
# ============================================================

def analyze_column_quality(
    df,
    schema,
    validity_report=None,
    consistency_report=None,
    anomaly_report=None,
):
    """
    Build explainable column-level quality information.
    """

    column_quality = {}

    anomaly_quality_columns = {}

    if isinstance(
        anomaly_report,
        dict
    ):

        anomaly_quality_columns = (
            anomaly_report.get(
                "columns",
                {}
            )
        )

    for column in df.columns:

        series = df[column]

        row_count = len(series)

        missing_count = int(
            _is_missing_like(
                series
            ).sum()
        )

        non_missing_count = (
            row_count
            - missing_count
        )

        missing_percentage = (
            missing_count
            / row_count
            * 100
            if row_count
            else 0.0
        )

        unique_values = int(
            series.nunique(
                dropna=True
            )
        )

        unique_ratio = (
            unique_values
            / non_missing_count
            if non_missing_count
            else 0.0
        )

        validity_info = (
            validity_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
            if isinstance(
                validity_report,
                dict
            )
            else {}
        )

        consistency_info = (
            consistency_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
            if isinstance(
                consistency_report,
                dict
            )
            else {}
        )

        anomaly_info = (
            anomaly_quality_columns.get(
                column,
                {}
            )
            if isinstance(
                anomaly_quality_columns,
                dict
            )
            else {}
        )

        column_quality[column] = {
            "semantic_type":
                _get_semantic_type(
                    schema,
                    column
                ),

            "rows":
                int(row_count),

            "missing_values":
                missing_count,

            "missing_percentage":
                _safe_round(
                    missing_percentage
                ),

            "completeness_score":
                _clamp_score(
                    100
                    - missing_percentage
                ),

            "unique_values":
                unique_values,

            "unique_ratio":
                _safe_round(
                    unique_ratio
                ),

            "validity_score":
                validity_info.get(
                    "validity_score",
                    100.0
                ),

            "consistency_score":
                consistency_info.get(
                    "consistency_score",
                    100.0
                ),

            "outliers":
                anomaly_info.get(
                    "outliers",
                    0
                ),

            "extreme_outliers":
                anomaly_info.get(
                    "extreme_outliers",
                    0
                ),

            "outlier_percentage":
                anomaly_info.get(
                    "outlier_percentage",
                    0.0
                ),

            "anomaly_quality_score":
                anomaly_info.get(
                    "anomaly_quality_score",
                    100.0
                ),
        }

    return column_quality


# ============================================================
# ISSUE GENERATION
# ============================================================

def generate_quality_issues(
    column_quality,
    uniqueness_report,
):
    """
    Convert quality metrics into explainable issues.
    """

    issues = []

    # --------------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_rows = int(
        uniqueness_report.get(
            "duplicate_rows",
            0
        )
    )

    duplicate_percentage = float(
        uniqueness_report.get(
            "duplicate_percentage",
            0.0
        )
    )

    if duplicate_rows > 0:

        issues.append({
            "type":
                "duplicate_rows",

            "severity":
                (
                    "high"
                    if duplicate_percentage >= 10
                    else "medium"
                ),

            "message":
                (
                    f"{duplicate_rows} duplicate "
                    "row(s) were detected "
                    f"({duplicate_percentage:.2f}%)."
                ),

            "count":
                duplicate_rows,
        })

    # --------------------------------------------------------
    # IDENTIFIER DUPLICATES
    # --------------------------------------------------------

    identifier_columns = (
        uniqueness_report.get(
            "identifier_columns",
            {}
        )
    )

    if isinstance(
        identifier_columns,
        dict
    ):

        for column, info in (
            identifier_columns.items()
        ):

            duplicate_ids = int(
                info.get(
                    "duplicate_identifier_rows",
                    0
                )
            )

            duplicate_rate = float(
                info.get(
                    "duplicate_percentage",
                    0.0
                )
            )

            if duplicate_ids > 0:

                issues.append({
                    "type":
                        "duplicate_identifiers",

                    "column":
                        column,

                    "severity":
                        (
                            "high"
                            if duplicate_rate >= 10
                            else "medium"
                        ),

                    "message":
                        (
                            f"Identifier column "
                            f"'{column}' contains "
                            f"{duplicate_ids} rows "
                            "with duplicated identifier "
                            f"values ({duplicate_rate:.2f}%)."
                        ),
                })

    # --------------------------------------------------------
    # COLUMN ISSUES
    # --------------------------------------------------------

    for column, info in (
        column_quality.items()
    ):

        missing_percentage = float(
            info.get(
                "missing_percentage",
                0.0
            )
        )

        if missing_percentage >= 50:

            issues.append({
                "type":
                    "high_missingness",

                "column":
                    column,

                "severity":
                    "high",

                "message":
                    (
                        f"Column '{column}' has "
                        f"{missing_percentage:.2f}% "
                        "missing values."
                    ),
            })

        elif missing_percentage >= 20:

            issues.append({
                "type":
                    "moderate_missingness",

                "column":
                    column,

                "severity":
                    "medium",

                "message":
                    (
                        f"Column '{column}' has "
                        f"{missing_percentage:.2f}% "
                        "missing values."
                    ),
            })

        validity_score = float(
            info.get(
                "validity_score",
                100.0
            )
        )

        if validity_score < 90:

            issues.append({
                "type":
                    "invalid_values",

                "column":
                    column,

                "severity":
                    (
                        "high"
                        if validity_score < 70
                        else "medium"
                    ),

                "message":
                    (
                        f"Column '{column}' has a "
                        "validity score of "
                        f"{validity_score:.2f}/100."
                    ),
            })

        consistency_score = float(
            info.get(
                "consistency_score",
                100.0
            )
        )

        if consistency_score < 90:

            issues.append({
                "type":
                    "inconsistent_values",

                "column":
                    column,

                "severity":
                    (
                        "high"
                        if consistency_score < 70
                        else "medium"
                    ),

                "message":
                    (
                        f"Column '{column}' has a "
                        "consistency score of "
                        f"{consistency_score:.2f}/100."
                    ),
            })

        outlier_percentage = float(
            info.get(
                "outlier_percentage",
                0.0
            )
        )

        extreme_outliers = int(
            info.get(
                "extreme_outliers",
                0
            )
        )

        if extreme_outliers > 0:

            issues.append({
                "type":
                    "extreme_outliers",

                "column":
                    column,

                "severity":
                    "high",

                "message":
                    (
                        f"Column '{column}' contains "
                        f"{extreme_outliers} extreme "
                        "statistical outlier(s)."
                    ),
            })

        elif outlier_percentage >= 10:

            issues.append({
                "type":
                    "high_outlier_rate",

                "column":
                    column,

                "severity":
                    (
                        "high"
                        if outlier_percentage >= 25
                        else "medium"
                    ),

                "message":
                    (
                        f"Column '{column}' contains "
                        f"{outlier_percentage:.2f}% "
                        "potential statistical outliers."
                    ),
            })

    return issues


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    quality_report,
):
    """
    Calculate weighted overall quality score.
    """

    if not isinstance(
        quality_report,
        dict
    ):

        raise TypeError(
            "quality_report must be a dictionary."
        )

    components = (
        quality_report.get(
            "components",
            quality_report
        )
    )

    score = 0.0
    used_weight = 0.0

    for component, weight in (
        QUALITY_WEIGHTS.items()
    ):

        component_data = (
            components.get(component)
        )

        if component_data is None:
            continue

        if isinstance(
            component_data,
            dict
        ):

            component_score = (
                component_data.get(
                    "score"
                )
            )

        else:

            component_score = (
                component_data
            )

        if component_score is None:
            continue

        try:

            component_score = float(
                component_score
            )

        except (TypeError, ValueError):
            continue

        score += (
            component_score
            * weight
        )

        used_weight += weight

    if used_weight == 0:
        return 0.0

    return _clamp_score(
        score / used_weight
    )


# ============================================================
# QUALITY RATING
# ============================================================

def get_quality_rating(score):
    """
    Convert quality score into a human-readable rating.
    """

    score = float(score)

    if score >= 95:
        return "Excellent"

    if score >= 85:
        return "Very Good"

    if score >= 75:
        return "Good"

    if score >= 60:
        return "Fair"

    if score >= 40:
        return "Poor"

    return "Critical"


# ============================================================
# COMPONENT EXPLANATIONS
# ============================================================

def generate_component_explanations(
    components,
):
    """
    Produce deterministic explanations for each
    quality component.
    """

    explanations = {}

    completeness = components.get(
        "completeness",
        {}
    )

    missing_percentage = float(
        completeness.get(
            "missing_percentage",
            0.0
        )
    )

    if missing_percentage == 0:

        explanations["completeness"] = (
            "No missing or missing-like cells were detected."
        )

    else:

        explanations["completeness"] = (
            f"{missing_percentage:.2f}% of dataset cells "
            "are missing or contain missing-value markers."
        )

    uniqueness = components.get(
        "uniqueness",
        {}
    )

    duplicate_rows = int(
        uniqueness.get(
            "duplicate_rows",
            0
        )
    )

    identifier_penalty = float(
        uniqueness.get(
            "identifier_penalty_percentage",
            0.0
        )
    )

    if (
        duplicate_rows == 0
        and identifier_penalty == 0
    ):

        explanations["uniqueness"] = (
            "No duplicate records or duplicate identifier "
            "problems were detected."
        )

    else:

        explanations["uniqueness"] = (
            f"{duplicate_rows} duplicate record(s) were "
            "detected. Average duplicate identifier rate "
            f"is {identifier_penalty:.2f}%."
        )

    validity = components.get(
        "validity",
        {}
    )

    invalid_values = int(
        validity.get(
            "invalid_values",
            0
        )
    )

    explanations["validity"] = (
        (
            "All checked non-missing values conform to "
            "their detected semantic types."
        )
        if invalid_values == 0
        else (
            f"{invalid_values} checked value(s) do not "
            "conform to their detected semantic type."
        )
    )

    consistency = components.get(
        "consistency",
        {}
    )

    inconsistent_values = int(
        consistency.get(
            "inconsistent_values",
            0
        )
    )

    explanations["consistency"] = (
        (
            "No representation inconsistencies were "
            "detected in the checked columns."
        )
        if inconsistent_values == 0
        else (
            f"{inconsistent_values} value(s) have "
            "representation or formatting inconsistencies."
        )
    )

    anomaly_quality = components.get(
        "anomaly_quality",
        {}
    )

    outliers = int(
        anomaly_quality.get(
            "outliers",
            0
        )
    )

    extreme = int(
        anomaly_quality.get(
            "extreme_outliers",
            0
        )
    )

    suspicious_rows = int(
        anomaly_quality.get(
            "suspicious_rows",
            0
        )
    )

    if (
        outliers == 0
        and extreme == 0
        and suspicious_rows == 0
    ):

        explanations["anomaly_quality"] = (
            "Anomaly Intelligence found no significant "
            "numeric or row-level anomaly evidence."
        )

    else:

        explanations["anomaly_quality"] = (
            f"Anomaly Intelligence found {outliers} "
            f"numeric outlier(s), {extreme} extreme "
            f"outlier(s), and {suspicious_rows} "
            "suspicious row(s)."
        )

    return explanations


# ============================================================
# PRIORITY FINDINGS
# ============================================================

def generate_priority_findings(
    issues,
    limit=10,
):
    """
    Return the most important quality findings first.
    """

    severity_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    sorted_issues = sorted(
        issues,
        key=lambda issue: (
            severity_rank.get(
                issue.get(
                    "severity",
                    "low"
                ),
                3
            ),
            str(
                issue.get(
                    "column",
                    ""
                )
            ),
        )
    )

    return sorted_issues[:limit]


# ============================================================
# MAIN QUALITY ANALYSIS
# ============================================================

def analyze_data_quality(
    df,
    schema=None,
    anomaly_report=None,
):
    """
    Run complete deterministic Quality Intelligence.

    Recommended call:

        anomalies = detect_anomalies(df, schema)

        quality = analyze_data_quality(
            df=df,
            schema=schema,
            anomaly_report=anomalies
        )

    anomaly_report remains optional for backward
    compatibility, but the full pipeline should provide it.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Cannot analyze quality of an empty dataset."
        )

    if schema is None:
        schema = {}

    # ========================================================
    # COMPONENTS
    # ========================================================

    completeness = (
        calculate_completeness(df)
    )

    uniqueness = (
        calculate_uniqueness(
            df=df,
            schema=schema,
            anomaly_report=anomaly_report,
        )
    )

    validity = (
        calculate_validity(
            df=df,
            schema=schema,
        )
    )

    consistency = (
        calculate_consistency(
            df=df,
            schema=schema,
        )
    )

    anomaly_quality = (
        calculate_anomaly_quality(
            df=df,
            anomaly_report=anomaly_report,
        )
    )

    components = {
        "completeness":
            completeness,

        "uniqueness":
            uniqueness,

        "validity":
            validity,

        "consistency":
            consistency,

        "anomaly_quality":
            anomaly_quality,
    }

    # ========================================================
    # COLUMN QUALITY
    # ========================================================

    column_quality = (
        analyze_column_quality(
            df=df,
            schema=schema,
            validity_report=validity,
            consistency_report=consistency,
            anomaly_report=anomaly_quality,
        )
    )

    # ========================================================
    # ISSUES
    # ========================================================

    issues = (
        generate_quality_issues(
            column_quality=column_quality,
            uniqueness_report=uniqueness,
        )
    )

    # ========================================================
    # OVERALL SCORE
    # ========================================================

    overall_score = (
        calculate_quality_score({
            "components":
                components
        })
    )

    rating = (
        get_quality_rating(
            overall_score
        )
    )

    # ========================================================
    # COMPONENT SCORES
    # ========================================================

    component_scores = {
        name:
            _safe_round(
                data.get(
                    "score",
                    0
                )
            )

        for name, data
        in components.items()
    }

    # ========================================================
    # EXPLANATIONS
    # ========================================================

    explanations = (
        generate_component_explanations(
            components
        )
    )

    priority_findings = (
        generate_priority_findings(
            issues
        )
    )

    # ========================================================
    # ISSUE COUNTS
    # ========================================================

    high_issues = sum(
        1
        for issue in issues
        if issue.get("severity") == "high"
    )

    medium_issues = sum(
        1
        for issue in issues
        if issue.get("severity") == "medium"
    )

    low_issues = sum(
        1
        for issue in issues
        if issue.get("severity") == "low"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    return {
        "overall_score":
            overall_score,

        # Backward compatibility with existing workflow/UI.
        "quality_score":
            overall_score,

        "rating":
            rating,

        "component_scores":
            component_scores,

        "component_explanations":
            explanations,

        "components":
            components,

        "column_quality":
            column_quality,

        "issues":
            issues,

        "priority_findings":
            priority_findings,

        "summary": {
            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "rating":
                rating,

            "high_severity_issues":
                int(high_issues),

            "medium_severity_issues":
                int(medium_issues),

            "low_severity_issues":
                int(low_issues),

            "total_issues":
                int(len(issues)),

            "anomaly_intelligence_used":
                isinstance(
                    anomaly_report,
                    dict
                ),
        },
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

def analyze_quality(
    df,
    schema=None,
    anomaly_report=None,
):
    """
    Alias for analyze_data_quality().
    """

    return analyze_data_quality(
        df=df,
        schema=schema,
        anomaly_report=anomaly_report,
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_quality_report(report):
    """
    Print compact Quality Intelligence results.
    """

    print(
        "\n================================"
    )

    print(
        "       DATA QUALITY REPORT"
    )

    print(
        "================================"
    )

    score = report.get(
        "overall_score",
        0
    )

    rating = report.get(
        "rating",
        "Unknown"
    )

    print(
        f"\nOverall Quality Score: "
        f"{score:.2f}/100"
    )

    print(
        f"Rating: {rating}"
    )

    print(
        "\nComponent Scores:"
    )

    component_scores = (
        report.get(
            "component_scores",
            {}
        )
    )

    explanations = (
        report.get(
            "component_explanations",
            {}
        )
    )

    for component, component_score in (
        component_scores.items()
    ):

        print(
            f"\n  {component}: "
            f"{component_score:.2f}/100"
        )

        explanation = (
            explanations.get(
                component
            )
        )

        if explanation:

            print(
                f"    {explanation}"
            )

    issues = report.get(
        "issues",
        []
    )

    print(
        f"\nDetected Issues: "
        f"{len(issues)}"
    )

    for issue in issues[:20]:

        print(
            " - "
            f"[{issue.get('severity', 'unknown').upper()}] "
            f"{issue.get('message', '')}"
        )