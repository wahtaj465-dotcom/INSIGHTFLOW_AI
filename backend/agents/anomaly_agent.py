"""
InsightFlow AI
Anomaly Intelligence Agent

Deterministic anomaly detection for structured datasets.

Detects:

1. Numeric outliers
2. Extreme numeric outliers
3. Rare categorical values
4. Identifier anomalies
5. Datetime anomalies
6. Missingness anomalies
7. Constant / near-constant columns
8. Duplicate rows
9. Suspicious rows
10. Column-level anomaly summaries

No LLM is required.
"""

import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IQR_MULTIPLIER = 1.5
EXTREME_IQR_MULTIPLIER = 3.0

RARE_CATEGORY_RATIO = 0.01
RARE_CATEGORY_MIN_COUNT = 5

HIGH_MISSING_THRESHOLD = 0.50
MODERATE_MISSING_THRESHOLD = 0.20

NEAR_CONSTANT_THRESHOLD = 0.95

MIN_NUMERIC_VALUES = 4


# ============================================================
# UTILITIES
# ============================================================

def _safe_round(value, digits=2):
    """
    Convert numeric values safely to Python floats.
    """

    try:

        value = float(value)

        if math.isnan(value):
            return 0.0

        if math.isinf(value):
            return 0.0

        return round(
            value,
            digits
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


def _get_semantic_type(
    schema,
    column
):
    """
    Retrieve semantic type from schema metadata.

    Supports both:

        semantic_type
        detected_type
    """

    if not isinstance(
        schema,
        dict
    ):
        return None

    info = schema.get(
        column,
        {}
    )

    if not isinstance(
        info,
        dict
    ):
        return None

    semantic_type = (
        info.get(
            "semantic_type"
        )
        or
        info.get(
            "detected_type"
        )
    )

    if semantic_type is None:
        return None

    return str(
        semantic_type
    ).strip().lower()


def _missing_mask(series):
    """
    Detect actual missing values plus common
    string representations of missing data.
    """

    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series)
        or
        pd.api.types.is_string_dtype(series)
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
            "-"
        }

        mask = (
            mask
            |
            normalized.isin(
                missing_tokens
            )
        )

    return mask


def _is_numeric_type(
    semantic_type
):
    """
    Check whether semantic type is numeric.
    """

    return semantic_type in {
        "numerical",
        "numeric",
        "number",
        "integer",
        "float",
        "continuous",
        "discrete"
    }


def _is_categorical_type(
    semantic_type
):
    """
    Check whether semantic type is categorical.
    """

    return semantic_type in {
        "categorical",
        "category",
        "boolean",
        "bool"
    }


def _is_identifier_type(
    semantic_type
):
    """
    Check whether semantic type represents an identifier.
    """

    return semantic_type in {
        "identifier",
        "id"
    }


def _is_datetime_type(
    semantic_type
):
    """
    Check whether semantic type represents datetime data.
    """

    return semantic_type in {
        "datetime",
        "date",
        "timestamp"
    }


# ============================================================
# 1. NUMERIC OUTLIER DETECTION
# ============================================================

def detect_numeric_outliers(
    df,
    schema
):
    """
    Detect numeric outliers using IQR.

    Two levels are reported:

    Standard outlier:
        Q1 - 1.5*IQR
        Q3 + 1.5*IQR

    Extreme outlier:
        Q1 - 3*IQR
        Q3 + 3*IQR
    """

    results = {}

    total_outliers = 0
    total_extreme = 0
    total_values = 0

    row_scores = {}


    for column in df.columns:

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )

        if not _is_numeric_type(
            semantic_type
        ):
            continue


        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )


        valid = numeric.dropna()


        if len(valid) < MIN_NUMERIC_VALUES:

            continue


        q1 = valid.quantile(
            0.25
        )

        q3 = valid.quantile(
            0.75
        )

        median = valid.median()

        iqr = (
            q3
            -
            q1
        )


        if (
            pd.isna(iqr)
            or
            iqr == 0
        ):

            results[column] = {

                "values_checked":
                    int(len(valid)),

                "outliers":
                    0,

                "extreme_outliers":
                    0,

                "outlier_percentage":
                    0.0,

                "q1":
                    _safe_round(q1),

                "median":
                    _safe_round(median),

                "q3":
                    _safe_round(q3),

                "iqr":
                    _safe_round(iqr),

                "lower_bound":
                    _safe_round(q1),

                "upper_bound":
                    _safe_round(q3),

                "extreme_lower_bound":
                    _safe_round(q1),

                "extreme_upper_bound":
                    _safe_round(q3),

                "outlier_indices":
                    [],

                "extreme_outlier_indices":
                    []
            }

            total_values += len(
                valid
            )

            continue


        lower_bound = (
            q1
            -
            IQR_MULTIPLIER
            *
            iqr
        )

        upper_bound = (
            q3
            +
            IQR_MULTIPLIER
            *
            iqr
        )


        extreme_lower = (
            q1
            -
            EXTREME_IQR_MULTIPLIER
            *
            iqr
        )

        extreme_upper = (
            q3
            +
            EXTREME_IQR_MULTIPLIER
            *
            iqr
        )


        outlier_mask = (
            numeric.notna()
            &
            (
                (numeric < lower_bound)
                |
                (numeric > upper_bound)
            )
        )


        extreme_mask = (
            numeric.notna()
            &
            (
                (numeric < extreme_lower)
                |
                (numeric > extreme_upper)
            )
        )


        outlier_indices = (
            df.index[
                outlier_mask
            ].tolist()
        )


        extreme_indices = (
            df.index[
                extreme_mask
            ].tolist()
        )


        outlier_count = len(
            outlier_indices
        )

        extreme_count = len(
            extreme_indices
        )


        for index in outlier_indices:

            row_scores[index] = (
                row_scores.get(
                    index,
                    0
                )
                +
                1
            )


        for index in extreme_indices:

            # Extra penalty for extreme values.

            row_scores[index] = (
                row_scores.get(
                    index,
                    0
                )
                +
                1
            )


        outlier_percentage = (
            outlier_count
            /
            len(valid)
            *
            100
        )


        results[column] = {

            "values_checked":
                int(len(valid)),

            "outliers":
                int(outlier_count),

            "extreme_outliers":
                int(extreme_count),

            "outlier_percentage":
                _safe_round(
                    outlier_percentage
                ),

            "q1":
                _safe_round(q1),

            "median":
                _safe_round(median),

            "q3":
                _safe_round(q3),

            "iqr":
                _safe_round(iqr),

            "lower_bound":
                _safe_round(
                    lower_bound
                ),

            "upper_bound":
                _safe_round(
                    upper_bound
                ),

            "extreme_lower_bound":
                _safe_round(
                    extreme_lower
                ),

            "extreme_upper_bound":
                _safe_round(
                    extreme_upper
                ),

            "outlier_indices":
                outlier_indices,

            "extreme_outlier_indices":
                extreme_indices
        }


        total_values += len(
            valid
        )

        total_outliers += (
            outlier_count
        )

        total_extreme += (
            extreme_count
        )


    overall_percentage = (

        total_outliers
        /
        total_values
        *
        100

        if total_values

        else 0.0
    )


    return {

        "columns":
            results,

        "values_checked":
            int(total_values),

        "outliers":
            int(total_outliers),

        "extreme_outliers":
            int(total_extreme),

        "outlier_percentage":
            _safe_round(
                overall_percentage
            ),

        "row_scores":
            row_scores
    }


# ============================================================
# 2. RARE CATEGORY DETECTION
# ============================================================

def detect_rare_categories(
    df,
    schema
):
    """
    Detect unusually rare values in categorical columns.

    A category is considered rare when its frequency is
    sufficiently small relative to the column size.

    Rare does NOT mean incorrect.

    These are flagged for analysis, not automatically removed.
    """

    results = {}

    total_rare_rows = 0

    row_scores = {}


    for column in df.columns:

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )


        if not _is_categorical_type(
            semantic_type
        ):
            continue


        missing = (
            _missing_mask(
                df[column]
            )
        )


        valid = (
            df.loc[
                ~missing,
                column
            ]
        )


        if valid.empty:
            continue


        counts = (
            valid.value_counts(
                dropna=True
            )
        )


        total = len(
            valid
        )


        rare_values = {}


        for value, count in counts.items():

            ratio = (
                count
                /
                total
            )


            # A category must be genuinely small.
            # The absolute threshold prevents tiny datasets
            # from classifying everything as rare.

            if (
                ratio <= RARE_CATEGORY_RATIO
                and
                count <= RARE_CATEGORY_MIN_COUNT
            ):

                rare_values[
                    str(value)
                ] = {

                    "count":
                        int(count),

                    "percentage":
                        _safe_round(
                            ratio * 100
                        )
                }


        rare_value_set = set(
            rare_values.keys()
        )


        rare_indices = []


        if rare_value_set:

            string_series = (
                df[column]
                .astype(str)
            )


            rare_mask = (
                string_series.isin(
                    rare_value_set
                )
                &
                ~missing
            )


            rare_indices = (
                df.index[
                    rare_mask
                ].tolist()
            )


            for index in rare_indices:

                row_scores[index] = (
                    row_scores.get(
                        index,
                        0
                    )
                    +
                    1
                )


        total_rare_rows += len(
            rare_indices
        )


        results[column] = {

            "unique_categories":
                int(
                    counts.size
                ),

            "rare_categories":
                rare_values,

            "rare_category_count":
                int(
                    len(
                        rare_values
                    )
                ),

            "rare_rows":
                int(
                    len(
                        rare_indices
                    )
                ),

            "rare_indices":
                rare_indices
        }


    return {

        "columns":
            results,

        "rare_rows":
            int(
                total_rare_rows
            ),

        "row_scores":
            row_scores
    }


# ============================================================
# 3. IDENTIFIER ANOMALIES
# ============================================================

def detect_identifier_anomalies(
    df,
    schema
):
    """
    Analyze identifier columns.

    Detect:

    - missing identifiers
    - duplicated identifiers
    - blank identifiers

    Important:

    Duplicate IDs are suspicious but not always invalid.
    Some datasets legitimately contain repeated IDs.
    """

    results = {}

    row_scores = {}


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


        series = (
            df[column]
        )


        missing = (
            _missing_mask(
                series
            )
        )


        missing_indices = (
            df.index[
                missing
            ].tolist()
        )


        non_missing = (
            series[
                ~missing
            ]
        )


        duplicate_mask = (
            non_missing.duplicated(
                keep=False
            )
        )


        duplicate_indices = (
            non_missing.index[
                duplicate_mask
            ].tolist()
        )


        duplicate_values = (
            non_missing[
                duplicate_mask
            ]
            .value_counts()
            .to_dict()
        )


        for index in missing_indices:

            row_scores[index] = (
                row_scores.get(
                    index,
                    0
                )
                +
                2
            )


        for index in duplicate_indices:

            row_scores[index] = (
                row_scores.get(
                    index,
                    0
                )
                +
                1
            )


        results[column] = {

            "missing_identifiers":
                int(
                    len(
                        missing_indices
                    )
                ),

            "duplicate_identifier_rows":
                int(
                    len(
                        duplicate_indices
                    )
                ),

            "duplicate_identifier_values":
                {
                    str(key): int(value)

                    for key, value
                    in duplicate_values.items()
                },

            "missing_indices":
                missing_indices,

            "duplicate_indices":
                duplicate_indices
        }


    return {

        "columns":
            results,

        "row_scores":
            row_scores
    }


# ============================================================
# 4. DATETIME ANOMALIES
# ============================================================

def detect_datetime_anomalies(
    df,
    schema
):
    """
    Detect invalid values in columns already identified
    by schema intelligence as datetime columns.

    Also reports min/max date.

    It does NOT automatically treat future dates as invalid,
    because future dates may be legitimate in forecasting,
    booking or scheduling datasets.
    """

    results = {}

    row_scores = {}


    for column in df.columns:

        semantic_type = (
            _get_semantic_type(
                schema,
                column
            )
        )


        if not _is_datetime_type(
            semantic_type
        ):
            continue


        series = (
            df[column]
        )


        missing = (
            _missing_mask(
                series
            )
        )


        non_missing = (
            series[
                ~missing
            ]
        )


        if non_missing.empty:

            results[column] = {

                "values_checked":
                    0,

                "invalid_dates":
                    0,

                "invalid_percentage":
                    0.0,

                "invalid_indices":
                    [],

                "minimum_date":
                    None,

                "maximum_date":
                    None
            }

            continue


        try:

            converted = pd.to_datetime(
                non_missing,
                errors="coerce",
                format="mixed"
            )

        except (
            TypeError,
            ValueError
        ):

            converted = pd.to_datetime(
                non_missing,
                errors="coerce"
            )


        invalid_mask = (
            converted.isna()
        )


        invalid_indices = (
            converted.index[
                invalid_mask
            ].tolist()
        )


        for index in invalid_indices:

            row_scores[index] = (
                row_scores.get(
                    index,
                    0
                )
                +
                2
            )


        valid_dates = (
            converted.dropna()
        )


        minimum_date = None
        maximum_date = None


        if not valid_dates.empty:

            minimum_date = (
                valid_dates.min().isoformat()
            )

            maximum_date = (
                valid_dates.max().isoformat()
            )


        invalid_percentage = (

            len(invalid_indices)
            /
            len(non_missing)
            *
            100
        )


        results[column] = {

            "values_checked":
                int(
                    len(
                        non_missing
                    )
                ),

            "invalid_dates":
                int(
                    len(
                        invalid_indices
                    )
                ),

            "invalid_percentage":
                _safe_round(
                    invalid_percentage
                ),

            "invalid_indices":
                invalid_indices,

            "minimum_date":
                minimum_date,

            "maximum_date":
                maximum_date
        }


    return {

        "columns":
            results,

        "row_scores":
            row_scores
    }


# ============================================================
# 5. MISSINGNESS ANOMALIES
# ============================================================

def detect_missingness_anomalies(
    df
):
    """
    Analyze missingness at column and row level.
    """

    columns = {}

    row_missing_counts = (
        pd.Series(
            0,
            index=df.index,
            dtype="int64"
        )
    )


    for column in df.columns:

        missing = (
            _missing_mask(
                df[column]
            )
        )


        missing_count = int(
            missing.sum()
        )


        missing_percentage = (

            missing_count
            /
            len(df)
            *
            100

            if len(df)

            else 0.0
        )


        if (
            missing_percentage
            >=
            HIGH_MISSING_THRESHOLD * 100
        ):

            severity = "high"

        elif (
            missing_percentage
            >=
            MODERATE_MISSING_THRESHOLD * 100
        ):

            severity = "medium"

        elif missing_count > 0:

            severity = "low"

        else:

            severity = "none"


        columns[column] = {

            "missing_values":
                missing_count,

            "missing_percentage":
                _safe_round(
                    missing_percentage
                ),

            "severity":
                severity,

            "missing_indices":
                df.index[
                    missing
                ].tolist()
        }


        row_missing_counts = (
            row_missing_counts
            +
            missing.astype(int)
        )


    column_count = len(
        df.columns
    )


    row_missing_percentage = (

        row_missing_counts
        /
        column_count
        *
        100

        if column_count

        else row_missing_counts.astype(
            float
        )
    )


    suspicious_rows = {}


    for index in df.index:

        percentage = float(
            row_missing_percentage.loc[
                index
            ]
        )


        if percentage >= 50:

            suspicious_rows[index] = {

                "missing_columns":
                    int(
                        row_missing_counts.loc[
                            index
                        ]
                    ),

                "missing_percentage":
                    _safe_round(
                        percentage
                    )
            }


    return {

        "columns":
            columns,

        "rows_with_high_missingness":
            suspicious_rows
    }


# ============================================================
# 6. CONSTANT / NEAR-CONSTANT COLUMNS
# ============================================================

def detect_low_variance_columns(
    df
):
    """
    Detect constant and near-constant columns.

    This is useful because such columns usually provide
    little analytical or predictive information.
    """

    results = {}


    for column in df.columns:

        missing = (
            _missing_mask(
                df[column]
            )
        )


        valid = (
            df.loc[
                ~missing,
                column
            ]
        )


        if valid.empty:

            results[column] = {

                "type":
                    "empty",

                "unique_values":
                    0,

                "dominant_percentage":
                    0.0
            }

            continue


        unique_count = int(
            valid.nunique(
                dropna=True
            )
        )


        if unique_count == 1:

            results[column] = {

                "type":
                    "constant",

                "unique_values":
                    1,

                "dominant_percentage":
                    100.0
            }

            continue


        counts = (
            valid.value_counts(
                dropna=True
            )
        )


        dominant_ratio = (
            counts.iloc[0]
            /
            len(valid)
        )


        if (
            dominant_ratio
            >=
            NEAR_CONSTANT_THRESHOLD
        ):

            results[column] = {

                "type":
                    "near_constant",

                "unique_values":
                    unique_count,

                "dominant_percentage":
                    _safe_round(
                        dominant_ratio
                        *
                        100
                    )
            }


    return results


# ============================================================
# 7. DUPLICATE ROWS
# ============================================================

def detect_duplicate_rows(
    df
):
    """
    Detect complete duplicate records.
    """

    duplicate_mask = (
        df.duplicated(
            keep=False
        )
    )


    duplicate_indices = (
        df.index[
            duplicate_mask
        ].tolist()
    )


    duplicate_count = int(
        df.duplicated().sum()
    )


    duplicate_percentage = (

        duplicate_count
        /
        len(df)
        *
        100

        if len(df)

        else 0.0
    )


    return {

        "duplicate_rows":
            duplicate_count,

        "duplicate_percentage":
            _safe_round(
                duplicate_percentage
            ),

        "affected_rows":
            int(
                len(
                    duplicate_indices
                )
            ),

        "duplicate_indices":
            duplicate_indices
    }


# ============================================================
# 8. MERGE ROW ANOMALY SCORES
# ============================================================

def _merge_row_scores(
    *score_maps
):
    """
    Combine anomaly evidence from different detectors.
    """

    merged = {}


    for score_map in score_maps:

        if not isinstance(
            score_map,
            dict
        ):
            continue


        for index, score in score_map.items():

            merged[index] = (
                merged.get(
                    index,
                    0
                )
                +
                score
            )


    return merged


# ============================================================
# 9. SUSPICIOUS ROW DETECTION
# ============================================================

def build_suspicious_rows(
    df,
    row_scores,
    missingness_report,
    duplicate_report
):
    """
    Build row-level anomaly summary.

    The score represents the amount of anomaly evidence
    associated with a row.

    It is NOT a probability.
    """

    scores = dict(
        row_scores
    )


    # --------------------------------------------------------
    # HIGH ROW MISSINGNESS
    # --------------------------------------------------------

    missing_rows = (
        missingness_report.get(
            "rows_with_high_missingness",
            {}
        )
    )


    for index in missing_rows:

        scores[index] = (
            scores.get(
                index,
                0
            )
            +
            2
        )


    # --------------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_indices = (
        duplicate_report.get(
            "duplicate_indices",
            []
        )
    )


    for index in duplicate_indices:

        scores[index] = (
            scores.get(
                index,
                0
            )
            +
            1
        )


    suspicious_rows = []


    for index, score in scores.items():

        if score <= 0:
            continue


        if score >= 5:

            severity = "high"

        elif score >= 3:

            severity = "medium"

        else:

            severity = "low"


        suspicious_rows.append({

            "row_index":
                index,

            "anomaly_score":
                int(score),

            "severity":
                severity
        })


    suspicious_rows.sort(
        key=lambda item: (
            -item[
                "anomaly_score"
            ],
            str(
                item[
                    "row_index"
                ]
            )
        )
    )


    return suspicious_rows


# ============================================================
# 10. COLUMN ANOMALY SUMMARY
# ============================================================

def build_column_summary(
    df,
    numeric_report,
    rare_report,
    identifier_report,
    datetime_report,
    missingness_report,
    low_variance_report
):
    """
    Combine anomaly information into one report per column.
    """

    summary = {}


    for column in df.columns:

        numeric_info = (
            numeric_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
        )


        rare_info = (
            rare_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
        )


        identifier_info = (
            identifier_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
        )


        datetime_info = (
            datetime_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
        )


        missing_info = (
            missingness_report
            .get(
                "columns",
                {}
            )
            .get(
                column,
                {}
            )
        )


        low_variance_info = (
            low_variance_report.get(
                column
            )
        )


        issues = []


        # ----------------------------------------------------
        # MISSING
        # ----------------------------------------------------

        if (
            missing_info.get(
                "missing_values",
                0
            )
            >
            0
        ):

            issues.append(
                "missing_values"
            )


        # ----------------------------------------------------
        # OUTLIERS
        # ----------------------------------------------------

        if (
            numeric_info.get(
                "outliers",
                0
            )
            >
            0
        ):

            issues.append(
                "numeric_outliers"
            )


        if (
            numeric_info.get(
                "extreme_outliers",
                0
            )
            >
            0
        ):

            issues.append(
                "extreme_numeric_outliers"
            )


        # ----------------------------------------------------
        # RARE CATEGORIES
        # ----------------------------------------------------

        if (
            rare_info.get(
                "rare_category_count",
                0
            )
            >
            0
        ):

            issues.append(
                "rare_categories"
            )


        # ----------------------------------------------------
        # IDENTIFIERS
        # ----------------------------------------------------

        if (
            identifier_info.get(
                "missing_identifiers",
                0
            )
            >
            0
        ):

            issues.append(
                "missing_identifiers"
            )


        if (
            identifier_info.get(
                "duplicate_identifier_rows",
                0
            )
            >
            0
        ):

            issues.append(
                "duplicate_identifiers"
            )


        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        if (
            datetime_info.get(
                "invalid_dates",
                0
            )
            >
            0
        ):

            issues.append(
                "invalid_dates"
            )


        # ----------------------------------------------------
        # LOW VARIANCE
        # ----------------------------------------------------

        if low_variance_info:

            issues.append(
                low_variance_info.get(
                    "type",
                    "low_variance"
                )
            )


        summary[column] = {

            "issues":
                issues,

            "issue_count":
                int(
                    len(
                        issues
                    )
                ),

            "missing_values":
                missing_info.get(
                    "missing_values",
                    0
                ),

            "missing_percentage":
                missing_info.get(
                    "missing_percentage",
                    0.0
                ),

            "outliers":
                numeric_info.get(
                    "outliers",
                    0
                ),

            "extreme_outliers":
                numeric_info.get(
                    "extreme_outliers",
                    0
                ),

            "outlier_percentage":
                numeric_info.get(
                    "outlier_percentage",
                    0.0
                ),

            "rare_categories":
                rare_info.get(
                    "rare_category_count",
                    0
                ),

            "missing_identifiers":
                identifier_info.get(
                    "missing_identifiers",
                    0
                ),

            "duplicate_identifier_rows":
                identifier_info.get(
                    "duplicate_identifier_rows",
                    0
                ),

            "invalid_dates":
                datetime_info.get(
                    "invalid_dates",
                    0
                ),

            "low_variance":
                low_variance_info
        }


    return summary


# ============================================================
# 11. GENERATE HUMAN-READABLE ISSUES
# ============================================================

def generate_anomaly_issues(
    column_summary,
    duplicate_report
):
    """
    Generate explainable anomaly findings.
    """

    issues = []


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = (
        duplicate_report.get(
            "duplicate_rows",
            0
        )
    )


    if duplicate_count > 0:

        issues.append({

            "type":
                "duplicate_rows",

            "severity":
                "medium",

            "message":
                (
                    f"{duplicate_count} complete "
                    "duplicate row(s) were detected."
                )
        })


    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    for column, info in column_summary.items():

        missing_percentage = (
            info.get(
                "missing_percentage",
                0
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
                        f"Column '{column}' contains "
                        f"{missing_percentage:.2f}% "
                        "missing values."
                    )
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
                        f"Column '{column}' contains "
                        f"{missing_percentage:.2f}% "
                        "missing values."
                    )
            })


        extreme_outliers = (
            info.get(
                "extreme_outliers",
                0
            )
        )


        if extreme_outliers > 0:

            issues.append({

                "type":
                    "extreme_numeric_outliers",

                "column":
                    column,

                "severity":
                    "high",

                "message":
                    (
                        f"Column '{column}' contains "
                        f"{extreme_outliers} extreme "
                        "numeric outlier(s)."
                    )
            })


        elif (
            info.get(
                "outliers",
                0
            )
            >
            0
        ):

            issues.append({

                "type":
                    "numeric_outliers",

                "column":
                    column,

                "severity":
                    "medium",

                "message":
                    (
                        f"Column '{column}' contains "
                        f"{info['outliers']} potential "
                        "numeric outlier(s)."
                    )
            })


        rare_count = (
            info.get(
                "rare_categories",
                0
            )
        )


        if rare_count > 0:

            issues.append({

                "type":
                    "rare_categories",

                "column":
                    column,

                "severity":
                    "low",

                "message":
                    (
                        f"Column '{column}' contains "
                        f"{rare_count} rare category "
                        "value(s)."
                    )
            })


        missing_ids = (
            info.get(
                "missing_identifiers",
                0
            )
        )


        if missing_ids > 0:

            issues.append({

                "type":
                    "missing_identifiers",

                "column":
                    column,

                "severity":
                    "high",

                "message":
                    (
                        f"Identifier column '{column}' "
                        f"contains {missing_ids} missing "
                        "identifier value(s)."
                    )
            })


        duplicate_ids = (
            info.get(
                "duplicate_identifier_rows",
                0
            )
        )


        if duplicate_ids > 0:

            issues.append({

                "type":
                    "duplicate_identifiers",

                "column":
                    column,

                "severity":
                    "medium",

                "message":
                    (
                        f"Identifier column '{column}' "
                        f"contains {duplicate_ids} rows "
                        "with duplicated identifiers."
                    )
            })


        invalid_dates = (
            info.get(
                "invalid_dates",
                0
            )
        )


        if invalid_dates > 0:

            issues.append({

                "type":
                    "invalid_dates",

                "column":
                    column,

                "severity":
                    "medium",

                "message":
                    (
                        f"Datetime column '{column}' "
                        f"contains {invalid_dates} "
                        "unparseable date value(s)."
                    )
            })


        low_variance = (
            info.get(
                "low_variance"
            )
        )


        if low_variance:

            variance_type = (
                low_variance.get(
                    "type"
                )
            )


            if variance_type == "constant":

                issues.append({

                    "type":
                        "constant_column",

                    "column":
                        column,

                    "severity":
                        "medium",

                    "message":
                        (
                            f"Column '{column}' is constant "
                            "and contains no analytical "
                            "variation."
                        )
                })


            elif variance_type == "near_constant":

                issues.append({

                    "type":
                        "near_constant_column",

                    "column":
                        column,

                    "severity":
                        "low",

                    "message":
                        (
                            f"Column '{column}' is "
                            "near-constant."
                        )
                })


    return issues


# ============================================================
# MAIN ANOMALY ANALYSIS
# ============================================================

def detect_anomalies(
    df,
    schema=None
):
    """
    Run complete deterministic anomaly analysis.

    Returns structured anomaly metadata that can be used by:

    - Cleaning Agent
    - Quality Agent
    - EDA Agent
    - Visualization Agent
    - Statistical Insight Engine
    - Future LLM reasoning layer
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
            "Cannot analyze anomalies in an empty dataset."
        )


    if schema is None:

        schema = {}


    # ========================================================
    # DETECTORS
    # ========================================================

    numeric_report = (
        detect_numeric_outliers(
            df,
            schema
        )
    )


    rare_report = (
        detect_rare_categories(
            df,
            schema
        )
    )


    identifier_report = (
        detect_identifier_anomalies(
            df,
            schema
        )
    )


    datetime_report = (
        detect_datetime_anomalies(
            df,
            schema
        )
    )


    missingness_report = (
        detect_missingness_anomalies(
            df
        )
    )


    low_variance_report = (
        detect_low_variance_columns(
            df
        )
    )


    duplicate_report = (
        detect_duplicate_rows(
            df
        )
    )


    # ========================================================
    # ROW ANOMALY SCORES
    # ========================================================

    row_scores = (
        _merge_row_scores(

            numeric_report.get(
                "row_scores",
                {}
            ),

            rare_report.get(
                "row_scores",
                {}
            ),

            identifier_report.get(
                "row_scores",
                {}
            ),

            datetime_report.get(
                "row_scores",
                {}
            )
        )
    )


    suspicious_rows = (
        build_suspicious_rows(

            df=df,

            row_scores=row_scores,

            missingness_report=missingness_report,

            duplicate_report=duplicate_report
        )
    )


    # ========================================================
    # COLUMN SUMMARY
    # ========================================================

    column_summary = (
        build_column_summary(

            df=df,

            numeric_report=numeric_report,

            rare_report=rare_report,

            identifier_report=identifier_report,

            datetime_report=datetime_report,

            missingness_report=missingness_report,

            low_variance_report=low_variance_report
        )
    )


    # ========================================================
    # EXPLAINABLE ISSUES
    # ========================================================

    issues = (
        generate_anomaly_issues(

            column_summary=
                column_summary,

            duplicate_report=
                duplicate_report
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    high_issues = sum(

        1

        for issue in issues

        if issue.get(
            "severity"
        ) == "high"
    )


    medium_issues = sum(

        1

        for issue in issues

        if issue.get(
            "severity"
        ) == "medium"
    )


    low_issues = sum(

        1

        for issue in issues

        if issue.get(
            "severity"
        ) == "low"
    )


    high_risk_rows = sum(

        1

        for row in suspicious_rows

        if row.get(
            "severity"
        ) == "high"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    return {

        "numeric_outliers":
            numeric_report,

        "rare_categories":
            rare_report,

        "identifier_anomalies":
            identifier_report,

        "datetime_anomalies":
            datetime_report,

        "missingness":
            missingness_report,

        "low_variance_columns":
            low_variance_report,

        "duplicates":
            duplicate_report,

        "column_summary":
            column_summary,

        "suspicious_rows":
            suspicious_rows,

        "issues":
            issues,

        "summary": {

            "rows":
                int(
                    len(df)
                ),

            "columns":
                int(
                    len(
                        df.columns
                    )
                ),

            "numeric_outliers":
                int(
                    numeric_report.get(
                        "outliers",
                        0
                    )
                ),

            "extreme_numeric_outliers":
                int(
                    numeric_report.get(
                        "extreme_outliers",
                        0
                    )
                ),

            "duplicate_rows":
                int(
                    duplicate_report.get(
                        "duplicate_rows",
                        0
                    )
                ),

            "suspicious_rows":
                int(
                    len(
                        suspicious_rows
                    )
                ),

            "high_risk_rows":
                int(
                    high_risk_rows
                ),

            "high_severity_issues":
                int(
                    high_issues
                ),

            "medium_severity_issues":
                int(
                    medium_issues
                ),

            "low_severity_issues":
                int(
                    low_issues
                ),

            "total_issues":
                int(
                    len(
                        issues
                    )
                )
        }
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

def analyze_anomalies(
    df,
    schema=None
):
    """
    Alias retained so other InsightFlow modules can call
    either:

        detect_anomalies(...)

    or:

        analyze_anomalies(...)
    """

    return detect_anomalies(
        df=df,
        schema=schema
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_anomaly_report(
    report
):
    """
    Print a compact anomaly report for development/testing.
    """

    print(
        "\n================================"
    )

    print(
        "       ANOMALY REPORT"
    )

    print(
        "================================"
    )


    summary = report.get(
        "summary",
        {}
    )


    print(
        "\nDataset:"
    )

    print(
        f"Rows: "
        f"{summary.get('rows', 0)}"
    )

    print(
        f"Columns: "
        f"{summary.get('columns', 0)}"
    )


    print(
        "\nAnomalies:"
    )

    print(
        "Numeric outliers: "
        f"{summary.get('numeric_outliers', 0)}"
    )

    print(
        "Extreme outliers: "
        f"{summary.get('extreme_numeric_outliers', 0)}"
    )

    print(
        "Duplicate rows: "
        f"{summary.get('duplicate_rows', 0)}"
    )

    print(
        "Suspicious rows: "
        f"{summary.get('suspicious_rows', 0)}"
    )


    print(
        "\nIssues:"
    )


    issues = report.get(
        "issues",
        []
    )


    if not issues:

        print(
            "No significant anomalies detected."
        )

        return


    for issue in issues[:30]:

        severity = (
            issue.get(
                "severity",
                "unknown"
            )
            .upper()
        )


        message = (
            issue.get(
                "message",
                ""
            )
        )


        print(
            f" - [{severity}] "
            f"{message}"
        )