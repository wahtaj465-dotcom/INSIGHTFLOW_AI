import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TOP_CATEGORIES = 10

MAX_NUMERIC_CORRELATION_COLUMNS = 30
MAX_CATEGORY_NUMERIC_RELATIONSHIPS = 50
MAX_CATEGORY_CATEGORY_RELATIONSHIPS = 30
MAX_TIME_RELATIONSHIPS = 30
MAX_IMPORTANT_RELATIONSHIPS = 25

MIN_CORRELATION_OBSERVATIONS = 3
MIN_GROUP_SIZE = 2

HIGH_CARDINALITY_THRESHOLD = 100
HIGH_CARDINALITY_RATIO = 0.50

CORRELATION_IMPORTANCE_THRESHOLD = 0.40
CATEGORY_EFFECT_THRESHOLD = 0.10
CRAMERS_V_THRESHOLD = 0.20


# ============================================================
# JSON-SAFE HELPERS
# ============================================================

def _safe_float(value, digits=6):
    """
    Convert a numeric value to a JSON-safe float.

    Returns None for:
    - NaN
    - infinity
    - invalid values
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return round(value, digits)


def _safe_int(value):
    """
    Convert a value to a JSON-safe integer.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_value(value):
    """
    Convert common Pandas / NumPy values into
    JSON-safe Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return _safe_float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return str(value) if not isinstance(
        value,
        (str, int, float, bool)
    ) else value


def _safe_dict(dictionary):
    """
    Convert dictionary keys and values to
    JSON-safe representations.
    """

    result = {}

    for key, value in dictionary.items():

        safe_key = (
            "Missing"
            if _is_missing_scalar(key)
            else str(key)
        )

        result[safe_key] = _safe_value(value)

    return result


def _is_missing_scalar(value):
    """
    Safely determine whether a scalar value is missing.
    """

    try:
        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

    except Exception:
        pass

    return False


# ============================================================
# SCHEMA HELPERS
# ============================================================

def _get_detected_type(schema, column):
    """
    Read semantic type from schema safely.
    """

    info = schema.get(column, {})

    detected_type = info.get(
        "detected_type",
        ""
    )

    return str(detected_type).strip().lower()


def _columns_by_type(schema, *semantic_types):
    """
    Return columns matching one or more semantic types.
    """

    wanted = {
        semantic_type.lower()
        for semantic_type in semantic_types
    }

    columns = []

    for column, info in schema.items():

        detected_type = str(
            info.get("detected_type", "")
        ).strip().lower()

        if detected_type in wanted:
            columns.append(column)

    return columns


def _existing_columns(df, columns):
    """
    Keep only schema columns that actually exist
    in the DataFrame.
    """

    return [
        column
        for column in columns
        if column in df.columns
    ]


# ============================================================
# GENERAL HELPERS
# ============================================================

def _missing_percentage(series):
    """
    Missing percentage for a Series.
    """

    if len(series) == 0:
        return 0.0

    return (
        float(series.isna().mean()) * 100
    )


def _numeric_series(series):
    """
    Convert Series to numeric and replace infinities
    with missing values.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return numeric


def _datetime_series(series):
    """
    Convert Series to datetime without producing
    unnecessary format inference warnings.

    The schema intelligence should already identify
    datetime columns, so EDA only converts those columns.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    cleaned = series.copy()

    try:
        return pd.to_datetime(
            cleaned,
            errors="coerce",
            format="mixed"
        )

    except (TypeError, ValueError):

        return pd.to_datetime(
            cleaned,
            errors="coerce"
        )


def _entropy_from_counts(counts):
    """
    Shannon entropy of a categorical distribution.
    """

    if counts.empty:
        return None

    probabilities = (
        counts / counts.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return _safe_float(entropy)


def _normalized_entropy(counts):
    """
    Entropy normalized to 0..1.

    0:
        one category dominates completely

    1:
        categories are evenly distributed
    """

    category_count = len(counts)

    if category_count <= 1:
        return 0.0

    entropy = _entropy_from_counts(counts)

    if entropy is None:
        return None

    maximum_entropy = math.log2(
        category_count
    )

    if maximum_entropy == 0:
        return 0.0

    return _safe_float(
        entropy / maximum_entropy
    )


def _iqr_outlier_summary(series):
    """
    Local IQR-based outlier profile.

    This does not replace AnomalyAgent.
    It provides EDA-level distribution information
    that can later be combined with anomaly intelligence.
    """

    clean = _numeric_series(
        series
    ).dropna()

    if len(clean) < 4:

        return {
            "method": "IQR",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr):

        return {
            "method": "IQR",
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0
        }

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_mask = (
        (clean < lower_bound)
        |
        (clean > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    outlier_percentage = (
        outlier_count
        / len(clean)
        * 100
    )

    return {
        "method": "IQR",
        "lower_bound": _safe_float(lower_bound),
        "upper_bound": _safe_float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": _safe_float(
            outlier_percentage
        )
    }


def _correlation_strength(value):
    """
    Human-readable correlation strength.
    """

    if value is None:
        return "unknown"

    absolute = abs(value)

    if absolute >= 0.80:
        return "very strong"

    if absolute >= 0.60:
        return "strong"

    if absolute >= 0.40:
        return "moderate"

    if absolute >= 0.20:
        return "weak"

    return "very weak"


def _correlation_direction(value):
    """
    Correlation direction.
    """

    if value is None:
        return "unknown"

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "none"


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

def _analyze_overview(df, schema):
    """
    General dataset-level profile.
    """

    row_count = len(df)
    column_count = len(df.columns)

    total_cells = (
        row_count * column_count
    )

    total_missing = int(
        df.isna().sum().sum()
    )

    missing_percentage = (
        total_missing
        / total_cells
        * 100
        if total_cells > 0
        else 0
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_rows
        / row_count
        * 100
        if row_count > 0
        else 0
    )

    semantic_type_counts = {}

    for column in df.columns:

        detected_type = (
            _get_detected_type(
                schema,
                column
            )
            or "unknown"
        )

        semantic_type_counts[
            detected_type
        ] = (
            semantic_type_counts.get(
                detected_type,
                0
            )
            + 1
        )

    return {
        "rows": row_count,
        "columns": column_count,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_percentage": _safe_float(
            missing_percentage
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": _safe_float(
            duplicate_percentage
        ),
        "semantic_type_counts": semantic_type_counts
    }


# ============================================================
# 2. MISSINGNESS INTELLIGENCE
# ============================================================

def _analyze_missingness(df):
    """
    Analyze missing values by column.

    Also detects columns that frequently become
    missing together.
    """

    result = {
        "total_missing": int(
            df.isna().sum().sum()
        ),
        "columns": {},
        "high_missing_columns": [],
        "co_missing_pairs": []
    }

    row_count = len(df)

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (
            missing_count
            / row_count
            * 100
            if row_count > 0
            else 0
        )

        result["columns"][column] = {
            "missing_count": missing_count,
            "missing_percentage": _safe_float(
                missing_percentage
            )
        }

        if missing_percentage >= 30:

            result[
                "high_missing_columns"
            ].append({
                "column": column,
                "missing_percentage":
                    _safe_float(
                        missing_percentage
                    )
            })

    # --------------------------------------------------------
    # Co-missingness
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in df.columns
        if df[column].isna().any()
    ]

    for i in range(
        len(missing_columns)
    ):

        for j in range(
            i + 1,
            len(missing_columns)
        ):

            column_1 = missing_columns[i]
            column_2 = missing_columns[j]

            both_missing = int(
                (
                    df[column_1].isna()
                    &
                    df[column_2].isna()
                ).sum()
            )

            if both_missing == 0:
                continue

            union_missing = int(
                (
                    df[column_1].isna()
                    |
                    df[column_2].isna()
                ).sum()
            )

            if union_missing == 0:
                continue

            jaccard = (
                both_missing
                / union_missing
            )

            if jaccard >= 0.50:

                result[
                    "co_missing_pairs"
                ].append({
                    "column_1": column_1,
                    "column_2": column_2,
                    "both_missing": both_missing,
                    "jaccard_similarity":
                        _safe_float(jaccard)
                })

    result["high_missing_columns"].sort(
        key=lambda item:
            item["missing_percentage"],
        reverse=True
    )

    result["co_missing_pairs"].sort(
        key=lambda item:
            item["jaccard_similarity"],
        reverse=True
    )

    return result


# ============================================================
# 3. NUMERICAL INTELLIGENCE
# ============================================================

def _analyze_numerical(
    df,
    numerical_columns
):
    """
    Rich numerical profiling.
    """

    result = {}

    for column in numerical_columns:

        series = _numeric_series(
            df[column]
        )

        clean = series.dropna()

        total_count = len(series)
        valid_count = len(clean)

        if valid_count == 0:

            result[column] = {
                "count": 0,
                "missing_count": total_count,
                "missing_percentage": 100.0
            }

            continue

        mean = clean.mean()
        median = clean.median()
        std = clean.std()
        variance = clean.var()

        minimum = clean.min()
        maximum = clean.max()

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)

        iqr = q3 - q1

        skewness = (
            clean.skew()
            if valid_count >= 3
            else None
        )

        kurtosis = (
            clean.kurt()
            if valid_count >= 4
            else None
        )

        zero_count = int(
            (clean == 0).sum()
        )

        negative_count = int(
            (clean < 0).sum()
        )

        positive_count = int(
            (clean > 0).sum()
        )

        unique_count = int(
            clean.nunique()
        )

        coefficient_of_variation = None

        if (
            mean is not None
            and pd.notna(mean)
            and mean != 0
            and std is not None
            and pd.notna(std)
        ):

            coefficient_of_variation = (
                abs(std / mean)
            )

        if skewness is None or pd.isna(skewness):

            distribution_shape = "unknown"

        elif skewness > 1:

            distribution_shape = (
                "highly right-skewed"
            )

        elif skewness > 0.5:

            distribution_shape = (
                "moderately right-skewed"
            )

        elif skewness < -1:

            distribution_shape = (
                "highly left-skewed"
            )

        elif skewness < -0.5:

            distribution_shape = (
                "moderately left-skewed"
            )

        else:

            distribution_shape = (
                "approximately symmetric"
            )

        result[column] = {

            "count":
                valid_count,

            "missing_count":
                int(series.isna().sum()),

            "missing_percentage":
                _safe_float(
                    _missing_percentage(series)
                ),

            "unique_values":
                unique_count,

            "mean":
                _safe_float(mean),

            "median":
                _safe_float(median),

            "min":
                _safe_float(minimum),

            "max":
                _safe_float(maximum),

            "range":
                _safe_float(
                    maximum - minimum
                ),

            "std":
                _safe_float(std),

            "variance":
                _safe_float(variance),

            "q1":
                _safe_float(q1),

            "q3":
                _safe_float(q3),

            "iqr":
                _safe_float(iqr),

            "skewness":
                _safe_float(skewness),

            "kurtosis":
                _safe_float(kurtosis),

            "distribution_shape":
                distribution_shape,

            "coefficient_of_variation":
                _safe_float(
                    coefficient_of_variation
                ),

            "zero_count":
                zero_count,

            "zero_percentage":
                _safe_float(
                    zero_count
                    / valid_count
                    * 100
                ),

            "negative_count":
                negative_count,

            "negative_percentage":
                _safe_float(
                    negative_count
                    / valid_count
                    * 100
                ),

            "positive_count":
                positive_count,

            "outliers":
                _iqr_outlier_summary(
                    clean
                )
        }

    return result


# ============================================================
# 4. CATEGORICAL INTELLIGENCE
# ============================================================

def _analyze_categorical(
    df,
    categorical_columns
):
    """
    Rich categorical profiling.
    """

    result = {}

    row_count = len(df)

    for column in categorical_columns:

        series = df[column]

        non_missing = series.dropna()

        unique_count = int(
            non_missing.nunique()
        )

        valid_count = len(
            non_missing
        )

        counts = (
            non_missing
            .astype(str)
            .value_counts()
        )

        top_counts = counts.head(
            TOP_CATEGORIES
        )

        most_common = (
            str(counts.index[0])
            if len(counts) > 0
            else None
        )

        most_common_count = (
            int(counts.iloc[0])
            if len(counts) > 0
            else 0
        )

        dominance_percentage = (
            most_common_count
            / valid_count
            * 100
            if valid_count > 0
            else 0
        )

        cardinality_ratio = (
            unique_count
            / valid_count
            if valid_count > 0
            else 0
        )

        high_cardinality = (
            unique_count
            >= HIGH_CARDINALITY_THRESHOLD
            or
            cardinality_ratio
            >= HIGH_CARDINALITY_RATIO
        )

        distribution = {
            str(key): int(value)
            for key, value
            in top_counts.items()
        }

        result[column] = {

            "count":
                valid_count,

            "missing_count":
                int(
                    series.isna().sum()
                ),

            "missing_percentage":
                _safe_float(
                    (
                        series.isna().sum()
                        / row_count
                        * 100
                    )
                    if row_count > 0
                    else 0
                ),

            "unique_values":
                unique_count,

            "cardinality_ratio":
                _safe_float(
                    cardinality_ratio
                ),

            "high_cardinality":
                bool(high_cardinality),

            "most_common":
                most_common,

            "most_common_count":
                most_common_count,

            "dominance_percentage":
                _safe_float(
                    dominance_percentage
                ),

            "entropy":
                _entropy_from_counts(
                    counts
                ),

            "normalized_entropy":
                _normalized_entropy(
                    counts
                ),

            "top_categories":
                distribution,

            # Compatibility with the previous EDA format.
            "distribution":
                distribution
        }

    return result


# ============================================================
# 5. NUMERIC CORRELATION INTELLIGENCE
# ============================================================

def _analyze_correlations(
    df,
    numerical_columns
):
    """
    Analyze Pearson correlations and rank relationships.
    """

    result = {
        "matrix": {},
        "pairs": [],
        "strongest_relationships": []
    }

    usable_columns = []

    for column in numerical_columns:

        series = _numeric_series(
            df[column]
        )

        if (
            series.notna().sum()
            >= MIN_CORRELATION_OBSERVATIONS
            and
            series.nunique(
                dropna=True
            ) > 1
        ):

            usable_columns.append(
                column
            )

    usable_columns = (
        usable_columns[
            :MAX_NUMERIC_CORRELATION_COLUMNS
        ]
    )

    if len(usable_columns) < 2:
        return result

    numeric_df = pd.DataFrame({
        column:
            _numeric_series(df[column])
        for column in usable_columns
    })

    correlation_matrix = (
        numeric_df.corr(
            method="pearson"
        )
    )

    # --------------------------------------------------------
    # JSON-safe matrix
    # --------------------------------------------------------

    for column in usable_columns:

        result["matrix"][column] = {}

        for other_column in usable_columns:

            value = (
                correlation_matrix
                .loc[
                    column,
                    other_column
                ]
            )

            result["matrix"][
                column
            ][
                other_column
            ] = _safe_float(value)

    # --------------------------------------------------------
    # Pair relationships
    # --------------------------------------------------------

    for i in range(
        len(usable_columns)
    ):

        for j in range(
            i + 1,
            len(usable_columns)
        ):

            column_1 = (
                usable_columns[i]
            )

            column_2 = (
                usable_columns[j]
            )

            paired = pd.concat(
                [
                    numeric_df[column_1],
                    numeric_df[column_2]
                ],
                axis=1
            ).dropna()

            if (
                len(paired)
                <
                MIN_CORRELATION_OBSERVATIONS
            ):
                continue

            correlation = (
                paired[column_1]
                .corr(
                    paired[column_2]
                )
            )

            correlation = (
                _safe_float(correlation)
            )

            if correlation is None:
                continue

            relationship = {

                "column_1":
                    column_1,

                "column_2":
                    column_2,

                "correlation":
                    correlation,

                "absolute_correlation":
                    _safe_float(
                        abs(correlation)
                    ),

                "strength":
                    _correlation_strength(
                        correlation
                    ),

                "direction":
                    _correlation_direction(
                        correlation
                    ),

                "observations":
                    len(paired)
            }

            result["pairs"].append(
                relationship
            )

    result["pairs"].sort(
        key=lambda item:
            item[
                "absolute_correlation"
            ],
        reverse=True
    )

    result[
        "strongest_relationships"
    ] = [
        relationship
        for relationship
        in result["pairs"]
        if (
            relationship[
                "absolute_correlation"
            ]
            >=
            CORRELATION_IMPORTANCE_THRESHOLD
        )
    ][:MAX_IMPORTANT_RELATIONSHIPS]

    return result


# ============================================================
# 6. CATEGORY -> NUMERIC RELATIONSHIPS
# ============================================================

def _analyze_category_numeric(
    df,
    categorical_columns,
    numerical_columns
):
    """
    Compare numerical values across categorical groups.

    Uses an eta-squared-like effect measure:

        between-group variance / total variance

    This is deterministic and does not require SciPy.
    """

    relationships = []

    for category_column in categorical_columns:

        category_series = (
            df[category_column]
        )

        category_count = int(
            category_series.nunique(
                dropna=True
            )
        )

        # Very high cardinality categories are not useful
        # for automatic group comparison.

        if (
            category_count < 2
            or
            category_count > 30
        ):
            continue

        for numeric_column in numerical_columns:

            numeric_series = (
                _numeric_series(
                    df[numeric_column]
                )
            )

            working = pd.DataFrame({
                "category":
                    category_series,
                "value":
                    numeric_series
            }).dropna()

            if len(working) < 3:
                continue

            working["category"] = (
                working["category"]
                .astype(str)
            )

            group_stats = (
                working
                .groupby(
                    "category"
                )["value"]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max"
                    ]
                )
            )

            group_stats = group_stats[
                group_stats["count"]
                >= MIN_GROUP_SIZE
            ]

            if len(group_stats) < 2:
                continue

            valid_categories = (
                group_stats.index
            )

            filtered = working[
                working["category"]
                .isin(
                    valid_categories
                )
            ]

            overall_mean = (
                filtered["value"]
                .mean()
            )

            total_ss = (
                (
                    filtered["value"]
                    - overall_mean
                ) ** 2
            ).sum()

            between_ss = 0.0

            for category, row in (
                group_stats.iterrows()
            ):

                between_ss += (
                    row["count"]
                    *
                    (
                        row["mean"]
                        - overall_mean
                    ) ** 2
                )

            effect_size = (
                between_ss
                / total_ss
                if total_ss > 0
                else 0
            )

            means = (
                group_stats["mean"]
            )

            highest_group = (
                str(
                    means.idxmax()
                )
            )

            lowest_group = (
                str(
                    means.idxmin()
                )
            )

            group_difference = (
                means.max()
                - means.min()
            )

            groups = {}

            for category, row in (
                group_stats.iterrows()
            ):

                groups[
                    str(category)
                ] = {
                    "count":
                        _safe_int(
                            row["count"]
                        ),

                    "mean":
                        _safe_float(
                            row["mean"]
                        ),

                    "median":
                        _safe_float(
                            row["median"]
                        ),

                    "std":
                        _safe_float(
                            row["std"]
                        ),

                    "min":
                        _safe_float(
                            row["min"]
                        ),

                    "max":
                        _safe_float(
                            row["max"]
                        )
                }

            relationships.append({

                "categorical_column":
                    category_column,

                "numerical_column":
                    numeric_column,

                "effect_size":
                    _safe_float(
                        effect_size
                    ),

                "highest_mean_group":
                    highest_group,

                "lowest_mean_group":
                    lowest_group,

                "mean_difference":
                    _safe_float(
                        group_difference
                    ),

                "groups":
                    groups
            })

    relationships.sort(
        key=lambda item:
            item["effect_size"],
        reverse=True
    )

    return relationships[
        :MAX_CATEGORY_NUMERIC_RELATIONSHIPS
    ]


# ============================================================
# 7. CATEGORY -> CATEGORY RELATIONSHIPS
# ============================================================

def _cramers_v(
    series_1,
    series_2
):
    """
    Compute Cramer's V without requiring SciPy.

    Uses the contingency table and chi-square formula.
    """

    working = pd.DataFrame({
        "a": series_1,
        "b": series_2
    }).dropna()

    if len(working) == 0:
        return None

    working["a"] = (
        working["a"].astype(str)
    )

    working["b"] = (
        working["b"].astype(str)
    )

    table = pd.crosstab(
        working["a"],
        working["b"]
    )

    if (
        table.shape[0] < 2
        or
        table.shape[1] < 2
    ):
        return None

    observed = (
        table.to_numpy(
            dtype=float
        )
    )

    total = observed.sum()

    if total <= 0:
        return None

    row_totals = (
        observed.sum(
            axis=1,
            keepdims=True
        )
    )

    column_totals = (
        observed.sum(
            axis=0,
            keepdims=True
        )
    )

    expected = (
        row_totals
        @
        column_totals
        /
        total
    )

    valid = expected > 0

    chi_square = np.sum(
        (
            (
                observed[valid]
                -
                expected[valid]
            ) ** 2
        )
        /
        expected[valid]
    )

    phi_squared = (
        chi_square
        / total
    )

    rows, columns = (
        observed.shape
    )

    denominator = min(
        rows - 1,
        columns - 1
    )

    if denominator <= 0:
        return None

    cramers_v = math.sqrt(
        phi_squared
        / denominator
    )

    return _safe_float(
        cramers_v
    )


def _analyze_category_category(
    df,
    categorical_columns
):
    """
    Analyze categorical associations using Cramer's V.
    """

    relationships = []

    usable_columns = []

    for column in categorical_columns:

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        if (
            unique_count >= 2
            and
            unique_count <= 30
        ):

            usable_columns.append(
                column
            )

    for i in range(
        len(usable_columns)
    ):

        for j in range(
            i + 1,
            len(usable_columns)
        ):

            column_1 = (
                usable_columns[i]
            )

            column_2 = (
                usable_columns[j]
            )

            value = _cramers_v(
                df[column_1],
                df[column_2]
            )

            if value is None:
                continue

            relationships.append({

                "column_1":
                    column_1,

                "column_2":
                    column_2,

                "cramers_v":
                    value,

                "strength":
                    (
                        "strong"
                        if value >= 0.50
                        else
                        "moderate"
                        if value >= 0.30
                        else
                        "weak"
                    )
            })

    relationships.sort(
        key=lambda item:
            item["cramers_v"],
        reverse=True
    )

    return relationships[
        :MAX_CATEGORY_CATEGORY_RELATIONSHIPS
    ]


# ============================================================
# 8. DATETIME INTELLIGENCE
# ============================================================

def _analyze_datetime(
    df,
    datetime_columns,
    numerical_columns
):
    """
    Analyze datetime columns and numerical trends over time.
    """

    datetime_result = {}
    time_relationships = []

    for date_column in datetime_columns:

        dates = _datetime_series(
            df[date_column]
        )

        valid_dates = (
            dates.dropna()
        )

        if valid_dates.empty:
            continue

        earliest = (
            valid_dates.min()
        )

        latest = (
            valid_dates.max()
        )

        unique_dates = int(
            valid_dates.nunique()
        )

        range_days = int(
            (
                latest
                - earliest
            ).days
        )

        datetime_result[
            date_column
        ] = {

            "valid_dates":
                int(
                    valid_dates.count()
                ),

            "missing_dates":
                int(
                    dates.isna().sum()
                ),

            "missing_percentage":
                _safe_float(
                    _missing_percentage(
                        dates
                    )
                ),

            "unique_dates":
                unique_dates,

            "earliest":
                earliest.isoformat(),

            "latest":
                latest.isoformat(),

            "range_days":
                range_days
        }

        # ----------------------------------------------------
        # Numerical trends
        # ----------------------------------------------------

        for numeric_column in numerical_columns:

            values = _numeric_series(
                df[numeric_column]
            )

            working = pd.DataFrame({
                "date": dates,
                "value": values
            }).dropna()

            if len(working) < 3:
                continue

            working = (
                working.sort_values(
                    "date"
                )
            )

            # Aggregate repeated timestamps before
            # measuring a trend.

            aggregated = (
                working
                .groupby(
                    "date",
                    as_index=False
                )["value"]
                .mean()
            )

            if len(aggregated) < 3:
                continue

            date_numeric = (
                aggregated["date"]
                .astype("int64")
                / 1_000_000_000
            )

            trend_correlation = (
                pd.Series(date_numeric)
                .corr(
                    aggregated["value"]
                )
            )

            trend_correlation = (
                _safe_float(
                    trend_correlation
                )
            )

            first_value = (
                aggregated[
                    "value"
                ].iloc[0]
            )

            last_value = (
                aggregated[
                    "value"
                ].iloc[-1]
            )

            absolute_change = (
                last_value
                - first_value
            )

            percentage_change = None

            if first_value != 0:

                percentage_change = (
                    absolute_change
                    / abs(first_value)
                    * 100
                )

            if trend_correlation is None:

                direction = "unknown"

            elif trend_correlation >= 0.30:

                direction = "increasing"

            elif trend_correlation <= -0.30:

                direction = "decreasing"

            else:

                direction = "stable_or_irregular"

            time_relationships.append({

                "datetime_column":
                    date_column,

                "numerical_column":
                    numeric_column,

                "observations":
                    len(aggregated),

                "trend_correlation":
                    trend_correlation,

                "trend_strength":
                    _correlation_strength(
                        trend_correlation
                    ),

                "direction":
                    direction,

                "first_value":
                    _safe_float(
                        first_value
                    ),

                "last_value":
                    _safe_float(
                        last_value
                    ),

                "absolute_change":
                    _safe_float(
                        absolute_change
                    ),

                "percentage_change":
                    _safe_float(
                        percentage_change
                    )
            })

    time_relationships.sort(
        key=lambda item:
            abs(
                item[
                    "trend_correlation"
                ]
                or 0
            ),
        reverse=True
    )

    return (
        datetime_result,
        time_relationships[
            :MAX_TIME_RELATIONSHIPS
        ]
    )


# ============================================================
# 9. IMPORTANT RELATIONSHIPS
# ============================================================

def _build_important_relationships(
    correlation_results,
    category_numeric_results,
    category_category_results,
    time_relationships
):
    """
    Produce a ranked, compact list of relationships
    useful to downstream agents.
    """

    findings = []

    # --------------------------------------------------------
    # Numeric <-> Numeric
    # --------------------------------------------------------

    for relationship in (
        correlation_results.get(
            "pairs",
            []
        )
    ):

        strength = relationship[
            "absolute_correlation"
        ]

        if (
            strength
            <
            CORRELATION_IMPORTANCE_THRESHOLD
        ):
            continue

        findings.append({

            "type":
                "numeric_correlation",

            "columns": [
                relationship["column_1"],
                relationship["column_2"]
            ],

            "score":
                strength,

            "details":
                relationship
        })

    # --------------------------------------------------------
    # Category -> Numeric
    # --------------------------------------------------------

    for relationship in (
        category_numeric_results
    ):

        score = (
            relationship[
                "effect_size"
            ]
            or 0
        )

        if (
            score
            <
            CATEGORY_EFFECT_THRESHOLD
        ):
            continue

        findings.append({

            "type":
                "category_numeric",

            "columns": [
                relationship[
                    "categorical_column"
                ],
                relationship[
                    "numerical_column"
                ]
            ],

            "score":
                score,

            "details":
                relationship
        })

    # --------------------------------------------------------
    # Category <-> Category
    # --------------------------------------------------------

    for relationship in (
        category_category_results
    ):

        score = (
            relationship[
                "cramers_v"
            ]
            or 0
        )

        if (
            score
            <
            CRAMERS_V_THRESHOLD
        ):
            continue

        findings.append({

            "type":
                "categorical_association",

            "columns": [
                relationship["column_1"],
                relationship["column_2"]
            ],

            "score":
                score,

            "details":
                relationship
        })

    # --------------------------------------------------------
    # Time -> Numeric
    # --------------------------------------------------------

    for relationship in (
        time_relationships
    ):

        score = abs(
            relationship[
                "trend_correlation"
            ]
            or 0
        )

        if score < 0.30:
            continue

        findings.append({

            "type":
                "time_trend",

            "columns": [
                relationship[
                    "datetime_column"
                ],
                relationship[
                    "numerical_column"
                ]
            ],

            "score":
                _safe_float(score),

            "details":
                relationship
        })

    findings.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    return findings[
        :MAX_IMPORTANT_RELATIONSHIPS
    ]


# ============================================================
# 10. EDA FLAGS / CAVEATS
# ============================================================

def _build_flags(
    overview,
    numerical_results,
    categorical_results,
    missingness_results
):
    """
    Generate deterministic analytical caveats.
    """

    flags = []

    if (
        overview[
            "duplicate_percentage"
        ]
        >= 5
    ):

        flags.append({

            "type":
                "duplicate_rows",

            "severity":
                "medium",

            "message":
                (
                    f"{overview['duplicate_percentage']}% "
                    "of rows are duplicated."
                )
        })

    for item in (
        missingness_results[
            "high_missing_columns"
        ]
    ):

        flags.append({

            "type":
                "high_missingness",

            "severity":
                (
                    "high"
                    if (
                        item[
                            "missing_percentage"
                        ]
                        >= 50
                    )
                    else
                    "medium"
                ),

            "column":
                item["column"],

            "message":
                (
                    f"{item['column']} has "
                    f"{item['missing_percentage']}% "
                    "missing values."
                )
        })

    for column, stats in (
        numerical_results.items()
    ):

        outliers = stats.get(
            "outliers",
            {}
        )

        outlier_percentage = (
            outliers.get(
                "outlier_percentage"
            )
            or 0
        )

        if outlier_percentage >= 5:

            flags.append({

                "type":
                    "outliers",

                "severity":
                    (
                        "high"
                        if outlier_percentage >= 15
                        else
                        "medium"
                    ),

                "column":
                    column,

                "message":
                    (
                        f"{column} contains approximately "
                        f"{outlier_percentage}% IQR outliers."
                    )
            })

        negative_percentage = (
            stats.get(
                "negative_percentage"
            )
            or 0
        )

        if negative_percentage > 0:

            flags.append({

                "type":
                    "negative_values",

                "severity":
                    "info",

                "column":
                    column,

                "message":
                    (
                        f"{column} contains "
                        f"{negative_percentage}% "
                        "negative values. Validate whether "
                        "negative values are meaningful "
                        "for this field."
                    )
            })

    for column, stats in (
        categorical_results.items()
    ):

        if stats.get(
            "high_cardinality"
        ):

            flags.append({

                "type":
                    "high_cardinality",

                "severity":
                    "info",

                "column":
                    column,

                "message":
                    (
                        f"{column} has high cardinality "
                        "and may not be suitable for "
                        "ordinary categorical charts."
                    )
            })

        dominance = (
            stats.get(
                "dominance_percentage"
            )
            or 0
        )

        if dominance >= 90:

            flags.append({

                "type":
                    "dominant_category",

                "severity":
                    "info",

                "column":
                    column,

                "message":
                    (
                        f"One category represents "
                        f"{dominance}% of non-missing "
                        f"values in {column}."
                    )
            })

    return flags


# ============================================================
# 11. LIMITED CHART RECOMMENDATIONS
# ============================================================

def _build_chart_recommendations(
    numerical_columns,
    categorical_columns,
    datetime_columns,
    correlation_results,
    category_numeric_results,
    time_relationships
):
    """
    Produce only high-level chart candidates.

    IMPORTANT:
    This is intentionally conservative.

    Visualization Intelligence will later decide:
    - exact chart type
    - priority
    - whether the chart is actually useful
    - aggregation
    - axis selection
    - chart limits

    We do NOT create every possible column combination here.
    """

    recommendations = []

    # --------------------------------------------------------
    # Basic univariate candidates
    # --------------------------------------------------------

    for column in numerical_columns[:8]:

        recommendations.append({

            "chart":
                "histogram",

            "x":
                column,

            "y":
                None,

            "intent":
                "distribution",

            "reason":
                (
                    f"Inspect the distribution "
                    f"of {column}."
                )
        })

    for column in categorical_columns[:8]:

        recommendations.append({

            "chart":
                "bar",

            "x":
                column,

            "y":
                "count",

            "intent":
                "category_frequency",

            "reason":
                (
                    f"Compare frequencies across "
                    f"{column} categories."
                )
        })

    # --------------------------------------------------------
    # Strong numeric relationships only
    # --------------------------------------------------------

    for relationship in (
        correlation_results.get(
            "strongest_relationships",
            []
        )[:5]
    ):

        recommendations.append({

            "chart":
                "scatter",

            "x":
                relationship[
                    "column_1"
                ],

            "y":
                relationship[
                    "column_2"
                ],

            "intent":
                "relationship",

            "reason":
                (
                    "Explore a "
                    f"{relationship['strength']} "
                    f"{relationship['direction']} "
                    "numeric relationship."
                )
        })

    # --------------------------------------------------------
    # Strong category -> numeric relationships
    # --------------------------------------------------------

    for relationship in (
        category_numeric_results[:5]
    ):

        if (
            relationship[
                "effect_size"
            ]
            <
            CATEGORY_EFFECT_THRESHOLD
        ):
            continue

        recommendations.append({

            "chart":
                "box",

            "x":
                relationship[
                    "categorical_column"
                ],

            "y":
                relationship[
                    "numerical_column"
                ],

            "intent":
                "group_comparison",

            "reason":
                (
                    "Compare numerical distributions "
                    "across categories."
                )
        })

    # --------------------------------------------------------
    # Meaningful time trends
    # --------------------------------------------------------

    for relationship in (
        time_relationships[:5]
    ):

        if (
            abs(
                relationship[
                    "trend_correlation"
                ]
                or 0
            )
            < 0.30
        ):
            continue

        recommendations.append({

            "chart":
                "line",

            "x":
                relationship[
                    "datetime_column"
                ],

            "y":
                relationship[
                    "numerical_column"
                ],

            "intent":
                "time_trend",

            "reason":
                (
                    f"Inspect the detected "
                    f"{relationship['direction']} "
                    "time trend."
                )
        })

    return recommendations


# ============================================================
# 12. AUTOMATED EDA
# ============================================================

def perform_eda(
    df,
    schema
):
    """
    Perform deterministic EDA Intelligence.

    Compatible with the existing InsightFlow workflow:

        perform_eda(df, schema)

    No LLM is required.

    Returns structured analytical metadata for:
    - Visualization Intelligence
    - Statistical Insight Engine
    - Analyst chatbot
    - LLM reasoning layer
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a Pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Cannot perform EDA on an empty dataset."
        )

    if not isinstance(
        schema,
        dict
    ):

        raise TypeError(
            "schema must be a dictionary."
        )

    # ========================================================
    # SEMANTIC COLUMN GROUPS
    # ========================================================

    numerical_columns = (
        _existing_columns(
            df,
            _columns_by_type(
                schema,
                "Numerical",
                "Numeric"
            )
        )
    )

    categorical_columns = (
        _existing_columns(
            df,
            _columns_by_type(
                schema,
                "Categorical",
                "Boolean"
            )
        )
    )

    datetime_columns = (
        _existing_columns(
            df,
            _columns_by_type(
                schema,
                "Datetime",
                "Date",
                "DateTime"
            )
        )
    )

    identifier_columns = (
        _existing_columns(
            df,
            _columns_by_type(
                schema,
                "Identifier",
                "ID"
            )
        )
    )

    text_columns = (
        _existing_columns(
            df,
            _columns_by_type(
                schema,
                "Text"
            )
        )
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    overview = (
        _analyze_overview(
            df,
            schema
        )
    )

    missingness = (
        _analyze_missingness(
            df
        )
    )

    numerical = (
        _analyze_numerical(
            df,
            numerical_columns
        )
    )

    categorical = (
        _analyze_categorical(
            df,
            categorical_columns
        )
    )

    correlations = (
        _analyze_correlations(
            df,
            numerical_columns
        )
    )

    category_numeric = (
        _analyze_category_numeric(
            df,
            categorical_columns,
            numerical_columns
        )
    )

    category_category = (
        _analyze_category_category(
            df,
            categorical_columns
        )
    )

    (
        datetime_analysis,
        time_relationships
    ) = _analyze_datetime(
        df,
        datetime_columns,
        numerical_columns
    )

    important_relationships = (
        _build_important_relationships(
            correlations,
            category_numeric,
            category_category,
            time_relationships
        )
    )

    flags = (
        _build_flags(
            overview,
            numerical,
            categorical,
            missingness
        )
    )

    chart_recommendations = (
        _build_chart_recommendations(
            numerical_columns,
            categorical_columns,
            datetime_columns,
            correlations,
            category_numeric,
            time_relationships
        )
    )

    # ========================================================
    # BACKWARD-COMPATIBLE FLAT CORRELATIONS
    # ========================================================

    flat_correlations = {}

    for relationship in (
        correlations.get(
            "pairs",
            []
        )
    ):

        key = (
            f"{relationship['column_1']} "
            f"vs "
            f"{relationship['column_2']}"
        )

        flat_correlations[
            key
        ] = relationship[
            "correlation"
        ]

    # ========================================================
    # BACKWARD-COMPATIBLE DISTRIBUTIONS
    # ========================================================

    distributions = {}

    for column, stats in (
        numerical.items()
    ):

        distributions[column] = {

            "skewness":
                stats.get(
                    "skewness"
                ),

            "shape":
                stats.get(
                    "distribution_shape"
                )
        }

    # ========================================================
    # FINAL STRUCTURED RESULT
    # ========================================================

    return {

        "overview":
            overview,

        "column_groups": {

            "numerical":
                numerical_columns,

            "categorical":
                categorical_columns,

            "datetime":
                datetime_columns,

            "identifier":
                identifier_columns,

            "text":
                text_columns
        },

        "missingness":
            missingness,

        "numerical":
            numerical,

        "categorical":
            categorical,

        # Existing workflow compatibility.
        "correlations":
            flat_correlations,

        # New richer correlation object.
        "correlation_analysis":
            correlations,

        "distributions":
            distributions,

        "datetime":
            datetime_analysis,

        "relationships": {

            "numeric_numeric":
                correlations.get(
                    "pairs",
                    []
                ),

            "categorical_numeric":
                category_numeric,

            "categorical_categorical":
                category_category,

            "time_numeric":
                time_relationships
        },

        "important_relationships":
            important_relationships,

        "flags":
            flags,

        "chart_recommendations":
            chart_recommendations
    }


# ============================================================
# 13. PRINT EDA REPORT
# ============================================================

def print_eda_report(
    eda_results
):
    """
    Print a concise deterministic EDA report.
    """

    print(
        "\n"
        "=========================================="
    )

    print(
        "           EDA INTELLIGENCE REPORT"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    overview = (
        eda_results.get(
            "overview",
            {}
        )
    )

    print(
        "\nDATASET OVERVIEW"
    )

    print(
        f"Rows: "
        f"{overview.get('rows', 0)}"
    )

    print(
        f"Columns: "
        f"{overview.get('columns', 0)}"
    )

    print(
        f"Missing values: "
        f"{overview.get('total_missing', 0)}"
    )

    print(
        "Missing percentage: "
        f"{overview.get('missing_percentage', 0)}%"
    )

    print(
        f"Duplicate rows: "
        f"{overview.get('duplicate_rows', 0)}"
    )

    # ========================================================
    # COLUMN GROUPS
    # ========================================================

    groups = (
        eda_results.get(
            "column_groups",
            {}
        )
    )

    print(
        "\nSEMANTIC COLUMN GROUPS"
    )

    for group_name, columns in (
        groups.items()
    ):

        print(
            f"{group_name.title()}: "
            f"{len(columns)}"
        )

    # ========================================================
    # NUMERICAL
    # ========================================================

    numerical = (
        eda_results.get(
            "numerical",
            {}
        )
    )

    print(
        "\nNUMERICAL ANALYSIS"
    )

    if not numerical:

        print(
            "No numerical columns detected."
        )

    for column, stats in (
        numerical.items()
    ):

        print(
            f"\n{column}"
        )

        print(
            f"  Mean: "
            f"{stats.get('mean')}"
        )

        print(
            f"  Median: "
            f"{stats.get('median')}"
        )

        print(
            f"  Std: "
            f"{stats.get('std')}"
        )

        print(
            f"  Q1: "
            f"{stats.get('q1')}"
        )

        print(
            f"  Q3: "
            f"{stats.get('q3')}"
        )

        print(
            f"  Skew: "
            f"{stats.get('skewness')}"
        )

        print(
            "  Distribution: "
            f"{stats.get('distribution_shape')}"
        )

        outliers = (
            stats.get(
                "outliers",
                {}
            )
        )

        print(
            "  IQR outliers: "
            f"{outliers.get('outlier_count', 0)} "
            "("
            f"{outliers.get('outlier_percentage', 0)}%"
            ")"
        )

    # ========================================================
    # CATEGORICAL
    # ========================================================

    categorical = (
        eda_results.get(
            "categorical",
            {}
        )
    )

    print(
        "\nCATEGORICAL ANALYSIS"
    )

    if not categorical:

        print(
            "No categorical columns detected."
        )

    for column, stats in (
        categorical.items()
    ):

        print(
            f"\n{column}"
        )

        print(
            "  Unique values: "
            f"{stats.get('unique_values')}"
        )

        print(
            "  Most common: "
            f"{stats.get('most_common')}"
        )

        print(
            "  Dominance: "
            f"{stats.get('dominance_percentage')}%"
        )

        print(
            "  Normalized entropy: "
            f"{stats.get('normalized_entropy')}"
        )

    # ========================================================
    # CORRELATIONS
    # ========================================================

    correlations = (
        eda_results.get(
            "correlation_analysis",
            {}
        )
    )

    strongest = (
        correlations.get(
            "strongest_relationships",
            []
        )
    )

    print(
        "\nIMPORTANT NUMERIC CORRELATIONS"
    )

    if not strongest:

        print(
            "No moderate or strong numeric "
            "correlations detected."
        )

    for relationship in strongest:

        print(
            "\n"
            f"{relationship['column_1']} vs "
            f"{relationship['column_2']}: "
            f"{relationship['correlation']}"
        )

        print(
            "  "
            f"{relationship['strength'].title()} "
            f"{relationship['direction']} relationship"
        )

    # ========================================================
    # CATEGORY -> NUMERIC
    # ========================================================

    relationships = (
        eda_results.get(
            "relationships",
            {}
        )
    )

    category_numeric = (
        relationships.get(
            "categorical_numeric",
            []
        )
    )

    print(
        "\nCATEGORY -> NUMERIC RELATIONSHIPS"
    )

    if not category_numeric:

        print(
            "No usable category/numeric "
            "relationships detected."
        )

    for relationship in (
        category_numeric[:10]
    ):

        print(
            "\n"
            f"{relationship['categorical_column']} "
            "-> "
            f"{relationship['numerical_column']}"
        )

        print(
            "  Effect size: "
            f"{relationship['effect_size']}"
        )

        print(
            "  Highest group: "
            f"{relationship['highest_mean_group']}"
        )

        print(
            "  Lowest group: "
            f"{relationship['lowest_mean_group']}"
        )

    # ========================================================
    # TIME TRENDS
    # ========================================================

    time_numeric = (
        relationships.get(
            "time_numeric",
            []
        )
    )

    print(
        "\nTIME TRENDS"
    )

    if not time_numeric:

        print(
            "No usable numerical time trends detected."
        )

    for relationship in (
        time_numeric[:10]
    ):

        print(
            "\n"
            f"{relationship['numerical_column']} "
            "over "
            f"{relationship['datetime_column']}"
        )

        print(
            "  Direction: "
            f"{relationship['direction']}"
        )

        print(
            "  Trend correlation: "
            f"{relationship['trend_correlation']}"
        )

        print(
            "  Percentage change: "
            f"{relationship['percentage_change']}"
        )

    # ========================================================
    # FLAGS
    # ========================================================

    flags = (
        eda_results.get(
            "flags",
            []
        )
    )

    print(
        "\nEDA FLAGS"
    )

    if not flags:

        print(
            "No major EDA caveats detected."
        )

    for flag in flags:

        print(
            f"- [{flag.get('severity', 'info').upper()}] "
            f"{flag.get('message')}"
        )

    # ========================================================
    # IMPORTANT RELATIONSHIPS
    # ========================================================

    important = (
        eda_results.get(
            "important_relationships",
            []
        )
    )

    print(
        "\nIMPORTANT RELATIONSHIPS"
    )

    if not important:

        print(
            "No important relationships passed "
            "the current thresholds."
        )

    for index, relationship in enumerate(
        important,
        start=1
    ):

        print(
            f"{index}. "
            f"{relationship['type']} | "
            f"{' vs '.join(relationship['columns'])} | "
            f"score={relationship['score']}"
        )

    # ========================================================
    # CHART CANDIDATES
    # ========================================================

    recommendations = (
        eda_results.get(
            "chart_recommendations",
            []
        )
    )

    print(
        "\nCHART CANDIDATES"
    )

    print(
        f"Generated: "
        f"{len(recommendations)}"
    )

    for index, chart in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"\n{index}. "
            f"{chart['chart'].upper()}"
        )

        print(
            f"   X: "
            f"{chart.get('x')}"
        )

        if chart.get("y") is not None:

            print(
                f"   Y: "
                f"{chart.get('y')}"
            )

        print(
            f"   Intent: "
            f"{chart.get('intent')}"
        )

        print(
            f"   Reason: "
            f"{chart.get('reason')}"
        )



    # ============================================================
# 14. PUBLIC EDA AGENT INTERFACE
# ============================================================

def run_eda(
    df,
    schema=None
):
    """
    Public entry point for EDA Intelligence.

    AnalyticsWorkflow calls this function.

    The actual deterministic EDA implementation remains
    perform_eda(df, schema).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to analyze.

    schema : dict, optional
        Semantic schema generated by Schema Intelligence.

    Returns
    -------
    dict
        Structured EDA intelligence.
    """

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame
    ):
        raise TypeError(
            "df must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot perform EDA on an empty dataset."
        )

    # --------------------------------------------------------
    # Normalize schema
    # --------------------------------------------------------

    if schema is None:
        schema = {}

    if not isinstance(
        schema,
        dict
    ):
        raise TypeError(
            "schema must be a dictionary."
        )

    # --------------------------------------------------------
    # Execute deterministic EDA
    # --------------------------------------------------------

    return perform_eda(
        df=df,
        schema=schema
    )