import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NULL_LIKE_VALUES = {
    "",
    " ",
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
    "--"
}


BOOLEAN_TRUE_VALUES = {
    "true",
    "yes",
    "y",
    "1",
    "t"
}


BOOLEAN_FALSE_VALUES = {
    "false",
    "no",
    "n",
    "0",
    "f"
}


# Columns where negative values are usually impossible
# or highly suspicious.

NON_NEGATIVE_KEYWORDS = {
    "age",
    "quantity",
    "qty",
    "price",
    "cost",
    "salary",
    "revenue",
    "sales",
    "income",
    "units",
    "count",
    "distance",
    "duration",
    "height",
    "weight",
    "rating",
    "radius",
    "speed"
}


# Columns where negative values can legitimately occur.

NEGATIVE_ALLOWED_KEYWORDS = {
    "profit",
    "loss",
    "balance",
    "change",
    "difference",
    "delta",
    "return",
    "adjustment",
    "variance",
    "growth",
    "margin",
    "cashflow",
    "cash_flow"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_column_name(column):
    """
    Normalize a column name for semantic checks.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().lower()
    ).strip("_")


def _get_schema_info(
    schema,
    column
):
    """
    Safely retrieve schema metadata for a column.
    """

    if not isinstance(
        schema,
        dict
    ):
        return {}

    info = schema.get(
        column,
        {}
    )

    if not isinstance(
        info,
        dict
    ):
        return {}

    return info


def _get_detected_type(
    schema,
    column
):
    """
    Retrieve semantic type from schema.
    """

    info = _get_schema_info(
        schema,
        column
    )

    detected_type = info.get(
        "detected_type",
        "Unknown"
    )

    return str(
        detected_type
    )


def _to_python_scalar(value):
    """
    Convert Pandas / NumPy scalar into a normal
    Python value for logging / JSON responses.
    """

    if isinstance(
        value,
        np.generic
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp
    ):
        return value.isoformat()

    return value


# ============================================================
# LOGGING HELPER
# ============================================================

def _add_log(
    cleaning_log,
    message,
    *,
    column=None,
    operation=None,
    count=None,
    strategy=None,
    reason=None,
    metadata=None
):
    """
    Add a backward-compatible string cleaning message.

    The current application expects cleaning_log to be a
    list of strings, so we preserve that contract.

    Structured metadata can later be introduced separately
    without breaking the API/frontend.
    """

    cleaning_log.append(
        message
    )


# ============================================================
# NULL NORMALIZATION
# ============================================================

def _normalize_null_like_values(
    df
):
    """
    Convert common textual missing-value markers into pd.NA.

    Only string/object columns are inspected.
    """

    cleaned_df = df.copy()

    changes = {}

    for column in cleaned_df.columns:

        series = cleaned_df[
            column
        ]

        if not (
            pd.api.types.is_object_dtype(
                series
            )
            or
            pd.api.types.is_string_dtype(
                series
            )
        ):
            continue

        original_missing = int(
            series.isna().sum()
        )

        def normalize(value):

            if pd.isna(value):
                return pd.NA

            if isinstance(
                value,
                str
            ):

                stripped = (
                    value.strip()
                )

                if (
                    stripped.lower()
                    in NULL_LIKE_VALUES
                ):
                    return pd.NA

                return stripped

            return value

        cleaned_df[column] = (
            series.map(
                normalize
            )
        )

        new_missing = int(
            cleaned_df[column]
            .isna()
            .sum()
        )

        converted = (
            new_missing
            - original_missing
        )

        if converted > 0:

            changes[column] = (
                converted
            )

    return (
        cleaned_df,
        changes
    )


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _numeric_series(
    series
):
    """
    Convert a Series to numeric without modifying
    the original Series.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def _numeric_skew(
    series
):
    """
    Calculate skewness safely.
    """

    numeric = (
        _numeric_series(
            series
        )
        .dropna()
    )

    if len(numeric) < 3:
        return 0.0

    skew = (
        numeric.skew()
    )

    if pd.isna(
        skew
    ):
        return 0.0

    return float(
        skew
    )


def _mode_value(
    series
):
    """
    Return most frequent non-null value safely.
    """

    non_null = (
        series.dropna()
    )

    if non_null.empty:
        return None

    mode = (
        non_null.mode(
            dropna=True
        )
    )

    if mode.empty:
        return None

    return mode.iloc[0]


def _mode_dominance(
    series
):
    """
    Return the fraction of valid values represented
    by the most common value.

    Example:

        A A A B C

    dominance = 3 / 5 = 0.60
    """

    non_null = (
        series.dropna()
    )

    if non_null.empty:
        return 0.0

    counts = (
        non_null
        .value_counts(
            dropna=True
        )
    )

    if counts.empty:
        return 0.0

    return float(
        counts.iloc[0]
        / len(non_null)
    )


# ============================================================
# SEMANTIC HELPERS
# ============================================================

def _negative_values_are_suspicious(
    column
):
    """
    Determine whether negative values are probably invalid
    based on column semantics.

    Conservative by design.
    """

    name = (
        _normalize_column_name(
            column
        )
    )

    tokens = set(
        name.split("_")
    )

    if tokens.intersection(
        NEGATIVE_ALLOWED_KEYWORDS
    ):
        return False

    if tokens.intersection(
        NON_NEGATIVE_KEYWORDS
    ):
        return True

    return False


def _is_high_cardinality(
    series
):
    """
    Determine whether a categorical/text column has
    high cardinality.
    """

    non_null = (
        series.dropna()
    )

    if non_null.empty:
        return False

    unique_count = int(
        non_null.nunique()
    )

    unique_ratio = (
        unique_count
        / len(non_null)
    )

    return (
        unique_count > 50
        and
        unique_ratio > 0.50
    )


# ============================================================
# NUMERIC IMPUTATION STRATEGY
# ============================================================

def _choose_numeric_imputation(
    series
):
    """
    Choose a deterministic numeric imputation strategy.

    Returns:

        strategy
        fill_value
        metadata
    """

    valid = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if valid.empty:

        return (
            None,
            None,
            {
                "reason":
                    "No valid numeric values."
            }
        )

    if len(valid) == 1:

        value = float(
            valid.iloc[0]
        )

        return (
            "single_value",
            value,
            {
                "reason":
                    "Only one valid numeric value exists."
            }
        )

    unique_count = int(
        valid.nunique()
    )

    if unique_count == 1:

        value = float(
            valid.iloc[0]
        )

        return (
            "constant",
            value,
            {
                "reason":
                    "Column contains one unique valid value."
            }
        )

    skewness = (
        _numeric_skew(
            valid
        )
    )

    mean = float(
        valid.mean()
    )

    median = float(
        valid.median()
    )

    metadata = {
        "skewness":
            round(
                skewness,
                4
            ),

        "mean":
            mean,

        "median":
            median,

        "valid_count":
            int(
                len(valid)
            )
    }

    # --------------------------------------------------------
    # Strongly skewed → median
    # --------------------------------------------------------

    if abs(
        skewness
    ) > 0.75:

        return (
            "median",
            median,
            metadata
        )

    # --------------------------------------------------------
    # Small sample → median is safer
    # --------------------------------------------------------

    if len(valid) < 10:

        return (
            "median",
            median,
            metadata
        )

    # --------------------------------------------------------
    # Fairly symmetric → mean
    # --------------------------------------------------------

    return (
        "mean",
        mean,
        metadata
    )


# ============================================================
# BOOLEAN STANDARDIZATION
# ============================================================

def _standardize_boolean(
    series
):
    """
    Standardize boolean-like values.

    Invalid values become pd.NA.

    Returns:
        converted_series
        invalid_count
    """

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series.astype(
                "boolean"
            ),
            0
        )

    normalized = (
        series
        .astype(
            "string"
        )
        .str.strip()
        .str.lower()
    )

    mapping = {}

    for value in (
        BOOLEAN_TRUE_VALUES
    ):
        mapping[value] = True

    for value in (
        BOOLEAN_FALSE_VALUES
    ):
        mapping[value] = False

    original_non_null = int(
        normalized.notna().sum()
    )

    converted = (
        normalized
        .map(
            mapping
        )
        .astype(
            "boolean"
        )
    )

    converted_non_null = int(
        converted.notna().sum()
    )

    invalid_count = max(
        original_non_null
        - converted_non_null,
        0
    )

    return (
        converted,
        invalid_count
    )


# ============================================================
# DATATYPE CLEANING
# ============================================================

def _clean_semantic_types(
    df,
    schema,
    cleaning_log
):
    """
    Convert columns according to semantic schema.
    """

    cleaned_df = (
        df.copy()
    )

    for column in (
        cleaned_df.columns
    ):

        detected_type = (
            _get_detected_type(
                schema,
                column
            )
        )

        series = (
            cleaned_df[
                column
            ]
        )

        # ====================================================
        # NUMERICAL
        # ====================================================

        if detected_type == "Numerical":

            before_non_null = int(
                series
                .notna()
                .sum()
            )

            converted = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
            )

            after_non_null = int(
                converted
                .notna()
                .sum()
            )

            invalid_count = max(
                before_non_null
                - after_non_null,
                0
            )

            cleaned_df[column] = (
                converted
            )

            if invalid_count > 0:

                _add_log(
                    cleaning_log,

                    (
                        f"Converted {invalid_count} "
                        f"invalid numeric value(s) "
                        f"in {column} to missing."
                    ),

                    column=column,
                    operation="numeric_conversion",
                    count=invalid_count
                )

        # ====================================================
        # DATETIME
        # ====================================================

        elif detected_type == "Datetime":

            before_non_null = int(
                series
                .notna()
                .sum()
            )

            # Schema Intelligence should already have
            # established that this is a datetime column,
            # therefore parsing here is intentional.

            converted = (
                pd.to_datetime(
                    series,
                    errors="coerce"
                )
            )

            after_non_null = int(
                converted
                .notna()
                .sum()
            )

            invalid_count = max(
                before_non_null
                - after_non_null,
                0
            )

            cleaned_df[column] = (
                converted
            )

            _add_log(
                cleaning_log,

                f"Converted {column} to datetime.",

                column=column,
                operation="datetime_conversion"
            )

            if invalid_count > 0:

                _add_log(
                    cleaning_log,

                    (
                        f"Converted {invalid_count} "
                        f"invalid datetime value(s) "
                        f"in {column} to missing."
                    ),

                    column=column,
                    operation="invalid_datetime",
                    count=invalid_count
                )

        # ====================================================
        # BOOLEAN
        # ====================================================

        elif detected_type == "Boolean":

            converted, invalid_count = (
                _standardize_boolean(
                    series
                )
            )

            cleaned_df[column] = (
                converted
            )

            _add_log(
                cleaning_log,

                (
                    f"Standardized boolean "
                    f"column: {column}."
                ),

                column=column,
                operation="boolean_standardization"
            )

            if invalid_count > 0:

                _add_log(
                    cleaning_log,

                    (
                        f"Converted {invalid_count} "
                        f"unrecognized boolean "
                        f"value(s) in {column} "
                        f"to missing."
                    ),

                    column=column,
                    operation="invalid_boolean",
                    count=invalid_count
                )

        # ====================================================
        # CATEGORICAL
        # ====================================================

        elif detected_type == "Categorical":

            cleaned_df[column] = (
                series
                .astype(
                    "string"
                )
                .str.strip()
            )

        # ====================================================
        # TEXT
        # ====================================================

        elif detected_type == "Text":

            cleaned_df[column] = (
                series
                .astype(
                    "string"
                )
                .str.strip()
            )

        # ====================================================
        # IDENTIFIER
        # ====================================================

        elif detected_type == "Identifier":

            # Keep numeric identifiers unchanged.
            #
            # Textual identifiers are standardized to strings
            # without changing their values.

            if (
                pd.api.types.is_object_dtype(
                    series
                )
                or
                pd.api.types.is_string_dtype(
                    series
                )
            ):

                cleaned_df[column] = (
                    series
                    .astype(
                        "string"
                    )
                    .str.strip()
                )

    return cleaned_df


# ============================================================
# INVALID NEGATIVE HANDLING
# ============================================================

def _handle_suspicious_negatives(
    df,
    schema,
    cleaning_log
):
    """
    Handle clearly suspicious negative numeric values.

    IMPORTANT:

    We do NOT automatically remove ordinary statistical
    outliers.

    For semantically non-negative measures such as age,
    distance, quantity, duration, speed, etc., negative
    values are converted to missing so the normal missing
    value strategy can subsequently handle them.
    """

    cleaned_df = (
        df.copy()
    )

    for column in (
        cleaned_df.columns
    ):

        if (
            _get_detected_type(
                schema,
                column
            )
            != "Numerical"
        ):
            continue

        if not (
            _negative_values_are_suspicious(
                column
            )
        ):
            continue

        numeric = (
            pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )
        )

        negative_mask = (
            numeric < 0
        )

        negative_count = int(
            negative_mask.sum()
        )

        if negative_count == 0:
            continue

        cleaned_df.loc[
            negative_mask,
            column
        ] = np.nan

        _add_log(
            cleaning_log,

            (
                f"Converted {negative_count} "
                f"semantically invalid negative "
                f"value(s) in {column} to missing."
            ),

            column=column,
            operation="invalid_negative",
            count=negative_count,
            reason=(
                "Column semantics indicate "
                "non-negative measurements."
            )
        )

    return cleaned_df


# ============================================================
# NUMERIC MISSING VALUES
# ============================================================

def _handle_numeric_missing(
    df,
    column,
    cleaning_log
):
    """
    Intelligent numerical missing-value treatment.
    """

    series = (
        df[column]
    )

    missing_count = int(
        series
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    row_count = max(
        len(df),
        1
    )

    missing_ratio = (
        missing_count
        / row_count
    )

    valid = (
        series.dropna()
    )

    if valid.empty:

        _add_log(
            cleaning_log,

            (
                f"Left {missing_count} missing "
                f"value(s) in {column} unchanged "
                f"because no valid numeric "
                f"values exist."
            ),

            column=column,
            operation="missing_preserved",
            count=missing_count
        )

        return

    # --------------------------------------------------------
    # HIGH MISSINGNESS
    # --------------------------------------------------------

    if missing_ratio > 0.60:

        _add_log(
            cleaning_log,

            (
                f"Left {missing_count} missing "
                f"value(s) in {column} unchanged "
                f"because missingness is "
                f"{missing_ratio * 100:.2f}%."
            ),

            column=column,
            operation="missing_preserved",
            count=missing_count,
            reason="High missingness"
        )

        return

    strategy, fill_value, metadata = (
        _choose_numeric_imputation(
            series
        )
    )

    if (
        strategy is None
        or fill_value is None
        or pd.isna(
            fill_value
        )
    ):

        return

    df[column] = (
        series.fillna(
            fill_value
        )
    )

    if strategy == "mean":

        reason = (
            "distribution is sufficiently symmetric"
        )

    elif strategy == "median":

        reason = (
            "median is more robust for skewed "
            "or limited numeric data"
        )

    elif strategy == "constant":

        reason = (
            "all valid values are identical"
        )

    else:

        reason = (
            "only one valid value was available"
        )

    _add_log(
        cleaning_log,

        (
            f"Filled {missing_count} missing "
            f"value(s) in {column} using "
            f"{strategy} = "
            f"{float(fill_value):.4g}; "
            f"{reason}."
        ),

        column=column,
        operation="numeric_imputation",
        count=missing_count,
        strategy=strategy,
        metadata=metadata
    )


# ============================================================
# CATEGORICAL MISSING VALUES
# ============================================================

def _handle_categorical_missing(
    df,
    column,
    cleaning_log
):
    """
    Handle categorical missing values using:

    - missing percentage
    - cardinality
    - dominant-category strength
    """

    series = (
        df[column]
    )

    missing_count = int(
        series
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    row_count = max(
        len(df),
        1
    )

    missing_ratio = (
        missing_count
        / row_count
    )

    non_null = (
        series.dropna()
    )

    if non_null.empty:

        df[column] = (
            series.fillna(
                "Unknown"
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Filled {missing_count} missing "
                f"value(s) in {column} with "
                f"'Unknown' because no valid "
                f"category values exist."
            ),

            column=column,
            operation="categorical_imputation",
            count=missing_count,
            strategy="Unknown"
        )

        return

    unique_count = int(
        non_null.nunique()
    )

    dominance = (
        _mode_dominance(
            series
        )
    )

    high_cardinality = (
        _is_high_cardinality(
            series
        )
    )

    # --------------------------------------------------------
    # HIGH CARDINALITY
    # --------------------------------------------------------

    if high_cardinality:

        df[column] = (
            series.fillna(
                "Unknown"
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Filled {missing_count} missing "
                f"value(s) in high-cardinality "
                f"categorical column {column} "
                f"with 'Unknown' rather than "
                f"mode imputation."
            ),

            column=column,
            operation="categorical_imputation",
            count=missing_count,
            strategy="Unknown"
        )

        return

    # --------------------------------------------------------
    # HIGH MISSINGNESS
    # --------------------------------------------------------

    if missing_ratio > 0.30:

        df[column] = (
            series.fillna(
                "Unknown"
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Filled {missing_count} missing "
                f"value(s) in {column} with "
                f"'Unknown' because missingness "
                f"is {missing_ratio * 100:.2f}%."
            ),

            column=column,
            operation="categorical_imputation",
            count=missing_count,
            strategy="Unknown"
        )

        return

    # --------------------------------------------------------
    # DOMINANT CATEGORY EXISTS
    # --------------------------------------------------------

    mode_value = (
        _mode_value(
            series
        )
    )

    # Only use mode when there is a reasonably dominant
    # category. Otherwise mode imputation creates artificial
    # certainty.

    if (
        mode_value is not None
        and dominance >= 0.40
    ):

        df[column] = (
            series.fillna(
                mode_value
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Filled {missing_count} missing "
                f"value(s) in {column} using "
                f"mode = {mode_value}; "
                f"dominant category represents "
                f"{dominance * 100:.2f}% of "
                f"valid values."
            ),

            column=column,
            operation="categorical_imputation",
            count=missing_count,
            strategy="mode"
        )

        return

    # --------------------------------------------------------
    # WEAK MODE → UNKNOWN
    # --------------------------------------------------------

    df[column] = (
        series.fillna(
            "Unknown"
        )
    )

    _add_log(
        cleaning_log,

        (
            f"Filled {missing_count} missing "
            f"value(s) in {column} with "
            f"'Unknown' because no category "
            f"was sufficiently dominant."
        ),

        column=column,
        operation="categorical_imputation",
        count=missing_count,
        strategy="Unknown"
    )


# ============================================================
# BOOLEAN MISSING VALUES
# ============================================================

def _handle_boolean_missing(
    df,
    column,
    cleaning_log
):
    """
    Boolean missing values are handled conservatively.

    Mode is only used when:
    - missingness is low
    - one boolean value is strongly dominant
    """

    series = (
        df[column]
    )

    missing_count = int(
        series
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    row_count = max(
        len(df),
        1
    )

    missing_ratio = (
        missing_count
        / row_count
    )

    dominance = (
        _mode_dominance(
            series
        )
    )

    mode_value = (
        _mode_value(
            series
        )
    )

    if (
        missing_ratio <= 0.10
        and
        dominance >= 0.70
        and
        mode_value is not None
    ):

        df[column] = (
            series.fillna(
                mode_value
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Filled {missing_count} missing "
                f"boolean value(s) in {column} "
                f"using mode = {mode_value}; "
                f"dominance was "
                f"{dominance * 100:.2f}%."
            ),

            column=column,
            operation="boolean_imputation",
            count=missing_count,
            strategy="mode"
        )

        return

    _add_log(
        cleaning_log,

        (
            f"Left {missing_count} missing "
            f"boolean value(s) in {column} "
            f"unchanged to avoid introducing "
            f"artificial True/False values."
        ),

        column=column,
        operation="missing_preserved",
        count=missing_count
    )


# ============================================================
# TEXT MISSING VALUES
# ============================================================

def _handle_text_missing(
    df,
    column,
    cleaning_log
):
    """
    Free text is not statistically imputed.

    Missing text remains missing because replacing it with
    mode/mean has no analytical meaning.
    """

    missing_count = int(
        df[column]
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    _add_log(
        cleaning_log,

        (
            f"Left {missing_count} missing "
            f"text value(s) in {column} "
            f"unchanged because free-text "
            f"values should not be statistically "
            f"imputed."
        ),

        column=column,
        operation="missing_preserved",
        count=missing_count
    )


# ============================================================
# DATETIME MISSING VALUES
# ============================================================

def _handle_datetime_missing(
    df,
    column,
    cleaning_log
):
    """
    Datetime values are not automatically imputed.

    Inventing timestamps can create false trends.
    """

    missing_count = int(
        df[column]
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    _add_log(
        cleaning_log,

        (
            f"Left {missing_count} missing "
            f"datetime value(s) in {column} "
            f"unchanged to avoid fabricating "
            f"time information."
        ),

        column=column,
        operation="missing_preserved",
        count=missing_count
    )


# ============================================================
# IDENTIFIER MISSING VALUES
# ============================================================

def _handle_identifier_missing(
    df,
    column,
    cleaning_log
):
    """
    Never invent identifiers.
    """

    missing_count = int(
        df[column]
        .isna()
        .sum()
    )

    if missing_count == 0:
        return

    _add_log(
        cleaning_log,

        (
            f"Left {missing_count} missing "
            f"identifier value(s) in {column} "
            f"unchanged; identifiers are never "
            f"automatically imputed."
        ),

        column=column,
        operation="missing_preserved",
        count=missing_count
    )


# ============================================================
# MISSING VALUE ORCHESTRATOR
# ============================================================

def _handle_missing_values(
    df,
    schema,
    cleaning_log
):
    """
    Apply semantic missing-value strategies.
    """

    for column in (
        df.columns
    ):

        if not (
            df[column]
            .isna()
            .any()
        ):
            continue

        detected_type = (
            _get_detected_type(
                schema,
                column
            )
        )

        if detected_type == "Numerical":

            _handle_numeric_missing(
                df,
                column,
                cleaning_log
            )

        elif detected_type == "Categorical":

            _handle_categorical_missing(
                df,
                column,
                cleaning_log
            )

        elif detected_type == "Boolean":

            _handle_boolean_missing(
                df,
                column,
                cleaning_log
            )

        elif detected_type == "Datetime":

            _handle_datetime_missing(
                df,
                column,
                cleaning_log
            )

        elif detected_type == "Identifier":

            _handle_identifier_missing(
                df,
                column,
                cleaning_log
            )

        elif detected_type == "Text":

            _handle_text_missing(
                df,
                column,
                cleaning_log
            )

        else:

            missing_count = int(
                df[column]
                .isna()
                .sum()
            )

            _add_log(
                cleaning_log,

                (
                    f"Left {missing_count} missing "
                    f"value(s) in {column} "
                    f"unchanged because its "
                    f"semantic type is unknown."
                ),

                column=column,
                operation="missing_preserved",
                count=missing_count
            )


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def clean_dataset(
    df,
    schema=None,
    quality_report=None,
    anomalies=None
):
    """
    InsightFlow Cleaning Intelligence.

    Cleaning pipeline:

    1. Validate inputs
    2. Normalize null-like strings
    3. Remove exact duplicate rows
    4. Convert semantic datatypes
    5. Handle semantically impossible negatives
    6. Apply intelligent missing-value strategies
    7. Reset DataFrame index
    8. Generate cleaning summary

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to clean.

    schema : dict, optional
        Semantic schema generated by Schema Intelligence.

    quality_report : dict, optional
        Quality Intelligence report for the original dataset.

        Currently used as analytical context and included in
        the cleaning summary. Cleaning decisions remain
        deterministic and datatype-driven.

    anomalies : dict, optional
        Anomaly Intelligence report for the original dataset.

        Statistical anomalies are intentionally NOT deleted
        automatically. The report is accepted so Cleaning
        Intelligence can cooperate with the wider pipeline.

    Important design decisions
    --------------------------
    - Statistical outliers are NOT automatically deleted.
    - Identifier values are NOT imputed.
    - Datetimes are NOT fabricated.
    - Free text is NOT mode-imputed.
    - High missingness avoids aggressive imputation.
    - Numerical imputation considers skewness.
    - Categorical imputation considers cardinality and
      category dominance.
    - Suspicious negative values are handled using semantic
      column meaning.
    - Quality/anomaly reports inform the pipeline without
      blindly modifying valid observations.

    Returns
    -------
    tuple

        cleaned_df,
        cleaning_log,
        cleaning_summary
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        df,
        pd.DataFrame
    ):
        raise TypeError(
            "df must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot clean an empty dataset."
        )

    # --------------------------------------------------------
    # Schema
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
    # Quality report
    # --------------------------------------------------------

    if quality_report is None:
        quality_report = {}

    if not isinstance(
        quality_report,
        dict
    ):
        raise TypeError(
            "quality_report must be a dictionary."
        )

    # --------------------------------------------------------
    # Anomaly report
    # --------------------------------------------------------

    if anomalies is None:
        anomalies = {}

    if not isinstance(
        anomalies,
        dict
    ):
        raise TypeError(
            "anomalies must be a dictionary."
        )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    cleaned_df = (
        df.copy()
    )

    cleaning_log = []

    original_rows = int(
        len(cleaned_df)
    )

    original_columns = int(
        len(cleaned_df.columns)
    )

    original_missing = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    original_duplicates = int(
        cleaned_df
        .duplicated()
        .sum()
    )

    # ========================================================
    # STEP 1
    # NORMALIZE NULL-LIKE VALUES
    # ========================================================

    cleaned_df, null_changes = (
        _normalize_null_like_values(
            cleaned_df
        )
    )

    normalized_null_count = 0

    for column, count in (
        null_changes.items()
    ):

        normalized_null_count += int(
            count
        )

        _add_log(
            cleaning_log,

            (
                f"Normalized {count} "
                f"null-like value(s) "
                f"in {column}."
            ),

            column=column,
            operation="null_normalization",
            count=count
        )

    # ========================================================
    # STEP 2
    # REMOVE EXACT DUPLICATES
    # ========================================================

    duplicate_count = int(
        cleaned_df
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        cleaned_df = (
            cleaned_df
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )

        _add_log(
            cleaning_log,

            (
                f"Removed {duplicate_count} "
                f"exact duplicate row(s)."
            ),

            operation="duplicate_removal",
            count=duplicate_count
        )

    # ========================================================
    # STEP 3
    # SEMANTIC DATATYPE CONVERSION
    # ========================================================

    cleaned_df = (
        _clean_semantic_types(
            cleaned_df,
            schema,
            cleaning_log
        )
    )

    # ========================================================
    # STEP 4
    # SEMANTIC INVALID VALUES
    # ========================================================

    missing_before_invalid_handling = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    cleaned_df = (
        _handle_suspicious_negatives(
            cleaned_df,
            schema,
            cleaning_log
        )
    )

    missing_after_invalid_handling = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    invalid_values_converted = max(
        missing_after_invalid_handling
        -
        missing_before_invalid_handling,
        0
    )

    # ========================================================
    # STEP 5
    # MISSING VALUES
    # ========================================================

    missing_before_imputation = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    _handle_missing_values(
        cleaned_df,
        schema,
        cleaning_log
    )

    missing_after_imputation = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    missing_values_filled = max(
        missing_before_imputation
        -
        missing_after_imputation,
        0
    )

    # ========================================================
    # STEP 6
    # FINAL INDEX NORMALIZATION
    # ========================================================

    cleaned_df = (
        cleaned_df
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # STEP 7
    # CLEANING SUMMARY
    # ========================================================

    final_rows = int(
        len(cleaned_df)
    )

    final_columns = int(
        len(cleaned_df.columns)
    )

    final_missing = int(
        cleaned_df
        .isna()
        .sum()
        .sum()
    )

    final_duplicates = int(
        cleaned_df
        .duplicated()
        .sum()
    )

    rows_removed = max(
        original_rows
        -
        final_rows,
        0
    )

    # --------------------------------------------------------
    # Determine whether upstream intelligence was available
    # --------------------------------------------------------

    quality_context_available = bool(
        quality_report
    )

    anomaly_context_available = bool(
        anomalies
    )

    schema_context_available = bool(
        schema
    )

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    cleaning_summary = {

        "original_rows":
            original_rows,

        "final_rows":
            final_rows,

        "rows_removed":
            rows_removed,

        "original_columns":
            original_columns,

        "final_columns":
            final_columns,

        "original_missing_values":
            original_missing,

        "normalized_null_like_values":
            normalized_null_count,

        "missing_before_imputation":
            missing_before_imputation,

        "missing_values_filled":
            missing_values_filled,

        "remaining_missing_values":
            final_missing,

        "original_duplicate_rows":
            original_duplicates,

        "duplicate_rows_removed":
            duplicate_count,

        "remaining_duplicate_rows":
            final_duplicates,

        "invalid_values_converted_to_missing":
            invalid_values_converted,

        "operation_count":
            int(
                len(cleaning_log)
            ),

        "schema_context_used":
            schema_context_available,

        "quality_context_available":
            quality_context_available,

        "anomaly_context_available":
            anomaly_context_available,

        "outliers_removed":
            0,

        "outlier_policy":
            (
                "Statistical outliers are preserved "
                "for analysis unless they violate "
                "clear semantic validity rules."
            )
    }

    # ========================================================
    # RETURN
    # ========================================================

    return (
        cleaned_df,
        cleaning_log,
        cleaning_summary
    )


# ============================================================
# PRINT CLEANING LOG
# ============================================================

def print_cleaning_log(
    cleaning_log
):
    """
    Display every cleaning decision.
    """

    print(
        "\n=============================="
    )

    print(
        "     CLEANING INTELLIGENCE"
    )

    print(
        "=============================="
    )

    if not cleaning_log:

        print(
            "\nNo cleaning operations required."
        )

        return

    for index, operation in enumerate(
        cleaning_log,
        start=1
    ):

        print(
            f"{index}. {operation}"
        )