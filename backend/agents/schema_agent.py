import re
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NULL_LIKE_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "nil",
    "missing",
    "unknown",
    "-",
    "--"
}


TRUE_VALUES = {
    "true",
    "yes",
    "y",
    "1",
    "on",
    "active",
    "enabled"
}


FALSE_VALUES = {
    "false",
    "no",
    "n",
    "0",
    "off",
    "inactive",
    "disabled"
}


ID_NAME_PATTERNS = (
    "id",
    "_id",
    "id_",
    "uuid",
    "guid",
    "code",
    "_code",
    "key",
    "_key",
    "number",
    "_number",
    "no",
    "_no"
)


DATE_NAME_HINTS = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "created",
    "updated",
    "modified",
    "joined",
    "registered",
    "birth",
    "dob",
    "order_date",
    "transaction_date"
}


TEXT_NAME_HINTS = {
    "description",
    "comment",
    "comments",
    "review",
    "message",
    "notes",
    "note",
    "address",
    "summary",
    "feedback",
    "text"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_column_name(column):
    """
    Convert a column name into a normalized lowercase form.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().lower()
    ).strip("_")


def _non_null_series(series):
    """
    Return values that are not missing.
    """

    return series.dropna()


def _string_series(series):
    """
    Convert non-null values to normalized strings.
    """

    return (
        series.dropna()
        .astype(str)
        .str.strip()
    )


def _safe_ratio(
    numerator,
    denominator
):
    """
    Avoid divide-by-zero problems.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# ============================================================
# NULL-LIKE VALUE DETECTION
# ============================================================

def detect_null_like_values(
    series
):
    """
    Detect textual values that probably represent missing data.

    Example:
        ""
        "N/A"
        "null"
        "None"
        "missing"
    """

    if series.empty:
        return {
            "count": 0,
            "ratio": 0.0,
            "values": []
        }

    string_values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    normalized = (
        string_values.str.lower()
    )

    mask = (
        normalized.isin(
            NULL_LIKE_VALUES
        )
    )

    detected = (
        string_values[
            mask
        ]
    )

    values = (
        detected
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    return {

        "count":
            int(
                mask.sum()
            ),

        "ratio":
            round(
                _safe_ratio(
                    mask.sum(),
                    len(series)
                ),
                4
            ),

        "values":
            values
    }


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def _has_identifier_name(
    column_name
):
    """
    Determine whether the column name strongly suggests
    an identifier.
    """

    name = (
        _normalize_column_name(
            column_name
        )
    )

    if name == "id":
        return True

    if name.startswith(
        "id_"
    ):
        return True

    if name.endswith(
        "_id"
    ):
        return True

    if (
        "uuid" in name
        or "guid" in name
    ):
        return True

    if name.endswith(
        "_code"
    ):
        return True

    if name.endswith(
        "_key"
    ):
        return True

    if name in {
        "customer_number",
        "account_number",
        "invoice_number",
        "order_number",
        "transaction_number",
        "employee_number"
    }:
        return True

    return False


def _looks_like_identifier_values(
    series
):
    """
    Detect highly unique values that behave like identifiers.
    """

    values = (
        series.dropna()
    )

    count = len(
        values
    )

    if count < 5:
        return False

    unique_count = (
        values.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count
        /
        count
    )

    if unique_ratio < 0.95:
        return False

    # Numeric sequential values can be IDs.

    if pd.api.types.is_integer_dtype(
        values
    ):

        return True

    # String identifiers such as:
    # CUST001
    # ORD-192
    # EMP1001

    string_values = (
        values
        .astype(str)
        .str.strip()
    )

    structured_ratio = (
        string_values
        .str.match(
            r"^[A-Za-z]*[-_]?\d+[A-Za-z0-9_-]*$"
        )
        .mean()
    )

    return (
        structured_ratio >= 0.8
    )


# ============================================================
# BOOLEAN DETECTION
# ============================================================

def _detect_boolean(
    series
):
    """
    Detect boolean-like columns.

    Handles:
        True / False
        Yes / No
        Y / N
        1 / 0
        Active / Inactive
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return True

    values = (
        _string_series(
            series
        )
        .str.lower()
    )

    if values.empty:
        return False

    unique_values = set(
        values.unique()
    )

    allowed = (
        TRUE_VALUES
        |
        FALSE_VALUES
    )

    if not unique_values.issubset(
        allowed
    ):
        return False

    has_true = bool(
        unique_values
        &
        TRUE_VALUES
    )

    has_false = bool(
        unique_values
        &
        FALSE_VALUES
    )

    return (
        has_true
        and
        has_false
    )


# ============================================================
# NUMERIC-LOOKING STRING DETECTION
# ============================================================

def _numeric_conversion(
    series
):
    """
    Attempt numeric conversion safely.

    Handles:
        1200
        "1200"
        "1,200"
        "$1200"
        "₹1,500"
        "25%"
    """

    values = (
        _string_series(
            series
        )
    )

    if values.empty:

        return (
            pd.Series(
                dtype=float
            ),
            0.0
        )

    cleaned = (
        values
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            r"[$₹€£]",
            "",
            regex=True
        )
        .str.replace(
            "%",
            "",
            regex=False
        )
        .str.strip()
    )

    numeric = (
        pd.to_numeric(
            cleaned,
            errors="coerce"
        )
    )

    ratio = (
        numeric.notna().mean()
    )

    return (
        numeric,
        float(ratio)
    )


# ============================================================
# DATETIME DETECTION
# ============================================================

def _has_date_name_hint(
    column_name
):
    """
    Detect date/time hints from the column name.
    """

    name = (
        _normalize_column_name(
            column_name
        )
    )

    tokens = set(
        name.split("_")
    )

    if (
        tokens
        &
        DATE_NAME_HINTS
    ):
        return True

    return any(
        hint in name
        for hint in (
            "date",
            "timestamp",
            "datetime",
            "created_at",
            "updated_at",
            "birth"
        )
    )


def _parse_datetime_values(
    series
):
    """
    Attempt datetime conversion while avoiding Pandas'
    ambiguous-format warning.

    We first try strict/common formats and only use mixed
    parsing as a controlled fallback.
    """

    values = (
        _string_series(
            series
        )
    )

    if values.empty:

        return (
            pd.Series(
                dtype="datetime64[ns]"
            ),
            0.0
        )

    # Avoid interpreting ordinary numeric columns as dates.

    numeric_ratio = (
        pd.to_numeric(
            values,
            errors="coerce"
        )
        .notna()
        .mean()
    )

    if numeric_ratio >= 0.95:

        return (
            pd.Series(
                pd.NaT,
                index=values.index
            ),
            0.0
        )

    formats = [

        "%Y-%m-%d",
        "%Y/%m/%d",

        "%d-%m-%Y",
        "%d/%m/%Y",

        "%m-%d-%Y",
        "%m/%d/%Y",

        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",

        "%Y-%m-%dT%H:%M:%S",

        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",

        "%Y-%m",
        "%Y/%m",

        "%b %d %Y",
        "%B %d %Y",

        "%d %b %Y",
        "%d %B %Y"
    ]

    best_result = None
    best_ratio = 0.0

    for date_format in formats:

        converted = (
            pd.to_datetime(
                values,
                format=date_format,
                errors="coerce"
            )
        )

        ratio = (
            converted
            .notna()
            .mean()
        )

        if ratio > best_ratio:

            best_ratio = (
                float(
                    ratio
                )
            )

            best_result = (
                converted
            )

        if ratio >= 0.95:
            break

    # Pandas 2.x supports format="mixed".
    # It avoids the warning produced by unconstrained parsing.

    if best_ratio < 0.8:

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    UserWarning
                )

                mixed = (
                    pd.to_datetime(
                        values,
                        format="mixed",
                        errors="coerce"
                    )
                )

            mixed_ratio = (
                mixed.notna().mean()
            )

            if mixed_ratio > best_ratio:

                best_result = mixed

                best_ratio = float(
                    mixed_ratio
                )

        except (
            TypeError,
            ValueError
        ):

            pass

    if best_result is None:

        best_result = (
            pd.Series(
                pd.NaT,
                index=values.index
            )
        )

    return (
        best_result,
        best_ratio
    )


# ============================================================
# TEXT / CATEGORY HELPERS
# ============================================================

def _average_string_length(
    series
):
    """
    Calculate average text length.
    """

    values = (
        _string_series(
            series
        )
    )

    if values.empty:
        return 0.0

    return float(
        values.str.len().mean()
    )


def _looks_like_long_text(
    series,
    column_name
):
    """
    Detect free-form text.
    """

    name = (
        _normalize_column_name(
            column_name
        )
    )

    if any(
        hint in name
        for hint in TEXT_NAME_HINTS
    ):
        return True

    average_length = (
        _average_string_length(
            series
        )
    )

    return (
        average_length >= 40
    )


def _is_categorical(
    series
):
    """
    Determine whether a column behaves like a category.
    """

    values = (
        series.dropna()
    )

    count = len(
        values
    )

    if count == 0:
        return False

    unique_count = (
        values.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count
        /
        count
    )

    # Small number of categories.

    if unique_count <= 20:
        return True

    # Larger datasets can have more than 20 legitimate
    # categories.

    if (
        unique_count <= 100
        and
        unique_ratio <= 0.20
    ):
        return True

    # Relative cardinality.

    if (
        count >= 50
        and
        unique_ratio <= 0.05
    ):
        return True

    return False


# ============================================================
# SEMANTIC TYPE DETECTION
# ============================================================

def detect_column_type(
    df,
    column
):
    """
    Detect the semantic type of a DataFrame column.

    Possible values:

        Identifier
        Boolean
        Datetime
        Numerical
        Categorical
        Text
        Empty
    """

    series = (
        df[column]
    )

    column_name = (
        _normalize_column_name(
            column
        )
    )

    non_null = (
        series.dropna()
    )

    # --------------------------------------------------------
    # EMPTY COLUMN
    # --------------------------------------------------------

    if non_null.empty:

        return "Empty"

    # --------------------------------------------------------
    # BOOLEAN
    #
    # Must happen before numeric because Python/Pandas bool
    # values can behave numerically.
    # --------------------------------------------------------

    if _detect_boolean(
        series
    ):

        return "Boolean"

    # --------------------------------------------------------
    # IDENTIFIER
    # --------------------------------------------------------

    if _has_identifier_name(
        column_name
    ):

        return "Identifier"

    # --------------------------------------------------------
    # NATIVE DATETIME
    # --------------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):

        return "Datetime"

    # --------------------------------------------------------
    # NATIVE NUMERIC
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        series
    ):

        # Numeric unique columns may still be identifiers.

        if _looks_like_identifier_values(
            series
        ):

            # Only infer an ID from values when the column
            # behaves very strongly like one.

            unique_ratio = (
                series.nunique(
                    dropna=True
                )
                /
                len(
                    non_null
                )
            )

            if (
                unique_ratio >= 0.98
                and
                any(
                    hint in column_name
                    for hint in (
                        "number",
                        "code",
                        "key",
                        "serial"
                    )
                )
            ):

                return "Identifier"

        return "Numerical"

    # --------------------------------------------------------
    # DATETIME STRING
    # --------------------------------------------------------

    _, datetime_ratio = (
        _parse_datetime_values(
            series
        )
    )

    if (
        datetime_ratio >= 0.90
        and
        _has_date_name_hint(
            column_name
        )
    ):

        return "Datetime"

    # Strong date data can still be detected without a
    # date-like column name.

    if datetime_ratio >= 0.98:

        return "Datetime"

    # --------------------------------------------------------
    # NUMERIC-LOOKING STRING
    # --------------------------------------------------------

    _, numeric_ratio = (
        _numeric_conversion(
            series
        )
    )

    if numeric_ratio >= 0.95:

        return "Numerical"

    # --------------------------------------------------------
    # LONG TEXT
    # --------------------------------------------------------

    if _looks_like_long_text(
        series,
        column_name
    ):

        return "Text"

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if _is_categorical(
        series
    ):

        return "Categorical"

    # --------------------------------------------------------
    # IDENTIFIER-LIKE STRING
    # --------------------------------------------------------

    if _looks_like_identifier_values(
        series
    ):

        return "Identifier"

    return "Text"


# ============================================================
# COLUMN PROFILE
# ============================================================

def analyze_column(
    df,
    column
):
    """
    Generate semantic metadata for one column.
    """

    series = (
        df[column]
    )

    row_count = (
        len(series)
    )

    non_null_count = int(
        series.notna().sum()
    )

    missing_count = int(
        series.isna().sum()
    )

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    missing_ratio = (
        _safe_ratio(
            missing_count,
            row_count
        )
    )

    unique_ratio = (
        _safe_ratio(
            unique_count,
            non_null_count
        )
    )

    detected_type = (
        detect_column_type(
            df,
            column
        )
    )

    null_like = (
        detect_null_like_values(
            series
        )
    )

    _, numeric_ratio = (
        _numeric_conversion(
            series
        )
    )

    _, datetime_ratio = (
        _parse_datetime_values(
            series
        )
    )

    profile = {

        # Existing fields preserved

        "pandas_dtype":
            str(
                series.dtype
            ),

        "detected_type":
            detected_type,

        "missing_values":
            missing_count,

        "unique_values":
            unique_count,

        # New semantic intelligence

        "non_null_values":
            non_null_count,

        "missing_ratio":
            round(
                missing_ratio,
                4
            ),

        "unique_ratio":
            round(
                unique_ratio,
                4
            ),

        "cardinality":
            unique_count,

        "null_like_values":
            null_like,

        "numeric_parse_ratio":
            round(
                numeric_ratio,
                4
            ),

        "datetime_parse_ratio":
            round(
                datetime_ratio,
                4
            ),

        "average_text_length":
            round(
                _average_string_length(
                    series
                ),
                2
            ),

        "is_identifier":
            detected_type
            == "Identifier",

        "is_numeric":
            detected_type
            == "Numerical",

        "is_datetime":
            detected_type
            == "Datetime",

        "is_categorical":
            detected_type
            == "Categorical",

        "is_text":
            detected_type
            == "Text",

        "is_boolean":
            detected_type
            == "Boolean"
    }

    # --------------------------------------------------------
    # NUMERIC PROFILE
    # --------------------------------------------------------

    if detected_type == "Numerical":

        if pd.api.types.is_numeric_dtype(
            series
        ):

            numeric_values = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
            )

        else:

            numeric_values, _ = (
                _numeric_conversion(
                    series
                )
            )

        numeric_values = (
            numeric_values.dropna()
        )

        if not numeric_values.empty:

            profile[
                "numeric_summary"
            ] = {

                "min":
                    float(
                        numeric_values.min()
                    ),

                "max":
                    float(
                        numeric_values.max()
                    ),

                "mean":
                    float(
                        numeric_values.mean()
                    ),

                "median":
                    float(
                        numeric_values.median()
                    ),

                "std":
                    (
                        float(
                            numeric_values.std()
                        )
                        if len(
                            numeric_values
                        ) > 1
                        else 0.0
                    )
            }

    # --------------------------------------------------------
    # CATEGORICAL / BOOLEAN PROFILE
    # --------------------------------------------------------

    if detected_type in {
        "Categorical",
        "Boolean"
    }:

        counts = (
            series
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
        )

        profile[
            "top_values"
        ] = {

            str(key):
                int(value)

            for key, value
            in counts.items()
        }

    # --------------------------------------------------------
    # DATETIME PROFILE
    # --------------------------------------------------------

    if detected_type == "Datetime":

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            dates = (
                pd.to_datetime(
                    series,
                    errors="coerce"
                )
            )

        else:

            dates, _ = (
                _parse_datetime_values(
                    series
                )
            )

        dates = (
            dates.dropna()
        )

        if not dates.empty:

            profile[
                "datetime_summary"
            ] = {

                "min":
                    dates.min().isoformat(),

                "max":
                    dates.max().isoformat()
            }

    return profile


# ============================================================
# DATASET SCHEMA ANALYSIS
# ============================================================

def analyze_schema(
    df
):
    """
    Analyze the semantic schema of the complete dataset.

    This function intentionally retains the same return shape
    expected by the rest of InsightFlow:

        {
            column_name: {
                ...
            }
        }

    Additional metadata is added inside each column profile.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "analyze_schema expects a Pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Cannot analyze the schema of an empty dataset."
        )

    schema = {}

    for column in df.columns:

        schema[
            column
        ] = (
            analyze_column(
                df,
                column
            )
        )

    return schema


# ============================================================
# SEMANTIC TYPE GROUPS
# ============================================================

def get_columns_by_type(
    schema
):
    """
    Group columns by semantic type.

    Useful for:
        cleaning
        EDA
        visualization
        statistical analysis
    """

    groups = {

        "Identifier": [],
        "Boolean": [],
        "Datetime": [],
        "Numerical": [],
        "Categorical": [],
        "Text": [],
        "Empty": []
    }

    for column, info in schema.items():

        detected_type = (
            info.get(
                "detected_type",
                "Text"
            )
        )

        groups.setdefault(
            detected_type,
            []
        ).append(
            column
        )

    return groups


# ============================================================
# PRINT REPORT
# ============================================================

def print_schema_report(
    df,
    schema
):

    print(
        "\n========================================"
    )

    print(
        "        DATASET SCHEMA INTELLIGENCE"
    )

    print(
        "========================================"
    )

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    groups = (
        get_columns_by_type(
            schema
        )
    )

    print(
        "\nSemantic Types:"
    )

    for semantic_type, columns in groups.items():

        if columns:

            print(
                f"  {semantic_type}: "
                f"{len(columns)}"
            )

    print(
        "\nColumn Analysis:\n"
    )

    for column, info in schema.items():

        print(
            f"Column: {column}"
        )

        print(
            "  Pandas Type      : "
            f"{info['pandas_dtype']}"
        )

        print(
            "  Semantic Type    : "
            f"{info['detected_type']}"
        )

        print(
            "  Missing          : "
            f"{info['missing_values']}"
        )

        print(
            "  Missing Ratio    : "
            f"{info['missing_ratio']:.2%}"
        )

        print(
            "  Unique           : "
            f"{info['unique_values']}"
        )

        print(
            "  Unique Ratio     : "
            f"{info['unique_ratio']:.2%}"
        )

        print(
            "  Numeric Parse    : "
            f"{info['numeric_parse_ratio']:.2%}"
        )

        print(
            "  Datetime Parse   : "
            f"{info['datetime_parse_ratio']:.2%}"
        )

        null_like = (
            info.get(
                "null_like_values",
                {}
            )
        )

        if null_like.get(
            "count",
            0
        ):

            print(
                "  Null-like Values : "
                f"{null_like['count']}"
            )

        print(
            "-" * 40
        )