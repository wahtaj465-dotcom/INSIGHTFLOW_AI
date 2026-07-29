import pandas as pd


# ============================================================
# 1. DATA QUALITY REPORT
# ============================================================

def generate_quality_report(df, schema):
    """
    Analyze a DataFrame for common data quality problems.

    Detects:
    1. Duplicate rows
    2. Missing values
    3. Invalid dates
    """

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {},
        "invalid_dates": {}
    }

    # --------------------------------------------------------
    # Detect missing values
    # --------------------------------------------------------

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count > 0:
            report["missing_values"][column] = missing_count

    # --------------------------------------------------------
    # Detect invalid dates
    # --------------------------------------------------------

    for column, info in schema.items():

        if info["detected_type"] == "Datetime":

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            # Invalid means:
            # original value exists, but conversion failed.
            invalid_mask = (
                converted.isna()
                & df[column].notna()
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count > 0:
                report["invalid_dates"][column] = invalid_count

    return report


# ============================================================
# 2. PRINT DATA QUALITY REPORT
# ============================================================

def print_quality_report(report):
    """
    Print the data quality report in a readable format.
    """

    print("\n==============================")
    print("      DATA QUALITY REPORT")
    print("==============================")

    print(f"\nRows: {report['total_rows']}")
    print(f"Columns: {report['total_columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing Values:")

    if report["missing_values"]:

        for column, count in report["missing_values"].items():
            print(f"  {column}: {count}")

    else:
        print("  None")

    # --------------------------------------------------------
    # Invalid dates
    # --------------------------------------------------------

    print("\nInvalid Dates:")

    if report["invalid_dates"]:

        for column, count in report["invalid_dates"].items():
            print(f"  {column}: {count}")

    else:
        print("  None")


# ============================================================
# 3. CLEAN DATASET
# ============================================================

def clean_dataset(df, schema):
    """
    Clean common data quality problems.

    Current strategies:
    1. Remove exact duplicate rows
    2. Standardize categorical / low-cardinality text columns
    3. Convert detected datetime columns
    4. Fill missing numerical values using median

    Returns:
        cleaned_df:
            Cleaned copy of the original DataFrame.

        cleaning_log:
            List describing operations performed.
    """

    # Preserve the original DataFrame
    cleaned_df = df.copy()

    cleaning_log = []

    # ========================================================
    # STEP 1 — REMOVE DUPLICATES
    # ========================================================

    duplicate_count = int(
        cleaned_df.duplicated().sum()
    )

    if duplicate_count > 0:

        cleaned_df = cleaned_df.drop_duplicates()

        cleaning_log.append(
            f"Removed {duplicate_count} duplicate row(s)."
        )

    # ========================================================
    # STEP 2 — STANDARDIZE CATEGORICAL / LOW-CARDINALITY TEXT
    # ========================================================

    for column, info in schema.items():

        detected_type = info["detected_type"]

        if detected_type in ["Categorical", "Text"]:

            unique_count = cleaned_df[column].nunique(
                dropna=True
            )

            # V1 heuristic:
            # Treat low-cardinality text as categorical.
            if unique_count <= 20:

                cleaned_df[column] = (
                    cleaned_df[column]
                    .astype("string")
                    .str.strip()
                    .str.title()
                )

                cleaning_log.append(
                    f"Standardized text/category column: {column}"
                )

    # ========================================================
    # STEP 3 — CONVERT DATETIME COLUMNS
    # ========================================================

    for column, info in schema.items():

        if info["detected_type"] == "Datetime":

            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                errors="coerce"
            )

            invalid_after_conversion = int(
                cleaned_df[column].isna().sum()
            )

            cleaning_log.append(
                f"Converted {column} to datetime "
                f"({invalid_after_conversion} "
                f"invalid/missing value(s))."
            )

    # ========================================================
    # STEP 4 — HANDLE MISSING NUMERICAL VALUES
    # ========================================================

    for column, info in schema.items():

        if info["detected_type"] == "Numerical":

            missing_count = int(
                cleaned_df[column].isna().sum()
            )

            if missing_count > 0:

                median_value = cleaned_df[column].median()

                if pd.notna(median_value):

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(median_value)
                    )

                    cleaning_log.append(
                        f"Filled {missing_count} missing value(s) "
                        f"in {column} using median = "
                        f"{median_value}."
                    )

                else:

                    cleaning_log.append(
                        f"Could not fill missing values in "
                        f"{column} because no valid median "
                        f"was available."
                    )

    return cleaned_df, cleaning_log


# ============================================================
# 4. PRINT CLEANING LOG
# ============================================================

def print_cleaning_log(cleaning_log):
    """
    Display every cleaning operation performed.
    """

    print("\n==============================")
    print("         CLEANING LOG")
    print("==============================")

    if not cleaning_log:

        print("\nNo cleaning operations required.")
        return

    for index, operation in enumerate(
        cleaning_log,
        start=1
    ):

        print(f"{index}. {operation}")


# ============================================================
# 5. DETECT NUMERIC ANOMALIES
# ============================================================

def detect_numeric_anomalies(df, schema):
    """
    Detect suspicious values in numerical columns.

    Detects:
    1. Negative values
    2. Statistical outliers using the IQR method

    Important:
    These values are FLAGGED only.
    They are NOT automatically modified.

    This is because a negative value or statistical outlier
    may be valid depending on the meaning of the column.
    """

    anomalies = {}

    for column, info in schema.items():

        # Only analyze numerical columns
        if info["detected_type"] != "Numerical":
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        column_anomalies = {
            "negative_values": [],
            "outliers": [],
            "lower_bound": None,
            "upper_bound": None
        }

        # ====================================================
        # 1. NEGATIVE VALUES
        # ====================================================

        negative_values = series[
            series < 0
        ]

        if not negative_values.empty:

            column_anomalies["negative_values"] = (
                negative_values.tolist()
            )

        # ====================================================
        # 2. IQR OUTLIER DETECTION
        # ====================================================

        # Avoid IQR analysis on extremely small samples
        if len(series) >= 4:

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr > 0:

                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)

                column_anomalies["lower_bound"] = float(
                    lower_bound
                )

                column_anomalies["upper_bound"] = float(
                    upper_bound
                )

                outliers = series[
                    (series < lower_bound)
                    | (series > upper_bound)
                ]

                column_anomalies["outliers"] = (
                    outliers.tolist()
                )

        # Store only columns where anomalies were found
        if (
            column_anomalies["negative_values"]
            or column_anomalies["outliers"]
        ):

            anomalies[column] = column_anomalies

    return anomalies


# ============================================================
# 6. PRINT ANOMALY REPORT
# ============================================================

def print_anomaly_report(anomalies):
    """
    Display suspicious numerical values.
    """

    print("\n==============================")
    print("       ANOMALY REPORT")
    print("==============================")

    if not anomalies:

        print("\nNo numerical anomalies detected.")
        return

    for column, details in anomalies.items():

        print(f"\nColumn: {column}")

        if details["negative_values"]:

            print(
                f"  Negative values: "
                f"{details['negative_values']}"
            )

        if details["outliers"]:

            print(
                f"  IQR outliers: "
                f"{details['outliers']}"
            )

        if (
            details["lower_bound"] is not None
            and details["upper_bound"] is not None
        ):

            print(
                f"  Expected IQR range: "
                f"{details['lower_bound']:.2f} "
                f"to {details['upper_bound']:.2f}"
            )


# ============================================================
# 7. COUNT UNIQUE ANOMALIES
# ============================================================

def count_anomalies(anomalies):
    """
    Count anomaly values without double-counting a value that
    appears in both negative_values and outliers.

    Example:
        -900 could be both negative AND an IQR outlier.

    It should count as one suspicious observation here,
    not two.
    """

    total = 0

    for details in anomalies.values():

        negative_values = details.get(
            "negative_values",
            []
        )

        outliers = details.get(
            "outliers",
            []
        )

        combined_values = (
            negative_values + outliers
        )

        # set() prevents the same value being counted twice
        unique_values = set(combined_values)

        total += len(unique_values)

    return total


# ============================================================
# 8. CALCULATE DATA QUALITY SCORE
# ============================================================

def calculate_quality_score(
    df,
    quality_report,
    anomalies
):
    """
    Calculate a heuristic data quality score from 0 to 100.

    The score considers:
    1. Missing values
    2. Duplicate rows
    3. Invalid dates
    4. Numerical anomalies

    IMPORTANT:
    This is a project-defined heuristic score.
    It is not a universal or standardized industry metric.
    """

    score = 100.0

    rows = max(
        len(df),
        1
    )

    columns = max(
        len(df.columns),
        1
    )

    total_cells = rows * columns

    # ========================================================
    # 1. MISSING VALUE PENALTY
    # Maximum penalty: 30
    # ========================================================

    total_missing = sum(
        quality_report["missing_values"].values()
    )

    missing_ratio = (
        total_missing / total_cells
    )

    missing_penalty = min(
        missing_ratio * 100,
        30
    )

    score -= missing_penalty

    # ========================================================
    # 2. DUPLICATE PENALTY
    # Maximum penalty: 20
    # ========================================================

    duplicate_ratio = (
        quality_report["duplicate_rows"]
        / rows
    )

    duplicate_penalty = min(
        duplicate_ratio * 100,
        20
    )

    score -= duplicate_penalty

    # ========================================================
    # 3. INVALID DATE PENALTY
    # Maximum penalty: 20
    # ========================================================

    total_invalid_dates = sum(
        quality_report["invalid_dates"].values()
    )

    invalid_date_ratio = (
        total_invalid_dates / rows
    )

    invalid_date_penalty = min(
        invalid_date_ratio * 100,
        20
    )

    score -= invalid_date_penalty

    # ========================================================
    # 4. NUMERIC ANOMALY PENALTY
    # Maximum penalty: 30
    # ========================================================

    anomaly_count = count_anomalies(
        anomalies
    )

    anomaly_ratio = (
        anomaly_count / total_cells
    )

    anomaly_penalty = min(
        anomaly_ratio * 100,
        30
    )

    score -= anomaly_penalty

    # Score cannot go below 0
    score = max(
        score,
        0
    )

    return round(
        score,
        2
    )


# ============================================================
# 9. PRINT QUALITY SCORE
# ============================================================

def print_quality_score(score, label="Data"):
    """
    Display the quality score with a simple rating.
    """

    if score >= 90:
        rating = "Excellent"

    elif score >= 75:
        rating = "Good"

    elif score >= 60:
        rating = "Needs Improvement"

    else:
        rating = "Poor"

    print("\n==============================")
    print("       DATA QUALITY SCORE")
    print("==============================")

    print(
        f"\n{label}: {score}/100"
    )

    print(
        f"Rating: {rating}"
    )


# ============================================================
# 10. BEFORE VS AFTER QUALITY COMPARISON
# ============================================================

def print_quality_comparison(
    before_score,
    after_score
):
    """
    Compare data quality before and after cleaning.
    """

    improvement = round(
        after_score - before_score,
        2
    )

    print("\n==============================")
    print("    DATA QUALITY COMPARISON")
    print("==============================")

    print(
        f"\nBefore Cleaning : "
        f"{before_score}/100"
    )

    print(
        f"After Cleaning  : "
        f"{after_score}/100"
    )

    if improvement > 0:

        print(
            f"Improvement     : "
            f"+{improvement} points"
        )

    elif improvement < 0:

        print(
            f"Change          : "
            f"{improvement} points"
        )

    else:

        print(
            "Improvement     : "
            "No change"
        )