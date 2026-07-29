import pandas as pd


# ============================================================
# 1. AUTOMATED EDA
# ============================================================

def perform_eda(df, schema):
    """
    Perform automated exploratory data analysis.

    Returns:
    - Dataset overview
    - Numerical statistics
    - Categorical statistics
    - Correlations
    - Distribution/skew analysis
    - Datetime analysis
    - Chart recommendations
    """

    eda_results = {
        "overview": {},
        "numerical": {},
        "categorical": {},
        "correlations": {},
        "distributions": {},
        "datetime": {},
        "chart_recommendations": []
    }

    # ========================================================
    # 1. DATASET OVERVIEW
    # ========================================================

    eda_results["overview"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing": int(df.isna().sum().sum())
    }

    # ========================================================
    # 2. NUMERICAL ANALYSIS
    # ========================================================

    numerical_columns = []

    for column, info in schema.items():

        if info["detected_type"] == "Numerical":

            numerical_columns.append(column)

            series = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            eda_results["numerical"][column] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std())
            }

    # ========================================================
    # 3. CATEGORICAL ANALYSIS
    # ========================================================

    categorical_columns = []

    for column, info in schema.items():

        if info["detected_type"] == "Categorical":

            categorical_columns.append(column)

            counts = df[column].value_counts(
                dropna=False
            )

            eda_results["categorical"][column] = {
                "unique_values": int(
                    df[column].nunique(dropna=True)
                ),

                "most_common": (
                    counts.index[0]
                    if len(counts) > 0
                    else None
                ),

                "most_common_count": (
                    int(counts.iloc[0])
                    if len(counts) > 0
                    else 0
                ),

                "distribution": counts.to_dict()
            }

    # ========================================================
    # 4. CORRELATION ANALYSIS
    # ========================================================

    if len(numerical_columns) >= 2:

        numeric_df = df[
            numerical_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        correlation_matrix = numeric_df.corr()

        for i in range(len(numerical_columns)):

            for j in range(i + 1, len(numerical_columns)):

                column_1 = numerical_columns[i]
                column_2 = numerical_columns[j]

                correlation = correlation_matrix.loc[
                    column_1,
                    column_2
                ]

                if pd.notna(correlation):

                    key = f"{column_1} vs {column_2}"

                    eda_results["correlations"][key] = float(
                        correlation
                    )

    # ========================================================
    # 5. DISTRIBUTION / SKEW ANALYSIS
    # ========================================================

    for column in numerical_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 3:
            continue

        skewness = float(series.skew())

        if skewness > 1:

            shape = "Highly right-skewed"

        elif skewness > 0.5:

            shape = "Moderately right-skewed"

        elif skewness < -1:

            shape = "Highly left-skewed"

        elif skewness < -0.5:

            shape = "Moderately left-skewed"

        else:

            shape = "Approximately symmetric"

        eda_results["distributions"][column] = {
            "skewness": skewness,
            "shape": shape
        }

    # ========================================================
    # 6. DATETIME ANALYSIS
    # ========================================================

    datetime_columns = []

    for column, info in schema.items():

        if info["detected_type"] == "Datetime":

            datetime_columns.append(column)

            dates = pd.to_datetime(
                df[column],
                errors="coerce"
            ).dropna()

            if dates.empty:
                continue

            eda_results["datetime"][column] = {
                "valid_dates": int(len(dates)),
                "missing_dates": int(
                    df[column].isna().sum()
                ),
                "earliest": dates.min().strftime(
                    "%Y-%m-%d"
                ),
                "latest": dates.max().strftime(
                    "%Y-%m-%d"
                ),
                "range_days": int(
                    (dates.max() - dates.min()).days
                )
            }

    # ========================================================
    # 7. CHART RECOMMENDATIONS
    # ========================================================

    # Numerical columns -> Histogram + Box Plot

    for column in numerical_columns:

        eda_results["chart_recommendations"].append({
            "chart": "histogram",
            "x": column,
            "y": None,
            "reason": (
                f"Analyze the distribution of {column}."
            )
        })

        eda_results["chart_recommendations"].append({
            "chart": "box",
            "x": column,
            "y": None,
            "reason": (
                f"Identify spread and potential outliers "
                f"in {column}."
            )
        })

    # Categorical columns -> Bar Chart

    for column in categorical_columns:

        eda_results["chart_recommendations"].append({
            "chart": "bar",
            "x": column,
            "y": "count",
            "reason": (
                f"Compare frequency across {column} categories."
            )
        })

    # Numerical relationships -> Scatter Plot

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            column_1 = numerical_columns[i]
            column_2 = numerical_columns[j]

            eda_results["chart_recommendations"].append({
                "chart": "scatter",
                "x": column_1,
                "y": column_2,
                "reason": (
                    f"Explore the relationship between "
                    f"{column_1} and {column_2}."
                )
            })

    # Datetime + Numerical -> Line Chart

    for date_column in datetime_columns:

        for numeric_column in numerical_columns:

            eda_results["chart_recommendations"].append({
                "chart": "line",
                "x": date_column,
                "y": numeric_column,
                "reason": (
                    f"Analyze how {numeric_column} changes "
                    f"over time."
                )
            })

    return eda_results


# ============================================================
# 2. PRINT EDA REPORT
# ============================================================

def print_eda_report(eda_results):
    """
    Print automated EDA results.
    """

    print("\n==============================")
    print("          EDA REPORT")
    print("==============================")

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    overview = eda_results["overview"]

    print("\nDATASET OVERVIEW")

    print(f"Rows: {overview['rows']}")
    print(f"Columns: {overview['columns']}")
    print(
        f"Missing Values: "
        f"{overview['total_missing']}"
    )

    # ========================================================
    # NUMERICAL ANALYSIS
    # ========================================================

    print("\nNUMERICAL ANALYSIS")

    if not eda_results["numerical"]:

        print("No numerical columns detected.")

    for column, stats in eda_results["numerical"].items():

        print(f"\n{column}")

        print(f"  Count  : {stats['count']}")
        print(f"  Mean   : {stats['mean']:.2f}")
        print(f"  Median : {stats['median']:.2f}")
        print(f"  Minimum: {stats['min']:.2f}")
        print(f"  Maximum: {stats['max']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")

    # ========================================================
    # CATEGORICAL ANALYSIS
    # ========================================================

    print("\nCATEGORICAL ANALYSIS")

    if not eda_results["categorical"]:

        print("No categorical columns detected.")

    for column, stats in eda_results["categorical"].items():

        print(f"\n{column}")

        print(
            f"  Unique values: "
            f"{stats['unique_values']}"
        )

        print(
            f"  Most common: "
            f"{stats['most_common']}"
        )

        print(
            f"  Frequency: "
            f"{stats['most_common_count']}"
        )

        print(
            f"  Distribution: "
            f"{stats['distribution']}"
        )

    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    print("\nCORRELATION ANALYSIS")

    if not eda_results["correlations"]:

        print(
            "Not enough numerical columns "
            "for correlation analysis."
        )

    for pair, value in eda_results["correlations"].items():

        if abs(value) >= 0.7:
            strength = "Strong"

        elif abs(value) >= 0.4:
            strength = "Moderate"

        else:
            strength = "Weak"

        if value > 0:
            direction = "positive"

        elif value < 0:
            direction = "negative"

        else:
            direction = "no"

        print(
            f"\n{pair}: {value:.3f}"
        )

        print(
            f"  {strength} {direction} correlation"
        )

    # ========================================================
    # DISTRIBUTION ANALYSIS
    # ========================================================

    print("\nDISTRIBUTION ANALYSIS")

    if not eda_results["distributions"]:

        print(
            "Not enough numerical data "
            "for distribution analysis."
        )

    for column, result in eda_results[
        "distributions"
    ].items():

        print(f"\n{column}")

        print(
            f"  Skewness: "
            f"{result['skewness']:.3f}"
        )

        print(
            f"  Shape: "
            f"{result['shape']}"
        )

    # ========================================================
    # DATETIME ANALYSIS
    # ========================================================

    print("\nDATETIME ANALYSIS")

    if not eda_results["datetime"]:

        print("No usable datetime columns detected.")

    for column, result in eda_results["datetime"].items():

        print(f"\n{column}")

        print(
            f"  Valid dates : "
            f"{result['valid_dates']}"
        )

        print(
            f"  Missing     : "
            f"{result['missing_dates']}"
        )

        print(
            f"  Earliest    : "
            f"{result['earliest']}"
        )

        print(
            f"  Latest      : "
            f"{result['latest']}"
        )

        print(
            f"  Range       : "
            f"{result['range_days']} days"
        )

    # ========================================================
    # CHART RECOMMENDATIONS
    # ========================================================

    print("\nCHART RECOMMENDATIONS")

    if not eda_results["chart_recommendations"]:

        print("No charts recommended.")

    for index, chart in enumerate(
        eda_results["chart_recommendations"],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{chart['chart'].upper()}"
        )

        print(
            f"   X: {chart['x']}"
        )

        if chart["y"] is not None:

            print(
                f"   Y: {chart['y']}"
            )

        print(
            f"   Reason: {chart['reason']}"
        )