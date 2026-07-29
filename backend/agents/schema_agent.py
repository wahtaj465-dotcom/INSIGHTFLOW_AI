import pandas as pd


def detect_column_type(df, column):
    """
    Detect the semantic type of a DataFrame column.
    """

    series = df[column]

    column_name = column.lower()

    # -----------------------------
    # 1. Detect possible ID columns
    # -----------------------------

    if (
        column_name == "id"
        or column_name.endswith("_id")
        or column_name.startswith("id_")
    ):
        return "Identifier"

    # -----------------------------
    # 2. Detect numeric columns
    # -----------------------------

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    # -----------------------------
    # 3. Detect date columns
    # -----------------------------

    if (
        "date" in column_name
        or "time" in column_name
        or "year" in column_name
    ):

        converted = pd.to_datetime(series, errors="coerce")

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.8:
            return "Datetime"

    # -----------------------------
    # 4. Detect categorical columns
    # -----------------------------

    unique_values = series.nunique(dropna=True)
    total_values = series.notna().sum()

    if total_values > 0:

        unique_ratio = unique_values / total_values

        # Few unique values usually indicate a category
        if unique_values <= 20 and unique_ratio <= 0.5:
            return "Categorical"

    # -----------------------------
    # 5. Otherwise treat as text
    # -----------------------------

    return "Text"


def analyze_schema(df):
    """
    Analyze the complete schema of a dataset.
    """

    schema = {}

    for column in df.columns:

        schema[column] = {
            "pandas_dtype": str(df[column].dtype),

            "detected_type": detect_column_type(
                df,
                column
            ),

            "missing_values": int(
                df[column].isna().sum()
            ),

            "unique_values": int(
                df[column].nunique(dropna=True)
            )
        }

    return schema


def print_schema_report(df, schema):

    print("\n==============================")
    print("       DATASET SCHEMA")
    print("==============================")

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn Analysis:\n")

    for column, info in schema.items():

        print(f"Column: {column}")
        print(f"  Pandas Type : {info['pandas_dtype']}")
        print(f"  Detected Type: {info['detected_type']}")
        print(f"  Missing     : {info['missing_values']}")
        print(f"  Unique      : {info['unique_values']}")

        print("-" * 30)