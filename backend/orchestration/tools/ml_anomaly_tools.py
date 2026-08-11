from langchain_core.tools import tool

import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.services.dataset_manager import (
    dataset_manager
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONTAMINATION = 0.05

MIN_NUMERIC_COLUMNS = 1

MIN_ROWS_REQUIRED = 10


# ============================================================
# ML ANOMALY DETECTION
# ============================================================

@tool
def run_ml_anomaly_detection(
    dataset_id: str,
    contamination: float = DEFAULT_CONTAMINATION
) -> dict:
    """
    Detect multivariate anomalies using
    Scikit-learn Isolation Forest.

    This tool is separate from the existing
    deterministic anomaly engine.

    Existing anomaly detection includes:
        - IQR outliers
        - rare categories
        - missingness
        - duplicates
        - identifier anomalies
        - datetime anomalies
        - low variance

    This tool adds:
        - machine-learning-based multivariate
          anomaly detection

    The LLM does NOT decide which rows are
    anomalous.

    Isolation Forest makes the mathematical
    anomaly detection decision.
    """

    # ========================================================
    # 1. VALIDATE DATASET ID
    # ========================================================

    if not isinstance(
        dataset_id,
        str
    ):

        return {
            "success": False,
            "error": (
                "dataset_id must be a string."
            )
        }

    dataset_id = dataset_id.strip()

    if not dataset_id:

        return {
            "success": False,
            "error": (
                "dataset_id cannot be empty."
            )
        }

    # ========================================================
    # 2. VALIDATE CONTAMINATION
    # ========================================================

    try:

        contamination = float(
            contamination
        )

    except (
        TypeError,
        ValueError
    ):

        return {
            "success": False,
            "error": (
                "contamination must be a number "
                "between 0 and 0.5."
            )
        }

    if not (
        0 < contamination <= 0.5
    ):

        return {
            "success": False,
            "error": (
                "contamination must be greater "
                "than 0 and less than or equal to 0.5."
            )
        }

    # ========================================================
    # 3. GET PREPARED DATASET
    # ========================================================

    dataset = (
        dataset_manager.get_dataset(
            dataset_id
        )
    )

    if dataset is None:

        return {
            "success": False,
            "error": (
                f"Dataset '{dataset_id}' does not exist "
                "or has expired."
            )
        }

    # ========================================================
    # 4. GET CLEANED DATAFRAME
    # ========================================================

    df = dataset.get(
        "cleaned_df"
    )

    if df is None:

        return {
            "success": False,
            "error": (
                "Prepared dataset does not contain "
                "a cleaned DataFrame."
            )
        }

    if not isinstance(
        df,
        pd.DataFrame
    ):

        return {
            "success": False,
            "error": (
                "Prepared dataset does not contain "
                "a valid Pandas DataFrame."
            )
        }

    if df.empty:

        return {
            "success": False,
            "error": (
                "Cannot perform ML anomaly detection "
                "on an empty dataset."
            )
        }

    if len(df) < MIN_ROWS_REQUIRED:

        return {
            "success": False,
            "error": (
                f"Isolation Forest requires at least "
                f"{MIN_ROWS_REQUIRED} rows for this tool."
            )
        }

    # ========================================================
    # 5. SELECT NUMERICAL COLUMNS
    # ========================================================

    numeric_df = (
        df.select_dtypes(
            include="number"
        ).copy()
    )

    if (
        numeric_df.shape[1]
        < MIN_NUMERIC_COLUMNS
    ):

        return {
            "success": False,
            "error": (
                "No suitable numerical columns were "
                "found for Isolation Forest."
            )
        }

    # ========================================================
    # 6. REMOVE CONSTANT COLUMNS
    # ========================================================

    variable_columns = [

        column

        for column
        in numeric_df.columns

        if numeric_df[
            column
        ].nunique(
            dropna=True
        ) > 1
    ]

    if not variable_columns:

        return {
            "success": False,
            "error": (
                "All numerical columns are constant. "
                "Isolation Forest cannot identify "
                "meaningful anomalies."
            )
        }

    numeric_df = (
        numeric_df[
            variable_columns
        ].copy()
    )

    # ========================================================
    # 7. HANDLE MISSING NUMERICAL VALUES
    # ========================================================

    for column in numeric_df.columns:

        median = (
            numeric_df[
                column
            ].median()
        )

        if pd.isna(
            median
        ):

            median = 0.0

        numeric_df[
            column
        ] = numeric_df[
            column
        ].fillna(
            median
        )

    # ========================================================
    # 8. TRAIN ISOLATION FOREST
    # ========================================================

    model = IsolationForest(

        n_estimators=100,

        contamination=contamination,

        random_state=42
    )

    predictions = (
        model.fit_predict(
            numeric_df
        )
    )

    # ========================================================
    # 9. GET ANOMALY SCORES
    # ========================================================

    decision_scores = (
        model.decision_function(
            numeric_df
        )
    )

    # Isolation Forest returns:
    #
    #  1  = normal
    # -1  = anomaly

    anomaly_mask = (
        predictions == -1
    )

    anomaly_indices = (
        df.index[
            anomaly_mask
        ].tolist()
    )

    # ========================================================
    # 10. BUILD ANOMALY ROWS
    # ========================================================

    anomaly_rows = []

    for position, index in enumerate(
        df.index
    ):

        if not anomaly_mask[
            position
        ]:

            continue

        row = {

            "row_index":
                index,

            "anomaly_score":
                round(
                    float(
                        decision_scores[
                            position
                        ]
                    ),
                    6
                )
        }

        # ----------------------------------------------------
        # Add numerical values from the original DataFrame
        # ----------------------------------------------------

        for column in numeric_df.columns:

            value = df.loc[
                index,
                column
            ]

            if pd.isna(
                value
            ):

                row[
                    str(column)
                ] = None

            else:

                try:

                    row[
                        str(column)
                    ] = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    row[
                        str(column)
                    ] = str(
                        value
                    )

        anomaly_rows.append(
            row
        )

    # ========================================================
    # 11. SORT MOST ANOMALOUS FIRST
    # ========================================================

    anomaly_rows.sort(

        key=lambda item:
            item.get(
                "anomaly_score",
                0
            )
    )

    # ========================================================
    # 12. SUMMARY
    # ========================================================

    anomaly_count = len(
        anomaly_rows
    )

    anomaly_percentage = (

        anomaly_count
        /
        len(df)
        *
        100

        if len(df)

        else 0.0
    )

    # ========================================================
    # 13. BUILD STRUCTURED ML RESULT
    # ========================================================

    ml_anomaly_result = {

        "dataset_id":
            dataset_id,

        "algorithm":
            "Isolation Forest",

        "model_type":
            "unsupervised machine learning",

        "rows_analyzed":
            int(
                len(df)
            ),

        "features_used":
            [
                str(column)
                for column
                in numeric_df.columns
            ],

        "feature_count":
            int(
                len(
                    numeric_df.columns
                )
            ),

        "contamination":
            contamination,

        "anomaly_count":
            int(
                anomaly_count
            ),

        "anomaly_percentage":
            round(
                anomaly_percentage,
                2
            ),

        "anomaly_indices":
            anomaly_indices,

        "anomalies":
            anomaly_rows
    }

    # ========================================================
    # 14. RETURN RESULT
    # ========================================================

    return {

        "success":
            True,

        "ml_anomaly_result":
            ml_anomaly_result
    }