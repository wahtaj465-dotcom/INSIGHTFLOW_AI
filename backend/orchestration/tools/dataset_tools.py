from langchain_core.tools import tool

from backend.services.dataset_manager import (
    dataset_manager
)


@tool
def get_dataset_context(
    dataset_id: str
) -> dict:
    """
    Retrieve analytical context for a prepared InsightFlow dataset.

    Returns dataset dimensions, columns, schema, data-quality
    information, anomalies, EDA results, statistical findings,
    and available EDA charts.

    Use this tool before performing analytics when information
    about the dataset structure or existing analysis is needed.
    """

    if not isinstance(dataset_id, str):
        return {
            "success": False,
            "error": "dataset_id must be a string."
        }

    dataset_id = dataset_id.strip()

    if not dataset_id:
        return {
            "success": False,
            "error": "dataset_id cannot be empty."
        }

    dataset = dataset_manager.get_dataset(
        dataset_id
    )

    if dataset is None:
        return {
            "success": False,
            "error": (
                f"Dataset '{dataset_id}' does not exist "
                "or has expired."
            )
        }

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

    return {
        "success": True,

        "dataset_id": dataset_id,

        "rows": int(len(df)),

        "columns": [
            str(column)
            for column in df.columns
        ],

        "column_count": int(
            len(df.columns)
        ),

        "schema": dataset.get(
            "cleaned_schema",
            {}
        ),

        "quality_report": dataset.get(
            "cleaned_quality_report",
            {}
        ),

        "anomalies": dataset.get(
            "cleaned_anomalies",
            {}
        ),

        "eda_results": dataset.get(
            "eda_results",
            {}
        ),

        "statistical_findings": dataset.get(
            "statistical_findings",
            []
        ),

        "eda_charts": dataset.get(
            "eda_charts",
            []
        )
    }