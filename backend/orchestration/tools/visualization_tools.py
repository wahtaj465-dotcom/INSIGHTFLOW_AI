from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from backend.services.dataset_manager import dataset_manager
from backend.services.visualization_service import (
    VisualizationService,
)


visualization_service = VisualizationService()


def _records_to_dataframe(
    records: Optional[list[dict[str, Any]]]
) -> Optional[pd.DataFrame]:
    """
    Convert SQL-result records back into a DataFrame.

    Returns None when no records were supplied.
    """

    if not records:
        return None

    return pd.DataFrame(records)


@tool
def generate_visualization(
    dataset_id: str,
    question: str,
    sql_result: Optional[list[dict[str, Any]]] = None,
) -> dict:
    """
    Generate a visualization for an InsightFlow analytical request.

    Use this tool when the user asks to plot, chart, visualize,
    compare distributions, show relationships, display trends,
    create KPIs, or otherwise visually represent dataset results.

    The tool can use both the SQL query result and the prepared
    source dataset when choosing and constructing the chart.
    """

    # -----------------------------------------
    # Validate arguments
    # -----------------------------------------

    if not isinstance(dataset_id, str):
        return {
            "success": False,
            "error": "dataset_id must be a string.",
        }

    if not isinstance(question, str):
        return {
            "success": False,
            "error": "question must be a string.",
        }

    dataset_id = dataset_id.strip()
    question = question.strip()

    if not dataset_id:
        return {
            "success": False,
            "error": "dataset_id cannot be empty.",
        }

    if not question:
        return {
            "success": False,
            "error": "question cannot be empty.",
        }

    # -----------------------------------------
    # Retrieve prepared dataset
    # -----------------------------------------

    dataset = dataset_manager.get_dataset(
        dataset_id
    )

    if dataset is None:
        return {
            "success": False,
            "error": (
                f"Dataset '{dataset_id}' does not "
                "exist or has expired."
            ),
        }

    source_df = dataset.get(
        "cleaned_df"
    )

    if source_df is None:
        return {
            "success": False,
            "error": (
                "Prepared dataset does not contain "
                "a cleaned DataFrame."
            ),
        }

    # -----------------------------------------
    # Build SQL-result DataFrame
    # -----------------------------------------

    result_df = _records_to_dataframe(
        sql_result
    )

    if result_df is None:
        result_df = source_df.copy()

    # -----------------------------------------
    # Generate chart
    # -----------------------------------------

    try:
        chart = (
            visualization_service
            .generate_result_chart(
                df=result_df,
                question=question,
                source_df=source_df,
                schema=dataset.get(
                    "cleaned_schema",
                    {}
                ),
            )
        )

        if chart is None:
            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "chart": None,
                "error": (
                    "No suitable visualization "
                    "could be generated."
                ),
            }

        return {
            "success": True,
            "dataset_id": dataset_id,
            "question": question,
            "chart": chart,
            "chart_type": chart.get(
                "chart_type"
            ),
            "title": chart.get(
                "title"
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "dataset_id": dataset_id,
            "question": question,
            "chart": None,
            "error": str(exc),
        }