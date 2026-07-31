from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from backend.services.dataset_manager import dataset_manager
from backend.services.llm_service import LLMService
from backend.agents.insight_agent import InsightAgent


def _records_to_dataframe(
    records: Optional[list[dict[str, Any]]]
) -> pd.DataFrame:
    """
    Convert SQL-result records into a DataFrame.
    """

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


@tool
def generate_analytical_insight(
    dataset_id: str,
    question: str,
    sql_result: Optional[list[dict[str, Any]]] = None,
    generated_sql: Optional[str] = None,
) -> dict:
    """
    Generate a natural-language analytical insight for an InsightFlow
    dataset.

    Use this tool after analytical operations such as SQL execution,
    statistical analysis, anomaly detection, or visualization when the
    user needs an explanation, interpretation, business insight, or
    summary of the analytical result.
    """

    # -------------------------------------------------
    # Validate inputs
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Retrieve prepared dataset
    # -------------------------------------------------

    dataset = dataset_manager.get_dataset(
        dataset_id
    )

    if dataset is None:
        return {
            "success": False,
            "dataset_id": dataset_id,
            "question": question,
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
            "dataset_id": dataset_id,
            "question": question,
            "error": (
                "Prepared dataset does not contain "
                "a cleaned DataFrame."
            ),
        }

    # -------------------------------------------------
    # Build analytical result DataFrame
    # -------------------------------------------------

    result_df = _records_to_dataframe(
        sql_result
    )

    # If no SQL result exists, give the insight agent
    # access to the prepared source dataset.
    if result_df.empty:
        result_df = source_df.copy()

    # -------------------------------------------------
    # Generate insight
    # -------------------------------------------------

    try:

        llm_service = LLMService()

        insight_agent = InsightAgent(
            llm_service
        )

        insight = insight_agent.generate_insight(
            question=question,
            sql=generated_sql or "",
            result_df=result_df,
        )

        if insight is None:
            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "insight": None,
                "insight_source": None,
                "llm_success": False,
                "error": (
                    "Insight Agent did not return "
                    "an analytical insight."
                ),
            }

        # ---------------------------------------------
        # Support current InsightAgent response shapes
        # ---------------------------------------------

        if isinstance(insight, dict):

            insight_text = (
                insight.get("insight")
                or insight.get("text")
                or insight.get("response")
            )

            insight_source = insight.get(
                "source",
                "llm"
            )

            llm_success = insight.get(
                "llm_success",
                insight_source == "llm"
            )

            llm_error = insight.get(
                "llm_error"
            )

        else:

            insight_text = str(
                insight
            )

            insight_source = "llm"

            llm_success = True

            llm_error = None

        return {
            "success": True,

            "dataset_id": dataset_id,

            "question": question,

            "insight": insight_text,

            "insight_source": insight_source,

            "llm_success": bool(
                llm_success
            ),

            "llm_error": llm_error,

            "error": None,
        }

    except Exception as exc:

        return {
            "success": False,

            "dataset_id": dataset_id,

            "question": question,

            "insight": None,

            "insight_source": None,

            "llm_success": False,

            "llm_error": str(exc),

            "error": str(exc),
        }