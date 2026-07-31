from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from backend.services.dataset_manager import dataset_manager
from backend.services.llm_service import LLMService
from backend.agents.insight_agent import InsightAgent


def _records_to_dataframe(records) -> pd.DataFrame:
    """
    Convert SQL results into a pandas DataFrame.
    """

    if isinstance(records, pd.DataFrame):
        return records.copy()

    if records is None:
        return pd.DataFrame()

    if isinstance(records, list):
        return pd.DataFrame(records)

    if isinstance(records, dict):
        return pd.DataFrame([records])

    return pd.DataFrame({
        "result": [records]
    })


@tool
def generate_analytical_insight(
    dataset_id: str,
    question: str,
    sql_result: Optional[list[dict[str, Any]]] = None,
    generated_sql: Optional[str] = None,
) -> dict:
    """
    Generate analytical insights for a prepared dataset.
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
    # Retrieve dataset
    # -------------------------------------------------

    dataset = dataset_manager.get_dataset(dataset_id)

    if dataset is None:
        return {
            "success": False,
            "dataset_id": dataset_id,
            "question": question,
            "error": f"Dataset '{dataset_id}' does not exist or has expired.",
        }

    source_df = dataset.get("cleaned_df")

    if source_df is None:
        return {
            "success": False,
            "dataset_id": dataset_id,
            "question": question,
            "error": "Prepared dataset does not contain a cleaned DataFrame.",
        }

    # -------------------------------------------------
    # Build DataFrame
    # -------------------------------------------------

    result_df = _records_to_dataframe(sql_result)

    if result_df.empty:
        result_df = source_df.copy()

    # -------------------------------------------------
    # Optional metadata
    # -------------------------------------------------

    quality_report = (
        dataset.get("quality_report")
        or dataset.get("quality")
    )

    anomalies = (
        dataset.get("anomalies")
        or dataset.get("anomaly_report")
    )

    eda_results = (
        dataset.get("eda_results")
        or dataset.get("eda")
    )

    # -------------------------------------------------
    # Generate insight
    # -------------------------------------------------

    try:

        llm_service = LLMService()

        insight_agent = InsightAgent(llm_service)

        insight = insight_agent.generate_insight(
            question=question,
            sql=generated_sql or "",
            result=result_df,
            quality_report=quality_report,
            anomalies=anomalies,
            eda_results=eda_results,
        )

        if insight is None:
            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "insight": None,
                "insight_source": None,
                "llm_success": False,
                "error": "Insight Agent did not return an analytical insight.",
            }

        # ---------------------------------------------
        # Normalize response
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

            llm_error = insight.get("llm_error")

        else:

            insight_text = str(insight)

            insight_source = "llm"

            llm_success = True

            llm_error = None

        return {
            "success": True,
            "dataset_id": dataset_id,
            "question": question,
            "insight": insight_text,
            "insight_source": insight_source,
            "llm_success": bool(llm_success),
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