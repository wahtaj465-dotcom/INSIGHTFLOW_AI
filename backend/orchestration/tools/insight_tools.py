from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from backend.services.dataset_manager import (
    dataset_manager
)

from backend.services.llm_service import (
    LLMService
)

from backend.agents.insight_agent import (
    InsightAgent
)


# ============================================================
# RECORDS → DATAFRAME
# ============================================================

def _records_to_dataframe(
    records
) -> pd.DataFrame:
    """
    Convert SQL results into a pandas DataFrame.
    """

    if isinstance(
        records,
        pd.DataFrame
    ):
        return records.copy()

    if records is None:
        return pd.DataFrame()

    if isinstance(
        records,
        list
    ):
        return pd.DataFrame(
            records
        )

    if isinstance(
        records,
        dict
    ):
        return pd.DataFrame(
            [records]
        )

    return pd.DataFrame({
        "result": [records]
    })


# ============================================================
# ANALYTICAL INSIGHT TOOL
# ============================================================

@tool
def generate_analytical_insight(
    dataset_id: str,
    question: str,
    sql_result: Optional[
        list[dict[str, Any]]
    ] = None,
    generated_sql: Optional[str] = None,
    ml_anomaly_result: Optional[
        dict[str, Any]
    ] = None,
) -> dict:
    """
    Generate analytical insights for a prepared dataset.

    The tool supports:

        - SQL analytical results
        - dataset quality information
        - existing deterministic anomaly detection
        - EDA results
        - Scikit-learn ML anomaly detection

    The ML anomaly result is supplied as additional
    mathematical evidence to the Insight Agent.
    """

    # ========================================================
    # 1. VALIDATE INPUTS
    # ========================================================

    if not isinstance(
        dataset_id,
        str
    ):
        return {
            "success": False,
            "error": (
                "dataset_id must be a string."
            ),
        }

    if not isinstance(
        question,
        str
    ):
        return {
            "success": False,
            "error": (
                "question must be a string."
            ),
        }

    dataset_id = dataset_id.strip()
    question = question.strip()

    if not dataset_id:
        return {
            "success": False,
            "error": (
                "dataset_id cannot be empty."
            ),
        }

    if not question:
        return {
            "success": False,
            "error": (
                "question cannot be empty."
            ),
        }

    # ========================================================
    # 2. RETRIEVE DATASET
    # ========================================================

    dataset = (
        dataset_manager.get_dataset(
            dataset_id
        )
    )

    if dataset is None:
        return {
            "success": False,
            "dataset_id": dataset_id,
            "question": question,
            "error": (
                f"Dataset '{dataset_id}' does not exist "
                "or has expired."
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

    # ========================================================
    # 3. BUILD SQL RESULT DATAFRAME
    # ========================================================

    result_df = (
        _records_to_dataframe(
            sql_result
        )
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # If SQL was not used but ML anomaly detection was used,
    # do not pretend the complete source dataset is a SQL
    # result.
    #
    # For normal SQL-based insight generation, the existing
    # behavior is preserved.
    # --------------------------------------------------------

    if (
        result_df.empty
        and
        not ml_anomaly_result
    ):

        result_df = (
            source_df.copy()
        )

    # ========================================================
    # 4. EXISTING DATASET METADATA
    # ========================================================

    quality_report = (
        dataset.get(
            "quality_report"
        )
        or dataset.get(
            "quality"
        )
    )

    # --------------------------------------------------------
    # EXISTING DETERMINISTIC ANOMALIES
    # --------------------------------------------------------

    anomalies = (
        dataset.get(
            "anomalies"
        )
        or dataset.get(
            "anomaly_report"
        )
    )

    # --------------------------------------------------------
    # EDA RESULTS
    # --------------------------------------------------------

    eda_results = (
        dataset.get(
            "eda_results"
        )
        or dataset.get(
            "eda"
        )
    )

    # ========================================================
    # 5. ADD ML ANOMALY RESULTS
    # ========================================================

    if ml_anomaly_result:

        if anomalies is None:

            anomalies = {}

        elif not isinstance(
            anomalies,
            dict
        ):

            anomalies = {
                "existing_anomaly_report":
                    anomalies
            }

        else:

            anomalies = (
                anomalies.copy()
            )

        anomalies[
            "ml_anomaly_detection"
        ] = ml_anomaly_result

    # ========================================================
    # 6. GENERATE INSIGHT
    # ========================================================

    try:

        llm_service = (
            LLMService()
        )

        insight_agent = (
            InsightAgent(
                llm_service
            )
        )

        insight = (
            insight_agent.generate_insight(

                question=
                    question,

                sql=(
                    generated_sql
                    or ""
                ),

                result=
                    result_df,

                quality_report=
                    quality_report,

                anomalies=
                    anomalies,

                eda_results=
                    eda_results,

                # IMPORTANT:
                # Pass the actual mathematical ML result
                # directly to the Insight Agent.
                ml_anomaly_result=
                    ml_anomaly_result,
            )
        )

        # ====================================================
        # 7. HANDLE EMPTY INSIGHT
        # ====================================================

        if insight is None:

            return {
                "success":
                    False,

                "dataset_id":
                    dataset_id,

                "question":
                    question,

                "insight":
                    None,

                "insight_source":
                    None,

                "llm_success":
                    False,

                "error": (
                    "Insight Agent did not return "
                    "an analytical insight."
                ),
            }

        # ====================================================
        # 8. NORMALIZE RESPONSE
        # ====================================================

        if isinstance(
            insight,
            dict
        ):

            insight_text = (
                insight.get(
                    "insight"
                )
                or insight.get(
                    "text"
                )
                or insight.get(
                    "response"
                )
            )

            insight_source = (
                insight.get(
                    "source",
                    "llm"
                )
            )

            llm_success = (
                insight.get(
                    "llm_success",
                    insight_source == "llm"
                )
            )

            llm_error = (
                insight.get(
                    "llm_error"
                )
            )

        else:

            insight_text = str(
                insight
            )

            insight_source = (
                "llm"
            )

            llm_success = True

            llm_error = None

        # ====================================================
        # 9. RETURN INSIGHT
        # ====================================================

        return {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "question":
                question,

            "insight":
                insight_text,

            "insight_source":
                insight_source,

            "llm_success":
                bool(
                    llm_success
                ),

            "llm_error":
                llm_error,

            "error":
                None,
        }

    # ========================================================
    # 10. ERROR HANDLING
    # ========================================================

    except Exception as exc:

        return {

            "success":
                False,

            "dataset_id":
                dataset_id,

            "question":
                question,

            "insight":
                None,

            "insight_source":
                None,

            "llm_success":
                False,

            "llm_error":
                str(exc),

            "error":
                str(exc),
        }