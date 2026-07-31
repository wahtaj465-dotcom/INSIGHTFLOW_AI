from typing import Any

from langchain_core.tools import tool

from backend.services.dataset_manager import dataset_manager
from backend.services.llm_service import LLMService
from backend.services.sql_engine import SQLEngine
from backend.agents.sql_agent import SQLAgent


def _dataframe_to_records(df) -> list[dict[str, Any]]:
    """
    Convert a Pandas DataFrame into JSON-friendly records.
    """

    if df is None:
        return []

    if df.empty:
        return []

    clean_df = df.copy()

    for column in clean_df.columns:

        if str(clean_df[column].dtype).startswith(
            "datetime"
        ):
            clean_df[column] = (
                clean_df[column]
                .astype(str)
            )

    clean_df = clean_df.where(
        clean_df.notna(),
        None
    )

    return clean_df.to_dict(
        orient="records"
    )


@tool
def run_sql_analysis(
    dataset_id: str,
    question: str
) -> dict:
    """
    Analyze a prepared InsightFlow dataset using natural-language SQL.

    Use this tool when the user asks a question requiring filtering,
    aggregation, grouping, ranking, comparison, counting, totals,
    averages, trends, or other SQL-based analysis.

    The tool retrieves the prepared dataset, registers it in DuckDB,
    generates SQL using the existing SQL agent, executes the query,
    and returns the SQL and query result.
    """

    # -----------------------------------------
    # Validate arguments
    # -----------------------------------------

    if not isinstance(dataset_id, str):
        return {
            "success": False,
            "error": "dataset_id must be a string."
        }

    if not isinstance(question, str):
        return {
            "success": False,
            "error": "question must be a string."
        }

    dataset_id = dataset_id.strip()
    question = question.strip()

    if not dataset_id:
        return {
            "success": False,
            "error": "dataset_id cannot be empty."
        }

    if not question:
        return {
            "success": False,
            "error": "question cannot be empty."
        }

    # -----------------------------------------
    # Retrieve dataset session
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

    # -----------------------------------------
    # SQL execution
    # -----------------------------------------

    try:

        sql_engine = SQLEngine()

        sql_engine.register_dataframe(
            df=df,
            table_name="dataset"
        )

        llm_service = LLMService()

        sql_agent = SQLAgent(
            llm_service=llm_service,
            sql_engine=sql_engine,
            table_name="dataset"
        )

        result = sql_agent.ask(
            question
        )

        if not isinstance(result, dict):
            return {
                "success": False,
                "error": (
                    "SQL Agent returned an invalid "
                    "response."
                )
            }

        sql_df = result.get(
            "result"
        )

        records = _dataframe_to_records(
            sql_df
        )

        return {
            "success": True,

            "dataset_id": dataset_id,

            "question": question,

            "generated_sql": result.get(
                "generated_sql"
            ),

            "result": records,

            "row_count": len(records),

            "columns": (
                [
                    str(column)
                    for column in sql_df.columns
                ]
                if sql_df is not None
                else []
            ),

            "relevant_columns": result.get(
                "relevant_columns",
                []
            ),

            "sql_attempts": result.get(
                "sql_attempts",
                []
            ),

            "sql_source": result.get(
                "sql_source"
            ),

            "fallback_used": result.get(
                "fallback_used",
                False
            ),

            "error": None
        }

    except Exception as exc:

        return {
            "success": False,

            "dataset_id": dataset_id,

            "question": question,

            "error": str(exc)
        }