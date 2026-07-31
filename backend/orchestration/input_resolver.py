from typing import Any

from backend.orchestration.state import AgentState


# ============================================================
# STATE VALUE RESOLUTION
# ============================================================

def _resolve_state_value(
    source: str,
    state: AgentState,
) -> Any:
    """
    Resolve an input value from the current AgentState.

    Examples:

        dataset_id
            -> state["dataset_id"]

        question
            -> state["question"]

        sql_result
            -> state["sql_result"]

        generated_sql
            -> state["generated_sql"]
    """

    return state.get(
        source
    )


# ============================================================
# BUILD TOOL INPUT
# ============================================================

def build_tool_input(
    input_mapping: dict,
    state: AgentState,
) -> dict:
    """
    Build the input dictionary for a tool dynamically
    from registry metadata and AgentState.

    Example metadata:

        {
            "dataset_id": "dataset_id",
            "question": "question",
            "sql_result": "sql_result",
        }

    produces:

        {
            "dataset_id": state["dataset_id"],
            "question": state["question"],
            "sql_result": state["sql_result"],
        }
    """

    tool_input = {}

    for (
        parameter_name,
        state_source
    ) in input_mapping.items():

        tool_input[
            parameter_name
        ] = _resolve_state_value(
            state_source,
            state,
        )

    return tool_input