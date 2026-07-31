from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.tool_registry import (
    get_tool_definition,
)


# ============================================================
# BUILD TOOL INPUT
# ============================================================

def build_tool_input(
    state: AgentState,
    input_mapping: dict,
) -> dict:
    """
    Build tool arguments dynamically from AgentState.

    Example:

        {
            "dataset_id": "dataset_id",
            "sql_result": "sql_result",
        }

    means:

        tool_input["dataset_id"] =
            state["dataset_id"]

        tool_input["sql_result"] =
            state["sql_result"]
    """

    tool_input = {}

    for (
        tool_argument,
        state_field
    ) in input_mapping.items():

        tool_input[
            tool_argument
        ] = state.get(
            state_field
        )

    return tool_input


# ============================================================
# PROMOTE TOOL OUTPUTS
# ============================================================

def promote_tool_outputs(
    result: dict,
    output_mapping: dict,
) -> dict:
    """
    Map important tool outputs back into AgentState.

    Example:

        {
            "generated_sql": "generated_sql",
            "result": "sql_result",
        }
    """

    updates = {}

    for (
        result_field,
        state_field
    ) in output_mapping.items():

        updates[
            state_field
        ] = result.get(
            result_field
        )

    return updates


# ============================================================
# TOOL EXECUTOR NODE
# ============================================================

def tool_executor_node(
    state: AgentState
) -> dict:
    """
    Dynamically execute the tool selected by the agent.

    The executor itself does not contain
    tool-specific SQL/visualization/insight logic.

    Tool behavior is defined through the
    central tool registry.
    """

    # ========================================================
    # READ STATE
    # ========================================================

    executed_tools = list(
        state.get(
            "executed_tools",
            []
        )
    )

    tool_results = dict(
        state.get(
            "tool_results",
            {}
        )
    )

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    tool_name = state.get(
        "current_step"
    )

    # ========================================================
    # NO TOOL SELECTED
    # ========================================================

    if not tool_name:

        trace.append({
            "node":
                "tool_executor",

            "status":
                "skipped",

            "reason":
                "No current tool step exists.",
        })

        return {
            "current_step":
                None,

            "trace":
                trace,
        }

    # ========================================================
    # GET TOOL DEFINITION
    # ========================================================

    definition = (
        get_tool_definition(
            tool_name
        )
    )

    if definition is None:

        error = (
            f"Unknown tool: {tool_name}"
        )

        trace.append({
            "node":
                "tool_executor",

            "tool":
                tool_name,

            "status":
                "error",

            "error":
                error,
        })

        return {
            "current_step":
                tool_name,

            "failed_tool":
                tool_name,

            "last_tool_error":
                error,

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # GET TOOL + METADATA
    # ========================================================

    tool = definition[
        "tool"
    ]

    input_mapping = definition.get(
        "inputs",
        {}
    )

    output_mapping = definition.get(
        "outputs",
        {}
    )

    # ========================================================
    # BUILD TOOL INPUT DYNAMICALLY
    # ========================================================

    tool_input = build_tool_input(
        state,
        input_mapping,
    )

    # ========================================================
    # EXECUTE TOOL
    # ========================================================

    try:

        result = tool.invoke(
            tool_input
        )

        if not isinstance(
            result,
            dict
        ):

            result = {
                "success": True,
                "result": result,
            }

    except Exception as exc:

        result = {
            "success":
                False,

            "error":
                str(exc),
        }

    # ========================================================
    # STORE RAW TOOL RESULT
    # ========================================================

    tool_results[
        tool_name
    ] = result

    # ========================================================
    # SUCCESSFUL EXECUTION
    # ========================================================

    success = (
        result.get(
            "success"
        )
        is True
    )

    if success:

        if (
            tool_name
            not in executed_tools
        ):

            executed_tools.append(
                tool_name
            )

    # ========================================================
    # TRACE
    # ========================================================

    trace.append({
        "node":
            "tool_executor",

        "tool":
            tool_name,

        "status": (
            "success"
            if success
            else "error"
        ),

        "error":
            result.get(
                "error"
            ),
    })

    # ========================================================
    # BASE STATE UPDATES
    # ========================================================

    updates = {
        "current_step":
            tool_name,

        "executed_tools":
            executed_tools,

        "tool_results":
            tool_results,

        "trace":
            trace,
    }

    # ========================================================
    # PROMOTE OUTPUTS DYNAMICALLY
    # ========================================================

    promoted_outputs = (
        promote_tool_outputs(
            result,
            output_mapping,
        )
    )

    updates.update(
        promoted_outputs
    )

    # ========================================================
    # FAILURE STATE
    # ========================================================

    if not success:

        error = (
            result.get(
                "error"
            )
            or
            f"{tool_name} failed."
        )

        updates[
            "failed_tool"
        ] = tool_name

        updates[
            "last_tool_error"
        ] = error

        updates[
            "error"
        ] = error

    # ========================================================
    # CLEAR PREVIOUS FAILURE AFTER SUCCESS
    # ========================================================

    else:

        updates[
            "failed_tool"
        ] = None

        updates[
            "last_tool_error"
        ] = None

        updates[
            "error"
        ] = None

    return updates