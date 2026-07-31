from backend.orchestration.state import AgentState

from backend.orchestration.tool_registry import (
    get_tool_definition,
)

from backend.orchestration.input_resolver import (
    build_tool_input,
)

from backend.orchestration.dependency_resolver import (
    check_tool_dependencies,
)


def tool_executor_node(
    state: AgentState
) -> dict:
    """
    Execute the current tool dynamically using
    metadata from the central tool registry.

    Responsibilities:
    - find the current tool
    - retrieve its registry definition
    - validate tool dependencies
    - build inputs dynamically
    - execute the tool
    - store the raw tool result
    - promote declared outputs into AgentState
    - record success/failure information
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

    current_step = state.get(
        "current_step"
    )

    # ========================================================
    # NO CURRENT TOOL
    # ========================================================

    if not current_step:

        trace.append({
            "node": "tool_executor",
            "status": "skipped",
            "reason": "No current tool step exists.",
        })

        return {
            "current_step": None,
            "trace": trace,
        }

    # ========================================================
    # GET TOOL DEFINITION
    # ========================================================

    definition = get_tool_definition(
        current_step
    )

    if definition is None:

        error = (
            f"Unknown tool: {current_step}"
        )

        trace.append({
            "node": "tool_executor",
            "tool": current_step,
            "status": "error",
            "error": error,
        })

        return {
            "current_step":
                current_step,

            "failed_tool":
                current_step,

            "last_tool_error":
                error,

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # CHECK TOOL DEPENDENCIES
    # ========================================================

    dependency_status = (
        check_tool_dependencies(
            current_step,
            state,
        )
    )

    if not dependency_status[
        "ready"
    ]:

        missing_dependencies = (
            dependency_status.get(
                "missing_dependencies",
                []
            )
        )

        error = (
            f"Tool '{current_step}' cannot execute because "
            f"required dependencies are not satisfied: "
            f"{missing_dependencies}"
        )

        result = {
            "success":
                False,

            "error":
                error,

            "missing_dependencies":
                missing_dependencies,
        }

        # Store the blocked execution result.
        tool_results[
            current_step
        ] = result

        trace.append({
            "node":
                "tool_executor",

            "tool":
                current_step,

            "status":
                "blocked",

            "missing_dependencies":
                missing_dependencies,

            "error":
                error,
        })

        return {
            "current_step":
                current_step,

            "executed_tools":
                executed_tools,

            "tool_results":
                tool_results,

            "failed_tool":
                current_step,

            "last_tool_error":
                error,

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # GET EXECUTABLE TOOL
    # ========================================================

    tool = definition.get(
        "tool"
    )

    if tool is None:

        error = (
            f"Tool '{current_step}' has no executable "
            "tool registered."
        )

        trace.append({
            "node": "tool_executor",
            "tool": current_step,
            "status": "error",
            "error": error,
        })

        return {
            "current_step":
                current_step,

            "failed_tool":
                current_step,

            "last_tool_error":
                error,

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # BUILD INPUT DYNAMICALLY
    # ========================================================

    input_mapping = definition.get(
        "inputs",
        {}
    )

    tool_input = build_tool_input(
        input_mapping,
        state,
    )

    # ========================================================
    # EXECUTE TOOL
    # ========================================================

    try:

        result = tool.invoke(
            tool_input
        )

    except Exception as exc:

        result = {
            "success": False,
            "error": str(exc),
        }

    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        result = {
            "success": True,
            "result": result,
        }

    # ========================================================
    # STORE RAW TOOL RESULT
    # ========================================================

    tool_results[
        current_step
    ] = result

    success = (
        result.get(
            "success"
        )
        is True
    )

    # ========================================================
    # MARK SUCCESSFUL EXECUTION
    # ========================================================

    if success:

        if (
            current_step
            not in executed_tools
        ):

            executed_tools.append(
                current_step
            )

    # ========================================================
    # BASE STATE UPDATE
    # ========================================================

    updates = {
        "current_step":
            current_step,

        "executed_tools":
            executed_tools,

        "tool_results":
            tool_results,
    }

    # ========================================================
    # PROMOTE OUTPUTS DYNAMICALLY
    # ========================================================

    if success:

        output_mapping = definition.get(
            "outputs",
            {}
        )

        for (
            result_key,
            state_key
        ) in output_mapping.items():

            updates[
                state_key
            ] = result.get(
                result_key
            )

    # ========================================================
    # HANDLE SUCCESS / FAILURE
    # ========================================================

    if success:

        updates[
            "failed_tool"
        ] = None

        updates[
            "last_tool_error"
        ] = None

        updates[
            "error"
        ] = None

    else:

        error = (
            result.get(
                "error"
            )
            or
            f"{current_step} failed."
        )

        updates[
            "failed_tool"
        ] = current_step

        updates[
            "last_tool_error"
        ] = error

        updates[
            "error"
        ] = error

    # ========================================================
    # TRACE
    # ========================================================

    trace.append({
        "node":
            "tool_executor",

        "tool":
            current_step,

        "status": (
            "success"
            if success
            else "error"
        ),

        "input_keys":
            list(
                tool_input.keys()
            ),

        "error": (
            None
            if success
            else updates.get(
                "last_tool_error"
            )
        ),
    })

    updates[
        "trace"
    ] = trace

    return updates