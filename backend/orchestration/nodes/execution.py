from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.tool_registry import (
    get_tool_definition,
)

from backend.orchestration.input_resolver import (
    build_tool_input,
)

from backend.orchestration.dependency_resolver import (
    check_tool_dependencies,
)


# ============================================================
# TOOL EXECUTOR NODE
# ============================================================

def tool_executor_node(
    state: AgentState
) -> dict:
    """
    Execute the current tool dynamically using metadata
    from the central tool registry.

    Responsibilities:
    - identify the current tool
    - retrieve its registry definition
    - validate dependencies
    - build tool inputs dynamically
    - execute the tool
    - store raw tool results
    - promote declared outputs into AgentState
    - record success/failure information
    """

    # ========================================================
    # 1. READ STATE
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
    # 2. NO CURRENT STEP
    # ========================================================

    if not current_step:

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
    # 3. GET TOOL DEFINITION
    # ========================================================

    definition = get_tool_definition(
        current_step
    )

    if definition is None:

        error = (
            f"Unknown tool: {current_step}"
        )

        trace.append({
            "node":
                "tool_executor",

            "tool":
                current_step,

            "status":
                "error",

            "error":
                error,
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
    # 4. CHECK DEPENDENCIES
    # ========================================================

    dependency_status = (
        check_tool_dependencies(
            current_step,
            state,
        )
    )

    dependencies_ready = (
        dependency_status.get(
            "ready",
            False,
        )
    )
    
    if not dependencies_ready:


        missing_dependencies = (
            dependency_status.get(
                "missing_dependencies",
                []
            )
        )

        failed_dependencies = (
            dependency_status.get(
                "failed_dependencies",
                []
            )
        )

        reason = dependency_status.get(
            "reason"
        )

        # ----------------------------------------------------
        # Build fallback reason if resolver did not provide one
        # ----------------------------------------------------

        if not reason:

            if failed_dependencies:

                reason = (
                    f"Tool '{current_step}' cannot execute "
                    "because required dependencies failed: "
                    f"{failed_dependencies}"
                )

            elif missing_dependencies:

                reason = (
                    f"Tool '{current_step}' cannot execute "
                    "because required dependencies are "
                    f"missing: {missing_dependencies}"
                )

            else:

                reason = (
                    f"Tool '{current_step}' dependencies "
                    "are not satisfied."
                )

        # ----------------------------------------------------
        # IMPORTANT:
        # Record blocked execution as failed tool result
        # ----------------------------------------------------

        blocked_result = {
            "success":
                False,

            "error":
                reason,

            "blocked":
                True,

            "missing_dependencies":
                missing_dependencies,

            "failed_dependencies":
                failed_dependencies,
        }

        tool_results[
            current_step
        ] = blocked_result

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        trace.append({
            "node":
                "tool_executor",

            "tool":
                current_step,

            "status":
                "blocked",

            "reason":
                reason,

            "missing_dependencies":
                missing_dependencies,

            "failed_dependencies":
                failed_dependencies,
        })

        # ----------------------------------------------------
        # Return failure state
        # ----------------------------------------------------

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
                reason,

            "error":
                reason,

            "trace":
                trace,
        }



    # ========================================================
    # 5. GET EXECUTABLE TOOL
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
            "node":
                "tool_executor",

            "tool":
                current_step,

            "status":
                "error",

            "error":
                error,
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
    # 6. BUILD TOOL INPUT DYNAMICALLY
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
    # 7. EXECUTE TOOL
    # ========================================================

    try:

        result = tool.invoke(
            tool_input
        )

    except Exception as exc:

        result = {
            "success":
                False,

            "error":
                str(exc),
        }

    # ========================================================
    # 8. NORMALIZE RESULT
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        result = {
            "success":
                True,

            "result":
                result,
        }

    # ========================================================
    # 9. STORE RAW RESULT
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
    # 10. RECORD SUCCESSFUL EXECUTION
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
    # 11. BASE STATE UPDATE
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
    # 12. PROMOTE DECLARED OUTPUTS
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
    # 13. SUCCESS / FAILURE STATE
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
    # 14. TRACE
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

    # ========================================================
    # 15. RETURN
    # ========================================================

    return updates