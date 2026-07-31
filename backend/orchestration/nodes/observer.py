from backend.orchestration.state import AgentState


def observer_node(
    state: AgentState
) -> dict:
    """
    Observe the latest tool execution and determine
    whether the workflow should:

    - continue to the next tool
    - re-plan after a failure
    - finish successfully
    - finish because recovery attempts are exhausted
    """

    # ========================================================
    # READ STATE
    # ========================================================

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    plan = list(
        state.get(
            "plan",
            []
        )
    )

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

    current_step = state.get(
        "current_step"
    )

    replan_count = state.get(
        "replan_count",
        0
    )

    max_replans = state.get(
        "max_replans",
        2
    )

    # ========================================================
    # NO CURRENT STEP
    # ========================================================

    if not current_step:

        trace.append({
            "node": "observer",
            "decision": "finish",
            "reason": "No current tool step exists.",
        })

        return {
            "current_step": None,
            "completed": True,
            "trace": trace,
        }

    # ========================================================
    # GET RESULT OF CURRENT TOOL
    # ========================================================

    result = tool_results.get(
        current_step
    )

    success = False

    if isinstance(
        result,
        dict
    ):

        success = (
            result.get(
                "success"
            )
            is True
        )

    elif result is not None:

        success = True

    # ========================================================
    # TOOL FAILED
    # ========================================================

    if not success:

        last_tool_error = (
            state.get(
                "last_tool_error"
            )
            or (
                result.get("error")
                if isinstance(result, dict)
                else None
            )
            or f"{current_step} failed."
        )

        # ----------------------------------------------------
        # Recovery is still available
        # ----------------------------------------------------

        if replan_count < max_replans:

            trace.append({
                "node": "observer",
                "tool": current_step,
                "status": "error",
                "decision": "replan",
                "reason": last_tool_error,
                "replan_count": replan_count,
            })

            return {
                "current_step":
                    current_step,

                "failed_tool":
                    current_step,

                "last_tool_error":
                    last_tool_error,

                "completed":
                    False,

                "trace":
                    trace,
            }

        # ----------------------------------------------------
        # Recovery attempts exhausted
        # ----------------------------------------------------

        error = (
            f"Tool '{current_step}' failed and "
            "maximum re-planning attempts were reached."
        )

        trace.append({
            "node": "observer",
            "tool": current_step,
            "status": "error",
            "decision": "finish",
            "reason": error,
            "replan_count": replan_count,
        })

        return {
            "current_step":
                None,

            "failed_tool":
                current_step,

            "last_tool_error":
                last_tool_error,

            "completed":
                True,

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # TOOL SUCCEEDED
    # ========================================================

    remaining_tools = [
        tool_name
        for tool_name in plan
        if tool_name not in executed_tools
    ]

    # ========================================================
    # MORE TOOLS REMAIN
    # ========================================================

    if remaining_tools:

        next_step = (
            remaining_tools[0]
        )

        trace.append({
            "node": "observer",
            "tool": current_step,
            "status": "success",
            "decision": "continue",
            "completed_tool": current_step,
            "next_step": next_step,
        })

        return {
            "current_step":
                next_step,

            "retry_count":
                0,

            "failed_tool":
                None,

            "last_tool_error":
                None,

            "error":
                None,

            "completed":
                False,

            "trace":
                trace,
        }

    # ========================================================
    # ALL TOOLS COMPLETED
    # ========================================================

    trace.append({
        "node": "observer",
        "tool": current_step,
        "status": "success",
        "decision": "finish",
        "completed_tool": current_step,
        "reason": (
            "All planned tools completed successfully."
        ),
    })

    return {
        "current_step":
            None,

        "retry_count":
            0,

        "failed_tool":
            None,

        "last_tool_error":
            None,

        "error":
            None,

        "completed":
            True,

        "trace":
            trace,
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_after_observation(
    state: AgentState
) -> str:
    """
    Decide where LangGraph should route after observation.

    Returns:
        execute -> execute the next tool
        replan  -> recover from a failed tool
        finish  -> terminate the workflow
    """

    # ========================================================
    # WORKFLOW COMPLETED
    # ========================================================

    if state.get(
        "completed",
        False
    ):

        return "finish"

    # ========================================================
    # CHECK OBSERVER'S MOST RECENT DECISION
    # ========================================================

    trace = state.get(
        "trace",
        []
    )

    if trace:

        last_event = (
            trace[-1]
        )

        if (
            last_event.get("node")
            == "observer"
            and
            last_event.get("decision")
            == "replan"
        ):

            return "replan"

    # ========================================================
    # NEXT TOOL AVAILABLE
    # ========================================================

    if state.get(
        "current_step"
    ):

        return "execute"

    # ========================================================
    # NOTHING LEFT TO DO
    # ========================================================

    return "finish"