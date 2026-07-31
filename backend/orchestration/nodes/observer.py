from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.step_resolver import (
    resolve_next_step,
)


# ============================================================
# OBSERVER NODE
# ============================================================

def observer_node(
    state: AgentState
) -> dict:
    """
    Observe the latest tool execution and decide
    what the agent should do next.

    Responsibilities:
    - inspect the latest tool result
    - detect execution failure
    - trigger replanning when recovery is required
    - use the step resolver to find the next
      executable tool
    - detect blocked/invalid plans
    - detect workflow completion

    The observer does NOT execute tools.
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

        resolution = resolve_next_step(
            plan=plan,
            executed_tools=executed_tools,
            tool_results=tool_results,
        )

        # ----------------------------------------------------
        # Plan already completed
        # ----------------------------------------------------

        if resolution.status == "complete":

            trace.append({
                "node":
                    "observer",

                "decision":
                    "finish",

                "reason":
                    resolution.reason,
            })

            return {
                "current_step":
                    None,

                "completed":
                    True,

                "error":
                    None,

                "trace":
                    trace,
            }

        # ----------------------------------------------------
        # A ready step exists
        # ----------------------------------------------------

        if resolution.status == "ready":

            trace.append({
                "node":
                    "observer",

                "decision":
                    "continue",

                "next_step":
                    resolution.next_step,

                "reason":
                    resolution.reason,
            })

            return {
                "current_step":
                    resolution.next_step,

                "completed":
                    False,

                "error":
                    None,

                "trace":
                    trace,
            }

        # ----------------------------------------------------
        # Plan is blocked/invalid
        # ----------------------------------------------------

        reason = (
            resolution.reason
            or
            "No executable tool could be resolved."
        )

        trace.append({
            "node":
                "observer",

            "decision":
                "replan",

            "status":
                resolution.status,

            "reason":
                reason,

            "blocked_tools":
                resolution.blocked_tools,

            "missing_dependencies":
                resolution.missing_dependencies,
        })

        return {
            "current_step":
                None,

            "failed_tool":
                None,

            "last_tool_error":
                reason,

            "completed":
                False,

            "error":
                None,

            "trace":
                trace,
        }

    # ========================================================
    # INSPECT CURRENT TOOL RESULT
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
    # CURRENT TOOL FAILED
    # ========================================================

    if not success:

        last_tool_error = (
            state.get(
                "last_tool_error"
            )
            or (
                result.get(
                    "error"
                )
                if isinstance(
                    result,
                    dict
                )
                else None
            )
            or
            f"{current_step} failed."
        )

        # ----------------------------------------------------
        # Recovery still available
        # ----------------------------------------------------

        if replan_count < max_replans:

            trace.append({
                "node":
                    "observer",

                "tool":
                    current_step,

                "status":
                    "error",

                "decision":
                    "replan",

                "reason":
                    last_tool_error,

                "replan_count":
                    replan_count,
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

                "error":
                    None,

                "trace":
                    trace,
            }

        # ----------------------------------------------------
        # Recovery exhausted
        # ----------------------------------------------------

        error = (
            f"Tool '{current_step}' failed and "
            "maximum re-planning attempts were reached."
        )

        trace.append({
            "node":
                "observer",

            "tool":
                current_step,

            "status":
                "error",

            "decision":
                "finish",

            "reason":
                error,

            "replan_count":
                replan_count,
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
    # CURRENT TOOL SUCCEEDED
    # ========================================================

    resolution = resolve_next_step(
        plan=plan,
        executed_tools=executed_tools,
        tool_results=tool_results,
    )

    # ========================================================
    # NEXT TOOL READY
    # ========================================================

    if resolution.status == "ready":

        trace.append({
            "node":
                "observer",

            "tool":
                current_step,

            "status":
                "success",

            "decision":
                "continue",

            "completed_tool":
                current_step,

            "next_step":
                resolution.next_step,

            "reason":
                resolution.reason,
        })

        return {
            "current_step":
                resolution.next_step,

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
    # PLAN COMPLETED
    # ========================================================

    if resolution.status == "complete":

        trace.append({
            "node":
                "observer",

            "tool":
                current_step,

            "status":
                "success",

            "decision":
                "finish",

            "completed_tool":
                current_step,

            "reason":
                resolution.reason,
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

    # ========================================================
    # PLAN BLOCKED / INVALID
    # ========================================================

    reason = (
        resolution.reason
        or
        "No executable tool could be resolved."
    )

    # --------------------------------------------------------
    # Recovery available
    # --------------------------------------------------------

    if replan_count < max_replans:

        trace.append({
            "node":
                "observer",

            "tool":
                current_step,

            "status":
                resolution.status,

            "decision":
                "replan",

            "completed_tool":
                current_step,

            "reason":
                reason,

            "blocked_tools":
                resolution.blocked_tools,

            "missing_dependencies":
                resolution.missing_dependencies,

            "replan_count":
                replan_count,
        })

        return {
            "current_step":
                None,

            "failed_tool":
                None,

            "last_tool_error":
                reason,

            "retry_count":
                0,

            "completed":
                False,

            "error":
                None,

            "trace":
                trace,
        }

    # ========================================================
    # BLOCKED AND RECOVERY EXHAUSTED
    # ========================================================

    error = (
        "Execution plan could not continue and "
        "maximum re-planning attempts were reached. "
        f"{reason}"
    )

    trace.append({
        "node":
            "observer",

        "tool":
            current_step,

        "status":
            resolution.status,

        "decision":
            "finish",

        "reason":
            error,

        "blocked_tools":
            resolution.blocked_tools,

        "missing_dependencies":
            resolution.missing_dependencies,

        "replan_count":
            replan_count,
    })

    return {
        "current_step":
            None,

        "failed_tool":
            None,

        "last_tool_error":
            reason,

        "completed":
            True,

        "error":
            error,

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

        execute
            A tool is ready for execution.

        replan
            Recovery planning is required.

        finish
            Workflow has completed or cannot recover.
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
    # CHECK MOST RECENT OBSERVER DECISION
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
            last_event.get(
                "node"
            )
            == "observer"
        ):

            decision = (
                last_event.get(
                    "decision"
                )
            )

            if decision == "replan":

                return "replan"

            if decision == "continue":

                return "execute"

            if decision == "finish":

                return "finish"

    # ========================================================
    # FALLBACK ROUTING
    # ========================================================

    if state.get(
        "current_step"
    ):

        return "execute"

    return "finish"

