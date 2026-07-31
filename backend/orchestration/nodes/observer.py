from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.execution_policy import (
    EXECUTE,
    REPLAN,
    FINISH,
    evaluate_execution_policy,
)


# ============================================================
# OBSERVER NODE
# ============================================================

def observer_node(
    state: AgentState
) -> dict:
    """
    Observe the current agent state and delegate the
    next-action decision to the centralized execution policy.

    Responsibilities:

    - inspect the latest execution state
    - defensively detect a failed current tool
    - call the execution policy
    - translate the policy decision into state updates
    - record the decision in the trace

    The observer does NOT:

    - execute tools
    - resolve dependencies directly
    - select tools directly
    - perform replanning
    - modify the execution plan
    """

    # ========================================================
    # 1. READ STATE
    # ========================================================

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    current_step = state.get(
        "current_step"
    )

    failed_tool = state.get(
        "failed_tool"
    )

    last_tool_error = state.get(
        "last_tool_error"
    )

    tool_results = dict(
        state.get(
            "tool_results",
            {}
        )
    )

    # ========================================================
    # 2. DEFENSIVE FAILURE DETECTION
    # ========================================================
    #
    # execution.py normally sets failed_tool and
    # last_tool_error.
    #
    # However, observer_node may also be invoked directly
    # in tests or by another caller.
    #
    # If the current tool result explicitly says
    # success=False, normalize that failure before policy
    # evaluation.
    #
    # IMPORTANT:
    # Do NOT create a modified state for normal successful
    # execution. This preserves the policy integration
    # contract:
    #
    # evaluate_execution_policy(state)
    # ========================================================

    detected_failure = False

    if (
        current_step
        and
        not failed_tool
    ):

        current_result = tool_results.get(
            current_step
        )

        if isinstance(
            current_result,
            dict
        ):

            if (
                current_result.get(
                    "success"
                )
                is False
            ):

                failed_tool = (
                    current_step
                )

                last_tool_error = (
                    current_result.get(
                        "error"
                    )
                    or
                    f"{current_step} failed."
                )

                detected_failure = True

    # ========================================================
    # 3. BUILD POLICY INPUT
    # ========================================================
    #
    # Normal case:
    #
    #     evaluate_execution_policy(state)
    #
    # Defensive failure case:
    #
    #     provide a copy containing the discovered failure.
    #
    # This means normal policy integration tests still receive
    # the exact original state.
    # ========================================================

    if detected_failure:

        policy_state = dict(
            state
        )

        policy_state[
            "failed_tool"
        ] = failed_tool

        policy_state[
            "last_tool_error"
        ] = last_tool_error

    else:

        policy_state = state

    # ========================================================
    # 4. EVALUATE EXECUTION POLICY
    # ========================================================

    decision = evaluate_execution_policy(
        policy_state
    )

    action = decision.get(
        "action"
    )

    next_step = decision.get(
        "current_step"
    )

    reason = (
        decision.get(
            "reason"
        )
        or
        "Execution policy produced no reason."
    )

    policy_error = decision.get(
        "error"
    )

    # ========================================================
    # 5. EXECUTE
    # ========================================================

    if action == EXECUTE:

        trace.append({
            "node":
                "observer",

            "tool":
                current_step,

            "status":
                "success",

            "decision":
                "continue",

            "next_step":
                next_step,

            "reason":
                reason,

            "error":
                None,
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
    # 6. REPLAN
    # ========================================================

    if action == REPLAN:

        recovery_error = (
            policy_error
            or
            last_tool_error
            or
            reason
        )

        recovery_tool = (
            failed_tool
            or
            next_step
            or
            current_step
        )

        trace.append({
            "node":
                "observer",

            "tool":
                recovery_tool,

            "status": (
                "error"
                if failed_tool
                else "blocked"
            ),

            "decision":
                "replan",

            "reason":
                reason,

            "error":
                recovery_error,

            "replan_count":
                state.get(
                    "replan_count",
                    0
                ),
        })

        return {
            "current_step": (
                failed_tool
                if failed_tool
                else None
            ),

            "failed_tool":
                failed_tool,

            "last_tool_error":
                recovery_error,

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
    # 7. FINISH
    # ========================================================

    if action == FINISH:

        final_error = (
            policy_error
        )

        # If we're finishing because a failed tool exhausted
        # recovery, preserve its error even if the policy
        # returned no separate error.
        if (
            failed_tool
            and
            not final_error
        ):

            final_error = (
                last_tool_error
                or
                f"{failed_tool} failed."
            )

        trace.append({
            "node":
                "observer",

            "tool": (
                failed_tool
                or
                current_step
            ),

            "status": (
                "error"
                if final_error
                else "success"
            ),

            "decision":
                "finish",

            "reason":
                reason,

            "error":
                final_error,
        })

        return {
            "current_step":
                None,

            "retry_count":
                0,

            "failed_tool":
                failed_tool,

            "last_tool_error": (
                last_tool_error
                if failed_tool
                else None
            ),

            "completed":
                True,

            "error":
                final_error,

            "trace":
                trace,
        }

    # ========================================================
    # 8. DEFENSIVE FALLBACK
    # ========================================================

    error = (
        "Execution policy returned an unsupported "
        f"action: {action}"
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

        "error":
            error,
    })

    return {
        "current_step":
            None,

        "failed_tool":
            failed_tool,

        "last_tool_error": (
            last_tool_error
            or
            error
        ),

        "retry_count":
            0,

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
    Route LangGraph after observer execution.

    Returns:

    execute
        Another tool should execute.

    replan
        Recovery planning is required.

    finish
        Workflow has completed or cannot recover.
    """

    # ========================================================
    # 1. COMPLETED
    # ========================================================

    if state.get(
        "completed",
        False
    ):

        return "finish"

    # ========================================================
    # 2. READ OBSERVER DECISION
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
    # 3. DEFENSIVE FALLBACK
    # ========================================================

    if state.get(
        "current_step"
    ):

        return "execute"

    return "finish"