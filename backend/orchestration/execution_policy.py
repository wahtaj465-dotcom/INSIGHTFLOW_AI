from typing import Any

from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.step_resolver import (
    resolve_next_step,
)


# ============================================================
# POLICY DECISIONS
# ============================================================

EXECUTE = "execute"
REPLAN = "replan"
FINISH = "finish"


# ============================================================
# BUILD DECISION
# ============================================================

def _build_decision(
    action: str,
    *,
    current_step: str | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Build a normalized execution-policy decision.
    """

    return {
        "action": action,
        "current_step": current_step,
        "reason": reason,
        "error": error,
    }


# ============================================================
# RECOVERY BUDGET
# ============================================================

def has_replan_budget(
    state: AgentState
) -> bool:
    """
    Return True when the agent is still allowed
    to perform another replan.
    """

    replan_count = state.get(
        "replan_count",
        0,
    )

    max_replans = state.get(
        "max_replans",
        2,
    )

    return (
        replan_count
        <
        max_replans
    )


# ============================================================
# RESOLUTION VALUE HELPER
# ============================================================

def _resolution_value(
    resolution,
    *names: str,
    default=None,
):
    """
    Read a value from either:

    - a dictionary resolution
    - a StepResolution object
    - another object exposing equivalent attributes

    This keeps the execution policy compatible with both
    the real step resolver and mocked test resolutions.
    """

    if isinstance(
        resolution,
        dict,
    ):

        for name in names:

            if name in resolution:

                value = resolution.get(
                    name
                )

                if value is not None:
                    return value

        return default

    for name in names:

        value = getattr(
            resolution,
            name,
            None,
        )

        if value is not None:
            return value

    return default


# ============================================================
# EXECUTION POLICY
# ============================================================

def evaluate_execution_policy(
    state: AgentState
) -> dict[str, Any]:
    """
    Decide what the orchestration runtime should do next.

    Possible actions:

    - execute
    - replan
    - finish

    The policy does not execute tools and does not
    modify the execution plan.

    It evaluates:

    - explicit completion
    - tool failure
    - recovery budget
    - plan readiness
    - blocked plans
    - invalid plans
    - completed plans
    """

    # ========================================================
    # 1. EXPLICITLY COMPLETED STATE
    # ========================================================

    if state.get(
        "completed",
        False,
    ):

        return _build_decision(
            FINISH,
            reason=(
                "Agent state is already marked completed."
            ),
        )

    # ========================================================
    # 2. TOOL FAILURE
    # ========================================================

    failed_tool = state.get(
        "failed_tool"
    )

    last_tool_error = state.get(
        "last_tool_error"
    )

    if failed_tool:

        if has_replan_budget(
            state
        ):

            return _build_decision(
                REPLAN,
                current_step=failed_tool,
                reason=(
                    f"Tool '{failed_tool}' failed and "
                    "recovery budget remains."
                ),
                error=last_tool_error,
            )

        return _build_decision(
            FINISH,
            current_step=failed_tool,
            reason=(
                f"Tool '{failed_tool}' failed and "
                "the maximum replan limit was reached."
            ),
            error=(
                last_tool_error
                or
                f"{failed_tool} failed."
            ),
        )

    # ========================================================
    # 3. READ EXECUTION STATE
    # ========================================================

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

    # ========================================================
    # 4. RESOLVE NEXT STEP
    # ========================================================

    resolution = resolve_next_step(
        plan=plan,
        executed_tools=executed_tools,
        tool_results=tool_results,
    )

    # ========================================================
    # 5. NORMALIZE RESOLVER RESULT
    # ========================================================
    #
    # The real step resolver may return a StepResolution
    # object, while unit tests may mock it with a dictionary.
    #
    # Also support both:
    #
    # next_step
    # current_step
    #
    # so the policy remains decoupled from representation
    # details of the resolver.
    # ========================================================

    status = _resolution_value(
        resolution,
        "status",
    )

    next_step = _resolution_value(
        resolution,
        "current_step",
        "next_step",
    )

    reason = _resolution_value(
        resolution,
        "reason",
    )

    error = _resolution_value(
        resolution,
        "error",
    )

    # ========================================================
    # 6. READY TOOL
    # ========================================================

    if status == "ready":

        if not next_step:

            policy_error = (
                "No executable current step was resolved."
            )

            if has_replan_budget(
                state
            ):

                return _build_decision(
                    REPLAN,
                    current_step=None,
                    reason=(
                        "Step resolver reported a ready "
                        "state without a tool to execute."
                    ),
                    error=policy_error,
                )

            return _build_decision(
                FINISH,
                current_step=None,
                reason=(
                    "Step resolver reported a ready state "
                    "without a tool and the maximum replan "
                    "limit was reached."
                ),
                error=policy_error,
            )

        return _build_decision(
            EXECUTE,
            current_step=next_step,
            reason=(
                reason
                or
                f"Tool '{next_step}' is ready for execution."
            ),
            error=None,
        )

    # ========================================================
    # 7. PLAN COMPLETE
    # ========================================================

    if status == "complete":

        return _build_decision(
            FINISH,
            current_step=None,
            reason=(
                reason
                or
                "All tools in the execution plan "
                "have completed successfully."
            ),
            error=None,
        )

    # ========================================================
    # 8. BLOCKED PLAN
    # ========================================================

    if status == "blocked":

        if has_replan_budget(
            state
        ):

            return _build_decision(
                REPLAN,
                current_step=next_step,
                reason=(
                    reason
                    or
                    "Execution plan is blocked."
                ),
                error=error,
            )

        return _build_decision(
            FINISH,
            current_step=next_step,
            reason=(
                "Execution plan is blocked and "
                "the maximum replan limit was reached."
            ),
            error=(
                error
                or
                reason
                or
                "Execution plan is blocked."
            ),
        )

    # ========================================================
    # 9. INVALID PLAN
    # ========================================================

    if status == "invalid":

        if has_replan_budget(
            state
        ):

            return _build_decision(
                REPLAN,
                current_step=next_step,
                reason=(
                    reason
                    or
                    "Execution plan is invalid."
                ),
                error=error,
            )

        return _build_decision(
            FINISH,
            current_step=next_step,
            reason=(
                "Execution plan is invalid and "
                "the maximum replan limit was reached."
            ),
            error=(
                error
                or
                reason
                or
                "Execution plan is invalid."
            ),
        )

    # ========================================================
    # 10. UNKNOWN RESOLVER STATUS
    # ========================================================

    unknown_error = (
        f"Unknown step resolver status: {status}"
    )

    if has_replan_budget(
        state
    ):

        return _build_decision(
            REPLAN,
            current_step=next_step,
            reason=(
                "Step resolver returned an unknown status."
            ),
            error=unknown_error,
        )

    return _build_decision(
        FINISH,
        current_step=next_step,
        reason=(
            "Step resolver returned an unknown status "
            "and the maximum replan limit was reached."
        ),
        error=unknown_error,
    )