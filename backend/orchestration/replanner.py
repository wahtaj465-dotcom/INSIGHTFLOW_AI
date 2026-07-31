import json

from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.planner import (
    _parse_plan,
    _fallback_plan,
)

from backend.services.llm_service import (
    LLMService,
)


# ============================================================
# REPLANNER NODE
# ============================================================

def replanner_node(
    state: AgentState
) -> dict:
    """
    Create a revised execution plan after a tool failure.

    The re-planner considers:

    - original user question
    - current execution plan
    - successfully executed tools
    - failed tool
    - failure reason

    It then creates a revised remaining execution plan.
    """

    # ========================================================
    # READ STATE
    # ========================================================

    question = state.get(
        "question",
        ""
    )

    current_plan = list(
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

    failed_tool = state.get(
        "failed_tool"
    )

    last_tool_error = state.get(
        "last_tool_error"
    )

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    current_replan_count = state.get(
        "replan_count",
        0
    )

    replan_count = (
        current_replan_count
        + 1
    )

    # ========================================================
    # BUILD RECOVERY PROMPT
    # ========================================================

    prompt = f"""
You are the recovery planner for an autonomous data analytics agent.

The agent was executing an analytics workflow and one of its tools failed.

USER QUESTION:
{question}

CURRENT PLAN:
{json.dumps(current_plan)}

SUCCESSFULLY EXECUTED TOOLS:
{json.dumps(executed_tools)}

FAILED TOOL:
{failed_tool}

ERROR:
{last_tool_error}

AVAILABLE TOOLS:

- dataset_context
- sql
- visualization
- insight

Create a revised plan for completing the user's request.

RULES:

1. Return ONLY valid JSON.
2. Do not use markdown.
3. Use only the available tools.
4. Do not repeat successfully executed tools unless absolutely necessary.
5. You may retry the failed tool if retrying it is appropriate.
6. Preserve logical tool dependencies.
7. Visualization may depend on SQL results.
8. Insight generation may depend on SQL results.
9. Keep the revised plan as small as possible while still completing the request.

Return exactly this JSON structure:

{{
    "intent": "recovery",
    "tools": ["tool_name"],
    "reasoning": "brief explanation of the recovery plan"
}}
"""

    # ========================================================
    # TRY LLM RE-PLANNING
    # ========================================================

    try:

        llm_service = (
            LLMService()
        )

        response = (
            llm_service.generate(
                prompt
            )
        )

        revised_plan = (
            _parse_plan(
                response
            )
        )

        # ----------------------------------------------------
        # Remove tools that already completed successfully
        # ----------------------------------------------------

        remaining_tools = [
            tool_name

            for tool_name
            in revised_plan.tools

            if tool_name
            not in executed_tools
        ]

        planner_source = (
            "llm"
        )

        reasoning = (
            revised_plan.reasoning
        )

    # ========================================================
    # FALLBACK IF LLM IS UNAVAILABLE
    # ========================================================

    except Exception as exc:

        fallback_plan = (
            _fallback_plan(
                question
            )
        )

        remaining_tools = [
            tool_name

            for tool_name
            in fallback_plan.tools

            if tool_name
            not in executed_tools
        ]

        planner_source = (
            "fallback"
        )

        reasoning = (
            "LLM re-planning failed. "
            "Fallback recovery plan used. "
            f"Reason: {exc}"
        )

    # ========================================================
    # ENSURE RECOVERY HAS SOMETHING TO EXECUTE
    # ========================================================

    if (
        not remaining_tools
        and
        failed_tool
    ):

        remaining_tools = [
            failed_tool
        ]

    # ========================================================
    # DETERMINE NEXT STEP
    # ========================================================

    next_step = (
        remaining_tools[0]
        if remaining_tools
        else None
    )

    # ========================================================
    # TRACE RE-PLANNING
    # ========================================================

    trace.append({
        "node":
            "replanner",

        "status":
            "success",

        "source":
            planner_source,

        "failed_tool":
            failed_tool,

        "previous_plan":
            current_plan,

        "revised_plan":
            remaining_tools,

        "replan_count":
            replan_count,

        "reasoning":
            reasoning,
    })

    # ========================================================
    # RETURN STATE UPDATE
    # ========================================================

    return {

        # ----------------------------------------------------
        # Revised plan
        # ----------------------------------------------------

        "plan":
            remaining_tools,

        "plan_reasoning":
            reasoning,

        "planner_source":
            planner_source,

        # ----------------------------------------------------
        # Next execution step
        # ----------------------------------------------------

        "current_step":
            next_step,

        # ----------------------------------------------------
        # Recovery counters
        # ----------------------------------------------------

        "replan_count":
            replan_count,

        "retry_count":
            0,

        # ----------------------------------------------------
        # Clear previous failure state
        # ----------------------------------------------------

        "failed_tool":
            None,

        "last_tool_error":
            None,

        "error":
            None,

        # ----------------------------------------------------
        # Workflow continues
        # ----------------------------------------------------

        "completed":
            False,

        # ----------------------------------------------------
        # Observability
        # ----------------------------------------------------

        "trace":
            trace,
    }