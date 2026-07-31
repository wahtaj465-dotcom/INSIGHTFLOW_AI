import json

from backend.orchestration.state import AgentState

from backend.orchestration.planner import (
    _parse_plan,
    _fallback_plan,
)

from backend.orchestration.tool_registry import (
    get_tool_descriptions,
)

from backend.orchestration.plan_resolver import (
    resolve_plan_dependencies,
)

from backend.services.llm_service import (
    LLMService,
)


# ============================================================
# BUILD DYNAMIC TOOL CATALOG
# ============================================================

def _build_replanner_tool_catalog() -> str:
    """
    Build the tool catalog dynamically from the
    central tool registry.

    This ensures the replanner automatically knows
    about newly registered tools.
    """

    tool_descriptions = (
        get_tool_descriptions()
    )

    catalog = []

    for (
        tool_name,
        metadata
    ) in tool_descriptions.items():

        catalog.append({
            "name":
                tool_name,

            "description":
                metadata.get(
                    "description",
                    ""
                ),

            "inputs":
                metadata.get(
                    "inputs",
                    {}
                ),

            "outputs":
                metadata.get(
                    "outputs",
                    {}
                ),

            "dependencies":
                metadata.get(
                    "dependencies",
                    []
                ),
        })

    return json.dumps(
        catalog,
        indent=2
    )


# ============================================================
# BUILD RECOVERY PROMPT
# ============================================================

def _build_replanner_prompt(
    question: str,
    current_plan: list[str],
    executed_tools: list[str],
    failed_tool: str | None,
    last_tool_error: str | None,
    tool_catalog: str | None = None,
) -> str:
    """
    Build the recovery prompt using the dynamic
    tool registry.
    """

    if tool_catalog is None:

        tool_catalog = (
            _build_replanner_tool_catalog()
        )

    return f"""
You are the recovery planner for an autonomous
data analytics agent.

The agent was executing an analytics workflow
and one of its tools failed.

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

AVAILABLE TOOL CATALOG:

{tool_catalog}

Create a revised plan for completing the user's request.

RULES:

1. Return ONLY valid JSON.

2. Do not use markdown.

3. Use only tools from the AVAILABLE TOOL CATALOG.

4. Do not repeat successfully executed tools unless
they must be executed again because recovery requires
new output.

5. You may retry the failed tool if appropriate.

6. Tool dependencies will be expanded automatically
after the recovery plan is generated.

7. Keep the recovery plan minimal.

8. Do not answer the user's analytical question yourself.

Return exactly:

{{
    "intent": "recovery",
    "tools": ["tool_name"],
    "reasoning": "brief explanation of the recovery plan"
}}
""".strip()


# ============================================================
# GET CURRENT AVAILABLE TOOLS
# ============================================================

def _get_replanner_available_tools(
    tool_descriptions: dict,
) -> set[str]:
    """
    Derive available tool names from the same registry
    metadata used to build the replanner prompt.

    This keeps prompt generation and validation based
    on one consistent registry snapshot.
    """

    return set(
        tool_descriptions.keys()
    )


# ============================================================
# REPLANNER NODE
# ============================================================

def replanner_node(
    state: AgentState
) -> dict:
    """
    Create a revised execution plan after a tool failure.

    Flow:

    LLM recovery selection
            ↓
    registry validation
            ↓
    dependency expansion
            ↓
    remove completed tools
            ↓
    recovery execution
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

    replan_count = (
        state.get(
            "replan_count",
            0
        )
        + 1
    )

    # ========================================================
    # LOAD REGISTRY METADATA ONCE
    # ========================================================

    tool_descriptions = (
        get_tool_descriptions()
    )

    available_tools = (
        _get_replanner_available_tools(
            tool_descriptions
        )
    )

    # Build the exact catalog from the same snapshot.

    catalog = []

    for (
        tool_name,
        metadata
    ) in tool_descriptions.items():

        catalog.append({
            "name":
                tool_name,

            "description":
                metadata.get(
                    "description",
                    ""
                ),

            "inputs":
                metadata.get(
                    "inputs",
                    {}
                ),

            "outputs":
                metadata.get(
                    "outputs",
                    {}
                ),

            "dependencies":
                metadata.get(
                    "dependencies",
                    []
                ),
        })

    tool_catalog = json.dumps(
        catalog,
        indent=2
    )

    # ========================================================
    # TRY LLM REPLANNING
    # ========================================================

    try:

        prompt = _build_replanner_prompt(
            question=question,
            current_plan=current_plan,
            executed_tools=executed_tools,
            failed_tool=failed_tool,
            last_tool_error=last_tool_error,
            tool_catalog=tool_catalog,
        )

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
        # Validate selected capabilities
        # ----------------------------------------------------

        requested_tools = [
            tool_name
            for tool_name in revised_plan.tools
            if tool_name in available_tools
        ]

        requested_tools = list(
            dict.fromkeys(
                requested_tools
            )
        )

        if not requested_tools:

            raise ValueError(
                "Replanner returned no valid recovery tools."
            )

        # ----------------------------------------------------
        # Expand dependencies
        # ----------------------------------------------------

        resolved_plan = (
            resolve_plan_dependencies(
                requested_tools
            )
        )

        # ----------------------------------------------------
        # Remove tools that already succeeded
        # ----------------------------------------------------

        remaining_tools = [
            tool_name
            for tool_name in resolved_plan
            if tool_name not in executed_tools
        ]

        planner_source = (
            "llm"
        )

        reasoning = (
            revised_plan.reasoning
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    except Exception as exc:

        fallback_plan = (
            _fallback_plan(
                question
            )
        )

        requested_tools = [
            tool_name
            for tool_name in fallback_plan.tools
            if tool_name in available_tools
        ]

        requested_tools = list(
            dict.fromkeys(
                requested_tools
            )
        )

        if requested_tools:

            try:

                resolved_plan = (
                    resolve_plan_dependencies(
                        requested_tools
                    )
                )

            except Exception:

                resolved_plan = (
                    requested_tools
                )

        else:

            resolved_plan = []

        remaining_tools = [
            tool_name
            for tool_name in resolved_plan
            if tool_name not in executed_tools
        ]

        planner_source = (
            "fallback"
        )

        reasoning = (
            "LLM re-planning failed. "
            f"Fallback recovery plan used: {exc}"
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    remaining_tools = list(
        dict.fromkeys(
            remaining_tools
        )
    )

    # ========================================================
    # RETRY FAILED TOOL IF NO ALTERNATIVE EXISTS
    # ========================================================

    if (
        not remaining_tools
        and failed_tool
        and failed_tool in available_tools
    ):

        remaining_tools = [
            failed_tool
        ]

    # ========================================================
    # TRACE
    # ========================================================

    trace.append({
        "node":
            "replanner",

        "source":
            planner_source,

        "failed_tool":
            failed_tool,

        "revised_plan":
            remaining_tools,

        "replan_count":
            replan_count,
    })

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "plan":
            remaining_tools,

        "plan_reasoning":
            reasoning,

        "planner_source":
            planner_source,

        "current_step": (
            remaining_tools[0]
            if remaining_tools
            else None
        ),

        "replan_count":
            replan_count,

        "retry_count":
            0,

        "error":
            None,

        "trace":
            trace,
    }