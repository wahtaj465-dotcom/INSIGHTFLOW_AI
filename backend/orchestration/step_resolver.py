from dataclasses import (
    dataclass,
)

from backend.orchestration.tool_registry import (
    get_tool_definition,
)


# ============================================================
# STEP RESOLUTION RESULT
# ============================================================

@dataclass
class StepResolution:
    """
    Result returned by the step resolver.

    status:
        ready
            A tool is available for execution.

        complete
            Every tool in the plan completed successfully.

        blocked
            Remaining work exists, but no remaining tool
            can currently execute.

        invalid
            The plan contains an unknown or invalid tool.
    """

    status: str

    next_step: str | None = None

    blocked_tools: list[str] | None = None

    missing_dependencies: dict[
        str,
        list[str],
    ] | None = None

    reason: str | None = None


# ============================================================
# TOOL SUCCESS
# ============================================================

def _tool_succeeded(
    tool_name: str,
    executed_tools: list[str],
    tool_results: dict,
) -> bool:
    """
    Determine whether a tool should be considered
    successfully completed.

    executed_tools is treated as the primary execution
    record, while tool_results confirms that the latest
    recorded result did not fail.
    """

    if tool_name not in executed_tools:
        return False

    result = tool_results.get(
        tool_name
    )

    if result is None:
        return True

    if isinstance(
        result,
        dict
    ):

        return (
            result.get(
                "success",
                True,
            )
            is True
        )

    return True


# ============================================================
# DEPENDENCY SUCCESS
# ============================================================

def _dependency_succeeded(
    dependency: str,
    executed_tools: list[str],
    tool_results: dict,
) -> bool:
    """
    Check whether a dependency completed successfully.
    """

    return _tool_succeeded(
        dependency,
        executed_tools,
        tool_results,
    )


# ============================================================
# GET TOOL DEPENDENCIES
# ============================================================

def _get_dependencies(
    tool_name: str,
) -> list[str] | None:
    """
    Return dependencies declared by a tool.

    None means the tool is unknown.
    """

    definition = (
        get_tool_definition(
            tool_name
        )
    )

    if definition is None:
        return None

    return list(
        definition.get(
            "dependencies",
            []
        )
    )


# ============================================================
# RESOLVE NEXT STEP
# ============================================================

def resolve_next_step(
    plan: list[str],
    executed_tools: list[str],
    tool_results: dict,
) -> StepResolution:
    """
    Determine the next executable tool in a plan.

    The resolver:

    1. walks through the plan in order
    2. skips successfully completed tools
    3. validates registered tools
    4. checks declared dependencies
    5. returns the first executable tool
    6. reports blocked plans when work remains but
       dependencies are not satisfied
    7. reports complete when all work succeeded

    This function does NOT execute tools and does NOT
    mutate the agent state.
    """

    plan = list(
        plan or []
    )

    executed_tools = list(
        executed_tools or []
    )

    tool_results = dict(
        tool_results or {}
    )

    # ========================================================
    # EMPTY PLAN
    # ========================================================

    if not plan:

        return StepResolution(
            status="complete",
            next_step=None,
            blocked_tools=[],
            missing_dependencies={},
            reason="Execution plan contains no remaining tools.",
        )

    # ========================================================
    # TRACK BLOCKED TOOLS
    # ========================================================

    blocked_tools = []

    missing_dependencies = {}

    remaining_work = False

    # ========================================================
    # WALK PLAN IN ORDER
    # ========================================================

    for tool_name in plan:

        # ----------------------------------------------------
        # Skip successful tools
        # ----------------------------------------------------

        if _tool_succeeded(
            tool_name,
            executed_tools,
            tool_results,
        ):

            continue

        remaining_work = True

        # ----------------------------------------------------
        # Validate tool
        # ----------------------------------------------------

        dependencies = (
            _get_dependencies(
                tool_name
            )
        )

        if dependencies is None:

            return StepResolution(
                status="invalid",
                next_step=None,
                blocked_tools=[
                    tool_name
                ],
                missing_dependencies={},
                reason=(
                    f"Unknown tool in execution plan: "
                    f"{tool_name}"
                ),
            )

        # ----------------------------------------------------
        # Find unsatisfied dependencies
        # ----------------------------------------------------

        unsatisfied = [
            dependency
            for dependency in dependencies
            if not _dependency_succeeded(
                dependency,
                executed_tools,
                tool_results,
            )
        ]

        # ----------------------------------------------------
        # Tool is executable
        # ----------------------------------------------------

        if not unsatisfied:

            return StepResolution(
                status="ready",
                next_step=tool_name,
                blocked_tools=[],
                missing_dependencies={},
                reason=(
                    f"Tool '{tool_name}' is ready "
                    "for execution."
                ),
            )

        # ----------------------------------------------------
        # Tool is blocked
        # ----------------------------------------------------

        blocked_tools.append(
            tool_name
        )

        missing_dependencies[
            tool_name
        ] = unsatisfied

    # ========================================================
    # ALL TOOLS COMPLETED
    # ========================================================

    if not remaining_work:

        return StepResolution(
            status="complete",
            next_step=None,
            blocked_tools=[],
            missing_dependencies={},
            reason=(
                "All planned tools completed successfully."
            ),
        )

    # ========================================================
    # REMAINING WORK EXISTS BUT NOTHING IS READY
    # ========================================================

    return StepResolution(
        status="blocked",
        next_step=None,
        blocked_tools=blocked_tools,
        missing_dependencies=missing_dependencies,
        reason=(
            "Remaining tools cannot execute because "
            "their dependencies are not satisfied."
        ),
    )