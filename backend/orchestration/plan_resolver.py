from backend.orchestration.tool_registry import (
    get_tool_definition,
)


# ============================================================
# RESOLVE TOOL DEPENDENCIES
# ============================================================

def resolve_plan_dependencies(
    plan: list[str],
) -> list[str]:
    """
    Expand a tool plan so that all declared dependencies
    appear before the tools that require them.

    Example:

        plan:
            ["visualization"]

        visualization dependencies:
            ["sql"]

        result:
            ["sql", "visualization"]

    Dependencies are resolved recursively.

    Raises:
        ValueError:
            if an unknown tool is encountered
            or a circular dependency exists.
    """

    resolved = []

    visiting = set()
    visited = set()

    for tool_name in plan:

        _resolve_tool(
            tool_name=tool_name,
            resolved=resolved,
            visiting=visiting,
            visited=visited,
        )

    return resolved


# ============================================================
# RESOLVE SINGLE TOOL
# ============================================================

def _resolve_tool(
    tool_name: str,
    resolved: list[str],
    visiting: set[str],
    visited: set[str],
):
    """
    Recursively resolve one tool and its dependencies.
    """

    # --------------------------------------------------------
    # Already completely resolved
    # --------------------------------------------------------

    if tool_name in visited:
        return

    # --------------------------------------------------------
    # Circular dependency detected
    # --------------------------------------------------------

    if tool_name in visiting:

        raise ValueError(
            f"Circular tool dependency detected "
            f"for '{tool_name}'."
        )

    # --------------------------------------------------------
    # Tool must exist
    # --------------------------------------------------------

    definition = get_tool_definition(
        tool_name
    )

    if definition is None:

        raise ValueError(
            f"Unknown tool in plan: {tool_name}"
        )

    # --------------------------------------------------------
    # Start resolving this tool
    # --------------------------------------------------------

    visiting.add(
        tool_name
    )

    dependencies = definition.get(
        "dependencies",
        []
    )

    # --------------------------------------------------------
    # Resolve dependencies first
    # --------------------------------------------------------

    for dependency in dependencies:

        _resolve_tool(
            tool_name=dependency,
            resolved=resolved,
            visiting=visiting,
            visited=visited,
        )

    # --------------------------------------------------------
    # Tool resolution complete
    # --------------------------------------------------------

    visiting.remove(
        tool_name
    )

    visited.add(
        tool_name
    )

    resolved.append(
        tool_name
    )