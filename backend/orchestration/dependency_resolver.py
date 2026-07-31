from typing import Any, Dict, List

from backend.orchestration.state import AgentState
from backend.orchestration.tool_registry import (
    get_tool_definition,
)


# ============================================================
# DEPENDENCY RESULT
# ============================================================

def _build_dependency_result(
    ready: bool,
    missing_dependencies: List[str],
) -> Dict[str, Any]:
    """
    Build a consistent dependency-check result.
    """

    return {
        "ready": ready,
        "missing_dependencies": missing_dependencies,
    }


# ============================================================
# CHECK TOOL DEPENDENCIES
# ============================================================

def check_tool_dependencies(
    tool_name: str,
    state: AgentState,
) -> Dict[str, Any]:
    """
    Check whether all dependencies required by a tool
    have already completed successfully.

    Dependency information comes entirely from the
    central tool registry.

    Example:

        forecasting:
            dependencies = ["sql"]

    The forecasting tool is ready only if SQL has
    successfully executed.

    Returns:

        {
            "ready": True,
            "missing_dependencies": [],
        }

    or:

        {
            "ready": False,
            "missing_dependencies": ["sql"],
        }
    """

    # ========================================================
    # GET TOOL DEFINITION
    # ========================================================

    definition = get_tool_definition(
        tool_name
    )

    # Unknown tools cannot be considered ready.
    if definition is None:

        return _build_dependency_result(
            ready=False,
            missing_dependencies=[
                tool_name
            ],
        )

    # ========================================================
    # GET DECLARED DEPENDENCIES
    # ========================================================

    dependencies = list(
        definition.get(
            "dependencies",
            []
        )
        or []
    )

    # Tool has no dependencies.
    if not dependencies:

        return _build_dependency_result(
            ready=True,
            missing_dependencies=[],
        )

    # ========================================================
    # READ EXECUTION STATE
    # ========================================================

    executed_tools = set(
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

    missing_dependencies = []

    # ========================================================
    # CHECK EACH DEPENDENCY
    # ========================================================

    for dependency in dependencies:

        # ----------------------------------------------------
        # Dependency has not executed
        # ----------------------------------------------------

        if dependency not in executed_tools:

            missing_dependencies.append(
                dependency
            )

            continue

        # ----------------------------------------------------
        # Dependency executed, verify successful result
        # ----------------------------------------------------

        result = tool_results.get(
            dependency
        )

        if isinstance(
            result,
            dict
        ):

            if result.get(
                "success"
            ) is not True:

                missing_dependencies.append(
                    dependency
                )

        elif result is None:

            missing_dependencies.append(
                dependency
            )

    # ========================================================
    # BUILD RESULT
    # ========================================================

    return _build_dependency_result(
        ready=not missing_dependencies,
        missing_dependencies=missing_dependencies,
    )


# ============================================================
# GET MISSING DEPENDENCIES
# ============================================================

def get_missing_dependencies(
    tool_name: str,
    state: AgentState,
) -> List[str]:
    """
    Convenience helper that returns only the names
    of unresolved dependencies.
    """

    result = check_tool_dependencies(
        tool_name,
        state,
    )

    return result[
        "missing_dependencies"
    ]


# ============================================================
# CHECK TOOL READINESS
# ============================================================

def is_tool_ready(
    tool_name: str,
    state: AgentState,
) -> bool:
    """
    Return True when all dependencies required by
    the tool have completed successfully.
    """

    result = check_tool_dependencies(
        tool_name,
        state,
    )

    return result[
        "ready"
    ]