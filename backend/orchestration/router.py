from backend.orchestration.state import AgentState


def route_after_observation(
    state: AgentState
) -> str:
    """
    Decide whether another tool should execute
    or the graph should finalize.
    """

    plan = state.get(
        "plan",
        []
    )

    executed_tools = state.get(
        "executed_tools",
        []
    )

    remaining = [
        tool_name
        for tool_name in plan
        if tool_name not in executed_tools
    ]

    if remaining:
        return "continue"

    return "finish"