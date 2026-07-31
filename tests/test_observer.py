from backend.orchestration.nodes.observer import (
    observer_node,
    route_after_observation,
)
from unittest.mock import (
    patch,
)

from backend.orchestration.step_resolver import (
    StepResolution,
)



# ============================================================
# TEST 1
# Successful SQL -> move to visualization
# ============================================================

def test_observer_moves_to_next_tool():

    state = {
        "plan": [
            "sql",
            "visualization",
            "insight",
        ],

        "current_step": "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {
            "sql": {
                "success": True,
                "generated_sql": (
                    "SELECT city, AVG(score) "
                    "FROM dataset GROUP BY city"
                ),
            }
        },

        "replan_count": 0,
        "max_replans": 2,
        "trace": [],
    }

    result = observer_node(
        state
    )

    assert (
        result["current_step"]
        ==
        "visualization"
    )

    assert result["completed"] is False

    

    assert (
        result["trace"][-1]["decision"]
        ==
        "continue"
    )


# ============================================================
# TEST 2
# Last tool succeeds -> finish
# ============================================================

def test_observer_finishes_completed_plan():

    state = {
        "plan": [
            "sql",
            "visualization",
            "insight",
        ],

        "current_step": "insight",

        "executed_tools": [
            "sql",
            "visualization",
            "insight",
        ],

        "tool_results": {
            "sql": {
                "success": True
            },

            "visualization": {
                "success": True
            },

            "insight": {
                "success": True,
                "insight": "Example insight.",
            },
        },

        "replan_count": 0,
        "max_replans": 2,
        "trace": [],
    }

    result = observer_node(
        state
    )

    assert result["completed"] is True

    assert result["current_step"] is None

    assert (
        result["trace"][-1]["decision"]
        ==
        "finish"
    )




# ============================================================
# TEST 5
# Router sends unfinished state to executor
# ============================================================

def test_router_returns_execute():

    state = {
        "completed": False,
        "current_step": "visualization",
    }

    assert (
        route_after_observation(state)
        ==
        "execute"
    )


# ============================================================
# TEST 6
# Router sends completed state to finish
# ============================================================

def test_router_returns_finish():

    state = {
        "completed": True,
        "current_step": None,
    }

    assert (
        route_after_observation(state)
        ==
        "finish"
    )
def test_observer_replans_failed_tool():

    state = {
        "plan": [
            "sql",
            "visualization",
        ],

        "current_step": "sql",

        "executed_tools": [],

        "tool_results": {
            "sql": {
                "success": False,
                "error": "Invalid SQL",
            }
        },

        "replan_count": 0,
        "max_replans": 2,

        "trace": [],
    }

    result = observer_node(
        state
    )

    assert result["completed"] is False

    assert result["current_step"] == "sql"

    assert result["failed_tool"] == "sql"

    assert (
        result["last_tool_error"]
        == "Invalid SQL"
    )

    assert (
        result["trace"][-1]["decision"]
        == "replan"
    )

def test_observer_stops_after_max_replans():

    state = {
        "plan": [
            "sql"
        ],

        "current_step": "sql",

        "executed_tools": [],

        "tool_results": {
            "sql": {
                "success": False,
                "error": "SQL failed",
            }
        },

        "replan_count": 2,
        "max_replans": 2,

        "trace": [],
    }

    result = observer_node(
        state
    )

    assert result["completed"] is True

    assert result["current_step"] is None

    assert result["failed_tool"] == "sql"

    assert (
        result["last_tool_error"]
        == "SQL failed"
    )

    assert result["error"] is not None

    assert (
        result["trace"][-1]["decision"]
        == "finish"
    )

def test_router_returns_replan():

    state = {
        "completed": False,

        "current_step": "sql",

        "trace": [
            {
                "node": "observer",
                "decision": "replan",
                "tool": "sql",
            }
        ],
    }

    assert (
        route_after_observation(state)
        ==
        "replan"
    )

# ============================================================
# TEST 8
# Observer uses step resolver for next tool
# ============================================================

def test_observer_uses_step_resolver():

    state = {

        "plan": [
            "sql",
            "forecasting",
            "insight",
        ],

        "current_step":
            "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {

            "sql": {
                "success": True
            }
        },

        "replan_count": 0,
        "max_replans": 2,

        "trace": [],
    }

    resolution = StepResolution(
        status="ready",
        next_step="forecasting",
        blocked_tools=[],
        missing_dependencies={},
        reason="Forecasting is ready.",
    )

    with patch(
        "backend.orchestration.nodes.observer."
        "resolve_next_step",
        return_value=resolution,
    ) as mock_resolver:

        result = observer_node(
            state
        )

    mock_resolver.assert_called_once_with(
        plan=[
            "sql",
            "forecasting",
            "insight",
        ],
        executed_tools=[
            "sql"
        ],
        tool_results={
            "sql": {
                "success": True
            }
        },
    )

    assert (
        result["current_step"]
        == "forecasting"
    )

    assert (
        result["completed"]
        is False
    )

    assert (
        result["trace"][-1]["decision"]
        == "continue"
    )


# ============================================================
# TEST 9
# Resolver complete -> observer finishes
# ============================================================

def test_observer_finishes_when_resolver_complete():

    state = {

        "plan": [
            "sql"
        ],

        "current_step":
            "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {

            "sql": {
                "success": True
            }
        },

        "replan_count": 0,
        "max_replans": 2,

        "trace": [],
    }

    resolution = StepResolution(
        status="complete",
        next_step=None,
        blocked_tools=[],
        missing_dependencies={},
        reason=(
            "All planned tools completed successfully."
        ),
    )

    with patch(
        "backend.orchestration.nodes.observer."
        "resolve_next_step",
        return_value=resolution,
    ):

        result = observer_node(
            state
        )

    assert (
        result["completed"]
        is True
    )

    assert (
        result["current_step"]
        is None
    )

    assert (
        result["trace"][-1]["decision"]
        == "finish"
    )


# ============================================================
# TEST 10
# Blocked plan triggers replanning
# ============================================================

def test_observer_replans_blocked_plan():

    state = {

        "plan": [
            "sql",
            "forecasting",
        ],

        "current_step":
            "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {

            "sql": {
                "success": True
            }
        },

        "replan_count": 0,
        "max_replans": 2,

        "trace": [],
    }

    resolution = StepResolution(
        status="blocked",
        next_step=None,
        blocked_tools=[
            "forecasting"
        ],
        missing_dependencies={
            "forecasting": [
                "dataset_context"
            ]
        },
        reason=(
            "Remaining tools cannot execute because "
            "their dependencies are not satisfied."
        ),
    )

    with patch(
        "backend.orchestration.nodes.observer."
        "resolve_next_step",
        return_value=resolution,
    ):

        result = observer_node(
            state
        )

    assert (
        result["completed"]
        is False
    )

    assert (
        result["current_step"]
        is None
    )

    assert (
        result["trace"][-1]["decision"]
        == "replan"
    )

    assert (
        result["trace"][-1]["blocked_tools"]
        ==
        [
            "forecasting"
        ]
    )


# ============================================================
# TEST 11
# Invalid plan triggers replanning
# ============================================================

def test_observer_replans_invalid_plan():

    state = {

        "plan": [
            "sql",
            "unknown_tool",
        ],

        "current_step":
            "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {

            "sql": {
                "success": True
            }
        },

        "replan_count": 0,
        "max_replans": 2,

        "trace": [],
    }

    resolution = StepResolution(
        status="invalid",
        next_step=None,
        blocked_tools=[
            "unknown_tool"
        ],
        missing_dependencies={},
        reason=(
            "Unknown tool in execution plan: "
            "unknown_tool"
        ),
    )

    with patch(
        "backend.orchestration.nodes.observer."
        "resolve_next_step",
        return_value=resolution,
    ):

        result = observer_node(
            state
        )

    assert (
        result["completed"]
        is False
    )

    assert (
        result["trace"][-1]["decision"]
        == "replan"
    )

    assert (
        "Unknown tool"
        in result["last_tool_error"]
    )


# ============================================================
# TEST 12
# Blocked plan stops after max replans
# ============================================================

def test_observer_stops_blocked_plan_after_max_replans():

    state = {

        "plan": [
            "sql",
            "forecasting",
        ],

        "current_step":
            "sql",

        "executed_tools": [
            "sql"
        ],

        "tool_results": {

            "sql": {
                "success": True
            }
        },

        "replan_count": 2,
        "max_replans": 2,

        "trace": [],
    }

    resolution = StepResolution(
        status="blocked",
        next_step=None,
        blocked_tools=[
            "forecasting"
        ],
        missing_dependencies={
            "forecasting": [
                "dataset_context"
            ]
        },
        reason="Forecasting dependency missing.",
    )

    with patch(
        "backend.orchestration.nodes.observer."
        "resolve_next_step",
        return_value=resolution,
    ):

        result = observer_node(
            state
        )

    assert (
        result["completed"]
        is True
    )

    assert (
        result["current_step"]
        is None
    )

    assert (
        result["error"]
        is not None
    )

    assert (
        result["trace"][-1]["decision"]
        == "finish"
    )
