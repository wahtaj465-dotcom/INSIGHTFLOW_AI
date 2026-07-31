from unittest.mock import (
    patch,
)

from backend.orchestration.nodes.observer import (
    observer_node,
    route_after_observation,
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

    assert (
        result["completed"]
        is False
    )

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
        ==
        "finish"
    )


# ============================================================
# TEST 3
# Router sends unfinished state to executor
# ============================================================

def test_router_returns_execute():

    state = {
        "completed": False,
        "current_step": "visualization",
    }

    assert (
        route_after_observation(
            state
        )
        ==
        "execute"
    )


# ============================================================
# TEST 4
# Router sends completed state to finish
# ============================================================

def test_router_returns_finish():

    state = {
        "completed": True,
        "current_step": None,
    }

    assert (
        route_after_observation(
            state
        )
        ==
        "finish"
    )


# ============================================================
# TEST 5
# Failed tool -> replan
# ============================================================

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

    assert (
        result["completed"]
        is False
    )

    assert (
        result["current_step"]
        ==
        "sql"
    )

    assert (
        result["failed_tool"]
        ==
        "sql"
    )

    assert (
        result["last_tool_error"]
        ==
        "Invalid SQL"
    )

    assert (
        result["trace"][-1]["decision"]
        ==
        "replan"
    )


# ============================================================
# TEST 6
# Failed tool stops after max replans
# ============================================================

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

    assert (
        result["completed"]
        is True
    )

    assert (
        result["current_step"]
        is None
    )

    assert (
        result["failed_tool"]
        ==
        "sql"
    )

    assert (
        result["last_tool_error"]
        ==
        "SQL failed"
    )

    assert (
        result["error"]
        is not None
    )

    assert (
        result["trace"][-1]["decision"]
        ==
        "finish"
    )


# ============================================================
# TEST 7
# Router sends replan decision to replanner
# ============================================================

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
        route_after_observation(
            state
        )
        ==
        "replan"
    )


# ============================================================
# TEST 8
# Observer uses execution policy
# ============================================================

def test_observer_uses_execution_policy():

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

    policy_decision = {
        "action":
            "execute",

        "current_step":
            "forecasting",

        "reason":
            "Forecasting is ready.",

        "error":
            None,
    }

    with patch(
        "backend.orchestration.nodes.observer."
        "evaluate_execution_policy",
        return_value=policy_decision,
    ) as mock_policy:

        result = observer_node(
            state
        )

    mock_policy.assert_called_once_with(
        state
    )

    assert (
        result["current_step"]
        ==
        "forecasting"
    )

    assert (
        result["completed"]
        is False
    )

    assert (
        result["trace"][-1]["decision"]
        ==
        "continue"
    )


# ============================================================
# TEST 9
# Policy finish -> observer finishes
# ============================================================

def test_observer_finishes_when_policy_finishes():

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

    policy_decision = {
        "action":
            "finish",

        "current_step":
            None,

        "reason":
            "All planned tools completed successfully.",

        "error":
            None,
    }

    with patch(
        "backend.orchestration.nodes.observer."
        "evaluate_execution_policy",
        return_value=policy_decision,
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
        ==
        "finish"
    )


# ============================================================
# TEST 10
# Blocked policy decision -> replan
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

    policy_decision = {
        "action":
            "replan",

        "current_step":
            None,

        "reason":
            (
                "Remaining tools cannot execute because "
                "their dependencies are not satisfied."
            ),

        "error":
            "Forecasting dependency missing.",
    }

    with patch(
        "backend.orchestration.nodes.observer."
        "evaluate_execution_policy",
        return_value=policy_decision,
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
        ==
        "replan"
    )

    assert (
        result["error"]
        is None
    )


# ============================================================
# TEST 11
# Invalid policy decision -> replan
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

    policy_decision = {
        "action":
            "replan",

        "current_step":
            None,

        "reason":
            (
                "Unknown tool in execution plan: "
                "unknown_tool"
            ),

        "error":
            (
                "Unknown tool in execution plan: "
                "unknown_tool"
            ),
    }

    with patch(
        "backend.orchestration.nodes.observer."
        "evaluate_execution_policy",
        return_value=policy_decision,
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
        ==
        "replan"
    )

    assert (
        "Unknown tool"
        in result["last_tool_error"]
    )


# ============================================================
# TEST 12
# Blocked plan stops after replan budget exhausted
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

    policy_decision = {
        "action":
            "finish",

        "current_step":
            None,

        "reason":
            (
                "Execution plan is blocked and "
                "the maximum replan limit was reached."
            ),

        "error":
            "Forecasting dependency missing.",
    }

    with patch(
        "backend.orchestration.nodes.observer."
        "evaluate_execution_policy",
        return_value=policy_decision,
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
        ==
        "finish"
    )