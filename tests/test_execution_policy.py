from unittest.mock import patch

from backend.orchestration.execution_policy import (
    EXECUTE,
    REPLAN,
    FINISH,
    evaluate_execution_policy,
    has_replan_budget,
)


# ============================================================
# TEST 1
# Recovery budget remains
# ============================================================

def test_has_replan_budget():

    state = {
        "replan_count": 0,
        "max_replans": 2,
    }

    assert (
        has_replan_budget(state)
        is True
    )


# ============================================================
# TEST 2
# Recovery budget exhausted
# ============================================================

def test_replan_budget_exhausted():

    state = {
        "replan_count": 2,
        "max_replans": 2,
    }

    assert (
        has_replan_budget(state)
        is False
    )


# ============================================================
# TEST 3
# Ready tool -> execute
# ============================================================

def test_policy_executes_ready_tool():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "ready",
        "current_step": "sql",
        "reason": "SQL is ready.",
        "error": None,
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == EXECUTE
    )

    assert (
        decision["current_step"]
        == "sql"
    )

    assert (
        decision["error"]
        is None
    )


# ============================================================
# TEST 4
# Completed plan -> finish
# ============================================================

def test_policy_finishes_completed_plan():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "complete",
        "current_step": None,
        "reason": "All tools completed.",
        "error": None,
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == FINISH
    )

    assert (
        decision["current_step"]
        is None
    )


# ============================================================
# TEST 5
# Failed tool -> replan
# ============================================================

def test_policy_replans_failed_tool():

    state = {
        "completed": False,

        "failed_tool":
            "sql",

        "last_tool_error":
            "Invalid SQL.",

        "replan_count":
            0,

        "max_replans":
            2,
    }

    decision = (
        evaluate_execution_policy(
            state
        )
    )

    assert (
        decision["action"]
        == REPLAN
    )

    assert (
        decision["current_step"]
        == "sql"
    )

    assert (
        decision["error"]
        == "Invalid SQL."
    )


# ============================================================
# TEST 6
# Failed tool + exhausted recovery -> finish
# ============================================================

def test_policy_stops_failed_tool_after_max_replans():

    state = {
        "completed": False,

        "failed_tool":
            "sql",

        "last_tool_error":
            "SQL failed.",

        "replan_count":
            2,

        "max_replans":
            2,
    }

    decision = (
        evaluate_execution_policy(
            state
        )
    )

    assert (
        decision["action"]
        == FINISH
    )

    assert (
        decision["error"]
        == "SQL failed."
    )


# ============================================================
# TEST 7
# Blocked plan -> replan
# ============================================================

def test_policy_replans_blocked_plan():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "blocked",

        "current_step":
            "visualization",

        "reason":
            "SQL dependency has not completed.",

        "error":
            "Missing dependency: sql",
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == REPLAN
    )

    assert (
        decision["current_step"]
        == "visualization"
    )


# ============================================================
# TEST 8
# Blocked plan + no recovery -> finish
# ============================================================

def test_policy_stops_blocked_plan_after_max_replans():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 2,
        "max_replans": 2,
    }

    resolution = {
        "status": "blocked",

        "current_step":
            "visualization",

        "reason":
            "Dependency unavailable.",

        "error":
            "Missing dependency: sql",
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == FINISH
    )

    assert (
        decision["error"]
        is not None
    )


# ============================================================
# TEST 9
# Invalid plan -> replan
# ============================================================

def test_policy_replans_invalid_plan():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "invalid",

        "current_step":
            None,

        "reason":
            "Plan contains unknown tool.",

        "error":
            "Unknown tool: forecasting",
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == REPLAN
    )

    assert (
        decision["error"]
        == "Unknown tool: forecasting"
    )


# ============================================================
# TEST 10
# Already completed state -> finish immediately
# ============================================================

def test_policy_finishes_already_completed_state():

    state = {
        "completed": True,

        "failed_tool":
            None,

        "replan_count":
            0,

        "max_replans":
            2,
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
    ) as mock_resolver:

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == FINISH
    )

    mock_resolver.assert_not_called()


# ============================================================
# TEST 11
# Resolver returns unknown state
# ============================================================

def test_policy_replans_unknown_resolver_status():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "something_new",
        "current_step": None,
        "reason": None,
        "error": None,
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == REPLAN
    )

    assert (
        decision["error"]
        ==
        "Unknown step resolver status: something_new"
    )


# ============================================================
# TEST 12
# Ready status without tool is unsafe
# ============================================================

def test_policy_replans_ready_status_without_tool():

    state = {
        "completed": False,
        "failed_tool": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    resolution = {
        "status": "ready",
        "current_step": None,
        "reason": None,
        "error": None,
    }

    with patch(
        "backend.orchestration.execution_policy."
        "resolve_next_step",
        return_value=resolution,
    ):

        decision = (
            evaluate_execution_policy(
                state
            )
        )

    assert (
        decision["action"]
        == REPLAN
    )

    assert (
        decision["current_step"]
        is None
    )

    assert (
        decision["error"]
        is not None
    )