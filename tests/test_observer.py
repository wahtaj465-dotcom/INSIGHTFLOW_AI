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
