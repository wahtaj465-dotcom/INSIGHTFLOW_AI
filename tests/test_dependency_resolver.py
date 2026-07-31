from unittest.mock import patch

from backend.orchestration.dependency_resolver import (
    check_tool_dependencies,
    get_missing_dependencies,
    is_tool_ready,
)


# ============================================================
# TEST 1
# Tool with no dependencies -> ready
# ============================================================

def test_tool_with_no_dependencies_is_ready():

    fake_definition = {
        "dependencies": [],
    }

    state = {
        "executed_tools": [],
        "tool_results": {},
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = check_tool_dependencies(
            "sql",
            state,
        )

    assert result["ready"] is True

    assert result["missing_dependencies"] == []


# ============================================================
# TEST 2
# Successful dependency -> ready
# ============================================================

def test_tool_with_successful_dependency_is_ready():

    fake_definition = {
        "dependencies": [
            "sql",
        ],
    }

    state = {
        "executed_tools": [
            "sql",
        ],

        "tool_results": {
            "sql": {
                "success": True,
                "result": [
                    {
                        "sales": 100
                    }
                ],
            }
        },
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = check_tool_dependencies(
            "forecasting",
            state,
        )

    assert result["ready"] is True

    assert result["missing_dependencies"] == []


# ============================================================
# TEST 3
# Dependency has not executed -> not ready
# ============================================================

def test_missing_dependency_is_detected():

    fake_definition = {
        "dependencies": [
            "sql",
        ],
    }

    state = {
        "executed_tools": [],
        "tool_results": {},
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = check_tool_dependencies(
            "forecasting",
            state,
        )

    assert result["ready"] is False

    assert result["missing_dependencies"] == [
        "sql",
    ]


# ============================================================
# TEST 4
# Dependency executed but failed -> not ready
# ============================================================

def test_failed_dependency_is_not_ready():

    fake_definition = {
        "dependencies": [
            "sql",
        ],
    }

    state = {
        "executed_tools": [
            "sql",
        ],

        "tool_results": {
            "sql": {
                "success": False,
                "error": "SQL execution failed.",
            }
        },
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = check_tool_dependencies(
            "forecasting",
            state,
        )

    assert result["ready"] is False

    assert result["missing_dependencies"] == [
        "sql",
    ]


# ============================================================
# TEST 5
# Multiple dependencies -> detect only unresolved ones
# ============================================================

def test_multiple_dependencies_detect_missing_ones():

    fake_definition = {
        "dependencies": [
            "dataset_context",
            "sql",
        ],
    }

    state = {
        "executed_tools": [
            "dataset_context",
        ],

        "tool_results": {
            "dataset_context": {
                "success": True,
            }
        },
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = check_tool_dependencies(
            "forecasting",
            state,
        )

    assert result["ready"] is False

    assert result["missing_dependencies"] == [
        "sql",
    ]


# ============================================================
# TEST 6
# Unknown tool -> not ready
# ============================================================

def test_unknown_tool_is_not_ready():

    state = {
        "executed_tools": [],
        "tool_results": {},
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=None,
    ):

        result = check_tool_dependencies(
            "unknown_tool",
            state,
        )

    assert result["ready"] is False

    assert result["missing_dependencies"] == [
        "unknown_tool",
    ]


# ============================================================
# TEST 7
# get_missing_dependencies helper
# ============================================================

def test_get_missing_dependencies():

    fake_definition = {
        "dependencies": [
            "sql",
            "dataset_context",
        ],
    }

    state = {
        "executed_tools": [
            "sql",
        ],

        "tool_results": {
            "sql": {
                "success": True,
            }
        },
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        missing = get_missing_dependencies(
            "forecasting",
            state,
        )

    assert missing == [
        "dataset_context",
    ]


# ============================================================
# TEST 8
# is_tool_ready helper -> True
# ============================================================

def test_is_tool_ready_returns_true():

    fake_definition = {
        "dependencies": [
            "sql",
        ],
    }

    state = {
        "executed_tools": [
            "sql",
        ],

        "tool_results": {
            "sql": {
                "success": True,
            }
        },
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        ready = is_tool_ready(
            "forecasting",
            state,
        )

    assert ready is True


# ============================================================
# TEST 9
# is_tool_ready helper -> False
# ============================================================

def test_is_tool_ready_returns_false():

    fake_definition = {
        "dependencies": [
            "sql",
        ],
    }

    state = {
        "executed_tools": [],
        "tool_results": {},
    }

    with patch(
        "backend.orchestration.dependency_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        ready = is_tool_ready(
            "forecasting",
            state,
        )

    assert ready is False