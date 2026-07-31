from unittest.mock import (
    patch,
)

from backend.orchestration.step_resolver import (
    StepResolution,
    resolve_next_step,
)


# ============================================================
# TEST 1
# First tool is ready
# ============================================================

def test_first_tool_is_ready():

    fake_definition = {
        "dependencies": [],
    }

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "sql",
            ],
            executed_tools=[],
            tool_results={},
        )

    assert isinstance(
        result,
        StepResolution,
    )

    assert (
        result.status
        == "ready"
    )

    assert (
        result.next_step
        == "sql"
    )


# ============================================================
# TEST 2
# Successful tool is skipped
# ============================================================

def test_successful_tool_is_skipped():

    definitions = {

        "sql": {
            "dependencies": [],
        },

        "visualization": {
            "dependencies": [
                "sql"
            ],
        },
    }

    def fake_definition(
        tool_name
    ):
        return definitions.get(
            tool_name
        )

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        side_effect=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "sql",
                "visualization",
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
        result.status
        == "ready"
    )

    assert (
        result.next_step
        == "visualization"
    )


# ============================================================
# TEST 3
# Dependency satisfied
# ============================================================

def test_dependency_satisfied():

    fake_definition = {
        "dependencies": [
            "sql"
        ],
    }

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "visualization"
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
        result.status
        == "ready"
    )

    assert (
        result.next_step
        == "visualization"
    )


# ============================================================
# TEST 4
# Missing dependency blocks tool
# ============================================================

def test_missing_dependency_blocks_tool():

    fake_definition = {
        "dependencies": [
            "sql"
        ],
    }

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "visualization"
            ],

            executed_tools=[],

            tool_results={},
        )

    assert (
        result.status
        == "blocked"
    )

    assert (
        result.next_step
        is None
    )

    assert (
        result.blocked_tools
        ==
        [
            "visualization"
        ]
    )

    assert (
        result.missing_dependencies
        ==
        {
            "visualization": [
                "sql"
            ]
        }
    )


# ============================================================
# TEST 5
# Failed dependency blocks downstream tool
# ============================================================

def test_failed_dependency_blocks_tool():

    fake_definition = {
        "dependencies": [
            "sql"
        ],
    }

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "visualization"
            ],

            executed_tools=[
                "sql"
            ],

            tool_results={
                "sql": {
                    "success": False,
                    "error": "SQL failed.",
                }
            },
        )

    assert (
        result.status
        == "blocked"
    )

    assert (
        result.next_step
        is None
    )

    assert (
        result.missing_dependencies
        ==
        {
            "visualization": [
                "sql"
            ]
        }
    )


# ============================================================
# TEST 6
# Resolver finds first ready tool
# ============================================================

def test_resolver_finds_first_ready_tool():

    definitions = {

        "forecasting": {
            "dependencies": [
                "sql"
            ],
        },

        "dataset_context": {
            "dependencies": [],
        },
    }

    def fake_definition(
        tool_name
    ):
        return definitions.get(
            tool_name
        )

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        side_effect=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "forecasting",
                "dataset_context",
            ],

            executed_tools=[],

            tool_results={},
        )

    assert (
        result.status
        == "ready"
    )

    assert (
        result.next_step
        == "dataset_context"
    )


# ============================================================
# TEST 7
# All tools completed
# ============================================================

def test_all_tools_completed():

    result = resolve_next_step(
        plan=[
            "sql",
            "insight",
        ],

        executed_tools=[
            "sql",
            "insight",
        ],

        tool_results={
            "sql": {
                "success": True
            },

            "insight": {
                "success": True
            },
        },
    )

    assert (
        result.status
        == "complete"
    )

    assert (
        result.next_step
        is None
    )


# ============================================================
# TEST 8
# Empty plan is complete
# ============================================================

def test_empty_plan_is_complete():

    result = resolve_next_step(
        plan=[],
        executed_tools=[],
        tool_results={},
    )

    assert (
        result.status
        == "complete"
    )

    assert (
        result.next_step
        is None
    )


# ============================================================
# TEST 9
# Unknown tool produces invalid result
# ============================================================

def test_unknown_tool_is_invalid():

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=None,
    ):

        result = resolve_next_step(
            plan=[
                "magic_tool"
            ],

            executed_tools=[],

            tool_results={},
        )

    assert (
        result.status
        == "invalid"
    )

    assert (
        result.next_step
        is None
    )

    assert (
        result.blocked_tools
        ==
        [
            "magic_tool"
        ]
    )

    assert (
        "Unknown tool"
        in result.reason
    )


# ============================================================
# TEST 10
# Multiple blocked tools are reported
# ============================================================

def test_multiple_blocked_tools():

    definitions = {

        "forecasting": {
            "dependencies": [
                "sql"
            ],
        },

        "visualization": {
            "dependencies": [
                "sql"
            ],
        },
    }

    def fake_definition(
        tool_name
    ):
        return definitions.get(
            tool_name
        )

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        side_effect=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "forecasting",
                "visualization",
            ],

            executed_tools=[],

            tool_results={},
        )

    assert (
        result.status
        == "blocked"
    )

    assert (
        result.blocked_tools
        ==
        [
            "forecasting",
            "visualization",
        ]
    )

    assert (
        result.missing_dependencies
        ==
        {
            "forecasting": [
                "sql"
            ],

            "visualization": [
                "sql"
            ],
        }
    )


# ============================================================
# TEST 11
# Future tool works without resolver modification
# ============================================================

def test_future_tool_supported_dynamically():

    fake_definition = {

        "description":
            "Forecast future values.",

        "dependencies": [
            "sql"
        ],
    }

    with patch(
        "backend.orchestration.step_resolver."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = resolve_next_step(
            plan=[
                "forecasting"
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
        result.status
        == "ready"
    )

    assert (
        result.next_step
        == "forecasting"
    )