from unittest.mock import patch

import pytest

from backend.orchestration.plan_resolver import (
    resolve_plan_dependencies,
)


# ============================================================
# TEST 1
# Tool with no dependencies remains unchanged
# ============================================================

def test_tool_without_dependencies():

    definitions = {
        "sql": {
            "dependencies": [],
        }
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies(
            ["sql"]
        )

    assert result == [
        "sql"
    ]


# ============================================================
# TEST 2
# Direct dependency is inserted first
# ============================================================

def test_direct_dependency_added():

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

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies(
            ["visualization"]
        )

    assert result == [
        "sql",
        "visualization",
    ]


# ============================================================
# TEST 3
# Recursive dependencies are resolved
# ============================================================

def test_recursive_dependencies():

    definitions = {
        "sql": {
            "dependencies": [],
        },

        "forecasting": {
            "dependencies": [
                "sql"
            ],
        },

        "report": {
            "dependencies": [
                "forecasting"
            ],
        },
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies(
            ["report"]
        )

    assert result == [
        "sql",
        "forecasting",
        "report",
    ]


# ============================================================
# TEST 4
# Shared dependencies are not duplicated
# ============================================================

def test_shared_dependency_not_duplicated():

    definitions = {
        "sql": {
            "dependencies": [],
        },

        "visualization": {
            "dependencies": [
                "sql"
            ],
        },

        "insight": {
            "dependencies": [
                "sql"
            ],
        },
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies([
            "visualization",
            "insight",
        ])

    assert result == [
        "sql",
        "visualization",
        "insight",
    ]


# ============================================================
# TEST 5
# Existing correct plan remains correct
# ============================================================

def test_existing_dependency_order_preserved():

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

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies([
            "sql",
            "visualization",
        ])

    assert result == [
        "sql",
        "visualization",
    ]


# ============================================================
# TEST 6
# Unknown tool raises error
# ============================================================

def test_unknown_tool_raises_error():

    definitions = {
        "sql": {
            "dependencies": [],
        }
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        with pytest.raises(
            ValueError,
            match="Unknown tool",
        ):

            resolve_plan_dependencies([
                "forecasting"
            ])


# ============================================================
# TEST 7
# Circular dependency is detected
# ============================================================

def test_circular_dependency_detected():

    definitions = {
        "tool_a": {
            "dependencies": [
                "tool_b"
            ],
        },

        "tool_b": {
            "dependencies": [
                "tool_a"
            ],
        },
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        with pytest.raises(
            ValueError,
            match="Circular tool dependency",
        ):

            resolve_plan_dependencies([
                "tool_a"
            ])


# ============================================================
# TEST 8
# Complex dependency graph resolves correctly
# ============================================================

def test_complex_dependency_graph():

    definitions = {
        "sql": {
            "dependencies": [],
        },

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

        "report": {
            "dependencies": [
                "forecasting",
                "visualization",
            ],
        },
    }

    with patch(
        "backend.orchestration.plan_resolver."
        "get_tool_definition",
        side_effect=lambda name: definitions.get(name),
    ):

        result = resolve_plan_dependencies(
            ["report"]
        )

    assert result == [
        "sql",
        "forecasting",
        "visualization",
        "report",
    ]