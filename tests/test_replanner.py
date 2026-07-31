from unittest.mock import patch

from backend.orchestration.replanner import (
    replanner_node,
)


# ============================================================
# TEST 1
# LLM creates recovery plan after SQL failure
# ============================================================

def test_replanner_creates_recovery_plan():

    state = {
        "question": (
            "Show average revenue by region "
            "and visualize the result."
        ),

        "plan": [
            "sql",
            "visualization",
            "insight",
        ],

        "executed_tools": [],

        "failed_tool": "sql",

        "last_tool_error": (
            "Invalid SQL query."
        ),

        "replan_count": 0,

        "max_replans": 2,

        "trace": [],
    }

    fake_response = """
    {
        "intent": "recovery",
        "tools": [
            "sql",
            "visualization",
            "insight"
        ],
        "reasoning": "Retry SQL and continue the analysis."
    }
    """

    with patch(
        "backend.orchestration.replanner.LLMService.generate",
        return_value=fake_response,
    ):

        result = replanner_node(
            state
        )

    assert result["plan"] == [
        "sql",
        "visualization",
        "insight",
    ]

    assert (
        result["current_step"]
        == "sql"
    )

    assert (
        result["replan_count"]
        == 1
    )

    assert (
        result["planner_source"]
        == "llm"
    )

    assert (
        result["trace"][-1]["node"]
        == "replanner"
    )


# ============================================================
# TEST 2
# Successfully executed tools should not run again
# ============================================================

def test_replanner_excludes_completed_tools():

    state = {
        "question": (
            "Analyze revenue and visualize it."
        ),

        "plan": [
            "sql",
            "visualization",
            "insight",
        ],

        "executed_tools": [
            "sql"
        ],

        "failed_tool": (
            "visualization"
        ),

        "last_tool_error": (
            "Unable to generate chart."
        ),

        "replan_count": 0,

        "max_replans": 2,

        "trace": [],
    }

    fake_response = """
    {
        "intent": "recovery",
        "tools": [
            "sql",
            "visualization",
            "insight"
        ],
        "reasoning": "Retry visualization and continue."
    }
    """

    with patch(
        "backend.orchestration.replanner.LLMService.generate",
        return_value=fake_response,
    ):

        result = replanner_node(
            state
        )

    assert "sql" not in result["plan"]

    assert result["plan"] == [
        "visualization",
        "insight",
    ]

    assert (
        result["current_step"]
        == "visualization"
    )


# ============================================================
# TEST 3
# LLM failure should use fallback recovery
# ============================================================

def test_replanner_falls_back_when_llm_fails():

    state = {
        "question": (
            "Create a scatter plot of age "
            "vs score and explain it."
        ),

        "plan": [
            "sql",
            "visualization",
            "insight",
        ],

        "executed_tools": [],

        "failed_tool": "sql",

        "last_tool_error": (
            "SQL execution failed."
        ),

        "replan_count": 0,

        "max_replans": 2,

        "trace": [],
    }

    with patch(
        "backend.orchestration.replanner.LLMService.generate",
        side_effect=RuntimeError(
            "LLM unavailable"
        ),
    ):

        result = replanner_node(
            state
        )

    assert (
        result["planner_source"]
        == "fallback"
    )

    assert (
        result["replan_count"]
        == 1
    )

    assert result["plan"]

    assert (
        result["current_step"]
        ==
        result["plan"][0]
    )

    assert (
        result["trace"][-1]["node"]
        == "replanner"
    )