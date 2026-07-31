from unittest.mock import patch

from backend.orchestration.replanner import (
    replanner_node,
    _build_replanner_tool_catalog,

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

# ============================================================
# TEST 4
# Replanner tool catalog comes from registry
# ============================================================

def test_build_replanner_tool_catalog():

    fake_tools = {

        "forecasting": {

            "description":
                "Forecast future values from historical data.",

            "inputs": {
                "dataset_id": "dataset_id",
                "question": "question",
            },

            "outputs": {
                "forecast": "forecast_result",
            },

            "dependencies": [],
        }
    }

    with patch(
        "backend.orchestration.replanner.get_tool_descriptions",
        return_value=fake_tools,
    ):

        catalog = (
            _build_replanner_tool_catalog()
        )

    assert "forecasting" in catalog

    assert (
        "Forecast future values from historical data."
        in catalog
    )

    assert "dataset_id" in catalog

    assert "forecast_result" in catalog


# ============================================================
# TEST 5
# Replanner can discover a newly registered tool
# ============================================================

def test_replanner_uses_dynamic_tool_catalog():

    fake_tools = {

        "forecasting": {

            "description":
                "Forecast future values from historical data.",

            "inputs": {
                "dataset_id": "dataset_id",
                "question": "question",
            },

            "outputs": {
                "forecast": "forecast_result",
            },

            "dependencies": [],
        }
    }

    fake_llm_response = """
    {
        "intent": "recovery",
        "tools": ["forecasting"],
        "reasoning": "Use forecasting to recover and complete the request."
    }
    """

    state = {

        "dataset_id":
            "test_dataset",

        "question":
            "Forecast next month's sales.",

        "plan": [
            "sql"
        ],

        "executed_tools": [],

        "current_step":
            "sql",

        "failed_tool":
            "sql",

        "last_tool_error":
            "SQL analysis failed.",

        "replan_count":
            0,

        "max_replans":
            2,

        "trace": [],
    }

    with patch(
        "backend.orchestration.replanner.get_tool_descriptions",
        return_value=fake_tools,
    ), patch(
        "backend.orchestration.replanner.get_available_tools",
        return_value=["forecasting"],
    ), patch(
        "backend.orchestration.replanner.LLMService.generate",
        return_value=fake_llm_response,
    ) as mock_generate:

        result = replanner_node(
            state
        )

    prompt = (
        mock_generate.call_args.args[0]
    )

    # The registry metadata must appear in the LLM prompt.
    assert "forecasting" in prompt

    assert (
        "Forecast future values from historical data."
        in prompt
    )

    # The LLM-selected registered tool must become
    # the new recovery plan.
    assert result["plan"] == [
        "forecasting"
    ]

    assert (
        result["current_step"]
        == "forecasting"
    )

    assert (
        result["planner_source"]
        == "llm"
    )

    assert (
        result["replan_count"]
        == 1
    )

    assert (
        result["trace"][-1]["node"]
        == "replanner"
    )

    assert (
        result["trace"][-1]["revised_plan"]
        == ["forecasting"]
    )