from unittest.mock import patch

from backend.orchestration.planner import (
    AnalyticsPlan,
    _parse_plan,
    _fallback_plan,
    _build_tool_catalog,
    planner_node,
)


def test_parse_structured_plan():

    response = """
    {
        "intent": "relationship_analysis",
        "tools": [
            "sql",
            "visualization",
            "insight"
        ],
        "reasoning": "Analyze the relationship, plot it, and explain it."
    }
    """

    plan = _parse_plan(response)

    assert isinstance(plan, AnalyticsPlan)
    assert plan.intent == "relationship_analysis"

    assert plan.tools == [
        "sql",
        "visualization",
        "insight",
    ]

    assert (
        plan.reasoning
        == "Analyze the relationship, plot it, and explain it."
    )


def test_parse_visualization_plan():

    response = """
    {
        "intent": "visualization",
        "tools": [
            "sql",
            "visualization"
        ],
        "reasoning": "Query data and visualize it."
    }
    """

    plan = _parse_plan(response)

    assert isinstance(plan, AnalyticsPlan)
    assert plan.intent == "visualization"

    assert plan.tools == [
        "sql",
        "visualization",
    ]

    assert (
        plan.reasoning
        == "Query data and visualize it."
    )


def test_fallback_visualization_plan():

    plan = _fallback_plan(
        "Create a scatter plot of age vs score "
        "and explain the relationship"
    )

    assert isinstance(plan, AnalyticsPlan)
    assert plan.intent == "fallback_analysis"

    assert plan.tools == [
        "sql",
        "visualization",
        "insight",
    ]


def test_fallback_dataset_context():

    plan = _fallback_plan(
        "What columns are in this dataset?"
    )

    assert isinstance(plan, AnalyticsPlan)
    assert plan.intent == "fallback_analysis"

    assert plan.tools == [
        "dataset_context"
    ]

def test_planner_node_with_mocked_llm():

    fake_llm_response = """
    {
        "intent": "relationship_analysis",
        "tools": [
            "sql",
            "visualization",
            "insight"
        ],
        "reasoning": "Analyze the relationship, visualize it, and explain the findings."
    }
    """

    state = {
        "dataset_id": "test_dataset",

        "question": (
            "Create a scatter plot of age vs score "
            "colored by active and explain the relationship."
        ),

        "trace": [],
    }

    with patch(
        "backend.orchestration.planner.LLMService.generate",
        return_value=fake_llm_response,
    ) as mock_generate:

        result = planner_node(
            state
        )

    assert result["intent"] == (
        "relationship_analysis"
    )

    assert result["plan"] == [
        "sql",
        "visualization",
        "insight",
    ]

    assert result["planner_source"] == "llm"

    assert result["planner_error"] is None

    assert result["current_step"] == "sql"

    assert (
        result["plan_reasoning"]
        ==
        "Analyze the relationship, visualize it, and explain the findings."
    )

    assert len(
        result["trace"]
    ) == 1

    assert (
        result["trace"][0]["node"]
        ==
        "planner"
    )

    assert (
        result["trace"][0]["source"]
        ==
        "llm"
    )

    mock_generate.assert_called_once()

def test_planner_node_falls_back_when_llm_fails():

    state = {
        "dataset_id": "test_dataset",

        "question": (
            "Create a scatter plot of age vs score "
            "and explain the relationship."
        ),

        "trace": [],
    }

    with patch(
        "backend.orchestration.planner.LLMService.generate",
        side_effect=RuntimeError(
            "LLM_QUOTA_EXCEEDED: Gemini quota exhausted."
        ),
    ) as mock_generate:

        result = planner_node(
            state
        )

    # --------------------------------------------------------
    # Planner should recover using deterministic fallback
    # --------------------------------------------------------

    assert result["planner_source"] == "fallback"

    assert result["intent"] == "fallback_analysis"

    assert result["plan"] == [
        "sql",
        "visualization",
        "insight",
    ]

    # --------------------------------------------------------
    # First tool should be ready for execution
    # --------------------------------------------------------

    assert result["current_step"] == "sql"

    # --------------------------------------------------------
    # Original LLM error should remain observable
    # --------------------------------------------------------

    assert result["planner_error"] is not None

    assert (
        "LLM_QUOTA_EXCEEDED"
        in result["planner_error"]
    )

    # --------------------------------------------------------
    # Fallback itself should have reasoning
    # --------------------------------------------------------

    assert result["plan_reasoning"] is not None

    # --------------------------------------------------------
    # Planner trace should record fallback usage
    # --------------------------------------------------------

    assert len(result["trace"]) == 1

    assert (
        result["trace"][0]["node"]
        ==
        "planner"
    )

    assert (
        result["trace"][0]["source"]
        ==
        "fallback"
    )


# ============================================================
# TEST DYNAMIC TOOL CATALOG
# ============================================================

def test_build_tool_catalog():

    catalog = (
        _build_tool_catalog()
    )

    assert isinstance(
        catalog,
        str
    )

    assert "dataset_context" in catalog
    assert "sql" in catalog
    assert "visualization" in catalog
    assert "insight" in catalog

    assert "description" in catalog
    assert "inputs" in catalog
    assert "outputs" in catalog
    assert "dependencies" in catalog

# ============================================================
# TEST PLANNER USES REGISTRY TOOL CATALOG
# ============================================================

def test_planner_uses_dynamic_tool_catalog():

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
        "intent": "forecasting",
        "tools": ["forecasting"],
        "reasoning": "Forecast future values."
    }
    """

    with patch(
        "backend.orchestration.planner.get_tool_descriptions",
        return_value=fake_tools,
    ), patch(
        "backend.orchestration.planner.get_available_tools",
        return_value=["forecasting"],
    ), patch(
        "backend.orchestration.planner.LLMService.generate",
        return_value=fake_llm_response,
    ) as mock_generate:

        result = planner_node({
            "dataset_id": "test_dataset",
            "question": "Forecast next month's sales.",
            "trace": [],
        })

    prompt = (
        mock_generate.call_args.args[0]
    )

    assert "forecasting" in prompt

    assert (
        "Forecast future values from historical data."
        in prompt
    )

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
        result["planner_error"]
        is None
    )


    # --------------------------------------------------------
    # Verify the LLM was attempted exactly once
    # --------------------------------------------------------

    mock_generate.assert_called_once()