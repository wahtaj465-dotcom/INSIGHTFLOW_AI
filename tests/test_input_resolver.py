from backend.orchestration.input_resolver import (
    _resolve_state_value,
    build_tool_input,
)


# ============================================================
# TEST 1
# Resolve a simple value from AgentState
# ============================================================

def test_resolve_state_value():

    state = {
        "dataset_id": "dataset_123",
        "question": "What is the average sales?",
    }

    result = _resolve_state_value(
        "dataset_id",
        state,
    )

    assert result == "dataset_123"


# ============================================================
# TEST 2
# Build SQL-style tool input
# ============================================================

def test_build_sql_tool_input():

    state = {
        "dataset_id": "dataset_123",
        "question": "What is the average sales?",
    }

    input_mapping = {
        "dataset_id": "dataset_id",
        "question": "question",
    }

    result = build_tool_input(
        input_mapping,
        state,
    )

    assert result == {
        "dataset_id": "dataset_123",
        "question": "What is the average sales?",
    }


# ============================================================
# TEST 3
# Build visualization-style tool input
# ============================================================

def test_build_visualization_tool_input():

    state = {
        "dataset_id": "dataset_123",

        "question":
            "Create a bar chart of sales by region.",

        "sql_result": [
            {
                "region": "North",
                "sales": 1000,
            },
            {
                "region": "South",
                "sales": 1500,
            },
        ],
    }

    input_mapping = {
        "dataset_id": "dataset_id",
        "question": "question",
        "sql_result": "sql_result",
    }

    result = build_tool_input(
        input_mapping,
        state,
    )

    assert result == {
        "dataset_id": "dataset_123",

        "question":
            "Create a bar chart of sales by region.",

        "sql_result": [
            {
                "region": "North",
                "sales": 1000,
            },
            {
                "region": "South",
                "sales": 1500,
            },
        ],
    }


# ============================================================
# TEST 4
# Missing state value should resolve to None
# ============================================================

def test_missing_state_value_returns_none():

    state = {
        "dataset_id": "dataset_123",
    }

    result = _resolve_state_value(
        "sql_result",
        state,
    )

    assert result is None


# ============================================================
# TEST 5
# Missing optional tool input should become None
# ============================================================

def test_build_tool_input_with_missing_value():

    state = {
        "dataset_id": "dataset_123",
        "question": "Explain the results.",
    }

    input_mapping = {
        "dataset_id": "dataset_id",
        "question": "question",
        "sql_result": "sql_result",
        "generated_sql": "generated_sql",
    }

    result = build_tool_input(
        input_mapping,
        state,
    )

    assert result == {
        "dataset_id": "dataset_123",
        "question": "Explain the results.",
        "sql_result": None,
        "generated_sql": None,
    }


# ============================================================
# TEST 6
# Future tools work without resolver changes
# ============================================================

def test_build_future_forecasting_tool_input():

    state = {
        "dataset_id": "dataset_123",

        "question":
            "Forecast sales for next month.",

        "sql_result": [
            {
                "month": "January",
                "sales": 1000,
            },
            {
                "month": "February",
                "sales": 1200,
            },
        ],
    }

    input_mapping = {
        "dataset_id": "dataset_id",
        "question": "question",
        "historical_data": "sql_result",
    }

    result = build_tool_input(
        input_mapping,
        state,
    )

    assert result == {
        "dataset_id": "dataset_123",

        "question":
            "Forecast sales for next month.",

        "historical_data": [
            {
                "month": "January",
                "sales": 1000,
            },
            {
                "month": "February",
                "sales": 1200,
            },
        ],
    }