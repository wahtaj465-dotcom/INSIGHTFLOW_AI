from unittest.mock import (
    MagicMock,
    patch,
)

from backend.orchestration.nodes.execution import (
    tool_executor_node,
)


# ============================================================
# TEST 1
# Executor uses registry input metadata
# ============================================================

def test_executor_builds_inputs_from_registry():

    fake_tool = (
        MagicMock()
    )

    fake_tool.invoke.return_value = {
        "success": True,
        "result": [
            {
                "total": 100
            }
        ],
    }

    fake_definition = {

        "tool":
            fake_tool,

        "inputs": {
            "dataset_id":
                "dataset_id",

            "question":
                "question",
        },

        "outputs": {
            "result":
                "sql_result",
        },
    }

    state = {

        "dataset_id":
            "dataset_123",

        "question":
            "Calculate total sales.",

        "current_step":
            "sql",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = tool_executor_node(
            state
        )

    fake_tool.invoke.assert_called_once_with({
        "dataset_id":
            "dataset_123",

        "question":
            "Calculate total sales.",
    })

    assert result["sql_result"] == [
        {
            "total": 100
        }
    ]

    assert "sql" in (
        result["executed_tools"]
    )


# ============================================================
# TEST 2
# Executor promotes multiple declared outputs
# ============================================================

def test_executor_promotes_outputs():

    fake_tool = (
        MagicMock()
    )

    fake_tool.invoke.return_value = {
        "success": True,

        "generated_sql":
            "SELECT AVG(sales) FROM dataset",

        "result": [
            {
                "average_sales": 500
            }
        ],
    }

    fake_definition = {

        "tool":
            fake_tool,

        "inputs": {
            "dataset_id":
                "dataset_id",

            "question":
                "question",
        },

        "outputs": {
            "generated_sql":
                "generated_sql",

            "result":
                "sql_result",
        },
    }

    state = {

        "dataset_id":
            "dataset_123",

        "question":
            "What is average sales?",

        "current_step":
            "sql",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = tool_executor_node(
            state
        )

    assert (
        result["generated_sql"]
        ==
        "SELECT AVG(sales) FROM dataset"
    )

    assert result["sql_result"] == [
        {
            "average_sales": 500
        }
    ]


# ============================================================
# TEST 3
# Future tool requires no executor branch
# ============================================================

def test_executor_supports_future_tool():

    fake_tool = (
        MagicMock()
    )

    fake_tool.invoke.return_value = {
        "success": True,

        "forecast": [
            1200,
            1300,
            1400,
        ],
    }

    fake_definition = {

        "tool":
            fake_tool,

        "inputs": {
            "dataset_id":
                "dataset_id",

            "historical_data":
                "sql_result",
        },

        "outputs": {
            "forecast":
                "forecast_result",
        },
    }

    state = {

        "dataset_id":
            "dataset_123",

        "sql_result": [
            900,
            1000,
            1100,
        ],

        "current_step":
            "forecasting",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = tool_executor_node(
            state
        )

    fake_tool.invoke.assert_called_once_with({
        "dataset_id":
            "dataset_123",

        "historical_data": [
            900,
            1000,
            1100,
        ],
    })

    assert result["forecast_result"] == [
        1200,
        1300,
        1400,
    ]

    assert (
        "forecasting"
        in result["executed_tools"]
    )


# ============================================================
# TEST 4
# Failed tool is recorded for observer/replanner
# ============================================================

def test_executor_records_tool_failure():

    fake_tool = (
        MagicMock()
    )

    fake_tool.invoke.return_value = {
        "success": False,
        "error": "Forecast failed.",
    }

    fake_definition = {

        "tool":
            fake_tool,

        "inputs": {
            "dataset_id":
                "dataset_id",
        },

        "outputs": {
            "forecast":
                "forecast_result",
        },
    }

    state = {

        "dataset_id":
            "dataset_123",

        "current_step":
            "forecasting",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = tool_executor_node(
            state
        )

    assert (
        result["failed_tool"]
        == "forecasting"
    )

    assert (
        result["last_tool_error"]
        == "Forecast failed."
    )

    assert (
        "forecasting"
        not in result["executed_tools"]
    )

    assert (
        result["tool_results"]
        ["forecasting"]
        ["success"]
        is False
    )


# ============================================================
# TEST 5
# Unknown tool is handled safely
# ============================================================

def test_executor_handles_unknown_tool():

    state = {

        "current_step":
            "unknown_tool",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=None,
    ):

        result = tool_executor_node(
            state
        )

    assert (
        result["failed_tool"]
        == "unknown_tool"
    )

    assert (
        result["last_tool_error"]
        == "Unknown tool: unknown_tool"
    )

    assert (
        result["error"]
        == "Unknown tool: unknown_tool"
    )

    assert (
        result["trace"][-1]["status"]
        == "error"
    )


# ============================================================
# TEST 6
# Executor handles tool exceptions
# ============================================================

def test_executor_handles_tool_exception():

    fake_tool = (
        MagicMock()
    )

    fake_tool.invoke.side_effect = (
        RuntimeError(
            "Unexpected failure"
        )
    )

    fake_definition = {

        "tool":
            fake_tool,

        "inputs": {
            "dataset_id":
                "dataset_id",
        },

        "outputs": {},
    }

    state = {

        "dataset_id":
            "dataset_123",

        "current_step":
            "sql",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ):

        result = tool_executor_node(
            state
        )

    assert (
        result["failed_tool"]
        == "sql"
    )

    assert (
        result["last_tool_error"]
        == "Unexpected failure"
    )

    assert (
        result["tool_results"]
        ["sql"]
        ["success"]
        is False
    )