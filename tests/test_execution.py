from unittest.mock import (
    MagicMock,
    patch,
)

from backend.orchestration.nodes.execution import (
    tool_executor_node,
)


# ============================================================
# HELPER
# ============================================================

def ready_dependencies():
    """
    Dependency resolver response for a tool whose
    dependencies are fully satisfied.
    """

    return {
        "ready": True,
        "missing_dependencies": [],
    }


# ============================================================
# TEST 1
# Executor uses registry input metadata
# ============================================================

def test_executor_builds_inputs_from_registry():

    fake_tool = MagicMock()

    fake_tool.invoke.return_value = {
        "success": True,
        "result": [
            {
                "total": 100
            }
        ],
    }

    fake_definition = {
        "tool": fake_tool,

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
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
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

    assert (
        "sql"
        in result["executed_tools"]
    )


# ============================================================
# TEST 2
# Executor promotes multiple declared outputs
# ============================================================

def test_executor_promotes_outputs():

    fake_tool = MagicMock()

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
        "tool": fake_tool,

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
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
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

    fake_tool = MagicMock()

    fake_tool.invoke.return_value = {
        "success": True,

        "forecast": [
            1200,
            1300,
            1400,
        ],
    }

    fake_definition = {
        "tool": fake_tool,

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
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
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

    fake_tool = MagicMock()

    fake_tool.invoke.return_value = {
        "success": False,
        "error": "Forecast failed.",
    }

    fake_definition = {
        "tool": fake_tool,

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
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
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

    fake_tool = MagicMock()

    fake_tool.invoke.side_effect = (
        RuntimeError(
            "Unexpected failure"
        )
    )

    fake_definition = {
        "tool": fake_tool,

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
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
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


# ============================================================
# TEST 7
# Executor checks dependencies before execution
# ============================================================

def test_executor_checks_dependencies():

    fake_tool = MagicMock()

    fake_tool.invoke.return_value = {
        "success": True,
        "chart": {
            "type": "bar"
        },
    }

    fake_definition = {
        "tool": fake_tool,

        "inputs": {
            "sql_result":
                "sql_result",
        },

        "outputs": {
            "chart":
                "visualization",
        },

        "dependencies": [
            "sql"
        ],
    }

    state = {
        "current_step":
            "visualization",

        "sql_result": [
            {
                "city": "Delhi",
                "sales": 100
            }
        ],

        "executed_tools": [
            "sql"
        ],

        "tool_results": {
            "sql": {
                "success": True,
            }
        },

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
    ) as mock_dependencies:

        result = tool_executor_node(
            state
        )

    mock_dependencies.assert_called_once_with(
        "visualization",
        state,
    )

    fake_tool.invoke.assert_called_once()

    assert (
        "visualization"
        in result["executed_tools"]
    )

    assert (
        result["visualization"]
        ==
        {
            "type": "bar"
        }
    )


# ============================================================
# TEST 8
# Missing dependency blocks execution
# ============================================================

def test_executor_blocks_missing_dependency():

    fake_tool = MagicMock()

    fake_definition = {
        "tool": fake_tool,

        "inputs": {
            "sql_result":
                "sql_result",
        },

        "outputs": {
            "chart":
                "visualization",
        },

        "dependencies": [
            "sql"
        ],
    }

    state = {
        "current_step":
            "visualization",

        "executed_tools": [],

        "tool_results": {},

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value={
            "ready": False,
            "missing_dependencies": [
                "sql"
            ],
        },
    ):

        result = tool_executor_node(
            state
        )

    # Most important assertion:
    # blocked tools must never execute.
    fake_tool.invoke.assert_not_called()

    assert (
        result["failed_tool"]
        == "visualization"
    )

    assert (
        result["tool_results"]
        ["visualization"]
        ["success"]
        is False
    )

    assert (
        result["tool_results"]
        ["visualization"]
        ["missing_dependencies"]
        ==
        ["sql"]
    )

    assert (
        "visualization"
        not in result["executed_tools"]
    )

    assert (
        result["trace"][-1]["status"]
        == "blocked"
    )

    assert (
        result["trace"][-1]
        ["missing_dependencies"]
        ==
        ["sql"]
    )


# ============================================================
# TEST 9
# Failed dependency blocks execution
# ============================================================

def test_executor_blocks_failed_dependency():

    fake_tool = MagicMock()

    fake_definition = {
        "tool": fake_tool,

        "inputs": {
            "sql_result":
                "sql_result",
        },

        "outputs": {
            "chart":
                "visualization",
        },

        "dependencies": [
            "sql"
        ],
    }

    state = {
        "current_step":
            "visualization",

        "executed_tools": [],

        "tool_results": {
            "sql": {
                "success": False,
                "error": "SQL failed.",
            }
        },

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value={
            "ready": False,
            "missing_dependencies": [
                "sql"
            ],
        },
    ):

        result = tool_executor_node(
            state
        )

    fake_tool.invoke.assert_not_called()

    assert (
        result["failed_tool"]
        == "visualization"
    )

    assert (
        result["tool_results"]
        ["visualization"]
        ["success"]
        is False
    )

    assert (
        result["tool_results"]
        ["visualization"]
        ["missing_dependencies"]
        ==
        ["sql"]
    )

    assert (
        result["trace"][-1]["status"]
        == "blocked"
    )


# ============================================================
# TEST 10
# Successful dependencies allow execution
# ============================================================

def test_executor_runs_when_dependencies_satisfied():

    fake_tool = MagicMock()

    fake_tool.invoke.return_value = {
        "success": True,

        "forecast": [
            120,
            140,
            160,
        ],
    }

    fake_definition = {
        "tool": fake_tool,

        "inputs": {
            "historical_data":
                "sql_result",
        },

        "outputs": {
            "forecast":
                "forecast_result",
        },

        "dependencies": [
            "sql"
        ],
    }

    state = {
        "current_step":
            "forecasting",

        "sql_result": [
            80,
            90,
            100,
        ],

        "executed_tools": [
            "sql"
        ],

        "tool_results": {
            "sql": {
                "success": True,
            }
        },

        "trace": [],
    }

    with patch(
        "backend.orchestration.nodes.execution."
        "get_tool_definition",
        return_value=fake_definition,
    ), patch(
        "backend.orchestration.nodes.execution."
        "check_tool_dependencies",
        return_value=ready_dependencies(),
    ):

        result = tool_executor_node(
            state
        )

    fake_tool.invoke.assert_called_once_with({
        "historical_data": [
            80,
            90,
            100,
        ]
    })

    assert (
        result["forecast_result"]
        ==
        [
            120,
            140,
            160,
        ]
    )

    assert (
        "forecasting"
        in result["executed_tools"]
    )

    assert result["failed_tool"] is None

    assert result["error"] is None