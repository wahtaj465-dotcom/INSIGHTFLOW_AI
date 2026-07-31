from unittest.mock import (
    MagicMock,
    patch,
)

from backend.orchestration.graph import (
    agent_graph,
)


# ============================================================
# HELPERS
# ============================================================

def build_initial_state(
    question: str,
    dataset_id: str = "test_dataset",
):
    """
    Build a standard initial state for end-to-end
    orchestration tests.
    """

    return {
        "dataset_id":
            dataset_id,

        "question":
            question,

        "intent":
            None,

        "plan": [],

        "plan_reasoning":
            None,

        "planner_source":
            None,

        "planner_error":
            None,

        "current_step":
            None,

        "executed_tools": [],

        "tool_results": {},

        # Generic promoted outputs for dynamically
        # registered tools.
        "tool_outputs": {},

        "generated_sql":
            None,

        "sql_result":
            None,

        "visualization":
            None,

        "insight":
            None,

        "failed_tool":
            None,

        "last_tool_error":
            None,

        "error":
            None,

        "completed":
            False,

        "replan_count":
            0,

        "max_replans":
            2,

        "retry_count":
            0,

        "trace": [],
    }


def build_fake_tool(
    result: dict,
):
    """
    Create a fake LangChain-style tool exposing
    .invoke().
    """

    tool = MagicMock()

    tool.invoke.return_value = result

    return tool


# ============================================================
# TEST 1
# SIMPLE SQL ANALYSIS
# ============================================================

def test_e2e_simple_sql_analysis():
    """
    User asks an analytical question.

    Expected workflow:

        planner
            ↓
        sql
            ↓
        observer
            ↓
        finish
    """

    planner_response = """
    {
        "intent": "aggregation",
        "tools": ["sql"],
        "reasoning": "SQL is required to calculate total sales."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT SUM(sales) AS total_sales FROM dataset",

        "result": [
            {
                "total_sales": 5000
            }
        ],
    })

    sql_definition = {
        "tool":
            sql_tool,

        "description":
            "Run analytical SQL queries.",

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

        "dependencies": [],
    }

    state = build_initial_state(
        "What is the total sales?"
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        {
            "sql":
                sql_definition,
        },
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert "sql" in (
        result["executed_tools"]
    )

    assert result["generated_sql"] == (
        "SELECT SUM(sales) AS total_sales FROM dataset"
    )

    assert result["sql_result"] == [
        {
            "total_sales": 5000
        }
    ]

    assert (
        result["failed_tool"]
        is None
    )


# ============================================================
# TEST 2
# SQL + INSIGHT
# ============================================================

def test_e2e_sql_and_insight():
    """
    Expected workflow:

        planner
            ↓
        sql
            ↓
        insight
            ↓
        finish
    """

    planner_response = """
    {
        "intent": "general_analysis",
        "tools": ["sql", "insight"],
        "reasoning": "Query the data and interpret the result."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT AVG(sales) AS average_sales FROM dataset",

        "result": [
            {
                "average_sales": 500
            }
        ],
    })

    insight_tool = build_fake_tool({
        "success": True,

        "insight":
            "Average sales are 500 units.",
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical queries.",

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

            "dependencies": [],
        },

        "insight": {
            "tool":
                insight_tool,

            "description":
                "Interpret analytical results.",

            "inputs": {
                "dataset_id":
                    "dataset_id",

                "question":
                    "question",

                "sql_result":
                    "sql_result",

                "generated_sql":
                    "generated_sql",
            },

            "outputs": {
                "insight":
                    "insight",
            },

            "dependencies": [
                "sql"
            ],
        },
    }

    state = build_initial_state(
        "Analyze average sales and explain the result."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert result["executed_tools"] == [
        "sql",
        "insight",
    ]

    assert (
        result["sql_result"][0]["average_sales"]
        ==
        500
    )

    assert result["insight"] == (
        "Average sales are 500 units."
    )


# ============================================================
# TEST 3
# SQL + VISUALIZATION
# ============================================================

def test_e2e_visualization_workflow():
    """
    Expected workflow:

        planner
            ↓
        sql
            ↓
        visualization
            ↓
        finish
    """

    planner_response = """
    {
        "intent": "visualization",
        "tools": ["sql", "visualization"],
        "reasoning": "Retrieve grouped sales and visualize them."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT region, SUM(sales) AS total_sales "
            "FROM dataset GROUP BY region",

        "result": [
            {
                "region": "North",
                "total_sales": 1000,
            },
            {
                "region": "South",
                "total_sales": 1500,
            },
        ],
    })

    visualization_tool = build_fake_tool({
        "success": True,

        "chart": {
            "type":
                "bar",

            "x":
                "region",

            "y":
                "total_sales",
        },
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical SQL queries.",

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

            "dependencies": [],
        },

        "visualization": {
            "tool":
                visualization_tool,

            "description":
                "Generate charts.",

            "inputs": {
                "dataset_id":
                    "dataset_id",

                "question":
                    "question",

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
        },
    }

    state = build_initial_state(
        "Create a bar chart of total sales by region."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert result["executed_tools"] == [
        "sql",
        "visualization",
    ]

    assert result["visualization"] == {
        "type":
            "bar",

        "x":
            "region",

        "y":
            "total_sales",
    }


# ============================================================
# TEST 4
# AUTOMATIC DEPENDENCY EXPANSION
# ============================================================

def test_e2e_dependency_auto_resolution():
    """
    Planner intentionally requests ONLY visualization.

    Registry says:

        visualization depends on sql

    Plan resolver must therefore produce:

        sql
        visualization
    """

    planner_response = """
    {
        "intent": "visualization",
        "tools": ["visualization"],
        "reasoning": "Create the requested visualization."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT category, SUM(sales) AS total "
            "FROM dataset GROUP BY category",

        "result": [
            {
                "category": "A",
                "total": 100,
            },
            {
                "category": "B",
                "total": 200,
            },
        ],
    })

    visualization_tool = build_fake_tool({
        "success": True,

        "chart": {
            "type": "bar"
        },
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical queries.",

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

            "dependencies": [],
        },

        "visualization": {
            "tool":
                visualization_tool,

            "description":
                "Generate charts.",

            "inputs": {
                "dataset_id":
                    "dataset_id",

                "question":
                    "question",

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
        },
    }

    state = build_initial_state(
        "Visualize sales by category."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert result["plan"] == [
        "sql",
        "visualization",
    ]

    assert result["executed_tools"] == [
        "sql",
        "visualization",
    ]

    sql_tool.invoke.assert_called_once()

    visualization_tool.invoke.assert_called_once()


# ============================================================
# TEST 5
# FAILURE → REPLAN → RECOVERY
# ============================================================

def test_e2e_failure_recovery():
    """
    First SQL execution fails.

    Observer detects failure.
    Execution policy requests replan.
    Replanner retries SQL.
    Second SQL execution succeeds.
    """

    planner_response = """
    {
        "intent": "aggregation",
        "tools": ["sql"],
        "reasoning": "Use SQL to calculate total sales."
    }
    """

    recovery_response = """
    {
        "intent": "recovery",
        "tools": ["sql"],
        "reasoning": "Retry SQL after the previous failure."
    }
    """

    sql_tool = MagicMock()

    sql_tool.invoke.side_effect = [

        {
            "success": False,
            "error": "Temporary SQL failure.",
        },

        {
            "success": True,

            "generated_sql":
                "SELECT SUM(sales) AS total_sales FROM dataset",

            "result": [
                {
                    "total_sales": 5000
                }
            ],
        },
    ]

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical SQL queries.",

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

            "dependencies": [],
        },
    }

    state = build_initial_state(
        "Calculate total sales."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.replanner."
        "LLMService.generate",
        return_value=recovery_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert result["replan_count"] == 1

    assert result["failed_tool"] is None

    assert result["last_tool_error"] is None

    assert "sql" in (
        result["executed_tools"]
    )

    assert result["sql_result"] == [
        {
            "total_sales": 5000
        }
    ]

    assert (
        sql_tool.invoke.call_count
        ==
        2
    )


# ============================================================
# TEST 6
# UNRECOVERABLE FAILURE
# ============================================================

def test_e2e_stops_after_replan_budget():
    """
    SQL continues failing.

    Agent must eventually stop instead of
    looping forever.
    """

    planner_response = """
    {
        "intent": "aggregation",
        "tools": ["sql"],
        "reasoning": "Use SQL."
    }
    """

    recovery_response = """
    {
        "intent": "recovery",
        "tools": ["sql"],
        "reasoning": "Retry SQL."
    }
    """

    sql_tool = MagicMock()

    sql_tool.invoke.return_value = {
        "success": False,
        "error": "SQL service unavailable.",
    }

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical SQL queries.",

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

            "dependencies": [],
        },
    }

    state = build_initial_state(
        "Calculate total sales."
    )

    state["max_replans"] = 2

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.replanner."
        "LLMService.generate",
        return_value=recovery_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert (
        result["replan_count"]
        ==
        2
    )

    assert (
        result["failed_tool"]
        ==
        "sql"
    )

    assert (
        result["last_tool_error"]
        ==
        "SQL service unavailable."
    )

    assert (
        "sql"
        not in result["executed_tools"]
    )


# ============================================================
# TEST 7
# FUTURE TOOL EXTENSIBILITY
# ============================================================

def test_e2e_future_forecasting_tool():
    """
    Prove that the orchestration architecture can execute
    a new tool without adding forecasting-specific branches
    to the executor or observer.

    forecasting depends on sql.

    Planner requests ONLY forecasting.

    Expected resolved plan:

        sql
          ↓
        forecasting

    The future tool output is stored through the generic
    tool_outputs state container.
    """

    planner_response = """
    {
        "intent": "forecasting",
        "tools": ["forecasting"],
        "reasoning": "Forecast future sales."
    }
    """

    historical_data = [
        {
            "month": "Jan",
            "sales": 1000,
        },
        {
            "month": "Feb",
            "sales": 1100,
        },
        {
            "month": "Mar",
            "sales": 1200,
        },
    ]

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT month, sales FROM dataset ORDER BY month",

        "result":
            historical_data,
    })

    forecasting_tool = build_fake_tool({
        "success": True,

        "forecast": [
            1300,
            1400,
            1500,
        ],
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Retrieve historical analytical data.",

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

            "dependencies": [],
        },

        "forecasting": {
            "tool":
                forecasting_tool,

            "description":
                "Forecast future values using historical data.",

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

            "dependencies": [
                "sql"
            ],
        },
    }

    state = build_initial_state(
        "Forecast sales for the next three months."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    # --------------------------------------------------------
    # Workflow completed
    # --------------------------------------------------------

    assert result["completed"] is True

    # --------------------------------------------------------
    # Dependency automatically inserted
    # --------------------------------------------------------

    assert result["plan"] == [
        "sql",
        "forecasting",
    ]

    # --------------------------------------------------------
    # Both tools executed
    # --------------------------------------------------------

    assert result["executed_tools"] == [
        "sql",
        "forecasting",
    ]

    # --------------------------------------------------------
    # Dynamic output survived graph state
    # --------------------------------------------------------

    assert result["tool_outputs"][
        "forecast_result"
    ] == [
        1300,
        1400,
        1500,
    ]

    # --------------------------------------------------------
    # Forecasting received SQL output dynamically
    # --------------------------------------------------------

    forecasting_tool.invoke.assert_called_once_with({
        "dataset_id":
            "test_dataset",

        "historical_data":
            historical_data,
    })


# ============================================================
# TEST 8
# TRACE SHOWS COMPLETE EXECUTION
# ============================================================

def test_e2e_trace_records_workflow():
    """
    Ensure the final trace provides observability
    across the agent lifecycle.
    """

    planner_response = """
    {
        "intent": "aggregation",
        "tools": ["sql"],
        "reasoning": "Calculate total sales."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT SUM(sales) FROM dataset",

        "result": [
            {
                "total": 1000
            }
        ],
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical queries.",

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

            "dependencies": [],
        },
    }

    state = build_initial_state(
        "Calculate total sales."
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    trace = result.get(
        "trace",
        []
    )

    assert trace

    node_names = [
        item.get("node")
        for item in trace
    ]

    assert "planner" in node_names

    assert "tool_executor" in node_names

    assert "observer" in node_names


# ============================================================
# TEST 9
# NO DUPLICATE EXECUTION
# ============================================================

def test_e2e_tool_executes_only_once_after_success():
    """
    A successfully completed tool must not execute
    repeatedly as the graph loops through observer.
    """

    planner_response = """
    {
        "intent": "aggregation",
        "tools": ["sql"],
        "reasoning": "Run SQL once."
    }
    """

    sql_tool = build_fake_tool({
        "success": True,

        "generated_sql":
            "SELECT COUNT(*) AS count FROM dataset",

        "result": [
            {
                "count": 100
            }
        ],
    })

    registry = {

        "sql": {
            "tool":
                sql_tool,

            "description":
                "Run analytical queries.",

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

            "dependencies": [],
        },
    }

    state = build_initial_state(
        "How many records are there?"
    )

    with patch(
        "backend.orchestration.planner."
        "LLMService.generate",
        return_value=planner_response,
    ), patch(
        "backend.orchestration.tool_registry."
        "TOOL_REGISTRY",
        registry,
    ):

        result = agent_graph.invoke(
            state
        )

    assert result["completed"] is True

    assert result["executed_tools"] == [
        "sql"
    ]

    assert (
        sql_tool.invoke.call_count
        ==
        1
    )