from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# HELPERS
# ============================================================

def build_agent_result(
    *,
    dataset_id="dataset_123",
    question="What is total sales?",
):
    """
    Build a representative successful response returned
    by AgentService.

    API tests mock AgentService because the orchestration
    graph already has its own unit and E2E test coverage.
    """

    return {
        "success": True,
        "completed": True,

        "dataset_id": dataset_id,
        "question": question,

        "intent": "aggregation",

        "plan": [
            "sql",
            "insight",
        ],

        "plan_reasoning": (
            "SQL calculates the requested value and "
            "insight explains the result."
        ),

        "planner_source": "llm",
        "planner_error": None,

        "executed_tools": [
            "sql",
            "insight",
        ],

        "tool_results": {
            "sql": {
                "success": True,
                "generated_sql": (
                    "SELECT SUM(sales) AS total_sales "
                    "FROM dataset"
                ),
                "result": [
                    {
                        "total_sales": 5000
                    }
                ],
            },

            "insight": {
                "success": True,
                "insight": (
                    "Total sales are 5000."
                ),
            },
        },

        "tool_outputs": {
            "generated_sql": (
                "SELECT SUM(sales) AS total_sales "
                "FROM dataset"
            ),

            "sql_result": [
                {
                    "total_sales": 5000
                }
            ],

            "insight": (
                "Total sales are 5000."
            ),
        },

        "generated_sql": (
            "SELECT SUM(sales) AS total_sales "
            "FROM dataset"
        ),

        "sql_result": [
            {
                "total_sales": 5000
            }
        ],

        "visualization": None,

        "statistical_findings": [],

        "insight": (
            "Total sales are 5000."
        ),

        "retry_count": 0,
        "max_retries": 2,

        "replan_count": 0,
        "max_replans": 2,

        "failed_tool": None,
        "last_tool_error": None,

        "error": None,

        "trace": [
            {
                "node": "planner",
                "status": "success",
            },
            {
                "node": "tool_executor",
                "tool": "sql",
                "status": "success",
            },
            {
                "node": "tool_executor",
                "tool": "insight",
                "status": "success",
            },
            {
                "node": "finish",
                "status": "success",
            },
        ],
    }


def build_preparation_result(
    dataset_id="dataset_123",
):
    """
    Build a representative response returned by
    AnalyticsWorkflow.prepare_dataset().
    """

    return {
        "success": True,

        "dataset_id":
            dataset_id,

        "rows":
            3,

        "columns": [
            "month",
            "sales",
        ],

        "column_count":
            2,

        "schema": {
            "columns": [
                {
                    "name": "month",
                    "dtype": "object",
                },
                {
                    "name": "sales",
                    "dtype": "int64",
                },
            ]
        },

        "quality": {
            "before_score": 90,
            "after_score": 100,
        },

        "anomalies": {},

        "cleaning": {
            "original_rows": 3,
            "cleaned_rows": 3,
            "rows_removed": 0,
            "cleaning_log": [],
        },

        "eda_results": {
            "summary": {}
        },

        "eda_charts": [],

        "chart_count": 0,

        "statistical_findings": [],

        "statistical_finding_count": 0,
    }


# ============================================================
# TEST 1
# APPLICATION ROOT
# ============================================================

def test_root_endpoint():

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        ==
        "InsightFlow AI API is running"
    )

    assert data["version"] == "1.0.0"

    assert data["docs"] == "/docs"


# ============================================================
# TEST 2
# APPLICATION HEALTH
# ============================================================

def test_application_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# ============================================================
# TEST 3
# ANALYTICS API HEALTH
# ============================================================

def test_analytics_api_health():

    with patch(
        "backend.api.routes."
        "dataset_manager.count",
        return_value=3,
    ):

        response = client.get(
            "/api/health"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert (
        data["service"]
        ==
        "InsightFlow Analytics API"
    )

    assert data["active_datasets"] == 3

    assert (
        data["agentic_layer"]
        ==
        "enabled"
    )


# ============================================================
# TEST 4
# ASK DATASET THROUGH AGENT
# ============================================================

def test_ask_dataset_uses_agent_service():

    agent_result = build_agent_result()

    with patch(
        "backend.api.routes."
        "dataset_manager.exists",
        return_value=True,
    ), patch(
        "backend.api.routes."
        "agent_service.run",
        return_value=agent_result,
    ) as mock_run:

        response = client.post(
            "/api/datasets/dataset_123/ask",
            data={
                "question":
                    "What is total sales?"
            },
        )

    assert response.status_code == 200

    mock_run.assert_called_once_with(
        dataset_id="dataset_123",
        question="What is total sales?",
    )

    data = response.json()

    assert data["success"] is True

    assert data["completed"] is True

    assert (
        data["dataset_id"]
        ==
        "dataset_123"
    )

    assert (
        data["question"]
        ==
        "What is total sales?"
    )

    assert (
        data["agent"]["intent"]
        ==
        "aggregation"
    )

    assert data["agent"]["plan"] == [
        "sql",
        "insight",
    ]

    assert (
        data["agent"]["executed_tools"]
        ==
        [
            "sql",
            "insight",
        ]
    )

    assert (
        data["analysis"]["generated_sql"]
        ==
        "SELECT SUM(sales) AS total_sales "
        "FROM dataset"
    )

    assert (
        data["analysis"]["sql_result"]
        ==
        [
            {
                "total_sales": 5000
            }
        ]
    )

    assert (
        data["insight"]
        ==
        "Total sales are 5000."
    )

    assert data["error"] is None


# ============================================================
# TEST 5
# ASK UNKNOWN DATASET
# ============================================================

def test_ask_unknown_dataset_returns_404():

    with patch(
        "backend.api.routes."
        "dataset_manager.exists",
        return_value=False,
    ), patch(
        "backend.api.routes."
        "agent_service.run",
    ) as mock_run:

        response = client.post(
            "/api/datasets/missing_dataset/ask",
            data={
                "question":
                    "Calculate total sales."
            },
        )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        ==
        "Dataset session not found."
    )

    mock_run.assert_not_called()


# ============================================================
# TEST 6
# EMPTY QUESTION
# ============================================================

def test_ask_dataset_rejects_empty_question():

    with patch(
        "backend.api.routes."
        "dataset_manager.exists",
        return_value=True,
    ), patch(
        "backend.api.routes."
        "agent_service.run",
    ) as mock_run:

        response = client.post(
            "/api/datasets/dataset_123/ask",
            data={
                "question": "   "
            },
        )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        ==
        "Question cannot be empty."
    )

    mock_run.assert_not_called()


# ============================================================
# TEST 7
# AGENT FAILURE
# ============================================================

def test_ask_dataset_handles_agent_failure():

    failed_result = {
        "success": False,
        "completed": True,

        "dataset_id":
            "dataset_123",

        "question":
            "Calculate sales.",

        "plan": [
            "sql"
        ],

        "executed_tools": [],

        "failed_tool":
            "sql",

        "last_tool_error":
            "SQL execution failed.",

        "replan_count":
            2,

        "error":
            "SQL execution failed.",

        "trace": [],
    }

    with patch(
        "backend.api.routes."
        "dataset_manager.exists",
        return_value=True,
    ), patch(
        "backend.api.routes."
        "agent_service.run",
        return_value=failed_result,
    ):

        response = client.post(
            "/api/datasets/dataset_123/ask",
            data={
                "question":
                    "Calculate sales."
            },
        )

    assert response.status_code == 500

    detail = (
        response.json()["detail"]
    )

    assert (
        detail["message"]
        ==
        "Agent analytics request failed."
    )

    assert (
        detail["error"]
        ==
        "SQL execution failed."
    )

    assert (
        detail["failed_tool"]
        ==
        "sql"
    )

    assert detail["plan"] == [
        "sql"
    ]

    assert detail["replan_count"] == 2


# ============================================================
# TEST 8
# PREPARE DATASET
# ============================================================

def test_prepare_dataset_endpoint():

    preparation_result = (
        build_preparation_result()
    )

    with patch(
        "backend.api.routes."
        "AnalyticsWorkflow.prepare_dataset",
        return_value=preparation_result,
    ) as mock_prepare:

        response = client.post(
            "/api/datasets",

            files={
                "file": (
                    "sales.csv",
                    b"month,sales\nJan,1000\nFeb,1100\nMar,1200\n",
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200

    mock_prepare.assert_called_once()

    data = response.json()

    assert data["success"] is True

    assert (
        data["dataset_id"]
        ==
        "dataset_123"
    )

    assert (
        data["file"]["original_name"]
        ==
        "sales.csv"
    )

    assert data["dataset"]["rows"] == 3

    assert data["dataset"]["columns"] == [
        "month",
        "sales",
    ]

    assert (
        data["dataset"]["column_count"]
        ==
        2
    )

    assert data["quality"] == {
        "before_score": 90,
        "after_score": 100,
    }

    assert (
        data["cleaning"]["rows_removed"]
        ==
        0
    )


# ============================================================
# TEST 9
# INVALID UPLOAD TYPE
# ============================================================

def test_prepare_dataset_rejects_invalid_file_type():

    response = client.post(
        "/api/datasets",

        files={
            "file": (
                "sales.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "Unsupported file type"
        in response.json()["detail"]
    )


# ============================================================
# TEST 10
# GET DATASET
# ============================================================

def test_get_dataset():

    dataset_info = {
        "dataset_id":
            "dataset_123",

        "filename":
            "sales.csv",

        "rows":
            3,

        "columns": [
            "month",
            "sales",
        ],
    }

    with patch(
        "backend.api.routes."
        "dataset_manager.get_dataset_info",
        return_value=dataset_info,
    ):

        response = client.get(
            "/api/datasets/dataset_123"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        data["dataset"]["dataset_id"]
        ==
        "dataset_123"
    )


# ============================================================
# TEST 11
# GET UNKNOWN DATASET
# ============================================================

def test_get_unknown_dataset():

    with patch(
        "backend.api.routes."
        "dataset_manager.get_dataset_info",
        return_value=None,
    ):

        response = client.get(
            "/api/datasets/missing_dataset"
        )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        ==
        "Dataset session not found."
    )


# ============================================================
# TEST 12
# DELETE DATASET
# ============================================================

def test_delete_dataset():

    with patch(
        "backend.api.routes."
        "dataset_manager.delete_dataset",
        return_value=True,
    ) as mock_delete:

        response = client.delete(
            "/api/datasets/dataset_123"
        )

    assert response.status_code == 200

    mock_delete.assert_called_once_with(
        "dataset_123"
    )

    data = response.json()

    assert data["success"] is True

    assert (
        data["dataset_id"]
        ==
        "dataset_123"
    )


# ============================================================
# TEST 13
# DELETE UNKNOWN DATASET
# ============================================================

def test_delete_unknown_dataset():

    with patch(
        "backend.api.routes."
        "dataset_manager.delete_dataset",
        return_value=False,
    ):

        response = client.delete(
            "/api/datasets/missing_dataset"
        )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        ==
        "Dataset session not found."
    )


# ============================================================
# TEST 14
# ONE-SHOT ANALYZE
# ============================================================

def test_analyze_dataset_uses_preparation_and_agent():

    preparation_result = (
        build_preparation_result()
    )

    agent_result = (
        build_agent_result(
            question=(
                "What is total sales?"
            )
        )
    )

    dataset_info = {
        "dataset_id":
            "dataset_123",

        "filename":
            "sales.csv",

        "rows":
            3,

        "columns": [
            "month",
            "sales",
        ],
    }

    with patch(
        "backend.api.routes."
        "AnalyticsWorkflow.prepare_dataset",
        return_value=preparation_result,
    ) as mock_prepare, patch(
        "backend.api.routes."
        "agent_service.run",
        return_value=agent_result,
    ) as mock_agent, patch(
        "backend.api.routes."
        "dataset_manager.get_dataset_info",
        return_value=dataset_info,
    ):

        response = client.post(
            "/api/analyze",

            data={
                "question":
                    "What is total sales?"
            },

            files={
                "file": (
                    "sales.csv",
                    b"month,sales\nJan,1000\nFeb,1100\nMar,1200\n",
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200

    mock_prepare.assert_called_once()

    mock_agent.assert_called_once_with(
        dataset_id="dataset_123",
        question="What is total sales?",
    )

    data = response.json()

    assert data["success"] is True

    assert (
        data["dataset_id"]
        ==
        "dataset_123"
    )

    assert (
        data["question"]
        ==
        "What is total sales?"
    )

    assert (
        data["agent"]["intent"]
        ==
        "aggregation"
    )

    assert data["agent"]["plan"] == [
        "sql",
        "insight",
    ]

    assert (
        data["analysis"]["sql_result"]
        ==
        [
            {
                "total_sales": 5000
            }
        ]
    )

    assert (
        data["insight"]
        ==
        "Total sales are 5000."
    )


# ============================================================
# TEST 15
# ONE-SHOT ANALYZE REJECTS EMPTY QUESTION
# ============================================================

def test_analyze_rejects_empty_question():

    with patch(
        "backend.api.routes."
        "AnalyticsWorkflow.prepare_dataset",
    ) as mock_prepare, patch(
        "backend.api.routes."
        "agent_service.run",
    ) as mock_agent:

        response = client.post(
            "/api/analyze",

            data={
                "question": "   "
            },

            files={
                "file": (
                    "sales.csv",
                    b"month,sales\nJan,1000\n",
                    "text/csv",
                )
            },
        )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        ==
        "Question cannot be empty."
    )

    mock_prepare.assert_not_called()

    mock_agent.assert_not_called()