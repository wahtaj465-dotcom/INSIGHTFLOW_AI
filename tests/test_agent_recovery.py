from unittest.mock import patch

from backend.orchestration.graph import (
    agent_graph,
)


def test_agent_recovers_from_failed_sql():

    # ========================================================
    # MOCK PLANNER RESPONSE
    # ========================================================

    planner_response = """
    {
        "intent": "analysis",
        "tools": [
            "sql",
            "insight"
        ],
        "reasoning": "Query the dataset and explain the result."
    }
    """

    # ========================================================
    # MOCK REPLANNER RESPONSE
    # ========================================================

    replanner_response = """
    {
        "intent": "recovery",
        "tools": [
            "sql",
            "insight"
        ],
        "reasoning": "Retry SQL and then generate the insight."
    }
    """

    # ========================================================
    # SQL TOOL BEHAVIOR
    #
    # First call  -> failure
    # Second call -> success
    # ========================================================

    sql_results = [
        {
            "success": False,
            "error": "Temporary SQL execution failure.",
        },
        {
            "success": True,
            "generated_sql": (
                "SELECT AVG(score) "
                "FROM dataset"
            ),
            "result": [
                {
                    "average_score": 85
                }
            ],
        },
    ]

    # ========================================================
    # INSIGHT TOOL RESULT
    # ========================================================

    insight_result = {
        "success": True,
        "insight": (
            "The average score is 85."
        ),
    }

    # ========================================================
    # TOOL LOOKUP
    # ========================================================

    class FakeSQLTool:

        def invoke(
            self,
            tool_input
        ):
            return sql_results.pop(0)

    class FakeInsightTool:

        def invoke(
            self,
            tool_input
        ):
            return insight_result

    fake_sql_tool = FakeSQLTool()

    fake_insight_tool = FakeInsightTool()
    
    def fake_get_tool_definition(tool_name):


        if tool_name == "sql":

            return {
                "tool": fake_sql_tool,

                "inputs": {
                    "dataset_id": "dataset_id",
                    "question": "question",
                },

                "outputs": {
                    "generated_sql": "generated_sql",
                    "result": "sql_result",
                },
            }

        if tool_name == "insight":

            return {
                "tool": fake_insight_tool,

                "inputs": {
                    "dataset_id": "dataset_id",
                    "question": "question",
                    "sql_result": "sql_result",
                    "generated_sql": "generated_sql",
                },

                "outputs": {
                    "insight": "insight",
                },
            }

        return None



    # ========================================================
    # LLM BEHAVIOR
    #
    # First LLM call  -> planner
    # Second LLM call -> replanner
    # ========================================================

    llm_responses = [
        planner_response,
        replanner_response,
    ]

    def fake_generate(
        self,
        prompt
    ):
        return llm_responses.pop(0)

    # ========================================================
    # RUN GRAPH
    # ========================================================

    with patch(
        "backend.orchestration.planner.LLMService.generate",
        new=fake_generate,
    ), patch(
        "backend.orchestration.replanner.LLMService.generate",
        new=fake_generate,
    ), patch(
        "backend.orchestration.nodes.execution.get_tool_definition",
        side_effect=fake_get_tool_definition,
    ):

        result = agent_graph.invoke({
            "dataset_id": "test_dataset",

            "question": (
                "What is the average score "
                "and explain the result?"
            ),

            "max_replans": 2,

            "trace": [],
        })

    # ========================================================
    # ASSERT FINAL WORKFLOW STATE
    # ========================================================

    assert result["completed"] is True

    assert result["error"] is None

    assert result["replan_count"] == 1

    # ========================================================
    # SQL EVENTUALLY SUCCEEDED
    # ========================================================

    assert (
        result["generated_sql"]
        ==
        "SELECT AVG(score) FROM dataset"
    )

    assert result["sql_result"] == [
        {
            "average_score": 85
        }
    ]

    # ========================================================
    # INSIGHT EXECUTED AFTER RECOVERY
    # ========================================================

    assert (
        result["insight"]
        ==
        "The average score is 85."
    )

    assert "sql" in result[
        "executed_tools"
    ]

    assert "insight" in result[
        "executed_tools"
    ]

    # ========================================================
    # VERIFY AGENT ACTUALLY REPLANNED
    # ========================================================

    decisions = [
        event.get("decision")
        for event in result["trace"]
        if event.get("node") == "observer"
    ]

    assert "replan" in decisions

    assert "finish" in decisions

    replanner_events = [
        event
        for event in result["trace"]
        if event.get("node") == "replanner"
    ]

    assert len(
        replanner_events
    ) == 1

    # ========================================================
    # VERIFY FINAL RESPONSE
    # ========================================================

    final_response = result[
        "final_response"
    ]

    assert final_response[
        "success"
    ] is True

    assert final_response[
        "replan_count"
    ] == 1