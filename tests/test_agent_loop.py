from unittest.mock import patch

from backend.orchestration.graph import (
    build_agent_graph,
)


# ============================================================
# FAKE TOOL
# ============================================================

class FakeTool:
    """
    Simple fake LangChain-style tool used to test
    orchestration without calling real services.
    """

    def __init__(
        self,
        result
    ):
        self.result = result

    def invoke(
        self,
        tool_input
    ):
        return self.result


# ============================================================
# TEST FULL AGENT LOOP
# ============================================================

def test_full_agent_execution_loop():

    # --------------------------------------------------------
    # Fake LLM planner response
    # --------------------------------------------------------

    fake_plan = """
    {
        "intent": "relationship_analysis",
        "tools": [
            "sql",
            "visualization",
            "insight"
        ],
        "reasoning": "Analyze, visualize and explain the relationship."
    }
    """

    # --------------------------------------------------------
    # Fake tool results
    # --------------------------------------------------------

    fake_tools = {

        "sql": FakeTool({
            "success": True,

            "generated_sql":
                "SELECT age, score, active FROM dataset",

            "result": [
                {
                    "age": 25,
                    "score": 82,
                    "active": "Yes",
                },
                {
                    "age": 35,
                    "score": 84,
                    "active": "Yes",
                },
            ],
        }),

        "visualization": FakeTool({
            "success": True,

            "chart": {
                "type": "scatter",
                "x": "age",
                "y": "score",
                "color": "active",
            },
        }),

        "insight": FakeTool({
            "success": True,

            "insight": (
                "Score increases slightly with age "
                "in this sample."
            ),
        }),
    }

    # --------------------------------------------------------
    # Fake registry lookup
    # --------------------------------------------------------
    
    def fake_get_tool_definition(tool_name):


        tool = fake_tools.get(
            tool_name
        )

        if tool is None:
            return None

        definitions = {

            "sql": {
                "tool": tool,

                "inputs": {
                    "dataset_id": "dataset_id",
                    "question": "question",
                },

                "outputs": {
                    "generated_sql": "generated_sql",
                    "result": "sql_result",
                },
            },

            "visualization": {
                "tool": tool,

                "inputs": {
                    "dataset_id": "dataset_id",
                    "question": "question",
                    "sql_result": "sql_result",
                },

                "outputs": {
                    "chart": "visualization",
                },
            },

            "insight": {
                "tool": tool,

                "inputs": {
                    "dataset_id": "dataset_id",
                    "question": "question",
                    "sql_result": "sql_result",
                    "generated_sql": "generated_sql",
                },

                "outputs": {
                    "insight": "insight",
                },
            },
        }

        return definitions.get(
            tool_name
        )

    

    # --------------------------------------------------------
    # Mock Gemini + tool registry
    # --------------------------------------------------------

    with patch(
        "backend.orchestration.planner.LLMService.generate",
        return_value=fake_plan,
    ), patch(
        "backend.orchestration.nodes.execution.get_tool_definition",
        side_effect=fake_get_tool_definition,
    ):

        graph = build_agent_graph()

        result = graph.invoke({
            "dataset_id": "test_dataset",

            "question": (
                "Create a scatter plot of age vs score "
                "colored by active and explain the relationship."
            ),

            "max_retries": 2,

            "trace": [],
        })

    # ========================================================
    # GRAPH COMPLETED
    # ========================================================

    assert result["completed"] is True

    assert result["error"] is None

    # ========================================================
    # PLANNER
    # ========================================================

    assert (
        result["planner_source"]
        ==
        "llm"
    )

    assert (
        result["intent"]
        ==
        "relationship_analysis"
    )

    assert result["plan"] == [
        "sql",
        "visualization",
        "insight",
    ]

    # ========================================================
    # TOOL EXECUTION ORDER
    # ========================================================

    assert result["executed_tools"] == [
        "sql",
        "visualization",
        "insight",
    ]

    # ========================================================
    # SQL OUTPUT
    # ========================================================

    assert (
        result["generated_sql"]
        ==
        "SELECT age, score, active FROM dataset"
    )

    assert result["sql_result"] is not None

    assert len(
        result["sql_result"]
    ) == 2

    # ========================================================
    # VISUALIZATION OUTPUT
    # ========================================================

    assert (
        result["visualization"]["type"]
        ==
        "scatter"
    )

    assert (
        result["visualization"]["x"]
        ==
        "age"
    )

    assert (
        result["visualization"]["y"]
        ==
        "score"
    )

    # ========================================================
    # INSIGHT OUTPUT
    # ========================================================

    assert (
        result["insight"]
        ==
        "Score increases slightly with age in this sample."
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    final_response = (
        result["final_response"]
    )

    assert (
        final_response["success"]
        is True
    )

    assert final_response[
        "executed_tools"
    ] == [
        "sql",
        "visualization",
        "insight",
    ]

    assert (
        final_response["planner_source"]
        ==
        "llm"
    )

    # ========================================================
    # TRACE - EXECUTOR
    # ========================================================

    trace = result["trace"]

    executor_tools = [
        event.get("tool")
        for event in trace
        if (
            event.get("node")
            ==
            "tool_executor"
        )
    ]

    assert executor_tools == [
        "sql",
        "visualization",
        "insight",
    ]

    # ========================================================
    # TRACE - OBSERVER
    # ========================================================

    observer_decisions = [
        event.get("decision")
        for event in trace
        if (
            event.get("node")
            ==
            "observer"
        )
    ]

    assert observer_decisions == [
        "continue",
        "continue",
        "finish",
    ]

    # ========================================================
    # TRACE - GRAPH LIFECYCLE
    # ========================================================

    node_sequence = [
        event.get("node")
        for event in trace
    ]

    assert node_sequence[0] == (
        "initialize"
    )

    assert "planner" in node_sequence

    assert node_sequence[-1] == (
        "finish"
    )