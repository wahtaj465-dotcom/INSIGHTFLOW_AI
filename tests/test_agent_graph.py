from unittest.mock import patch

from backend.orchestration.graph import (
    agent_graph,
)


class FakeDatasetTool:

    def invoke(
        self,
        tool_input
    ):

        return {
            "success": True,

            "context": {
                "columns": [
                    "region",
                    "revenue",
                ]
            },
        }


def test_basic_agent_graph():

    # ========================================================
    # FAKE PLANNER
    # ========================================================

    fake_plan = """
    {
        "intent": "dataset_understanding",
        "tools": [
            "dataset_context"
        ],
        "reasoning": "Inspect the dataset structure."
    }
    """

    # ========================================================
    # FAKE TOOL
    # ========================================================

    fake_dataset_tool = (
        FakeDatasetTool()
    )

    def fake_get_tool_definition(
        tool_name
    ):

        if (
            tool_name
            ==
            "dataset_context"
        ):

            return {
                "tool":
                    fake_dataset_tool,

                "inputs": {
                    "dataset_id":
                        "dataset_id",
                },

                "outputs": {},
            }

        return None

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state = {

        "dataset_id":
            "test_dataset",

        "question":
            "What columns are in this dataset?",

        "trace": [],
    }

    # ========================================================
    # RUN GRAPH
    # ========================================================

    with patch(
        "backend.orchestration.planner.LLMService.generate",
        return_value=fake_plan,
    ), patch(
        "backend.orchestration.nodes.execution.get_tool_definition",
        side_effect=fake_get_tool_definition,
    ):

        result = (
            agent_graph.invoke(
                initial_state
            )
        )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert (
        result["dataset_id"]
        ==
        "test_dataset"
    )

    assert (
        result["question"]
        ==
        "What columns are in this dataset?"
    )

    assert (
        result["completed"]
        is True
    )

    assert (
        result["error"]
        is None
    )

    assert (
        "dataset_context"
        in result[
            "executed_tools"
        ]
    )

    assert (
        result[
            "final_response"
        ][
            "success"
        ]
        is True
    )