from backend.orchestration.state import AgentState


def test_agent_state_creation():

    state: AgentState = {

        "dataset_id": "test_dataset",

        "question":
            "Show average revenue by region",

        "intent":
            None,

        "plan":
            [],

        "executed_tools":
            [],

        "tool_results":
            {},

        "retry_count":
            0,

        "max_retries":
            2,

        "error":
            None,

        "completed":
            False,

        "trace":
            []
    }

    assert state["dataset_id"] == "test_dataset"

    assert state["question"] == (
        "Show average revenue by region"
    )

    assert state["plan"] == []

    assert state["executed_tools"] == []

    assert state["retry_count"] == 0

    assert state["completed"] is False