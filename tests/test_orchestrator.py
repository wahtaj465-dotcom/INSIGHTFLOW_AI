import pandas as pd

from backend.services.dataset_manager import (
    dataset_manager,
)

from backend.orchestration.graph import (
    agent_graph,
)


def test_orchestrator_dataset_context():

    df = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai",
            "Delhi",
        ],

        "sales": [
            100,
            200,
            150,
        ],
    })

    dataset_id = (
        dataset_manager
        .create_dataset(
            cleaned_df=df,
            original_filename="agent_test.csv",
        )
    )

    try:

        result = agent_graph.invoke({
            "dataset_id": dataset_id,

            "question":
                "What columns are in this dataset?",

            "max_retries": 2,

            "trace": [],
        })

        assert (
            result["completed"]
            is True
        )

        assert (
            result["plan"]
            ==
            ["dataset_context"]
        )

        assert (
            result["executed_tools"]
            ==
            ["dataset_context"]
        )

        context = (
            result[
                "tool_results"
            ][
                "dataset_context"
            ]
        )

        assert (
            context["success"]
            is True
        )

        assert (
            context["columns"]
            ==
            [
                "city",
                "sales",
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

    finally:

        dataset_manager.delete_dataset(
            dataset_id
        )