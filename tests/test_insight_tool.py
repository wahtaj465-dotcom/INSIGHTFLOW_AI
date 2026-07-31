import pandas as pd

from backend.services.dataset_manager import (
    dataset_manager,
)

from backend.orchestration.tools.insight_tools import (
    generate_analytical_insight,
)


def test_insight_tool_missing_dataset():

    result = generate_analytical_insight.invoke({
        "dataset_id": "does_not_exist",

        "question":
            "Explain sales performance by region",

        "sql_result": [],
    })

    assert result["success"] is False

    assert "does not exist" in result["error"]


def test_insight_tool_empty_question():

    df = pd.DataFrame({
        "region": [
            "North",
            "South",
        ],

        "sales": [
            100,
            200,
        ],
    })

    dataset_id = dataset_manager.create_dataset(
        cleaned_df=df,
        original_filename="insight_test.csv",
    )

    try:

        result = generate_analytical_insight.invoke({
            "dataset_id": dataset_id,
            "question": "",
            "sql_result": [],
        })

        assert result["success"] is False

        assert (
            result["error"]
            ==
            "question cannot be empty."
        )

    finally:

        dataset_manager.delete_dataset(
            dataset_id
        )


def test_insight_tool_invalid_dataset_id():

    result = generate_analytical_insight.invoke({
        "dataset_id": "",
        "question": "Explain sales",
        "sql_result": [],
    })

    assert result["success"] is False

    assert (
        result["error"]
        ==
        "dataset_id cannot be empty."
    )