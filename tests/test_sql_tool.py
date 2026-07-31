import pandas as pd

from backend.services.dataset_manager import (
    dataset_manager
)

from backend.orchestration.tools.sql_tools import (
    run_sql_analysis
)


def test_sql_tool_missing_dataset():

    result = run_sql_analysis.invoke({
        "dataset_id": "does_not_exist",
        "question": "Average sales by city"
    })

    assert result["success"] is False

    assert "does not exist" in result["error"]


def test_sql_tool_empty_question():

    df = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai"
        ],

        "sales": [
            100,
            200
        ]
    })

    dataset_id = dataset_manager.create_dataset(
        cleaned_df=df,
        original_filename="test.csv"
    )

    try:

        result = run_sql_analysis.invoke({
            "dataset_id": dataset_id,
            "question": ""
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