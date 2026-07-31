import pandas as pd

from backend.services.dataset_manager import (
    dataset_manager,
)

from backend.orchestration.tools.visualization_tools import (
    generate_visualization,
)


def create_test_dataset():

    df = pd.DataFrame({
        "age": [
            25,
            30,
            35,
            40,
            45,
        ],

        "score": [
            70,
            75,
            85,
            82,
            90,
        ],

        "city": [
            "Delhi",
            "Mumbai",
            "Delhi",
            "Mumbai",
            "Delhi",
        ],

        "active": [
            "Yes",
            "No",
            "Yes",
            "No",
            "Yes",
        ],
    })

    return dataset_manager.create_dataset(
        cleaned_df=df,
        original_filename="visualization_test.csv",
        cleaned_schema={},
    )


def test_generate_scatter_visualization():

    dataset_id = create_test_dataset()

    try:

        result = generate_visualization.invoke({
            "dataset_id": dataset_id,

            "question":
                "Create a scatter plot of age vs score "
                "colored by active",
        })

        assert result["success"] is True

        assert result["chart"] is not None

        assert result["chart_type"] == "scatter"

        assert (
            result["chart"]["x"]
            ==
            "age"
        )

        assert (
            result["chart"]["y"]
            ==
            "score"
        )

    finally:

        dataset_manager.delete_dataset(
            dataset_id
        )


def test_generate_boxplot_visualization():

    dataset_id = create_test_dataset()

    try:

        result = generate_visualization.invoke({
            "dataset_id": dataset_id,

            "question":
                "Create a box plot of score grouped by city",
        })

        assert result["success"] is True

        assert result["chart"] is not None

        assert result["chart_type"] == "box"

    finally:

        dataset_manager.delete_dataset(
            dataset_id
        )


def test_visualization_missing_dataset():

    result = generate_visualization.invoke({
        "dataset_id": "does_not_exist",

        "question":
            "Create a bar chart of score by city",
    })

    assert result["success"] is False

    assert "does not exist" in result["error"]