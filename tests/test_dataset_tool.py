import pandas as pd

from backend.services.dataset_manager import (
    dataset_manager
)

from backend.orchestration.tools.dataset_tools import (
    get_dataset_context
)


def test_get_dataset_context():

    df = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai",
            "Delhi"
        ],

        "sales": [
            100,
            200,
            150
        ]
    })

    dataset_id = dataset_manager.create_dataset(

        cleaned_df=df,

        original_filename="test.csv",

        cleaned_schema={
            "city": {
                "detected_type": "Categorical"
            },

            "sales": {
                "detected_type": "Numerical"
            }
        },

        cleaned_quality_report={
            "overall_score": 95
        },

        cleaned_anomalies={},

        eda_results={},

        eda_charts=[],

        statistical_findings=[]
    )

    try:

        result = get_dataset_context.invoke({
            "dataset_id": dataset_id
        })

        assert result["success"] is True

        assert result["dataset_id"] == dataset_id

        assert result["rows"] == 3

        assert result["column_count"] == 2

        assert result["columns"] == [
            "city",
            "sales"
        ]

        assert (
            result["schema"]["sales"]["detected_type"]
            ==
            "Numerical"
        )

    finally:

        dataset_manager.delete_dataset(
            dataset_id
        )