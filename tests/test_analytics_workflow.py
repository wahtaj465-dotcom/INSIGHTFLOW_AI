from pathlib import Path

import pytest

from backend.workflows.analytics_workflow import (
    AnalyticsWorkflow
)

from backend.services.dataset_manager import (
    dataset_manager
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

TEST_DIR = Path(__file__).parent

DATASET_DIR = (
    TEST_DIR
    / "datasets"
)


# ============================================================
# TEST DATASETS
# ============================================================

TEST_DATASETS = [

    "sales_test.csv",

    "messy_test.csv",

    "customer_test.csv",

    "timeseries_test.csv"
]


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def workflow():
    """
    Create a fresh AnalyticsWorkflow for each test.
    """

    return AnalyticsWorkflow(
        max_charts=10
    )


# ============================================================
# COMPLETE PIPELINE TEST
# ============================================================

@pytest.mark.parametrize(
    "filename",
    TEST_DATASETS
)
def test_complete_analytics_pipeline(
    workflow,
    filename
):
    """
    Verify that every structurally different dataset
    can pass through the complete deterministic
    analytics pipeline.
    """

    file_path = (
        DATASET_DIR
        / filename
    )

    assert file_path.exists(), (
        f"Test dataset missing: {file_path}"
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=filename
        )
    )

    # --------------------------------------------------------
    # BASIC RESULT
    # --------------------------------------------------------

    assert isinstance(
        result,
        dict
    )

    assert result.get(
        "success"
    ) is True

    # --------------------------------------------------------
    # DATASET ID
    # --------------------------------------------------------

    dataset_id = result.get(
        "dataset_id"
    )

    assert isinstance(
        dataset_id,
        str
    )

    assert dataset_id.strip()

    # --------------------------------------------------------
    # DATASET SHOULD EXIST IN MANAGER
    # --------------------------------------------------------

    assert dataset_manager.exists(
        dataset_id
    )

    # --------------------------------------------------------
    # ROWS / COLUMNS
    # --------------------------------------------------------

    assert result.get(
        "rows",
        0
    ) > 0

    columns = result.get(
        "columns",
        []
    )

    assert isinstance(
        columns,
        list
    )

    assert len(
        columns
    ) > 0

    assert result.get(
        "column_count"
    ) == len(
        columns
    )

    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    schema = result.get(
        "schema",
        {}
    )

    assert isinstance(
        schema,
        dict
    )

    original_schema = schema.get(
        "original",
        {}
    )

    cleaned_schema = schema.get(
        "cleaned",
        {}
    )

    assert isinstance(
        original_schema,
        dict
    )

    assert isinstance(
        cleaned_schema,
        dict
    )

    assert len(
        cleaned_schema
    ) > 0

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality = result.get(
        "quality",
        {}
    )

    assert isinstance(
        quality,
        dict
    )

    before_score = quality.get(
        "before_score"
    )

    after_score = quality.get(
        "after_score"
    )

    if before_score is not None:

        assert (
            0
            <= before_score
            <= 100
        )

    if after_score is not None:

        assert (
            0
            <= after_score
            <= 100
        )

    # --------------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------------

    anomalies = result.get(
        "anomalies",
        {}
    )

    assert isinstance(
        anomalies,
        dict
    )

    assert isinstance(
        anomalies.get(
            "before",
            {}
        ),
        dict
    )

    assert isinstance(
        anomalies.get(
            "after",
            {}
        ),
        dict
    )

    # --------------------------------------------------------
    # CLEANING
    # --------------------------------------------------------

    cleaning = result.get(
        "cleaning",
        {}
    )

    assert isinstance(
        cleaning,
        dict
    )

    operations = cleaning.get(
        "operations",
        []
    )

    assert isinstance(
        operations,
        list
    )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    eda_results = result.get(
        "eda_results",
        {}
    )

    assert isinstance(
        eda_results,
        dict
    )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    charts = result.get(
        "eda_charts",
        []
    )

    assert isinstance(
        charts,
        list
    )

    assert result.get(
        "chart_count"
    ) == len(
        charts
    )

    assert len(
        charts
    ) <= 10

    # --------------------------------------------------------
    # STATISTICAL FINDINGS
    # --------------------------------------------------------

    findings = result.get(
        "statistical_findings",
        []
    )

    assert isinstance(
        findings,
        list
    )

    assert result.get(
        "statistical_finding_count"
    ) == len(
        findings
    )


# ============================================================
# CHART CONTRACT TEST
# ============================================================

@pytest.mark.parametrize(
    "filename",
    TEST_DATASETS
)
def test_visualization_contract(
    workflow,
    filename
):
    """
    Ensure visualization metadata generated by the
    backend can be consumed by the frontend renderer.
    """

    file_path = (
        DATASET_DIR
        / filename
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=filename
        )
    )

    charts = result.get(
        "eda_charts",
        []
    )

    supported_chart_types = {

        "kpi",
        "metric",

        "bar",
        "bar_chart",

        "stacked_bar",
        "stacked_bar_chart",

        "histogram",
        "hist",

        "line",
        "line_chart",

        "scatter",
        "scatter_plot",

        "box",
        "boxplot",
        "box_plot",

        "violin",
        "violin_plot",

        "heatmap",
        "correlation_heatmap",

        "pie",
        "pie_chart"
    }

    for chart in charts:

        assert isinstance(
            chart,
            dict
        )

        chart_type = (
            chart.get(
                "chart_type"
            )
            or
            chart.get(
                "type"
            )
        )

        assert chart_type is not None

        normalized_type = (
            str(
                chart_type
            )
            .strip()
            .lower()
            .replace(
                "-",
                "_"
            )
            .replace(
                " ",
                "_"
            )
        )

        assert normalized_type in (
            supported_chart_types
        ), (
            "Unsupported chart type generated: "
            f"{chart_type}"
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        assert chart.get(
            "title"
        )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        if normalized_type not in {
            "kpi",
            "metric"
        }:

            assert (
                "data"
                in chart
            ), (
                "Visualization missing data: "
                f"{chart}"
            )


# ============================================================
# DATASET MANAGER TEST
# ============================================================

def test_dataset_manager_storage(
    workflow
):

    file_path = (
        DATASET_DIR
        / "sales_test.csv"
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=(
                "sales_test.csv"
            )
        )
    )

    dataset_id = result[
        "dataset_id"
    ]

    stored = (
        dataset_manager.get_dataset(
            dataset_id
        )
    )

    assert stored is not None

    assert (
        stored.get(
            "original_filename"
        )
        ==
        "sales_test.csv"
    )

    assert (
        stored.get(
            "cleaned_df"
        )
        is not None
    )

    assert isinstance(
        stored.get(
            "eda_results",
            {}
        ),
        dict
    )

    assert isinstance(
        stored.get(
            "eda_charts",
            []
        ),
        list
    )

    assert isinstance(
        stored.get(
            "statistical_findings",
            []
        ),
        list
    )


# ============================================================
# SALES SEMANTIC TYPE TEST
# ============================================================

def test_sales_semantic_types(
    workflow
):

    file_path = (
        DATASET_DIR
        / "sales_test.csv"
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=(
                "sales_test.csv"
            )
        )
    )

    schema = (
        result[
            "schema"
        ][
            "original"
        ]
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    assert (
        schema[
            "order_id"
        ][
            "detected_type"
        ]
        ==
        "Identifier"
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    assert (
        schema[
            "order_date"
        ][
            "detected_type"
        ]
        ==
        "Datetime"
    )

    # --------------------------------------------------------
    # NUMERICAL
    # --------------------------------------------------------

    assert (
        schema[
            "quantity"
        ][
            "detected_type"
        ]
        ==
        "Numerical"
    )

    assert (
        schema[
            "revenue"
        ][
            "detected_type"
        ]
        ==
        "Numerical"
    )


# ============================================================
# CUSTOMER SEMANTIC TYPE TEST
# ============================================================

def test_customer_semantic_types(
    workflow
):

    file_path = (
        DATASET_DIR
        / "customer_test.csv"
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=(
                "customer_test.csv"
            )
        )
    )

    schema = (
        result[
            "schema"
        ][
            "original"
        ]
    )

    assert (
        schema[
            "customer_id"
        ][
            "detected_type"
        ]
        ==
        "Identifier"
    )

    assert (
        schema[
            "age"
        ][
            "detected_type"
        ]
        ==
        "Numerical"
    )


# ============================================================
# TIME SERIES TEST
# ============================================================

def test_timeseries_detection(
    workflow
):

    file_path = (
        DATASET_DIR
        / "timeseries_test.csv"
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=(
                "timeseries_test.csv"
            )
        )
    )

    schema = (
        result[
            "schema"
        ][
            "original"
        ]
    )

    assert (
        schema[
            "date"
        ][
            "detected_type"
        ]
        ==
        "Datetime"
    )

    assert (
        schema[
            "revenue"
        ][
            "detected_type"
        ]
        ==
        "Numerical"
    )


# ============================================================
# QUALITY COMPONENT TEST
# ============================================================

@pytest.mark.parametrize(
    "filename",
    TEST_DATASETS
)
def test_quality_components(
    workflow,
    filename
):

    file_path = (
        DATASET_DIR
        / filename
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=filename
        )
    )

    quality = result[
        "quality"
    ]

    before_components = (
        quality.get(
            "before_components",
            {}
        )
    )

    after_components = (
        quality.get(
            "after_components",
            {}
        )
    )

    assert isinstance(
        before_components,
        dict
    )

    assert isinstance(
        after_components,
        dict
    )


# ============================================================
# CLEANING PRESERVES DATA TEST
# ============================================================

@pytest.mark.parametrize(
    "filename",
    TEST_DATASETS
)
def test_cleaning_preserves_dataset(
    workflow,
    filename
):

    file_path = (
        DATASET_DIR
        / filename
    )

    result = (
        workflow.prepare_dataset(
            file_path=file_path,
            original_filename=filename
        )
    )

    assert result[
        "rows"
    ] > 0

    assert result[
        "column_count"
    ] > 0