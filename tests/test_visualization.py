import pandas as pd
import pytest

# Change this import ONLY if your VisualizationService is stored elsewhere.
from backend.services.visualization_service import VisualizationService


@pytest.fixture
def service():
    return VisualizationService()


@pytest.fixture
def source_df():
    """
    Row-level cleaned dataset used to test visualizations.
    Includes:
    - numerical columns
    - categorical columns
    - boolean-like category
    - datetime
    - one extreme income value
    """
    return pd.DataFrame({
        "customer_id": [
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11
        ],
        "age": [
            25, 29, 31, 33, 33, 35,
            38, 42, 45, 50, 55
        ],
        "income": [
            45000,
            48000,
            52000,
            55000,
            47000,
            61000,
            43000,
            49000,
            58000,
            9999999,
            65000
        ],
        "city": [
            "Delhi",
            "Delhi",
            "Mumbai",
            "Mumbai",
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Delhi",
            "Bangalore",
            "Bangalore",
            "Mumbai"
        ],
        "signup_date": pd.to_datetime([
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
            "2025-04-01",
            "2025-05-01",
            "2025-06-01",
            "2025-07-01",
            "2025-08-01",
            "2025-09-01",
            "2025-10-01",
            "2025-11-01"
        ]),
        "active": [
            "Yes", "No", "Yes", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes", "Yes"
        ],
        "score": [
            82, 91, 75, 80, 88,
            84, 73, 69, 77, 86, 81
        ]
    })


@pytest.fixture
def schema():
    return {
        "customer_id": {
            "detected_type": "Identifier"
        },
        "age": {
            "detected_type": "Numerical"
        },
        "income": {
            "detected_type": "Numerical"
        },
        "city": {
            "detected_type": "Categorical"
        },
        "signup_date": {
            "detected_type": "Datetime"
        },
        "active": {
            "detected_type": "Categorical"
        },
        "score": {
            "detected_type": "Numerical"
        }
    }


# ============================================================
# HELPERS
# ============================================================

def assert_basic_chart(chart, expected_type):
    assert chart is not None
    assert chart["chart_type"] == expected_type
    assert "title" in chart
    assert "data" in chart
    assert len(chart["data"]) > 0
    assert "metadata" in chart


# ============================================================
# 1. CHART TYPE DETECTION
# ============================================================

@pytest.mark.parametrize(
    "question, expected",
    [
        ("Create a box plot of income", "box"),
        ("Show an income boxplot", "box"),
        ("Create a violin plot of income", "violin"),
        ("Create a histogram of age", "histogram"),
        ("Create a scatter plot of age vs score", "scatter"),
        ("Show a heatmap", "heatmap"),
        ("Create a correlation matrix", "heatmap"),
        ("Create a stacked bar chart", "stacked_bar"),
        ("Create a line chart", "line"),
        ("Create a bar chart", "bar"),
    ]
)
def test_detect_requested_chart_type(
    service,
    question,
    expected
):
    assert (
        service._detect_requested_chart_type(question)
        == expected
    )


# ============================================================
# 2. QUESTION COLUMN DETECTION
# ============================================================

def test_question_column_detection(
    service,
    source_df,
    schema
):
    question = (
        "Create a scatter plot of age vs score "
        "colored by active status"
    )

    columns = service._find_question_columns(
        question,
        source_df,
        schema
    )

    assert "age" in columns
    assert "score" in columns
    assert "active" in columns

    # Should preserve question order.
    assert columns.index("age") < columns.index("score")


# ============================================================
# 3. BOX PLOT
# ============================================================

def test_boxplot_income_by_city(
    service,
    source_df,
    schema
):
    # Simulates SQL Agent output.
    sql_result = source_df[
        ["city", "income"]
    ].copy()

    chart = service.generate_result_chart(
        df=sql_result,
        question=(
            "Create a box plot of income grouped by city "
            "and explain any outliers."
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "box")

    assert chart["x"] == "city"
    assert chart["y"] == "income"

    assert chart["metadata"]["metric"] == "income"
    assert chart["metadata"]["category"] == "city"

    # Critical:
    # box plots should use row-level source dataset.
    assert (
        chart["metadata"]["data_source"]
        == "source_dataset"
    )

    assert len(chart["data"]) == len(source_df)


# ============================================================
# 4. BOX PLOT WITH SECOND CATEGORY / COLOR
# ============================================================

def test_boxplot_with_color_group(
    service,
    source_df,
    schema
):
    chart = service.generate_result_chart(
        df=source_df[
            ["city", "active", "income"]
        ],
        question=(
            "Create a box plot of income grouped by city "
            "and active"
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "box")

    assert chart["x"] == "city"
    assert chart["y"] == "income"
    assert chart["color"] == "active"

    assert (
        chart["metadata"]["color_group"]
        == "active"
    )


# ============================================================
# 5. VIOLIN PLOT
# ============================================================

def test_violin_income_by_city(
    service,
    source_df,
    schema
):
    chart = service.generate_result_chart(
        df=source_df[["city", "income"]],
        question=(
            "Create a violin plot of income grouped by city"
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "violin")

    assert chart["x"] == "city"
    assert chart["y"] == "income"

    assert (
        chart["metadata"]["data_source"]
        == "source_dataset"
    )


# ============================================================
# 6. SCATTER
# ============================================================

def test_scatter_age_score_colored_active(
    service,
    source_df,
    schema
):
    sql_result = source_df[
        ["age", "score", "active"]
    ].copy()

    chart = service.generate_result_chart(
        df=sql_result,
        question=(
            "Create a scatter plot of age vs score "
            "colored by active status."
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "scatter")

    assert chart["x"] == "age"
    assert chart["y"] == "score"
    assert chart["color"] == "active"

    assert (
        chart["metadata"]["x_metric"]
        == "age"
    )

    assert (
        chart["metadata"]["y_metric"]
        == "score"
    )

    assert (
        chart["metadata"]["color_group"]
        == "active"
    )

    assert (
        chart["metadata"]["data_source"]
        == "source_dataset"
    )


# ============================================================
# 7. HISTOGRAM
# ============================================================

def test_histogram_age(
    service,
    source_df,
    schema
):
    chart = service.generate_result_chart(
        df=source_df[["age"]],
        question="Create a histogram of age",
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "histogram")

    assert chart["x"] == "bin"
    assert chart["y"] == "frequency"

    assert chart["metadata"]["metric"] == "age"

    total_frequency = sum(
        row["frequency"]
        for row in chart["data"]
    )

    assert total_frequency == len(source_df)


# ============================================================
# 8. HEATMAP
# ============================================================

def test_heatmap(
    service,
    source_df,
    schema
):
    chart = service.generate_result_chart(
        df=source_df[
            ["age", "income", "score"]
        ],
        question=(
            "Create a correlation heatmap of "
            "age income and score"
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "heatmap")

    assert chart["x"] == "x"
    assert chart["y"] == "y"

    assert (
        chart["metadata"]["intent"]
        == "correlation"
    )

    assert set(
        chart["metadata"]["columns"]
    ) == {
        "age",
        "income",
        "score"
    }

    # 3 x 3 correlation matrix.
    assert len(chart["data"]) == 9


# ============================================================
# 9. BAR CHART
# ============================================================

def test_bar_chart(
    service
):
    sql_result = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai",
            "Bangalore"
        ],
        "average_income": [
            50000,
            57000,
            62000
        ]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question=(
            "Create a bar chart of average_income by city"
        )
    )

    assert_basic_chart(chart, "bar")

    assert chart["x"] == "city"
    assert chart["y"] == "average_income"


# ============================================================
# 10. STACKED BAR
# ============================================================

def test_stacked_bar(
    service
):
    sql_result = pd.DataFrame({
        "city": [
            "Delhi",
            "Delhi",
            "Mumbai",
            "Mumbai"
        ],
        "active": [
            "Yes",
            "No",
            "Yes",
            "No"
        ],
        "count": [
            5,
            2,
            7,
            3
        ]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question=(
            "Create a stacked bar chart of active by city"
        )
    )

    assert_basic_chart(chart, "stacked_bar")

    assert chart["x"] == "active"
    assert chart["color"] == "city"
    assert chart["y"] == "count"


# ============================================================
# 11. LINE CHART
# ============================================================

def test_line_chart(
    service
):
    sql_result = pd.DataFrame({
        "month": pd.to_datetime([
            "2025-01-01",
            "2025-02-01",
            "2025-03-01"
        ]),
        "revenue": [
            100000,
            120000,
            150000
        ]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question=(
            "Create a line chart of revenue by month"
        )
    )

    assert_basic_chart(chart, "line")

    assert chart["x"] == "month"
    assert chart["y"] == "revenue"

    assert (
        chart["metadata"]["data_source"]
        == "sql_result"
    )


# ============================================================
# 12. KPI
# ============================================================

def test_kpi(
    service
):
    sql_result = pd.DataFrame({
        "total_customers": [120]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question="How many customers are there?"
    )

    assert chart is not None

    assert chart["chart_type"] == "kpi"
    assert chart["value"] == 120

    assert (
        chart["metadata"]["intent"]
        == "summary"
    )


# ============================================================
# 13. AUTOMATIC CATEGORY + NUMERIC
# ============================================================

def test_automatic_bar(
    service
):
    sql_result = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai",
            "Bangalore"
        ],
        "sales": [
            100,
            150,
            120
        ]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question="Compare sales by city"
    )

    assert_basic_chart(chart, "bar")

    assert chart["x"] == "city"
    assert chart["y"] == "sales"


# ============================================================
# 14. AUTOMATIC NUMERIC + NUMERIC
# ============================================================

def test_automatic_scatter(
    service
):
    sql_result = pd.DataFrame({
        "age": [20, 30, 40],
        "score": [70, 80, 90]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question="Show relationship"
    )

    assert_basic_chart(chart, "scatter")

    assert chart["x"] == "age"
    assert chart["y"] == "score"


# ============================================================
# 15. SINGLE CATEGORY DISTRIBUTION
# ============================================================

def test_single_category(
    service
):
    sql_result = pd.DataFrame({
        "city": [
            "Delhi",
            "Delhi",
            "Mumbai",
            "Bangalore"
        ]
    })

    chart = service.generate_result_chart(
        df=sql_result,
        question="Show city distribution"
    )

    assert_basic_chart(chart, "bar")

    assert chart["x"] == "city"
    assert chart["y"] == "count"

    total = sum(
        row["count"]
        for row in chart["data"]
    )

    assert total == 4


# ============================================================
# 16. NUMERIC STRING DETECTION
# ============================================================

def test_numeric_strings_detected_as_numeric(
    service
):
    series = pd.Series([
        "10",
        "20",
        "30",
        "invalid",
        "40"
    ])

    assert service._is_numeric(series) is True


# ============================================================
# 17. IDENTIFIER SHOULD NOT BECOME METRIC
# ============================================================

def test_identifier_excluded_from_explicit_scatter(
    service,
    source_df,
    schema
):
    chart = service.generate_result_chart(
        df=source_df[
            [
                "customer_id",
                "age",
                "score"
            ]
        ],
        question=(
            "Create a scatter plot of age vs score"
        ),
        source_df=source_df,
        schema=schema
    )

    assert_basic_chart(chart, "scatter")

    assert chart["x"] == "age"
    assert chart["y"] == "score"

    assert chart["x"] != "customer_id"
    assert chart["y"] != "customer_id"


# ============================================================
# 18. EMPTY DATAFRAME
# ============================================================

def test_empty_dataframe(
    service
):
    chart = service.generate_result_chart(
        df=pd.DataFrame(),
        question="Create a bar chart"
    )

    assert chart is None


# ============================================================
# 19. INVALID INPUT
# ============================================================

def test_invalid_dataframe_input(
    service
):
    with pytest.raises(TypeError):

        service.generate_result_chart(
            df=[1, 2, 3],
            question="Create a chart"
        )


# ============================================================
# 20. EDA GENERATION
# ============================================================

def test_generate_eda_charts(
    service,
    source_df,
    schema
):
    charts = service.generate_eda_charts(
        df=source_df,
        schema=schema
    )

    assert isinstance(charts, list)
    assert len(charts) > 0

    # Service default is max 10.
    assert len(charts) <= 10

    for chart in charts:

        assert "chart_type" in chart
        assert "title" in chart
        assert "data" in chart
        assert len(chart["data"]) > 0
        assert "metadata" in chart

        # Ranking should have attached priority.
        assert (
            "priority_score"
            in chart["metadata"]
        )


# ============================================================
# 21. EDA MUST EXCLUDE IDENTIFIER
# ============================================================

def test_eda_does_not_use_customer_id_as_axis(
    service,
    source_df,
    schema
):
    charts = service.generate_eda_charts(
        source_df,
        schema
    )

    for chart in charts:

        assert chart.get("x") != "customer_id"
        assert chart.get("y") != "customer_id"


# ============================================================
# 22. BOX PLOT OUTLIER HELPER
# ============================================================

def test_boxplot_outlier_detection(
    service
):
    series = pd.Series([
        10,
        11,
        12,
        13,
        14,
        1000
    ])

    result = service._create_boxplot_data(
        series,
        "value"
    )

    assert len(result) == 1

    stats = result[0]

    assert stats["column"] == "value"
    assert stats["outlier_count"] == 1
    assert 1000 in stats["outliers"]


# ============================================================
# 23. JSON SAFE DATETIME
# ============================================================

def test_dataframe_to_records_datetime(
    service
):
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2025-01-01")
        ],
        "value": [10]
    })

    records = service.dataframe_to_records(df)

    assert len(records) == 1

    assert (
        records[0]["date"]
        == "2025-01-01T00:00:00"
    )

    assert records[0]["value"] == 10