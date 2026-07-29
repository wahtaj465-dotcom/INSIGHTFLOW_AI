import pandas as pd

from backend.agents.schema_agent import analyze_schema
from backend.services.visualization_service import VisualizationService


# ============================================================
# TEST DATASET
# ============================================================

def create_test_dataset():
    """
    Create a small test dataset for visualization testing.
    """

    data = {

        "order_id": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        ],

        "region": [
            "North",
            "South",
            "North",
            "East",
            "West",
            "South",
            "North",
            "East",
            "West",
            "South"
        ],

        "product": [
            "Laptop",
            "Mouse",
            "Laptop",
            "Keyboard",
            "Monitor",
            "Mouse",
            "Laptop",
            "Keyboard",
            "Monitor",
            "Mouse"
        ],

        "quantity": [
            1, 3, 2, 4, 1,
            5, 2, 3, 1, 4
        ],

        "price": [
            55000,
            500,
            60000,
            1500,
            7000,
            600,
            58000,
            1400,
            7200,
            550
        ],

        "order_date": pd.to_datetime([
            "2026-01-10",
            "2026-01-11",
            "2026-01-12",
            "2026-01-13",
            "2026-01-14",
            "2026-01-15",
            "2026-01-16",
            "2026-01-17",
            "2026-01-18",
            "2026-01-19"
        ])
    }

    return pd.DataFrame(data)


# ============================================================
# PRINT CHART
# ============================================================

def print_chart(chart):
    """
    Pretty-print one chart specification.
    """

    print("\n" + "-" * 60)

    print(
        "Chart Type:",
        chart.get("chart_type")
    )

    print(
        "Title:",
        chart.get("title")
    )

    print(
        "X:",
        chart.get("x")
    )

    print(
        "Y:",
        chart.get("y")
    )

    print(
        "Reason:",
        chart.get("reason")
    )

    print(
        "Data:",
        chart.get("data")
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("       INSIGHTFLOW VISUALIZATION SERVICE TEST")
    print("=" * 60)


    # ========================================================
    # STEP 1 — CREATE DATASET
    # ========================================================

    print("\n[1/5] Creating test dataset...")

    df = create_test_dataset()

    print(
        f"Dataset created: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )


    # ========================================================
    # STEP 2 — ANALYZE SCHEMA
    # ========================================================

    print("\n[2/5] Analyzing schema...")

    schema = analyze_schema(
        df
    )

    for column, info in schema.items():

        print(
            f"{column}: "
            f"{info['detected_type']}"
        )


    # ========================================================
    # STEP 3 — CREATE VISUALIZATION SERVICE
    # ========================================================

    print(
        "\n[3/5] Creating "
        "VisualizationService..."
    )

    visualization_service = (
        VisualizationService(
            max_categories=15,
            max_charts=10
        )
    )

    print(
        "VisualizationService created."
    )


    # ========================================================
    # STEP 4 — GENERATE EDA CHARTS
    # ========================================================

    print("\n[4/5] Generating EDA charts...")

    charts = (
        visualization_service
        .generate_eda_charts(
            df=df,
            schema=schema
        )
    )

    print(
        f"\nGenerated {len(charts)} chart(s)."
    )

    for index, chart in enumerate(
        charts,
        start=1
    ):

        print(
            f"\nCHART {index}"
        )

        print_chart(
            chart
        )


    # ========================================================
    # STEP 5 — TEST SQL RESULT VISUALIZATION
    # ========================================================

    print(
        "\n[5/5] Testing SQL result "
        "visualization..."
    )

    sql_result = pd.DataFrame({

        "product": [
            "Laptop",
            "Monitor",
            "Keyboard",
            "Mouse"
        ],

        "average_price": [
            57666.67,
            7000.00,
            1450.00,
            216.67
        ]
    })

    question = (
        "What is the average price "
        "for each product?"
    )

    result_chart = (
        visualization_service
        .generate_result_chart(
            df=sql_result,
            question=question
        )
    )

    print(
        "\nSQL RESULT CHART"
    )

    if result_chart:

        print_chart(
            result_chart
        )

    else:

        print(
            "No chart was generated."
        )


    # ========================================================
    # TEST SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("                  TEST SUMMARY")
    print("=" * 60)

    if (
        charts
        and
        result_chart is not None
    ):

        print(
            "\nVisualizationService "
            "worked successfully."
        )

        print(
            "\nEDA charts were generated."
        )

        print(
            "SQL result chart was generated."
        )

    else:

        print(
            "\nVisualizationService test failed."
        )


if __name__ == "__main__":
    main()