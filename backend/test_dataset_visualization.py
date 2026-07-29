from backend.workflows.analytics_workflow import (
    AnalyticsWorkflow
)


# ============================================================
# PRINT CHART
# ============================================================

def print_chart(
    chart
):

    if chart is None:

        print(
            "No visualization generated."
        )

        return


    print(
        "Chart Type:",
        chart.get(
            "chart_type"
        )
    )

    print(
        "Title:",
        chart.get(
            "title"
        )
    )

    print(
        "X:",
        chart.get(
            "x"
        )
    )

    print(
        "Y:",
        chart.get(
            "y"
        )
    )

    print(
        "Reason:",
        chart.get(
            "reason"
        )
    )

    print(
        "Data:",
        chart.get(
            "data"
        )
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "    INSIGHTFLOW DATASET + VISUALIZATION TEST"
    )

    print(
        "=" * 60
    )


    workflow = (
        AnalyticsWorkflow()
    )


    try:

        # ====================================================
        # STEP 1 — PREPARE DATASET
        # ====================================================

        print(
            "\n[1/4] Preparing dataset..."
        )


        preparation = (
            workflow.prepare_dataset(

                file_path=
                    "data/sales.csv",

                original_filename=
                    "sales.csv"
            )
        )


        print(
            "\nPreparation Success:"
        )

        print(
            preparation.get(
                "success"
            )
        )


        if not preparation.get(
            "success",
            False
        ):

            print(
                "\nError:"
            )

            print(
                preparation.get(
                    "error"
                )
            )

            return


        dataset_id = (
            preparation.get(
                "dataset_id"
            )
        )


        print(
            "\nDataset ID:"
        )

        print(
            dataset_id
        )


        # ====================================================
        # STEP 2 — TEST EDA CHARTS
        # ====================================================

        print(
            "\n[2/4] Checking EDA charts..."
        )


        eda_charts = (
            preparation.get(
                "eda_charts",
                []
            )
        )


        print(
            "\nGenerated EDA Charts:",
            len(
                eda_charts
            )
        )


        for index, chart in enumerate(
            eda_charts,
            start=1
        ):

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"EDA CHART {index}"
            )

            print(
                "-" * 60
            )

            print_chart(
                chart
            )


        # ====================================================
        # STEP 3 — CHECK STORED DATASET
        # ====================================================

        print(
            "\n[3/4] Checking stored dataset metadata..."
        )


        dataset_info = (
            workflow.get_dataset_info(
                dataset_id
            )
        )


        stored_charts = (
            dataset_info.get(
                "eda_charts",
                []
            )
        )


        print(
            "\nCharts stored in DatasetManager:",
            len(
                stored_charts
            )
        )


        # ====================================================
        # STEP 4 — ASK QUESTION
        # ====================================================

        print(
            "\n[4/4] Asking analytical question..."
        )


        question = (
            "What is the average price "
            "for each product?"
        )


        result = (
            workflow.ask_dataset(

                dataset_id=
                    dataset_id,

                question=
                    question
            )
        )


        print(
            "\nQuestion:"
        )

        print(
            question
        )


        print(
            "\nSuccess:"
        )

        print(
            result.get(
                "success"
            )
        )


        if not result.get(
            "success",
            False
        ):

            print(
                "\nError:"
            )

            print(
                result.get(
                    "error"
                )
            )

            return


        print(
            "\nGenerated SQL:"
        )

        print(
            result.get(
                "generated_sql"
            )
        )


        print(
            "\nSQL Result:"
        )


        sql_result = (
            result.get(
                "sql_result"
            )
        )


        if sql_result is not None:

            print(
                sql_result.to_string(
                    index=False
                )
            )


        print(
            "\n"
            + "=" * 60
        )

        print(
            "QUESTION VISUALIZATION"
        )

        print(
            "=" * 60
        )


        result_chart = (
            result.get(
                "result_chart"
            )
        )


        print_chart(
            result_chart
        )


        print(
            "\n"
            + "=" * 60
        )

        print(
            "GENERATED INSIGHT"
        )

        print(
            "=" * 60
        )


        print(
            result.get(
                "insight"
            )
        )


        # ====================================================
        # FINAL CHECK
        # ====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "TEST SUMMARY"
        )

        print(
            "=" * 60
        )


        if (
            eda_charts
            and
            stored_charts
            and
            result_chart is not None
        ):

            print(
                "\nVisualization integration "
                "worked successfully."
            )

            print(
                "\nDataset prepared ONCE."
            )

            print(
                "EDA charts generated ONCE."
            )

            print(
                "EDA charts stored in DatasetManager."
            )

            print(
                "SQL question executed."
            )

            print(
                "Question-specific chart generated."
            )

            print(
                "AI insight generated."
            )

        else:

            print(
                "\nVisualization integration "
                "is incomplete."
            )


    finally:

        workflow.close()


if __name__ == "__main__":

    main()