from workflows.analytics_workflow import AnalyticsWorkflow


def main():

    print("\n" + "=" * 60)
    print("          INSIGHTFLOW LANGGRAPH TEST")
    print("=" * 60)

    # ========================================================
    # CREATE WORKFLOW
    # ========================================================

    workflow = AnalyticsWorkflow(
        table_name="sales",
        max_sql_retries=2
    )

    try:

        # ====================================================
        # TEST INPUT
        # ====================================================

        file_path = (
            "data/sales.csv"
        )

        question = (
            "What is the average price "
            "for each product?"
        )

        print("\nDataset:")
        print(file_path)

        print("\nUser Question:")
        print(question)

        print(
            "\nStarting InsightFlow workflow..."
        )

        # ====================================================
        # RUN LANGGRAPH
        # ====================================================

        result = workflow.run(
            file_path=file_path,
            question=question
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        print("\n")
        print("=" * 60)
        print("             WORKFLOW RESULT")
        print("=" * 60)

        print(
            "\nSuccess:",
            result.get("success")
        )

        if result.get("error"):

            print(
                "\nError:"
            )

            print(
                result.get("error")
            )

        # ====================================================
        # QUALITY SCORES
        # ====================================================

        print("\n")
        print("=" * 60)
        print("           DATA QUALITY")
        print("=" * 60)

        before_score = result.get(
            "before_score"
        )

        after_score = result.get(
            "after_score"
        )

        print(
            "\nBefore Cleaning:",
            before_score
        )

        print(
            "After Cleaning:",
            after_score
        )

        if (
            before_score is not None
            and after_score is not None
        ):

            improvement = round(
                after_score - before_score,
                2
            )

            print(
                "Improvement:",
                improvement
            )

        # ====================================================
        # CLEANING LOG
        # ====================================================

        print("\n")
        print("=" * 60)
        print("            CLEANING LOG")
        print("=" * 60)

        cleaning_log = result.get(
            "cleaning_log",
            []
        )

        if cleaning_log:

            for index, operation in enumerate(
                cleaning_log,
                start=1
            ):

                print(
                    f"{index}. {operation}"
                )

        else:

            print(
                "No cleaning operations."
            )

        # ====================================================
        # SQL
        # ====================================================

        print("\n")
        print("=" * 60)
        print("            GENERATED SQL")
        print("=" * 60)

        generated_sql = result.get(
            "generated_sql"
        )

        if generated_sql:

            print(
                "\n" + generated_sql
            )

        else:

            print(
                "\nNo SQL generated."
            )

        # ====================================================
        # SQL ATTEMPTS
        # ====================================================

        print("\n")
        print("=" * 60)
        print("           SQL ATTEMPTS")
        print("=" * 60)

        attempts = result.get(
            "sql_attempts",
            []
        )

        if attempts:

            for attempt in attempts:

                print(
                    f"\nAttempt "
                    f"{attempt.get('attempt')}"
                )

                print(
                    "Stage:",
                    attempt.get("stage")
                )

                print(
                    "SQL:"
                )

                print(
                    attempt.get("sql")
                )

                if attempt.get("error"):

                    print(
                        "Error:"
                    )

                    print(
                        attempt.get("error")
                    )

        else:

            print(
                "\nNo SQL attempt information."
            )

        # ====================================================
        # SQL RESULT
        # ====================================================

        print("\n")
        print("=" * 60)
        print("             SQL RESULT")
        print("=" * 60)

        sql_result = result.get(
            "sql_result"
        )

        if sql_result is not None:

            print(
                "\n",
                sql_result.to_string(
                    index=False
                )
            )

        else:

            print(
                "\nNo SQL result."
            )

        # ====================================================
        # RELEVANT COLUMNS
        # ====================================================

        print("\n")
        print("=" * 60)
        print("          RELEVANT COLUMNS")
        print("=" * 60)

        relevant_columns = result.get(
            "relevant_columns",
            []
        )

        print(
            "\n",
            relevant_columns
        )

        # ====================================================
        # FINAL INSIGHT
        # ====================================================

        print("\n")
        print("=" * 60)
        print("           FINAL INSIGHT")
        print("=" * 60)

        insight = result.get(
            "insight"
        )

        if insight:

            print(
                "\n" + insight
            )

        else:

            print(
                "\nNo insight generated."
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print("\n")
        print("=" * 60)
        print("          WORKFLOW SUMMARY")
        print("=" * 60)

        if result.get("success"):

            print(
                "\nInsightFlow LangGraph workflow "
                "completed successfully."
            )

        else:

            print(
                "\nInsightFlow workflow failed."
            )

            print(
                "Reason:",
                result.get("error")
            )

    finally:

        # Always close DuckDB
        workflow.close()


if __name__ == "__main__":
    main()