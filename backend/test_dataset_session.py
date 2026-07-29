from backend.workflows.analytics_workflow import AnalyticsWorkflow


def print_question_result(number, question, result):
    """
    Pretty-print the result of one dataset question.
    """

    print("\n" + "=" * 60)
    print(f"              QUESTION {number}")
    print("=" * 60)

    print(f"\nUser Question:\n{question}")

    print("\nSuccess:")
    print(result.get("success"))

    if not result.get("success"):

        print("\nError:")
        print(result.get("error"))

        print("\nGenerated SQL:")
        print(result.get("generated_sql"))

        print("\nSQL Attempts:")
        print(result.get("sql_attempts"))

        return

    print("\nGenerated SQL:")
    print(result.get("generated_sql"))

    print("\nSQL Result:")

    sql_result = result.get("sql_result")

    if sql_result is not None:
        print(sql_result.to_string(index=False))
    else:
        print("No SQL result.")

    print("\nRelevant Columns:")
    print(result.get("relevant_columns"))

    print("\nGenerated Insight:")
    print(result.get("insight"))


def main():

    print("\n" + "=" * 60)
    print("       INSIGHTFLOW DATASET SESSION TEST")
    print("=" * 60)

    workflow = AnalyticsWorkflow()

    try:

        # ====================================================
        # STEP 1 — PREPARE DATASET
        # ====================================================

        print("\nPreparing dataset...")

        preparation = workflow.prepare_dataset(
            file_path="data/sales.csv",
            original_filename="sales.csv"
        )

        print("\n" + "=" * 60)
        print("          DATASET PREPARATION RESULT")
        print("=" * 60)

        print("\nSuccess:")
        print(preparation.get("success"))

        if not preparation.get("success"):

            print("\nPreparation Error:")
            print(preparation.get("error"))

            return

        dataset_id = preparation.get(
            "dataset_id"
        )

        print("\nDataset ID:")
        print(dataset_id)

        print("\nOriginal Filename:")
        print(
            preparation.get(
                "original_filename"
            )
        )

        print("\nOriginal Rows:")
        print(
            preparation.get(
                "original_rows"
            )
        )

        print("\nCleaned Rows:")
        print(
            preparation.get(
                "cleaned_rows"
            )
        )

        print("\nColumns:")
        print(
            preparation.get(
                "columns"
            )
        )

        print("\nQuality Score:")
        print(
            "Before:",
            preparation.get(
                "before_score"
            )
        )

        print(
            "After :",
            preparation.get(
                "after_score"
            )
        )

        print("\nCleaning Log:")

        for index, item in enumerate(
            preparation.get(
                "cleaning_log",
                []
            ),
            start=1
        ):

            print(
                f"{index}. {item}"
            )


        # ====================================================
        # STEP 2 — GET DATASET INFORMATION
        # ====================================================

        print("\n" + "=" * 60)
        print("              DATASET INFO")
        print("=" * 60)

        dataset_info = (
            workflow.get_dataset_info(
                dataset_id
            )
        )

        print(dataset_info)


        # ====================================================
        # STEP 3 — QUESTION 1
        # ====================================================

        question1 = (
            "What is the average price "
            "for each product?"
        )

        result1 = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question1
        )

        print_question_result(
            1,
            question1,
            result1
        )


        # ====================================================
        # STEP 4 — QUESTION 2
        # ====================================================

        question2 = (
            "How many orders are there "
            "in each region?"
        )

        result2 = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question2
        )

        print_question_result(
            2,
            question2,
            result2
        )


        # ====================================================
        # STEP 5 — QUESTION 3
        # ====================================================

        question3 = (
            "Which product has the highest "
            "average price?"
        )

        result3 = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question3
        )

        print_question_result(
            3,
            question3,
            result3
        )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        print("\n" + "=" * 60)
        print("               TEST SUMMARY")
        print("=" * 60)

        all_successful = all([
            preparation.get(
                "success",
                False
            ),

            result1.get(
                "success",
                False
            ),

            result2.get(
                "success",
                False
            ),

            result3.get(
                "success",
                False
            )
        ])

        if all_successful:

            print(
                "\nDataset session system "
                "worked successfully."
            )

            print(
                "\nThe dataset was prepared ONCE "
                "and queried MULTIPLE TIMES."
            )

        else:

            print(
                "\nOne or more tests failed."
            )


        # ====================================================
        # IMPORTANT
        # Do NOT delete the dataset yet.
        #
        # We'll test delete_dataset() separately after
        # confirming repeated questions work correctly.
        # ====================================================


    finally:

        workflow.close()


if __name__ == "__main__":
    main()