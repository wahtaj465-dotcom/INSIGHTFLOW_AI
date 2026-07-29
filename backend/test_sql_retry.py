import pandas as pd

from services.sql_engine import SQLEngine
from services.llm_service import LLMService
from agents.sql_agent import SQLAgent


class RetryTestSQLAgent(SQLAgent):
    """
    Testing-only SQL Agent.

    It intentionally generates incorrect SQL on the first
    attempt so we can verify the correction mechanism.
    """

    def generate_sql(
        self,
        question,
        schema_context=None
    ):
        # Intentionally wrong column.
        # The real column is "price".
        return """
        SELECT
            product,
            AVG(revenue) AS average_revenue
        FROM sales
        GROUP BY product;
        """


def main():

    print("\n==============================")
    print("      SQL RETRY TEST")
    print("==============================")

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        "data/sales_processed.csv"
    )

    # ========================================================
    # SQL ENGINE
    # ========================================================

    sql_engine = SQLEngine()

    sql_engine.register_dataframe(
        df,
        table_name="sales"
    )

    # ========================================================
    # LLM
    # ========================================================

    llm_service = LLMService()

    # ========================================================
    # TEST AGENT
    # ========================================================

    sql_agent = RetryTestSQLAgent(
        llm_service=llm_service,
        sql_engine=sql_engine,
        table_name="sales",
        max_retries=2
    )

    question = (
        "What is the average price for each product?"
    )

    print("\nUSER QUESTION:")
    print(question)

    print(
        "\nFirst SQL will intentionally contain "
        "the wrong column 'revenue'."
    )

    # ========================================================
    # RUN AGENT
    # ========================================================

    response = sql_agent.ask(
        question
    )

    # ========================================================
    # ATTEMPT HISTORY
    # ========================================================

    print("\n==============================")
    print("        ATTEMPT HISTORY")
    print("==============================")

    for attempt in response["attempts"]:

        print(
            f"\nAttempt {attempt['attempt']}"
        )

        print(
            f"Stage: {attempt['stage']}"
        )

        print("SQL:")
        print(attempt["sql"])

        if attempt["error"]:

            print("Error:")
            print(attempt["error"])

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n==============================")
    print("          FINAL SQL")
    print("==============================")

    print(response["sql"])

    if response["success"]:

        print("\n==============================")
        print("         FINAL RESULT")
        print("==============================")

        print(response["result"])

        print(
            "\nRetry mechanism worked successfully."
        )

    else:

        print("\n==============================")
        print("            FAILED")
        print("==============================")

        print(response["error"])

    sql_engine.close()


if __name__ == "__main__":
    main()