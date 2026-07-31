import pandas as pd

from services.sql_engine import SQLEngine
from services.llm_service import LLMService

from agents.sql_agent import SQLAgent
from agents.insight_agent import InsightAgent

from agents.schema_agent import (
    analyze_schema
)

from agents.cleaning_agent import (
    generate_quality_report,
    detect_numeric_anomalies
)

from agents.eda_agent import (
    perform_eda
)


def main():

    print("\n==============================")
    print("      INSIGHT AGENT V3 TEST")
    print("==============================")


    # ========================================================
    # STEP 1 — LOAD DATASET
    # ========================================================

    df = pd.read_csv(
        "data/sales_processed.csv"
    )

    print(
        f"\nDataset loaded: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )


    # ========================================================
    # STEP 2 — SCHEMA
    # ========================================================

    schema = analyze_schema(
        df
    )


    # ========================================================
    # STEP 3 — QUALITY REPORT
    # ========================================================

    quality_report = (
        generate_quality_report(
            df,
            schema
        )
    )


    # ========================================================
    # STEP 4 — ANOMALY DETECTION
    # ========================================================

    anomalies = (
        detect_numeric_anomalies(
            df,
            schema
        )
    )


    # ========================================================
    # STEP 5 — AUTOMATED EDA
    # ========================================================

    eda_results = perform_eda(
        df,
        schema
    )

    print("\nEDA generated successfully.")


    # ========================================================
    # STEP 6 — SQL ENGINE
    # ========================================================

    sql_engine = SQLEngine()

    sql_engine.register_dataframe(
        df,
        table_name="sales"
    )


    # ========================================================
    # STEP 7 — LLM
    # ========================================================

    llm_service = LLMService()


    # ========================================================
    # STEP 8 — AGENTS
    # ========================================================

    sql_agent = SQLAgent(
        llm_service=llm_service,
        sql_engine=sql_engine,
        table_name="sales"
    )

    insight_agent = InsightAgent(
        llm_service=llm_service
    )


    # ========================================================
    # STEP 9 — QUESTION
    # ========================================================

    question = (
        "What is the average price for each product?"
    )

    print("\n==============================")
    print("         USER QUESTION")
    print("==============================")

    print(question)


    # ========================================================
    # STEP 10 — SQL AGENT
    # ========================================================

    print("\nRunning SQL Agent...")

    sql_response = sql_agent.ask(
        question
    )

    if not sql_response["success"]:

        print("\nSQL Agent failed:")

        print(
            sql_response["error"]
        )

        sql_engine.close()

        return


    # ========================================================
    # STEP 11 — SQL
    # ========================================================

    print("\n==============================")
    print("        GENERATED SQL")
    print("==============================")

    print(
        sql_response["sql"]
    )


    # ========================================================
    # STEP 12 — RESULT
    # ========================================================

    print("\n==============================")
    print("          SQL RESULT")
    print("==============================")

    print(
        sql_response["result"]
    )


    # ========================================================
    # STEP 13 — INSIGHT AGENT
    # ========================================================

    print(
        "\nGenerating EDA-aware insight..."
    )

    insight_response = (
        insight_agent.analyze(
            sql_response=sql_response,
            quality_report=quality_report,
            anomalies=anomalies,
            eda_results=eda_results
        )
    )


    # ========================================================
    # STEP 14 — RELEVANT COLUMNS
    # ========================================================

    print("\n==============================")
    print("       RELEVANT COLUMNS")
    print("==============================")

    print(
        insight_response.get(
            "relevant_columns",
            []
        )
    )


    # ========================================================
    # STEP 15 — FINAL INSIGHT
    # ========================================================

    print("\n==============================")
    print("       GENERATED INSIGHT")
    print("==============================")

    if insight_response["success"]:

        print(
            insight_response["insight"]
        )

    else:

        print(
            "Insight generation failed:"
        )

        print(
            insight_response["error"]
        )


    # ========================================================
    # STEP 16 — CLOSE DATABASE
    # ========================================================

    sql_engine.close()


if __name__ == "__main__":
    main()