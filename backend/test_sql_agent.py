import pandas as pd

from services.sql_engine import SQLEngine
from services.llm_service import LLMService
from agents.sql_agent import SQLAgent


def main():

    print("\n==============================")
    print("         SQL AGENT TEST")
    print("==============================")

    # ========================================================
    # LOAD CLEANED DATA
    # ========================================================

    df = pd.read_csv(
        "data/sales_processed.csv"
    )

    # ========================================================
    # CREATE SQL ENGINE
    # ========================================================

    sql_engine = SQLEngine()

    sql_engine.register_dataframe(
        df,
        table_name="sales"
    )

    # ========================================================
    # CREATE LLM SERVICE
    # ========================================================

    llm_service = LLMService()

    # ========================================================
    # CREATE SQL AGENT
    # ========================================================

    sql_agent = SQLAgent(
        llm_service=llm_service,
        sql_engine=sql_engine,
        table_name="sales"
    )

    # ========================================================
    # USER QUESTION
    # ========================================================

    question = (
        
        "What is the average price for each product?"
        
        
    )

    print("\nUSER QUESTION:")
    print(question)

    print("\nGenerating SQL using Gemini...")

    # ========================================================
    # ASK AGENT
    # ========================================================

    response = sql_agent.ask(
        question
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n==============================")
    print("       GENERATED SQL")
    print("==============================")

    print(response["sql"])

    if response["success"]:

        print("\n==============================")
        print("         SQL RESULT")
        print("==============================")

        print(response["result"])

    else:

        print("\n==============================")
        print("           ERROR")
        print("==============================")

        print(response["error"])

    # ========================================================
    # CLOSE DATABASE
    # ========================================================

    sql_engine.close()


if __name__ == "__main__":
    main()