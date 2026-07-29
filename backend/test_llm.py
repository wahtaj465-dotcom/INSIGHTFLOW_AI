from services.llm_service import LLMService


def main():

    print("\n==============================")
    print("       LLM CONNECTION TEST")
    print("==============================")

    # ========================================================
    # CREATE LLM SERVICE
    # ========================================================

    llm_service = LLMService()

    # ========================================================
    # TEST GEMINI
    # ========================================================

    prompt = """
Reply with exactly:

InsightFlow LLM connected
"""

    print("\nSending request to Gemini...")

    try:

        response = llm_service.generate(
            prompt
        )

        print("\nLLM RESPONSE:")
        print(response)

    except Exception as error:

        print("\nLLM CONNECTION FAILED")
        print(error)


if __name__ == "__main__":
    main()