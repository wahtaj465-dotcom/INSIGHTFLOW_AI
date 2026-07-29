import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


class LLMService:
    """
    Central LLM service for InsightFlow AI.

    Responsibilities:
    - Connect to Gemini
    - Generate text responses
    - Normalize LangChain/Gemini output
    - Detect quota/rate-limit failures
    - Expose LLM availability state

    Agents should use this service instead of connecting
    directly to Gemini.
    """

    def __init__(
        self,
        model_name="gemini-3.6-flash"
    ):

        self.model_name = model_name

        # Used to avoid repeatedly calling Gemini after
        # a quota/rate-limit failure during this process.
        self.available = True

        self.last_error = None

        # ====================================================
        # GET API KEY
        # ====================================================

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GOOGLE_API_KEY was not found. "
                "Check your .env file."
            )

        # ====================================================
        # CREATE GEMINI CLIENT
        # ====================================================

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key
        )


    # ========================================================
    # NORMALIZE RESPONSE CONTENT
    # ========================================================

    @staticmethod
    def _extract_text(content):
        """
        Convert Gemini/LangChain response content into
        a plain string.
        """

        if isinstance(
            content,
            str
        ):

            return content


        if isinstance(
            content,
            list
        ):

            text_parts = []

            for block in content:

                if isinstance(
                    block,
                    str
                ):

                    text_parts.append(
                        block
                    )

                    continue


                if isinstance(
                    block,
                    dict
                ):

                    text = block.get(
                        "text"
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )

            return "\n".join(
                text_parts
            )


        return str(
            content
        )


    # ========================================================
    # CHECK QUOTA/RATE LIMIT ERROR
    # ========================================================

    @staticmethod
    def is_quota_error(error):
        """
        Detect Gemini quota/rate-limit errors.

        Examples:
        - RESOURCE_EXHAUSTED
        - HTTP 429
        - quota exceeded
        - rate limit
        """

        error_text = str(
            error
        ).lower()

        indicators = [

            "resource_exhausted",

            "quota exceeded",

            "quota_exceeded",

            "rate limit",

            "rate_limit",

            "429",

            "too many requests",

            "generativelanguage.googleapis.com/"
            "generate_content_free_tier_requests"
        ]

        return any(
            indicator in error_text
            for indicator in indicators
        )


    # ========================================================
    # EXTRACT RETRY DELAY
    # ========================================================

    @staticmethod
    def extract_retry_delay(error):
        """
        Try to extract Gemini's suggested retry delay.

        Example:

            retryDelay: '56s'

        Returns:
            56

        or:
            None
        """

        text = str(
            error
        )

        patterns = [

            r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s",

            r"retry in\s+([\d.]+)s",

            r"retry after\s+([\d.]+)s"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                try:

                    return int(
                        float(
                            match.group(1)
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

        return None


    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate(
        self,
        prompt
    ):
        """
        Send a prompt to Gemini.

        Raises RuntimeError with a predictable message
        when Gemini quota is unavailable.
        """

        if not isinstance(
            prompt,
            str
        ):

            raise TypeError(
                "Prompt must be a string."
            )


        prompt = prompt.strip()


        if not prompt:

            raise ValueError(
                "Prompt cannot be empty."
            )


        # ----------------------------------------------------
        # If quota already failed during this application
        # process, do not repeatedly hit Gemini.
        # ----------------------------------------------------

        if not self.available:

            raise RuntimeError(
                "LLM_UNAVAILABLE: Gemini is temporarily "
                "unavailable because a previous request "
                "hit a quota or rate limit."
            )


        try:

            response = (
                self.llm.invoke(
                    prompt
                )
            )


            content = (
                response.content
            )


            text = (
                self._extract_text(
                    content
                )
            )


            if not text.strip():

                raise RuntimeError(
                    "Gemini returned an empty response."
                )


            self.last_error = None


            return text.strip()


        except Exception as error:

            self.last_error = str(
                error
            )


            # =================================================
            # QUOTA / RATE LIMIT
            # =================================================

            if self.is_quota_error(
                error
            ):

                self.available = False

                retry_delay = (
                    self.extract_retry_delay(
                        error
                    )
                )


                message = (
                    "LLM_QUOTA_EXCEEDED: "
                    f"Gemini model '{self.model_name}' "
                    "is temporarily unavailable because "
                    "its API quota/rate limit was reached."
                )


                if retry_delay is not None:

                    message += (
                        f" Suggested retry delay: "
                        f"{retry_delay} seconds."
                    )


                raise RuntimeError(
                    message
                ) from error


            # =================================================
            # OTHER LLM ERROR
            # =================================================

            raise RuntimeError(
                "LLM_REQUEST_FAILED: "
                f"Error calling model "
                f"'{self.model_name}': "
                f"{error}"
            ) from error


    # ========================================================
    # RESET AVAILABILITY
    # ========================================================

    def reset_availability(
        self
    ):
        """
        Allow Gemini requests again.

        Useful after the quota/rate-limit window resets.
        """

        self.available = True
        self.last_error = None


    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self
    ):

        return {

            "model":
                self.model_name,

            "available":
                self.available,

            "last_error":
                self.last_error
        }