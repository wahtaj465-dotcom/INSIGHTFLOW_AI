import os
import re
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


class LLMService:
    """
    Central LLM service for InsightFlow AI.

    Provider priority:

        1. Groq  -> Primary
        2. Gemini -> Backup

    Responsibilities:
    - Connect to Groq
    - Connect to Gemini
    - Automatically fail over from Groq to Gemini
    - Generate text responses
    - Normalize LangChain response output
    - Detect quota/rate-limit/temporary provider failures
    - Track provider availability
    - Apply provider cooldowns
    - Expose LLM status

    Agents should use this service instead of connecting
    directly to Groq or Gemini.
    """

    def __init__(
        self,
        model_name=None
    ):

        # ====================================================
        # MODEL CONFIGURATION
        # ====================================================

        # Gemini model
        #
        # model_name is kept for backward compatibility.
        # If supplied, it overrides GEMINI_MODEL.
        self.gemini_model_name = (
            model_name
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.6-flash"
            )
        )

        # Groq model
        self.groq_model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.8-27b"
        )

        # ====================================================
        # PROVIDER STATE
        # ====================================================

        # ----------------------------------------------------
        # Groq = PRIMARY
        # ----------------------------------------------------

        self.groq_available = True

        # ----------------------------------------------------
        # Gemini = BACKUP
        # ----------------------------------------------------

        self.gemini_available = True

        # Provider that successfully answered
        # the most recent request.
        self.last_provider = None

        # Last provider error.
        self.last_error = None

        # ====================================================
        # COOLDOWN STATE
        # ====================================================

        # When a provider encounters a temporary failure,
        # we temporarily stop sending requests to it.

        self.groq_unavailable_until = 0
        self.gemini_unavailable_until = 0

        # Default cooldown duration.
        #
        # If the provider itself gives us a retry delay,
        # that value will be used instead.
        self.default_cooldown_seconds = 60

        # ====================================================
        # API KEYS
        # ====================================================

        google_api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        # At least one provider must be configured.
        if not google_api_key and not groq_api_key:

            raise ValueError(
                "Neither GOOGLE_API_KEY nor GROQ_API_KEY "
                "was found. Check your .env file."
            )

        # ====================================================
        # CREATE GROQ CLIENT
        # ====================================================

        self.groq_llm = None

        if groq_api_key:

            self.groq_llm = ChatGroq(
                model=self.groq_model_name,
                groq_api_key=groq_api_key
            )

        else:

            # Groq is the primary provider.
            # If its API key is missing, mark it unavailable.
            self.groq_available = False

        # ====================================================
        # CREATE GEMINI CLIENT
        # ====================================================

        self.gemini_llm = None

        if google_api_key:

            self.gemini_llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=google_api_key
            )

        else:

            # Gemini is the backup provider.
            # If its API key is missing, mark it unavailable.
            self.gemini_available = False

    # ========================================================
    # NORMALIZE RESPONSE CONTENT
    # ========================================================

    @staticmethod
    def _extract_text(content):
        """
        Convert LangChain response content into a plain string.

        Handles:
        - str
        - list
        - dict
        - other response formats
        """

        # ----------------------------------------------------
        # STRING RESPONSE
        # ----------------------------------------------------

        if isinstance(
            content,
            str
        ):

            return content

        # ----------------------------------------------------
        # LIST RESPONSE
        # ----------------------------------------------------

        if isinstance(
            content,
            list
        ):

            text_parts = []

            for block in content:

                # --------------------------------------------
                # Plain string block
                # --------------------------------------------

                if isinstance(
                    block,
                    str
                ):

                    text_parts.append(
                        block
                    )

                    continue

                # --------------------------------------------
                # Dictionary content block
                # --------------------------------------------

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

        # ----------------------------------------------------
        # OTHER RESPONSE TYPES
        # ----------------------------------------------------

        return str(
            content
        )

    # ========================================================
    # CHECK TEMPORARY / QUOTA ERROR
    # ========================================================

    @staticmethod
    def is_temporary_error(error):
        """
        Determine whether an error is appropriate for
        provider failover.

        Failover is triggered for things such as:

        - quota exhaustion
        - rate limits
        - HTTP 429
        - temporary service failures
        - HTTP 502
        - HTTP 503
        - HTTP 504
        - timeout
        - connection errors

        Normal application errors should NOT automatically
        trigger provider failover.
        """

        error_text = str(
            error
        ).lower()

        indicators = [

            # =================================================
            # QUOTA / RATE LIMIT
            # =================================================

            "resource_exhausted",

            "quota exceeded",

            "quota_exceeded",

            "rate limit",

            "rate_limit",

            "too many requests",

            "429",

            # =================================================
            # TEMPORARY SERVER ERRORS
            # =================================================

            "service unavailable",

            "temporarily unavailable",

            "bad gateway",

            "502",

            "503",

            "504",

            "internal server error",

            # =================================================
            # NETWORK / TIMEOUT
            # =================================================

            "timeout",

            "timed out",

            "connection error",

            "connection reset",

            "connection refused",

            "remote disconnected",

        ]

        return any(
            indicator in error_text
            for indicator in indicators
        )

    # ========================================================
    # BACKWARD-COMPATIBLE QUOTA CHECK
    # ========================================================

    @staticmethod
    def is_quota_error(error):
        """
        Backward-compatible quota/rate-limit detector.

        Existing code can continue calling this method.
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
        Try to extract a provider-suggested retry delay.

        Examples:

            retryDelay: '56s'

            retry in 56s

            retry after 56s

        Returns:
            integer number of seconds

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
    # CHECK GROQ COOLDOWN
    # ========================================================

    def _is_groq_in_cooldown(
        self
    ):

        return (
            time.time()
            < self.groq_unavailable_until
        )

    # ========================================================
    # CHECK GEMINI COOLDOWN
    # ========================================================

    def _is_gemini_in_cooldown(
        self
    ):

        return (
            time.time()
            < self.gemini_unavailable_until
        )

    # ========================================================
    # MARK GROQ UNAVAILABLE
    # ========================================================

    def _disable_groq(
        self,
        error
    ):

        self.groq_available = False

        retry_delay = (
            self.extract_retry_delay(
                error
            )
        )

        cooldown = (
            retry_delay
            if retry_delay is not None
            else self.default_cooldown_seconds
        )

        self.groq_unavailable_until = (
            time.time()
            + cooldown
        )

    # ========================================================
    # MARK GEMINI UNAVAILABLE
    # ========================================================

    def _disable_gemini(
        self,
        error
    ):

        self.gemini_available = False

        retry_delay = (
            self.extract_retry_delay(
                error
            )
        )

        cooldown = (
            retry_delay
            if retry_delay is not None
            else self.default_cooldown_seconds
        )

        self.gemini_unavailable_until = (
            time.time()
            + cooldown
        )

    # ========================================================
    # RESTORE PROVIDER AFTER COOLDOWN
    # ========================================================

    def _refresh_provider_availability(
        self
    ):

        current_time = time.time()

        # ====================================================
        # GROQ
        # ====================================================

        if (
            not self.groq_available
            and current_time
            >= self.groq_unavailable_until
            and self.groq_llm is not None
        ):

            self.groq_available = True

        # ====================================================
        # GEMINI
        # ====================================================

        if (
            not self.gemini_available
            and current_time
            >= self.gemini_unavailable_until
            and self.gemini_llm is not None
        ):

            self.gemini_available = True

    # ========================================================
    # GENERATE USING GROQ
    # ========================================================

    def _generate_with_groq(
        self,
        prompt
    ):

        if self.groq_llm is None:

            raise RuntimeError(
                "Groq client is not configured."
            )

        response = (
            self.groq_llm.invoke(
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
                "Groq returned an empty response."
            )

        return text.strip()

    # ========================================================
    # GENERATE USING GEMINI
    # ========================================================

    def _generate_with_gemini(
        self,
        prompt
    ):

        if self.gemini_llm is None:

            raise RuntimeError(
                "Gemini client is not configured."
            )

        response = (
            self.gemini_llm.invoke(
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

        return text.strip()

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate(
        self,
        prompt
    ):
        """
        Generate a response using:

            1. Groq  -> Primary
            2. Gemini -> Backup

        Groq is attempted first.

        If Groq experiences a quota, rate-limit,
        timeout, connection, or temporary service error,
        the request automatically fails over to Gemini.

        If Gemini experiences a temporary failure as well,
        both providers are marked unavailable and an
        LLM_UNAVAILABLE error is raised.

        Normal application errors do NOT automatically
        trigger provider switching.
        """

        # ====================================================
        # VALIDATE PROMPT
        # ====================================================

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

        # ====================================================
        # REFRESH PROVIDER AVAILABILITY
        # ====================================================

        self._refresh_provider_availability()

        # ====================================================
        # TRY GROQ FIRST
        # ====================================================

        if (
            self.groq_available
            and self.groq_llm is not None
            and not self._is_groq_in_cooldown()
        ):

            try:

                text = (
                    self._generate_with_groq(
                        prompt
                    )
                )

                # --------------------------------------------
                # GROQ SUCCESS
                # --------------------------------------------

                self.last_provider = "groq"
                self.last_error = None

                return text

            except Exception as error:

                self.last_error = str(
                    error
                )

                # --------------------------------------------
                # TEMPORARY GROQ FAILURE
                # --------------------------------------------

                if self.is_temporary_error(
                    error
                ):

                    self._disable_groq(
                        error
                    )

                else:

                    # ----------------------------------------
                    # NORMAL GROQ ERROR
                    #
                    # Do not blindly switch providers for
                    # programming/application errors.
                    # ----------------------------------------

                    raise RuntimeError(
                        "LLM_REQUEST_FAILED: "
                        f"Error calling Groq model "
                        f"'{self.groq_model_name}': "
                        f"{error}"
                    ) from error

        # ====================================================
        # FAILOVER TO GEMINI
        # ====================================================

        if (
            self.gemini_available
            and self.gemini_llm is not None
            and not self._is_gemini_in_cooldown()
        ):

            try:

                text = (
                    self._generate_with_gemini(
                        prompt
                    )
                )

                # --------------------------------------------
                # GEMINI SUCCESS
                # --------------------------------------------

                self.last_provider = "gemini"
                self.last_error = None

                return text

            except Exception as error:

                self.last_error = str(
                    error
                )

                # --------------------------------------------
                # TEMPORARY GEMINI FAILURE
                # --------------------------------------------

                if self.is_temporary_error(
                    error
                ):

                    self._disable_gemini(
                        error
                    )

                else:

                    # ----------------------------------------
                    # NORMAL GEMINI ERROR
                    # ----------------------------------------

                    raise RuntimeError(
                        "LLM_REQUEST_FAILED: "
                        f"Error calling Gemini model "
                        f"'{self.gemini_model_name}': "
                        f"{error}"
                    ) from error

        # ====================================================
        # BOTH PROVIDERS UNAVAILABLE
        # ====================================================

        raise RuntimeError(
            "LLM_UNAVAILABLE: "
            "Both Groq and Gemini are currently unavailable. "
            f"Last provider error: {self.last_error}"
        )

    # ========================================================
    # RESET AVAILABILITY
    # ========================================================

    def reset_availability(
        self
    ):
        """
        Allow both providers to be used again.

        Useful after quota/rate-limit windows reset.
        """

        self.groq_available = (
            self.groq_llm is not None
        )

        self.gemini_available = (
            self.gemini_llm is not None
        )

        self.groq_unavailable_until = 0

        self.gemini_unavailable_until = 0

        self.last_error = None

        self.last_provider = None

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self
    ):
        """
        Return current provider status.
        """

        self._refresh_provider_availability()

        return {

            # ------------------------------------------------
            # PROVIDER PRIORITY
            # ------------------------------------------------

            "primary_provider":
                "groq",

            "backup_provider":
                "gemini",

            # ------------------------------------------------
            # LAST ACTIVE PROVIDER
            # ------------------------------------------------

            "active_provider":
                self.last_provider,

            # ------------------------------------------------
            # MODELS
            # ------------------------------------------------

            "groq_model":
                self.groq_model_name,

            "gemini_model":
                self.gemini_model_name,

            # ------------------------------------------------
            # AVAILABILITY
            # ------------------------------------------------

            "groq_available":
                self.groq_available,

            "gemini_available":
                self.gemini_available,

            # ------------------------------------------------
            # LAST ERROR
            # ------------------------------------------------

            "last_error":
                self.last_error
        }