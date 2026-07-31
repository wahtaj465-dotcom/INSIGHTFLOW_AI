import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000/api"

DEFAULT_TIMEOUT = 60
ANALYSIS_TIMEOUT = 180


class InsightFlowAPI:
    """
    HTTP client used by the InsightFlow frontend.

    This class is the frontend boundary to the FastAPI backend.

    Responsibilities:
    - backend health checks
    - dataset preparation/upload
    - dataset retrieval
    - agentic analytical questions
    - one-shot upload + analysis
    - dataset deletion
    - consistent API error handling
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
    ):
        self.base_url = base_url.rstrip("/")

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    @staticmethod
    def _validate_dataset_id(
        dataset_id,
    ) -> str:
        """
        Validate and normalize a dataset ID.
        """

        if not isinstance(
            dataset_id,
            str,
        ):
            raise ValueError(
                "dataset_id must be a string."
            )

        dataset_id = dataset_id.strip()

        if not dataset_id:
            raise ValueError(
                "dataset_id cannot be empty."
            )

        return dataset_id

    @staticmethod
    def _validate_question(
        question,
    ) -> str:
        """
        Validate and normalize a user question.
        """

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        return question

    @staticmethod
    def _prepare_file(
        uploaded_file,
    ):
        """
        Convert a Streamlit-style uploaded file into the
        multipart format expected by requests.

        Expected uploaded_file interface:

            uploaded_file.name
            uploaded_file.type
            uploaded_file.getvalue()
        """

        if uploaded_file is None:
            raise ValueError(
                "uploaded_file cannot be None."
            )

        filename = getattr(
            uploaded_file,
            "name",
            "dataset.csv",
        )

        content_type = (
            getattr(
                uploaded_file,
                "type",
                None,
            )
            or
            "application/octet-stream"
        )

        try:
            file_content = (
                uploaded_file.getvalue()
            )

        except AttributeError as exc:
            raise ValueError(
                "uploaded_file must provide getvalue()."
            ) from exc

        if not file_content:
            raise ValueError(
                "Uploaded file is empty."
            )

        return {
            "file": (
                filename,
                file_content,
                content_type,
            )
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ):
        """
        Check whether the analytics API is available.

        This endpoint targets:

            GET /api/health

        Health checks intentionally return a dictionary
        instead of raising when the backend is unreachable,
        allowing the UI to show backend status cleanly.
        """

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10,
            )

            response.raise_for_status()

            payload = response.json()

            if isinstance(
                payload,
                dict,
            ):
                return payload

            return {
                "status": "healthy",
                "response": payload,
            }

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            return {
                "status": "error",
                "error": str(error),
            }

    # ========================================================
    # PREPARE / UPLOAD DATASET
    # ========================================================

    def upload_dataset(
        self,
        uploaded_file,
    ):
        """
        Upload and prepare a CSV/XLS/XLSX dataset.

        Endpoint:

            POST /api/datasets

        Returns the COMPLETE API response so app.py can retain
        both dataset_id and preparation results.
        """

        files = self._prepare_file(
            uploaded_file
        )

        try:
            response = requests.post(
                f"{self.base_url}/datasets",
                files=files,
                timeout=ANALYSIS_TIMEOUT,
            )

            payload = self._handle_response(
                response
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Invalid dataset upload response "
                    "received from backend."
                )

            return payload

        except requests.RequestException as error:
            raise RuntimeError(
                "Could not connect to InsightFlow API: "
                f"{error}"
            ) from error

    # ========================================================
    # ALIAS: PREPARE DATASET
    # ========================================================

    def prepare_dataset(
        self,
        uploaded_file,
    ):
        """
        Semantic alias for upload_dataset().

        The backend /datasets endpoint performs dataset
        preparation, so both names represent the same action.

        Keeping upload_dataset preserves compatibility with
        the existing frontend.
        """

        return self.upload_dataset(
            uploaded_file
        )

    # ========================================================
    # GET DATASET
    # ========================================================

    def get_dataset(
        self,
        dataset_id,
    ):
        """
        Retrieve prepared dataset information.

        Endpoint:

            GET /api/datasets/{dataset_id}

        The backend normally returns:

            {
                "success": True,
                "dataset": {...}
            }

        Frontend components generally need the inner dataset
        object, so this method unwraps it.
        """

        dataset_id = (
            self._validate_dataset_id(
                dataset_id
            )
        )

        try:
            response = requests.get(
                (
                    f"{self.base_url}"
                    f"/datasets/{dataset_id}"
                ),
                timeout=DEFAULT_TIMEOUT,
            )

            payload = self._handle_response(
                response
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Invalid dataset response "
                    "received from backend."
                )

            dataset = payload.get(
                "dataset"
            )

            if isinstance(
                dataset,
                dict,
            ):
                return dataset

            # Compatibility fallback if the backend later
            # returns the dataset object directly.
            return payload

        except requests.RequestException as error:
            raise RuntimeError(
                "Could not retrieve dataset: "
                f"{error}"
            ) from error

    # ========================================================
    # ASK AGENT
    # ========================================================

    def ask_dataset(
        self,
        dataset_id,
        question,
    ):
        """
        Ask the completed InsightFlow agent graph a question
        about an already prepared dataset.

        Endpoint:

            POST /api/datasets/{dataset_id}/ask

        Flow:

            Frontend
                ↓
            FastAPI
                ↓
            AgentService
                ↓
            LangGraph Agent
                ↓
            Planner
                ↓
            Tool Execution
                ↓
            Observer / Recovery
                ↓
            Final Response

        Returns the complete backend response.
        """

        dataset_id = (
            self._validate_dataset_id(
                dataset_id
            )
        )

        question = (
            self._validate_question(
                question
            )
        )

        try:
            response = requests.post(
                (
                    f"{self.base_url}"
                    f"/datasets/{dataset_id}/ask"
                ),
                data={
                    "question": question,
                },
                timeout=ANALYSIS_TIMEOUT,
            )

            payload = self._handle_response(
                response
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Invalid agent response "
                    "received from backend."
                )

            return payload

        except requests.RequestException as error:
            raise RuntimeError(
                "Could not analyze question: "
                f"{error}"
            ) from error

    # ========================================================
    # ONE-SHOT ANALYSIS
    # ========================================================

    def analyze_dataset(
        self,
        uploaded_file,
        question,
    ):
        """
        Upload a dataset and ask an analytical question in
        one request.

        Endpoint:

            POST /api/analyze

        Useful when the frontend wants:

            upload
              ↓
            preparation
              ↓
            agent analysis

        as one operation.

        The regular upload_dataset() + ask_dataset() flow
        remains available for interactive sessions where the
        user asks multiple questions about one dataset.
        """

        question = (
            self._validate_question(
                question
            )
        )

        files = self._prepare_file(
            uploaded_file
        )

        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                files=files,
                data={
                    "question": question,
                },
                timeout=ANALYSIS_TIMEOUT,
            )

            payload = self._handle_response(
                response
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Invalid analysis response "
                    "received from backend."
                )

            return payload

        except requests.RequestException as error:
            raise RuntimeError(
                "Could not perform dataset analysis: "
                f"{error}"
            ) from error

    # ========================================================
    # DELETE DATASET
    # ========================================================

    def delete_dataset(
        self,
        dataset_id,
    ):
        """
        Delete a prepared dataset session.

        Endpoint:

            DELETE /api/datasets/{dataset_id}
        """

        dataset_id = (
            self._validate_dataset_id(
                dataset_id
            )
        )

        try:
            response = requests.delete(
                (
                    f"{self.base_url}"
                    f"/datasets/{dataset_id}"
                ),
                timeout=30,
            )

            payload = self._handle_response(
                response
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "Invalid delete response "
                    "received from backend."
                )

            return payload

        except requests.RequestException as error:
            raise RuntimeError(
                "Could not delete dataset: "
                f"{error}"
            ) from error

    # ========================================================
    # RESPONSE HANDLER
    # ========================================================

    @staticmethod
    def _handle_response(
        response,
    ):
        """
        Convert an HTTP response into Python data and produce
        readable errors for frontend components.

        FastAPI errors may look like:

            {
                "detail": "Dataset not found."
            }

        or:

            {
                "detail": {
                    "message": "...",
                    "error": "..."
                }
            }

        or validation errors where detail is a list.
        """

        # ----------------------------------------------------
        # Parse response
        # ----------------------------------------------------

        try:
            payload = response.json()

        except ValueError:
            payload = {
                "detail": (
                    response.text
                    or
                    "Backend returned a non-JSON response."
                )
            }

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        if response.ok:
            return payload

        # ----------------------------------------------------
        # Extract FastAPI detail
        # ----------------------------------------------------

        if isinstance(
            payload,
            dict,
        ):
            detail = payload.get(
                "detail",
                payload,
            )

        else:
            detail = payload

        # ----------------------------------------------------
        # Dictionary error
        # ----------------------------------------------------

        if isinstance(
            detail,
            dict,
        ):
            message = detail.get(
                "message"
            )

            error = detail.get(
                "error"
            )

            if message and error:
                error_text = (
                    f"{message} {error}"
                )

            elif message:
                error_text = str(
                    message
                )

            elif error:
                error_text = str(
                    error
                )

            else:
                error_text = str(
                    detail
                )

        # ----------------------------------------------------
        # FastAPI validation errors
        # ----------------------------------------------------

        elif isinstance(
            detail,
            list,
        ):
            messages = []

            for item in detail:

                if isinstance(
                    item,
                    dict,
                ):
                    message = item.get(
                        "msg"
                    )

                    location = item.get(
                        "loc"
                    )

                    if location:
                        location_text = ".".join(
                            str(part)
                            for part in location
                        )

                        if message:
                            messages.append(
                                f"{location_text}: "
                                f"{message}"
                            )

                        else:
                            messages.append(
                                location_text
                            )

                    elif message:
                        messages.append(
                            str(message)
                        )

                    else:
                        messages.append(
                            str(item)
                        )

                else:
                    messages.append(
                        str(item)
                    )

            error_text = "; ".join(
                messages
            )

        # ----------------------------------------------------
        # String / other error
        # ----------------------------------------------------

        else:
            error_text = str(
                detail
            )

        # ----------------------------------------------------
        # Raise frontend-friendly error
        # ----------------------------------------------------

        raise RuntimeError(
            f"API request failed "
            f"({response.status_code}): "
            f"{error_text}"
        )


# ============================================================
# SHARED API CLIENT
# ============================================================

api = InsightFlowAPI()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def health_check():
    """
    Check backend health using the shared API client.
    """

    return api.health_check()


def upload_dataset(
    uploaded_file,
):
    """
    Upload and prepare a dataset.
    """

    return api.upload_dataset(
        uploaded_file
    )


def prepare_dataset(
    uploaded_file,
):
    """
    Prepare a dataset.
    """

    return api.prepare_dataset(
        uploaded_file
    )


def get_dataset(
    dataset_id,
):
    """
    Retrieve a prepared dataset.
    """

    return api.get_dataset(
        dataset_id
    )


def ask_dataset(
    dataset_id,
    question,
):
    """
    Ask the agent an analytical question.
    """

    return api.ask_dataset(
        dataset_id,
        question,
    )


def analyze_dataset(
    uploaded_file,
    question,
):
    """
    Upload + prepare + analyze in one request.
    """

    return api.analyze_dataset(
        uploaded_file,
        question,
    )


def delete_dataset(
    dataset_id,
):
    """
    Delete a dataset session.
    """

    return api.delete_dataset(
        dataset_id
    )