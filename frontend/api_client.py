import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000/api"


class InsightFlowAPI:
    """
    Frontend API client for InsightFlow.

    Responsibilities:
        - Check backend health
        - Upload datasets
        - Retrieve prepared dataset metadata
        - Ask analytical questions
        - Delete dataset sessions
        - Normalize API responses
    """

    def __init__(
        self,
        base_url=API_BASE_URL
    ):

        self.base_url = (
            base_url.rstrip("/")
        )


    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self):
        """
        Check whether the InsightFlow backend is available.
        """

        try:

            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )

            response.raise_for_status()

            return response.json()


        except requests.RequestException as error:

            return {
                "status": "error",
                "error": str(error)
            }


    # ========================================================
    # UPLOAD DATASET
    # ========================================================

    def upload_dataset(
        self,
        uploaded_file
    ):
        """
        Upload a CSV/XLS/XLSX dataset to the backend.

        The backend prepares the dataset and returns:

            {
                "success": True,
                "dataset_id": "...",
                "dataset": {...},
                "quality": {...},
                "eda": {...},
                "visualizations": [...]
            }

        This method intentionally returns the COMPLETE response
        because app.py needs the dataset_id and preparation
        information.
        """

        if uploaded_file is None:

            raise ValueError(
                "uploaded_file cannot be None."
            )


        # ----------------------------------------------------
        # Prepare multipart upload
        # ----------------------------------------------------

        filename = getattr(
            uploaded_file,
            "name",
            "dataset.csv"
        )


        content_type = (
            getattr(
                uploaded_file,
                "type",
                None
            )
            or
            "application/octet-stream"
        )


        try:

            file_content = (
                uploaded_file.getvalue()
            )

        except AttributeError:

            raise ValueError(
                "uploaded_file must provide getvalue()."
            )


        files = {

            "file": (
                filename,
                file_content,
                content_type
            )
        }


        try:

            response = requests.post(
                f"{self.base_url}/datasets",
                files=files,
                timeout=180
            )


            payload = (
                self._handle_response(
                    response
                )
            )


            if not isinstance(
                payload,
                dict
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
    # GET DATASET
    # ========================================================

    def get_dataset(
        self,
        dataset_id
    ):
        """
        Retrieve prepared dataset metadata.

        Backend response:

            {
                "success": True,
                "dataset": {
                    "dataset_id": "...",
                    "rows": ...,
                    "columns": [...],
                    "schema": {...},
                    "quality": {...},
                    "eda_results": {...},
                    "eda_charts": [...]
                }
            }

        IMPORTANT:

        Frontend components should receive the INNER
        dataset object rather than the API envelope.

        Therefore:

            api.get_dataset(...)

        returns:

            {
                "dataset_id": "...",
                "rows": ...,
                "eda_charts": [...]
            }
        """

        if not isinstance(
            dataset_id,
            str
        ):

            raise ValueError(
                "dataset_id must be a string."
            )


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            raise ValueError(
                "dataset_id cannot be empty."
            )


        try:

            response = requests.get(
                f"{self.base_url}/datasets/{dataset_id}",
                timeout=60
            )


            payload = (
                self._handle_response(
                    response
                )
            )


            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if not isinstance(
                payload,
                dict
            ):

                raise RuntimeError(
                    "Invalid dataset response "
                    "received from backend."
                )


            # ------------------------------------------------
            # Current backend format:
            #
            # {
            #     "success": True,
            #     "dataset": {...}
            # }
            # ------------------------------------------------

            dataset = (
                payload.get(
                    "dataset"
                )
            )


            if isinstance(
                dataset,
                dict
            ):

                return dataset


            # ------------------------------------------------
            # Compatibility fallback
            #
            # If backend later returns dataset metadata
            # directly, do not break the frontend.
            # ------------------------------------------------

            return payload


        except requests.RequestException as error:

            raise RuntimeError(
                "Could not retrieve dataset: "
                f"{error}"
            ) from error


    # ========================================================
    # ASK DATASET
    # ========================================================

    def ask_dataset(
        self,
        dataset_id,
        question
    ):
        """
        Ask a natural-language analytical question about
        an already prepared dataset.

        Returns the COMPLETE backend response because the
        question component needs:

            question
            analysis
            visualization
            insight
            insight_status
        """

        # ----------------------------------------------------
        # Validate dataset ID
        # ----------------------------------------------------

        if not isinstance(
            dataset_id,
            str
        ):

            raise ValueError(
                "dataset_id must be a string."
            )


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            raise ValueError(
                "dataset_id cannot be empty."
            )


        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not isinstance(
            question,
            str
        ):

            raise ValueError(
                "Question must be a string."
            )


        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        # ----------------------------------------------------
        # Form data
        # ----------------------------------------------------

        data = {
            "question": question
        }


        try:

            response = requests.post(
                (
                    f"{self.base_url}"
                    f"/datasets/{dataset_id}/ask"
                ),
                data=data,
                timeout=180
            )


            payload = (
                self._handle_response(
                    response
                )
            )


            if not isinstance(
                payload,
                dict
            ):

                raise RuntimeError(
                    "Invalid analysis response "
                    "received from backend."
                )


            return payload


        except requests.RequestException as error:

            raise RuntimeError(
                "Could not analyze question: "
                f"{error}"
            ) from error


    # ========================================================
    # DELETE DATASET
    # ========================================================

    def delete_dataset(
        self,
        dataset_id
    ):
        """
        Delete a prepared dataset session.
        """

        if not isinstance(
            dataset_id,
            str
        ):

            raise ValueError(
                "dataset_id must be a string."
            )


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            raise ValueError(
                "dataset_id cannot be empty."
            )


        try:

            response = requests.delete(
                f"{self.base_url}/datasets/{dataset_id}",
                timeout=30
            )


            return (
                self._handle_response(
                    response
                )
            )


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
        response
    ):
        """
        Convert an HTTP response into Python data and
        raise a clean RuntimeError when the API reports
        an error.

        FastAPI commonly returns errors as:

            {
                "detail": "..."
            }

        or:

            {
                "detail": {
                    "message": "...",
                    "error": "..."
                }
            }
        """

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            payload = (
                response.json()
            )


        except ValueError:

            payload = {

                "detail":
                    (
                        response.text
                        or
                        "Backend returned a non-JSON response."
                    )
            }


        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        if response.ok:

            return payload


        # ----------------------------------------------------
        # Extract FastAPI error detail
        # ----------------------------------------------------

        if isinstance(
            payload,
            dict
        ):

            detail = (
                payload.get(
                    "detail",
                    payload
                )
            )

        else:

            detail = payload


        # ----------------------------------------------------
        # Make nested error messages readable
        # ----------------------------------------------------

        if isinstance(
            detail,
            dict
        ):

            message = (
                detail.get(
                    "message"
                )
            )


            error = (
                detail.get(
                    "error"
                )
            )


            if (
                message
                and
                error
            ):

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