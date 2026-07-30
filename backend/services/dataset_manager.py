from uuid import uuid4
from datetime import datetime, timezone
from threading import RLock

import pandas as pd


class DatasetManager:
    """
    In-memory dataset session manager for InsightFlow AI.

    Stores all prepared analytical artifacts belonging to
    one uploaded dataset.

    Stored information:
        - cleaned dataframe
        - original and cleaned schemas
        - original and cleaned quality reports
        - quality component scores
        - anomaly reports
        - cleaning decisions / operations
        - EDA results
        - visualization metadata
        - local statistical findings
        - session metadata

    V1:
        In-memory Python storage

    Future:
        Redis / database / object storage
    """

    def __init__(self):

        self.datasets = {}

        self.lock = RLock()


    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _safe_dict(value):
        """
        Normalize a value expected to be a dictionary.
        """

        if isinstance(value, dict):
            return value

        return {}


    @staticmethod
    def _safe_list(value):
        """
        Normalize a value expected to be a list.
        """

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        try:
            return list(value)

        except Exception:
            return []


    @staticmethod
    def _safe_score(value):
        """
        Convert numeric score-like values to regular Python
        floats so they can safely be serialized by FastAPI.
        """

        if value is None:
            return None

        try:
            return round(
                float(value),
                2
            )

        except (
            TypeError,
            ValueError
        ):
            return None


    # ========================================================
    # CREATE DATASET SESSION
    # ========================================================

    def create_dataset(
        self,
        cleaned_df,
        original_filename=None,

        # Original dataset
        original_schema=None,
        original_quality_report=None,
        original_quality_components=None,
        original_anomalies=None,
        before_score=None,

        # Cleaned dataset
        cleaned_schema=None,
        cleaned_quality_report=None,
        cleaned_quality_components=None,
        cleaned_anomalies=None,
        after_score=None,

        # Cleaning
        cleaning_log=None,
        cleaning_summary=None,

        # EDA
        eda_results=None,

        # Visualization
        eda_charts=None,

        # Local statistical intelligence
        statistical_findings=None
    ):
        """
        Create and store a new prepared dataset session.
        """

        # ----------------------------------------------------
        # Validate dataframe
        # ----------------------------------------------------

        if cleaned_df is None:

            raise ValueError(
                "cleaned_df cannot be None."
            )


        if not isinstance(
            cleaned_df,
            pd.DataFrame
        ):

            raise TypeError(
                "cleaned_df must be a Pandas DataFrame."
            )


        if cleaned_df.empty:

            raise ValueError(
                "Cannot store an empty dataset."
            )


        # ----------------------------------------------------
        # Dataset ID
        # ----------------------------------------------------

        dataset_id = uuid4().hex


        now = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )


        # ----------------------------------------------------
        # Normalize values
        # ----------------------------------------------------

        before_score = (
            self._safe_score(
                before_score
            )
        )


        after_score = (
            self._safe_score(
                after_score
            )
        )


        original_schema = (
            self._safe_dict(
                original_schema
            )
        )


        cleaned_schema = (
            self._safe_dict(
                cleaned_schema
            )
        )


        original_quality_report = (
            self._safe_dict(
                original_quality_report
            )
        )


        cleaned_quality_report = (
            self._safe_dict(
                cleaned_quality_report
            )
        )


        original_quality_components = (
            self._safe_dict(
                original_quality_components
            )
        )


        cleaned_quality_components = (
            self._safe_dict(
                cleaned_quality_components
            )
        )


        original_anomalies = (
            self._safe_dict(
                original_anomalies
            )
        )


        cleaned_anomalies = (
            self._safe_dict(
                cleaned_anomalies
            )
        )


        cleaning_log = (
            self._safe_list(
                cleaning_log
            )
        )


        cleaning_summary = (
            self._safe_dict(
                cleaning_summary
            )
        )


        eda_results = (
            self._safe_dict(
                eda_results
            )
        )


        eda_charts = (
            self._safe_list(
                eda_charts
            )
        )


        statistical_findings = (
            self._safe_list(
                statistical_findings
            )
        )


        # ----------------------------------------------------
        # Dataset session
        # ----------------------------------------------------

        dataset_session = {

            "dataset_id":
                dataset_id,

            "original_filename":
                original_filename,

            # ------------------------------------------------
            # Prepared DataFrame
            # ------------------------------------------------

            "cleaned_df":
                cleaned_df.copy(),

            # ------------------------------------------------
            # Original schema
            # ------------------------------------------------

            "original_schema":
                original_schema,

            # ------------------------------------------------
            # Original quality intelligence
            # ------------------------------------------------

            "original_quality_report":
                original_quality_report,

            "original_quality_components":
                original_quality_components,

            "original_anomalies":
                original_anomalies,

            "before_score":
                before_score,

            # ------------------------------------------------
            # Cleaned schema
            # ------------------------------------------------

            "cleaned_schema":
                cleaned_schema,

            # ------------------------------------------------
            # Cleaned quality intelligence
            # ------------------------------------------------

            "cleaned_quality_report":
                cleaned_quality_report,

            "cleaned_quality_components":
                cleaned_quality_components,

            "cleaned_anomalies":
                cleaned_anomalies,

            "after_score":
                after_score,

            # ------------------------------------------------
            # Cleaning intelligence
            # ------------------------------------------------

            "cleaning_log":
                cleaning_log,

            "cleaning_summary":
                cleaning_summary,

            # ------------------------------------------------
            # EDA intelligence
            # ------------------------------------------------

            "eda_results":
                eda_results,

            # ------------------------------------------------
            # Visualization intelligence
            # ------------------------------------------------

            "eda_charts":
                eda_charts,

            # ------------------------------------------------
            # Statistical insight engine
            # ------------------------------------------------

            "statistical_findings":
                statistical_findings,

            # ------------------------------------------------
            # Session metadata
            # ------------------------------------------------

            "created_at":
                now,

            "last_accessed_at":
                now
        }


        # ----------------------------------------------------
        # Store dataset
        # ----------------------------------------------------

        with self.lock:

            self.datasets[
                dataset_id
            ] = dataset_session


        print(
            f"Dataset session created: {dataset_id}"
        )

        print(
            f"Stored EDA charts: {len(eda_charts)}"
        )

        print(
            "Stored statistical findings: "
            f"{len(statistical_findings)}"
        )


        return dataset_id


    # ========================================================
    # GET DATASET
    # ========================================================

    def get_dataset(
        self,
        dataset_id
    ):
        """
        Retrieve the complete internal dataset session.

        This includes the DataFrame and should therefore
        normally be used only by backend services.
        """

        if not isinstance(
            dataset_id,
            str
        ):

            return None


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            return None


        with self.lock:

            dataset = (
                self.datasets.get(
                    dataset_id
                )
            )


            if dataset is None:

                return None


            dataset[
                "last_accessed_at"
            ] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )


            return dataset


    # ========================================================
    # CHECK DATASET
    # ========================================================

    def exists(
        self,
        dataset_id
    ):
        """
        Check whether a dataset session exists.
        """

        if not isinstance(
            dataset_id,
            str
        ):

            return False


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            return False


        with self.lock:

            return (
                dataset_id
                in self.datasets
            )


    # ========================================================
    # DELETE DATASET
    # ========================================================

    def delete_dataset(
        self,
        dataset_id
    ):
        """
        Delete a dataset session.
        """

        if not isinstance(
            dataset_id,
            str
        ):

            return False


        dataset_id = (
            dataset_id.strip()
        )


        if not dataset_id:

            return False


        with self.lock:

            if (
                dataset_id
                not in self.datasets
            ):

                return False


            del self.datasets[
                dataset_id
            ]


            return True


    # ========================================================
    # GET DATAFRAME
    # ========================================================

    def get_dataframe(
        self,
        dataset_id
    ):
        """
        Return a copy of the cleaned DataFrame.
        """

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return None


        dataframe = (
            dataset.get(
                "cleaned_df"
            )
        )


        if dataframe is None:

            return None


        return dataframe.copy()


    # ========================================================
    # GET DATASET INFORMATION
    # ========================================================

    def get_dataset_info(
        self,
        dataset_id
    ):
        """
        Return frontend/API-safe analytical metadata.

        The actual DataFrame is intentionally excluded.
        """

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return None


        df = (
            dataset[
                "cleaned_df"
            ]
        )


        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        before_score = (
            self._safe_score(
                dataset.get(
                    "before_score"
                )
            )
        )


        after_score = (
            self._safe_score(
                dataset.get(
                    "after_score"
                )
            )
        )


        improvement = None


        if (
            before_score is not None
            and
            after_score is not None
        ):

            improvement = round(
                after_score
                -
                before_score,
                2
            )


        # ----------------------------------------------------
        # Charts
        # ----------------------------------------------------

        eda_charts = (
            self._safe_list(
                dataset.get(
                    "eda_charts"
                )
            )
        )


        # ----------------------------------------------------
        # Statistical findings
        # ----------------------------------------------------

        statistical_findings = (
            self._safe_list(
                dataset.get(
                    "statistical_findings"
                )
            )
        )


        # ----------------------------------------------------
        # API-safe response
        # ----------------------------------------------------

        return {

            "dataset_id":
                dataset_id,

            "original_filename":
                dataset.get(
                    "original_filename"
                ),

            # ------------------------------------------------
            # Dataset dimensions
            # ------------------------------------------------

            "rows":
                int(
                    len(df)
                ),

            "columns":
                [
                    str(column)
                    for column
                    in df.columns
                ],

            "column_count":
                int(
                    len(df.columns)
                ),

            # ------------------------------------------------
            # Schema intelligence
            # ------------------------------------------------

            "schema": {

                "original":
                    dataset.get(
                        "original_schema",
                        {}
                    ),

                "cleaned":
                    dataset.get(
                        "cleaned_schema",
                        {}
                    )
            },

            # ------------------------------------------------
            # Quality intelligence
            # ------------------------------------------------

            "quality": {

                "before_score":
                    before_score,

                "after_score":
                    after_score,

                "improvement":
                    improvement,

                "before_components":
                    dataset.get(
                        "original_quality_components",
                        {}
                    ),

                "after_components":
                    dataset.get(
                        "cleaned_quality_components",
                        {}
                    ),

                "original_report":
                    dataset.get(
                        "original_quality_report",
                        {}
                    ),

                "cleaned_report":
                    dataset.get(
                        "cleaned_quality_report",
                        {}
                    ),

                "original_anomalies":
                    dataset.get(
                        "original_anomalies",
                        {}
                    ),

                "cleaned_anomalies":
                    dataset.get(
                        "cleaned_anomalies",
                        {}
                    )
            },

            # ------------------------------------------------
            # Cleaning intelligence
            # ------------------------------------------------

            "cleaning": {

                "operations":
                    dataset.get(
                        "cleaning_log",
                        []
                    ),

                "summary":
                    dataset.get(
                        "cleaning_summary",
                        {}
                    )
            },

            # ------------------------------------------------
            # Compatibility fields
            #
            # Keep these because the existing Streamlit
            # frontend may still expect cleaning_log inside
            # quality.
            # ------------------------------------------------

            "cleaning_log":
                dataset.get(
                    "cleaning_log",
                    []
                ),

            # ------------------------------------------------
            # EDA intelligence
            # ------------------------------------------------

            "eda_results":
                dataset.get(
                    "eda_results",
                    {}
                ),

            # ------------------------------------------------
            # Visualization intelligence
            # ------------------------------------------------

            "eda_charts":
                eda_charts,

            "eda_chart_count":
                len(
                    eda_charts
                ),

            # ------------------------------------------------
            # Statistical insight engine
            # ------------------------------------------------

            "statistical_findings":
                statistical_findings,

            "statistical_finding_count":
                len(
                    statistical_findings
                ),

            # ------------------------------------------------
            # Session metadata
            # ------------------------------------------------

            "created_at":
                dataset.get(
                    "created_at"
                ),

            "last_accessed_at":
                dataset.get(
                    "last_accessed_at"
                )
        }


    # ========================================================
    # GET EDA CHARTS
    # ========================================================

    def get_eda_charts(
        self,
        dataset_id
    ):
        """
        Retrieve chart specifications for a dataset.
        """

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return []


        return (
            self._safe_list(
                dataset.get(
                    "eda_charts"
                )
            )
        )


    # ========================================================
    # GET STATISTICAL FINDINGS
    # ========================================================

    def get_statistical_findings(
        self,
        dataset_id
    ):
        """
        Retrieve locally generated statistical findings.
        """

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return []


        return (
            self._safe_list(
                dataset.get(
                    "statistical_findings"
                )
            )
        )


    # ========================================================
    # UPDATE STATISTICAL FINDINGS
    # ========================================================

    def update_statistical_findings(
        self,
        dataset_id,
        findings
    ):
        """
        Update findings after running the statistical
        intelligence engine.
        """

        findings = (
            self._safe_list(
                findings
            )
        )


        with self.lock:

            dataset = (
                self.datasets.get(
                    dataset_id
                )
            )


            if dataset is None:

                return False


            dataset[
                "statistical_findings"
            ] = findings


            dataset[
                "last_accessed_at"
            ] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )


            return True


    # ========================================================
    # UPDATE EDA RESULTS
    # ========================================================

    def update_eda_results(
        self,
        dataset_id,
        eda_results
    ):
        """
        Update EDA results if additional analysis is
        generated after dataset creation.
        """

        eda_results = (
            self._safe_dict(
                eda_results
            )
        )


        with self.lock:

            dataset = (
                self.datasets.get(
                    dataset_id
                )
            )


            if dataset is None:

                return False


            dataset[
                "eda_results"
            ] = eda_results


            dataset[
                "last_accessed_at"
            ] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )


            return True


    # ========================================================
    # UPDATE EDA CHARTS
    # ========================================================

    def update_eda_charts(
        self,
        dataset_id,
        charts
    ):
        """
        Replace visualization metadata for a dataset.
        """

        charts = (
            self._safe_list(
                charts
            )
        )


        with self.lock:

            dataset = (
                self.datasets.get(
                    dataset_id
                )
            )


            if dataset is None:

                return False


            dataset[
                "eda_charts"
            ] = charts


            dataset[
                "last_accessed_at"
            ] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )


            return True


    # ========================================================
    # COUNT ACTIVE DATASETS
    # ========================================================

    def count(
        self
    ):
        """
        Number of active in-memory dataset sessions.
        """

        with self.lock:

            return len(
                self.datasets
            )


# ============================================================
# SHARED APPLICATION INSTANCE
# ============================================================

dataset_manager = DatasetManager()