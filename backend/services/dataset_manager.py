from uuid import uuid4
from datetime import datetime, timezone
from threading import RLock

import pandas as pd


class DatasetManager:
    """
    In-memory dataset session manager for InsightFlow AI.

    Stores:
        - cleaned dataframe
        - schema information
        - quality reports
        - anomaly reports
        - cleaning operations
        - EDA results
        - automated EDA chart specifications

    This allows the prepared dataset to be reused for
    multiple analytical questions without preprocessing
    the uploaded file again.

    V1:
        In-memory Python storage

    Future:
        Redis / database / object storage
    """

    def __init__(self):

        self.datasets = {}

        self.lock = RLock()


    # ========================================================
    # CREATE DATASET SESSION
    # ========================================================

    def create_dataset(
        self,
        cleaned_df,
        original_filename=None,
        original_schema=None,
        original_quality_report=None,
        original_anomalies=None,
        before_score=None,
        cleaned_schema=None,
        cleaned_quality_report=None,
        cleaned_anomalies=None,
        after_score=None,
        cleaning_log=None,
        eda_results=None,
        eda_charts=None
    ):

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
        # Create dataset ID
        # ----------------------------------------------------

        dataset_id = uuid4().hex


        now = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )


        # ----------------------------------------------------
        # Normalize EDA charts
        # ----------------------------------------------------

        if eda_charts is None:

            eda_charts = []


        if not isinstance(
            eda_charts,
            list
        ):

            try:

                eda_charts = list(
                    eda_charts
                )

            except Exception:

                eda_charts = []


        # ----------------------------------------------------
        # Build dataset session
        # ----------------------------------------------------

        dataset_session = {

            "dataset_id":
                dataset_id,

            "original_filename":
                original_filename,

            # Keep a private copy so later processing cannot
            # accidentally mutate the stored dataset.

            "cleaned_df":
                cleaned_df.copy(),

            # ------------------------------------------------
            # Original dataset metadata
            # ------------------------------------------------

            "original_schema":
                original_schema or {},

            "original_quality_report":
                original_quality_report or {},

            "original_anomalies":
                original_anomalies or {},

            "before_score":
                before_score,

            # ------------------------------------------------
            # Cleaned dataset metadata
            # ------------------------------------------------

            "cleaned_schema":
                cleaned_schema or {},

            "cleaned_quality_report":
                cleaned_quality_report or {},

            "cleaned_anomalies":
                cleaned_anomalies or {},

            "after_score":
                after_score,

            # ------------------------------------------------
            # Cleaning
            # ------------------------------------------------

            "cleaning_log":
                cleaning_log or [],

            # ------------------------------------------------
            # EDA
            # ------------------------------------------------

            "eda_results":
                eda_results or {},

            # ------------------------------------------------
            # Automated EDA visualizations
            # ------------------------------------------------

            "eda_charts":
                eda_charts,

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


        return dataset_id


    # ========================================================
    # GET DATASET
    # ========================================================

    def get_dataset(
        self,
        dataset_id
    ):

        if not isinstance(
            dataset_id,
            str
        ):

            return None


        dataset_id = dataset_id.strip()


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

        if not isinstance(
            dataset_id,
            str
        ):

            return False


        dataset_id = dataset_id.strip()


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

        if not isinstance(
            dataset_id,
            str
        ):

            return False


        dataset_id = dataset_id.strip()


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
    # GET DATASET INFORMATION
    # ========================================================

    def get_dataset_info(
        self,
        dataset_id
    ):

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return None


        df = dataset[
            "cleaned_df"
        ]


        # ----------------------------------------------------
        # Quality scores
        # ----------------------------------------------------

        before_score = (
            dataset.get(
                "before_score"
            )
        )


        after_score = (
            dataset.get(
                "after_score"
            )
        )


        improvement = None


        if (
            before_score is not None
            and
            after_score is not None
        ):

            try:

                improvement = round(
                    float(after_score)
                    -
                    float(before_score),
                    2
                )

            except (
                TypeError,
                ValueError
            ):

                improvement = None


        # ----------------------------------------------------
        # Retrieve charts
        # ----------------------------------------------------

        eda_charts = (
            dataset.get(
                "eda_charts",
                []
            )
        )


        if not isinstance(
            eda_charts,
            list
        ):

            eda_charts = []


        # ----------------------------------------------------
        # API-safe dataset information
        # ----------------------------------------------------

        return {

            "dataset_id":
                dataset_id,

            "original_filename":
                dataset.get(
                    "original_filename"
                ),

            "rows":
                int(
                    len(df)
                ),

            "columns":
                [
                    str(column)
                    for column in df.columns
                ],

            "column_count":
                int(
                    len(df.columns)
                ),

            # ------------------------------------------------
            # Schema
            # ------------------------------------------------

            "schema":
                dataset.get(
                    "cleaned_schema",
                    {}
                ),

            # ------------------------------------------------
            # Quality
            # ------------------------------------------------

            "quality": {

                "before_score":
                    before_score,

                "after_score":
                    after_score,

                "improvement":
                    improvement,

                "cleaning_log":
                    dataset.get(
                        "cleaning_log",
                        []
                    ),

                "quality_report":
                    dataset.get(
                        "cleaned_quality_report",
                        {}
                    ),

                "anomalies":
                    dataset.get(
                        "cleaned_anomalies",
                        {}
                    )
            },

            # ------------------------------------------------
            # EDA analysis
            # ------------------------------------------------

            "eda_results":
                dataset.get(
                    "eda_results",
                    {}
                ),

            # ------------------------------------------------
            # IMPORTANT:
            # Automated dashboard chart specifications
            # ------------------------------------------------

            "eda_charts":
                eda_charts,

            # ------------------------------------------------
            # Helpful count for frontend/debugging
            # ------------------------------------------------

            "eda_chart_count":
                len(
                    eda_charts
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
    # GET CLEANED DATAFRAME
    # ========================================================

    def get_dataframe(
        self,
        dataset_id
    ):

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
    # GET EDA CHARTS
    # ========================================================

    def get_eda_charts(
        self,
        dataset_id
    ):

        dataset = (
            self.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            return []


        charts = (
            dataset.get(
                "eda_charts",
                []
            )
        )


        if not isinstance(
            charts,
            list
        ):

            return []


        return charts


    # ========================================================
    # COUNT ACTIVE DATASETS
    # ========================================================

    def count(self):

        with self.lock:

            return len(
                self.datasets
            )


# ============================================================
# SHARED APPLICATION INSTANCE
# ============================================================

dataset_manager = DatasetManager()