from uuid import uuid4
from datetime import datetime, timezone
from threading import RLock


class DatasetManager:
    """
    In-memory dataset session manager for InsightFlow.

    Stores prepared datasets and their analytical metadata
    so that the user can ask multiple questions without
    preprocessing the dataset again.

    V1 implementation:
        Python memory

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
        eda_results=None
    ):
        """
        Store a prepared dataset and return its dataset ID.
        """

        if cleaned_df is None:
            raise ValueError(
                "cleaned_df cannot be None."
            )

        if cleaned_df.empty:
            raise ValueError(
                "Cannot store an empty dataset."
            )

        dataset_id = uuid4().hex

        now = datetime.now(
            timezone.utc
        ).isoformat()

        dataset_session = {
            "dataset_id": dataset_id,

            "original_filename":
                original_filename,

            "cleaned_df":
                cleaned_df.copy(),

            "original_schema":
                original_schema or {},

            "original_quality_report":
                original_quality_report or {},

            "original_anomalies":
                original_anomalies or {},

            "before_score":
                before_score,

            "cleaned_schema":
                cleaned_schema or {},

            "cleaned_quality_report":
                cleaned_quality_report or {},

            "cleaned_anomalies":
                cleaned_anomalies or {},

            "after_score":
                after_score,

            "cleaning_log":
                cleaning_log or [],

            "eda_results":
                eda_results or {},

            "created_at":
                now,

            "last_accessed_at":
                now
        }

        with self.lock:
            self.datasets[
                dataset_id
            ] = dataset_session

        return dataset_id


    # ========================================================
    # GET DATASET
    # ========================================================

    def get_dataset(
        self,
        dataset_id
    ):
        """
        Retrieve a prepared dataset session.
        """

        if not isinstance(
            dataset_id,
            str
        ):
            return None

        dataset_id = dataset_id.strip()

        if not dataset_id:
            return None

        with self.lock:

            dataset = self.datasets.get(
                dataset_id
            )

            if dataset is None:
                return None

            dataset[
                "last_accessed_at"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            return dataset


    # ========================================================
    # CHECK DATASET
    # ========================================================

    def exists(
        self,
        dataset_id
    ):

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

        Returns True when deleted.
        """

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
        """
        Return metadata without returning the complete
        Pandas DataFrame.
        """

        dataset = self.get_dataset(
            dataset_id
        )

        if dataset is None:
            return None

        df = dataset[
            "cleaned_df"
        ]

        before_score = dataset.get(
            "before_score"
        )

        after_score = dataset.get(
            "after_score"
        )

        improvement = None

        if (
            before_score is not None
            and after_score is not None
        ):
            improvement = round(
                after_score - before_score,
                2
            )

        return {
            "dataset_id":
                dataset_id,

            "original_filename":
                dataset.get(
                    "original_filename"
                ),

            "rows":
                len(df),

            "columns":
                list(df.columns),

            "column_count":
                len(df.columns),

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

            "eda_results":
                dataset.get(
                    "eda_results",
                    {}
                ),

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