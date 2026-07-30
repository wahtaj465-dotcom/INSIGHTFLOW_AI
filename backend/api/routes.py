from pathlib import Path
from uuid import uuid4
from datetime import datetime, date

import shutil
import math

import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from backend.workflows.analytics_workflow import AnalyticsWorkflow
from backend.services.dataset_manager import dataset_manager


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Analytics"]
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls"
}


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(value):
    """
    Recursively convert Pandas, NumPy, datetime and missing
    values into JSON-safe Python objects.
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    if isinstance(value, pd.DataFrame):

        return make_json_safe(
            value.to_dict(
                orient="records"
            )
        )

    # --------------------------------------------------------
    # SERIES
    # --------------------------------------------------------

    if isinstance(value, pd.Series):

        return make_json_safe(
            value.tolist()
        )

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    if isinstance(value, dict):

        safe = {}

        for key, item in value.items():

            if isinstance(
                key,
                (
                    pd.Timestamp,
                    datetime,
                    date
                )
            ):

                safe_key = key.isoformat()

            else:

                safe_key = str(key)

            safe[safe_key] = make_json_safe(
                item
            )

        return safe

    # --------------------------------------------------------
    # LIST / TUPLE / SET
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
            set
        )
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    # --------------------------------------------------------
    # NUMPY ARRAY
    # --------------------------------------------------------

    if isinstance(value, np.ndarray):

        return make_json_safe(
            value.tolist()
        )

    # --------------------------------------------------------
    # NUMPY INTEGER
    # --------------------------------------------------------

    if isinstance(value, np.integer):

        return int(value)

    # --------------------------------------------------------
    # NUMPY FLOAT
    # --------------------------------------------------------

    if isinstance(value, np.floating):

        converted = float(value)

        if not math.isfinite(converted):
            return None

        return converted

    # --------------------------------------------------------
    # PYTHON FLOAT
    # --------------------------------------------------------

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    # --------------------------------------------------------
    # NUMPY BOOLEAN
    # --------------------------------------------------------

    if isinstance(value, np.bool_):

        return bool(value)

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if isinstance(value, pd.Timestamp):

        if pd.isna(value):
            return None

        return value.isoformat()

    # --------------------------------------------------------
    # DATETIME / DATE
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            datetime,
            date
        )
    ):

        return value.isoformat()

    # --------------------------------------------------------
    # GENERIC MISSING VALUE
    # --------------------------------------------------------

    try:

        missing = pd.isna(value)

        if isinstance(
            missing,
            (
                bool,
                np.bool_
            )
        ):

            if missing:
                return None

    except (
        TypeError,
        ValueError
    ):

        pass

    # --------------------------------------------------------
    # NORMAL JSON TYPES
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            str,
            int,
            bool
        )
    ):

        return value

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return str(value)


# ============================================================
# DATAFRAME -> RECORDS
# ============================================================

def dataframe_to_records(df):

    if df is None:
        return None

    if not isinstance(
        df,
        pd.DataFrame
    ):

        return make_json_safe(df)

    return make_json_safe(
        df.to_dict(
            orient="records"
        )
    )


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_uploaded_file(file):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    original_filename = Path(
        file.filename
    ).name

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Currently supported: CSV, XLSX and XLS."
            )
        )

    unique_filename = (
        f"{uuid4().hex}{extension}"
    )

    file_path = (
        UPLOAD_DIR /
        unique_filename
    )

    try:

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save uploaded "
                f"file: {error}"
            )
        )

    finally:

        await file.close()

    return (
        original_filename,
        unique_filename,
        file_path
    )


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def api_health():

    return {
        "status": "healthy",
        "service": "InsightFlow Analytics API",
        "active_datasets": dataset_manager.count()
    }


# ============================================================
# PREPARE DATASET
# ============================================================

@router.post("/datasets")
async def prepare_dataset(
    file: UploadFile = File(...)
):

    (
        original_filename,
        stored_filename,
        file_path
    ) = await save_uploaded_file(file)

    try:

        workflow = AnalyticsWorkflow()

        result = workflow.prepare_dataset(
            file_path=str(file_path),
            original_filename=original_filename
        )

        if not result.get(
            "success",
            False
        ):

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Dataset preparation failed.",

                    "error":
                        result.get("error")
                }
            )

        dataset_id = result.get(
            "dataset_id"
        )

        if not dataset_id:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Dataset preparation completed "
                    "without creating a dataset session."
                )
            )

        before_score = result.get(
            "before_score"
        )

        after_score = result.get(
            "after_score"
        )

        improvement = None

        if (
            before_score is not None
            and
            after_score is not None
        ):

            improvement = round(
                after_score - before_score,
                2
            )

        response = {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "file": {

                "original_name":
                    original_filename,

                "stored_name":
                    stored_filename
            },

            "dataset": {

                "original_rows":
                    result.get(
                        "original_rows"
                    ),

                "cleaned_rows":
                    result.get(
                        "cleaned_rows"
                    ),

                "columns":
                    result.get(
                        "columns",
                        []
                    )
            },

            "quality": {

                "before_score":
                    before_score,

                "after_score":
                    after_score,

                "improvement":
                    improvement,

                "cleaning_log":
                    result.get(
                        "cleaning_log",
                        []
                    ),

                "quality_report":
                    result.get(
                        "quality_report",
                        {}
                    ),

                "anomalies":
                    result.get(
                        "anomalies",
                        {}
                    )
            },

            "eda":
                result.get(
                    "eda_results",
                    {}
                ),

            "visualizations":
                result.get(
                    "eda_charts",
                    []
                ),

            "statistical_findings":
                result.get(
                    "statistical_findings",
                    []
                )
        }

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Unexpected dataset preparation error.",

                "error":
                    str(error)
            }
        ) from error


# ============================================================
# GET DATASET
# ============================================================

@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str
):

    """
    Retrieve a prepared dataset session.

    Dataset sessions belong to DatasetManager, not
    AnalyticsWorkflow.
    """

    dataset_id = (
        dataset_id.strip()
        if isinstance(
            dataset_id,
            str
        )
        else ""
    )

    if not dataset_id:

        raise HTTPException(
            status_code=400,
            detail="Dataset ID cannot be empty."
        )

    try:

        dataset_info = (
            dataset_manager.get_dataset_info(
                dataset_id
            )
        )

        if dataset_info is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Dataset session not found."
                )
            )

        return make_json_safe({

            "success":
                True,

            "dataset":
                dataset_info
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Could not retrieve dataset session.",

                "error":
                    str(error)
            }
        ) from error


# ============================================================
# ASK DATASET
# ============================================================

@router.post(
    "/datasets/{dataset_id}/ask"
)
def ask_dataset(
    dataset_id: str,
    question: str = Form(...)
):

    if not isinstance(
        question,
        str
    ):

        raise HTTPException(
            status_code=400,
            detail="Question must be a string."
        )

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    dataset_id = (
        dataset_id.strip()
        if isinstance(
            dataset_id,
            str
        )
        else ""
    )

    if not dataset_id:

        raise HTTPException(
            status_code=400,
            detail="Dataset ID cannot be empty."
        )

    # --------------------------------------------------------
    # CHECK SESSION BEFORE RUNNING ANALYTICS
    # --------------------------------------------------------

    if not dataset_manager.exists(
        dataset_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Dataset session not found."
        )

    try:

        workflow = AnalyticsWorkflow()

        result = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question
        )

        # ----------------------------------------------------
        # TRUE WORKFLOW FAILURE
        # ----------------------------------------------------

        if not result.get(
            "success",
            False
        ):

            error = result.get(
                "error"
            )

            if (
                error
                ==
                "Dataset session not found."
            ):

                raise HTTPException(
                    status_code=404,
                    detail=error
                )

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Dataset question failed.",

                    "error":
                        error,

                    "generated_sql":
                        make_json_safe(
                            result.get(
                                "generated_sql"
                            )
                        ),

                    "sql_attempts":
                        make_json_safe(
                            result.get(
                                "sql_attempts",
                                []
                            )
                        )
                }
            )

        # ----------------------------------------------------
        # SQL RESULT
        # ----------------------------------------------------

        sql_records = dataframe_to_records(
            result.get(
                "sql_result"
            )
        )

        insight_source = result.get(
            "insight_source"
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "question":
                question,

            "analysis": {

                "generated_sql":
                    result.get(
                        "generated_sql"
                    ),

                "sql_result":
                    sql_records,

                "sql_attempts":
                    result.get(
                        "sql_attempts",
                        []
                    ),

                "relevant_columns":
                    result.get(
                        "relevant_columns",
                        []
                    )
            },

            "visualization":
                result.get(
                    "result_chart"
                ),

            "insight":
                result.get(
                    "insight"
                ),

            "statistical_findings":
                result.get(
                    "statistical_findings",
                    []
                ),

            "insight_status": {

                "available":
                    bool(
                        result.get(
                            "insight"
                        )
                    ),

                "source":
                    insight_source,

                "llm_success":
                    (
                        insight_source
                        ==
                        "llm"
                    ),

                "fallback_used":
                    (
                        insight_source
                        ==
                        "fallback"
                    )
            }
        }

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Unexpected dataset question error.",

                "error":
                    str(error)
            }
        ) from error


# ============================================================
# DELETE DATASET
# ============================================================

@router.delete(
    "/datasets/{dataset_id}"
)
def delete_dataset(
    dataset_id: str
):

    """
    Delete a dataset directly from DatasetManager.
    """

    dataset_id = (
        dataset_id.strip()
        if isinstance(
            dataset_id,
            str
        )
        else ""
    )

    if not dataset_id:

        raise HTTPException(
            status_code=400,
            detail="Dataset ID cannot be empty."
        )

    try:

        deleted = (
            dataset_manager.delete_dataset(
                dataset_id
            )
        )

        if not deleted:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Dataset session not found."
                )
            )

        return {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "message":
                "Dataset session deleted."
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Could not delete dataset session.",

                "error":
                    str(error)
            }
        ) from error


# ============================================================
# LEGACY ANALYZE
# ============================================================

@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    """
    Legacy one-shot endpoint.

    Upload dataset -> prepare -> ask question -> return result.

    Kept for backward compatibility.
    """

    if not isinstance(
        question,
        str
    ):

        raise HTTPException(
            status_code=400,
            detail="Question must be a string."
        )

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    (
        original_filename,
        unique_filename,
        file_path
    ) = await save_uploaded_file(file)

    try:

        workflow = AnalyticsWorkflow()

        # ----------------------------------------------------
        # PREPARE DATASET
        # ----------------------------------------------------

        preparation_result = (
            workflow.prepare_dataset(
                file_path=str(file_path),
                original_filename=original_filename
            )
        )

        if not preparation_result.get(
            "success",
            False
        ):

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Dataset preparation failed.",

                    "error":
                        preparation_result.get(
                            "error"
                        )
                }
            )

        dataset_id = (
            preparation_result.get(
                "dataset_id"
            )
        )

        if not dataset_id:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Dataset preparation completed "
                    "without creating a session."
                )
            )

        # ----------------------------------------------------
        # ASK QUESTION
        # ----------------------------------------------------

        result = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question
        )

        if not result.get(
            "success",
            False
        ):

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Analytics workflow failed.",

                    "error":
                        result.get(
                            "error"
                        ),

                    "generated_sql":
                        make_json_safe(
                            result.get(
                                "generated_sql"
                            )
                        ),

                    "sql_attempts":
                        make_json_safe(
                            result.get(
                                "sql_attempts",
                                []
                            )
                        )
                }
            )

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        dataset_info = (
            dataset_manager.get_dataset_info(
                dataset_id
            )
        )

        sql_records = (
            dataframe_to_records(
                result.get(
                    "sql_result"
                )
            )
        )

        before_score = (
            preparation_result.get(
                "before_score"
            )
        )

        after_score = (
            preparation_result.get(
                "after_score"
            )
        )

        improvement = None

        if (
            before_score is not None
            and
            after_score is not None
        ):

            improvement = round(
                after_score - before_score,
                2
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "file": {

                "original_name":
                    original_filename,

                "stored_name":
                    unique_filename
            },

            "question":
                question,

            "dataset": {

                "original_rows":
                    preparation_result.get(
                        "original_rows"
                    ),

                "cleaned_rows":
                    preparation_result.get(
                        "cleaned_rows"
                    ),

                "columns":
                    preparation_result.get(
                        "columns",
                        []
                    )
            },

            "dataset_info":
                dataset_info,

            "quality": {

                "before_score":
                    before_score,

                "after_score":
                    after_score,

                "improvement":
                    improvement,

                "cleaning_log":
                    preparation_result.get(
                        "cleaning_log",
                        []
                    ),

                "quality_report":
                    preparation_result.get(
                        "quality_report",
                        {}
                    ),

                "anomalies":
                    preparation_result.get(
                        "anomalies",
                        {}
                    )
            },

            "eda":
                preparation_result.get(
                    "eda_results",
                    {}
                ),

            "visualizations":
                preparation_result.get(
                    "eda_charts",
                    []
                ),

            "statistical_findings":
                preparation_result.get(
                    "statistical_findings",
                    []
                ),

            "analysis": {

                "generated_sql":
                    result.get(
                        "generated_sql"
                    ),

                "sql_result":
                    sql_records,

                "sql_attempts":
                    result.get(
                        "sql_attempts",
                        []
                    ),

                "relevant_columns":
                    result.get(
                        "relevant_columns",
                        []
                    )
            },

            "visualization":
                result.get(
                    "result_chart"
                ),

            "insight":
                result.get(
                    "insight"
                ),

            "insight_status": {

                "available":
                    bool(
                        result.get(
                            "insight"
                        )
                    ),

                "source":
                    result.get(
                        "insight_source"
                    ),

                "llm_success":
                    (
                        result.get(
                            "insight_source"
                        )
                        ==
                        "llm"
                    ),

                "fallback_used":
                    (
                        result.get(
                            "insight_source"
                        )
                        ==
                        "fallback"
                    )
            }
        }

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Unexpected analytics workflow error.",

                "error":
                    str(error)
            }
        ) from error