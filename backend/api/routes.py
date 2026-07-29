from pathlib import Path
from uuid import uuid4
import shutil

import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from backend.workflows.analytics_workflow import AnalyticsWorkflow


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Analytics"]
)


# ============================================================
# SHARED ANALYTICS WORKFLOW
# ============================================================

# IMPORTANT:
#
# The workflow must remain alive between API requests because
# prepared datasets are currently stored in memory.
#
# Flow:
#
# POST /datasets
#       ↓
# workflow.prepare_dataset()
#       ↓
# dataset_id
#
# POST /datasets/{dataset_id}/ask
#       ↓
# workflow.ask_dataset()
#
# If we created and closed a workflow inside every request,
# the stored dataset would disappear.

workflow = AnalyticsWorkflow()


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
# REQUEST MODEL
# ============================================================

class DatasetQuestion(BaseModel):
    """
    Request body used when asking a question
    about an already prepared dataset.
    """

    question: str


# ============================================================
# HELPER — DATAFRAME TO JSON-SAFE RECORDS
# ============================================================

def dataframe_to_records(df):
    """
    Convert a Pandas DataFrame into JSON-safe records.
    """

    if df is None:
        return None

    if not isinstance(df, pd.DataFrame):
        return df

    safe_df = df.astype(object).where(
        pd.notnull(df),
        None
    )

    return safe_df.to_dict(
        orient="records"
    )


# ============================================================
# HELPER — VALIDATE DATASET ID
# ============================================================

def validate_dataset_id(dataset_id):
    """
    Validate dataset ID before passing it
    to the analytics workflow.
    """

    if not isinstance(dataset_id, str):
        raise HTTPException(
            status_code=400,
            detail="Dataset ID must be a string."
        )

    dataset_id = dataset_id.strip()

    if not dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Dataset ID cannot be empty."
        )

    return dataset_id


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def api_health():
    """
    Verify that the analytics API is running.
    """

    return {
        "status": "healthy",
        "service": "InsightFlow Analytics API"
    }


# ============================================================
# 1. UPLOAD + PREPARE DATASET
# ============================================================

@router.post("/datasets")
async def create_dataset(
    file: UploadFile = File(...)
):
    """
    Upload and prepare a dataset.

    The expensive preparation pipeline runs only once:

    Upload
        ↓
    Ingestion
        ↓
    Schema Analysis
        ↓
    Quality Analysis
        ↓
    Anomaly Detection
        ↓
    Cleaning
        ↓
    EDA
        ↓
    Store Dataset Session
        ↓
    Return dataset_id

    Future questions reuse the prepared dataset.
    """

    # ========================================================
    # 1. VALIDATE FILE
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )


    # ========================================================
    # 2. VALIDATE FILE EXTENSION
    # ========================================================

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


    # ========================================================
    # 3. CREATE SAFE STORED FILE NAME
    # ========================================================

    unique_filename = (
        f"{uuid4().hex}{extension}"
    )

    file_path = (
        UPLOAD_DIR /
        unique_filename
    )


    # ========================================================
    # 4. SAVE UPLOADED FILE
    # ========================================================

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
                f"Could not save uploaded file: {error}"
            )
        )

    finally:

        await file.close()


    # ========================================================
    # 5. PREPARE DATASET
    # ========================================================

    try:

        result = workflow.prepare_dataset(
            file_path=str(file_path),
            original_filename=original_filename
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset preparation failed: {error}"
            )
        )


    # ========================================================
    # 6. CHECK RESULT
    # ========================================================

    if not result.get(
        "success",
        False
    ):

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Dataset preparation failed.",
                "error": result.get("error")
            }
        )


    # ========================================================
    # 7. QUALITY INFORMATION
    # ========================================================

    before_score = result.get(
        "before_score"
    )

    after_score = result.get(
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


    # ========================================================
    # 8. RESPONSE
    # ========================================================

    return {

        "success": True,

        "dataset_id":
            result.get(
                "dataset_id"
            ),

        "file": {

            "original_name":
                original_filename,

            "stored_name":
                unique_filename
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
            )
    }


# ============================================================
# 2. ASK QUESTION ABOUT PREPARED DATASET
# ============================================================

@router.post("/datasets/{dataset_id}/ask")
def ask_dataset(
    dataset_id: str,
    request: DatasetQuestion
):
    """
    Ask a natural-language question about an
    already prepared dataset.

    Dataset ID
        ↓
    Retrieve Prepared Dataset
        ↓
    DuckDB
        ↓
    SQL Agent
        ↓
    SQL Execution
        ↓
    Insight Agent
        ↓
    Response
    """

    dataset_id = validate_dataset_id(
        dataset_id
    )


    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    question = request.question

    if not isinstance(question, str):

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


    # ========================================================
    # ASK WORKFLOW
    # ========================================================

    try:

        result = workflow.ask_dataset(
            dataset_id=dataset_id,
            question=question
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset analysis failed: {error}"
            )
        )


    # ========================================================
    # CHECK RESULT
    # ========================================================

    if not result.get(
        "success",
        False
    ):

        error_message = result.get(
            "error"
        )

        # The workflow may report a missing dataset
        # as a normal failure dictionary instead of
        # raising KeyError.

        if (
            error_message
            and
            "not found" in str(error_message).lower()
        ):

            raise HTTPException(
                status_code=404,
                detail=error_message
            )

        raise HTTPException(
            status_code=500,
            detail={

                "message":
                    "Dataset analysis failed.",

                "error":
                    error_message,

                "generated_sql":
                    result.get(
                        "generated_sql"
                    ),

                "sql_attempts":
                    result.get(
                        "sql_attempts",
                        []
                    )
            }
        )


    # ========================================================
    # CONVERT SQL RESULT
    # ========================================================

    sql_records = dataframe_to_records(
        result.get(
            "sql_result"
        )
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

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

        "insight":
            result.get(
                "insight"
            )
    }


# ============================================================
# 3. GET DATASET INFORMATION
# ============================================================

@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str
):
    """
    Return information about a prepared dataset.
    """

    dataset_id = validate_dataset_id(
        dataset_id
    )

    try:

        result = workflow.get_dataset_info(
            dataset_id
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not retrieve dataset: {error}"
            )
        )


    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )


    return {
        "success": True,
        "dataset": result
    }


# ============================================================
# 4. DELETE DATASET SESSION
# ============================================================

@router.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: str
):
    """
    Delete a prepared dataset from the current
    in-memory session store.
    """

    dataset_id = validate_dataset_id(
        dataset_id
    )

    try:

        result = workflow.delete_dataset(
            dataset_id
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not delete dataset: {error}"
            )
        )


    # Support either:
    #
    # True / False
    #
    # OR
    #
    # {
    #     "success": True,
    #     ...
    # }

    if result is False or result is None:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )


    if isinstance(result, dict):

        if not result.get(
            "success",
            False
        ):

            error_message = result.get(
                "error",
                "Dataset could not be deleted."
            )

            if "not found" in str(
                error_message
            ).lower():

                raise HTTPException(
                    status_code=404,
                    detail=error_message
                )

            raise HTTPException(
                status_code=500,
                detail=error_message
            )


    return {

        "success": True,

        "dataset_id":
            dataset_id,

        "message":
            "Dataset session deleted successfully."
    }