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
    HTTPException,
)

from backend.workflows.analytics_workflow import (
    AnalyticsWorkflow,
)

from backend.services.dataset_manager import (
    dataset_manager,
)

from backend.services.agent_service import (
    agent_service,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Analytics"],
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(
    value
):
    """
    Recursively convert Pandas, NumPy, datetime and
    missing values into JSON-safe Python objects.
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.DataFrame,
    ):

        return make_json_safe(
            value.to_dict(
                orient="records"
            )
        )

    # --------------------------------------------------------
    # SERIES
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Series,
    ):

        return make_json_safe(
            value.tolist()
        )

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        safe = {}

        for (
            key,
            item
        ) in value.items():

            if isinstance(
                key,
                (
                    pd.Timestamp,
                    datetime,
                    date,
                ),
            ):

                safe_key = (
                    key.isoformat()
                )

            else:

                safe_key = str(
                    key
                )

            safe[
                safe_key
            ] = make_json_safe(
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
            set,
        ),
    ):

        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    # --------------------------------------------------------
    # NUMPY ARRAY
    # --------------------------------------------------------

    if isinstance(
        value,
        np.ndarray,
    ):

        return make_json_safe(
            value.tolist()
        )

    # --------------------------------------------------------
    # NUMPY INTEGER
    # --------------------------------------------------------

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    # --------------------------------------------------------
    # NUMPY FLOAT
    # --------------------------------------------------------

    if isinstance(
        value,
        np.floating,
    ):

        converted = float(
            value
        )

        if not math.isfinite(
            converted
        ):

            return None

        return converted

    # --------------------------------------------------------
    # PYTHON FLOAT
    # --------------------------------------------------------

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):

            return None

        return value

    # --------------------------------------------------------
    # NUMPY BOOLEAN
    # --------------------------------------------------------

    if isinstance(
        value,
        np.bool_,
    ):

        return bool(
            value
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Timestamp,
    ):

        if pd.isna(
            value
        ):

            return None

        return value.isoformat()

    # --------------------------------------------------------
    # DATETIME / DATE
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):

        return value.isoformat()

    # --------------------------------------------------------
    # GENERIC MISSING VALUE
    # --------------------------------------------------------

    try:

        missing = pd.isna(
            value
        )

        if isinstance(
            missing,
            (
                bool,
                np.bool_,
            ),
        ):

            if missing:
                return None

    except (
        TypeError,
        ValueError,
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
            bool,
        ),
    ):

        return value

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return str(
        value
    )


# ============================================================
# DATAFRAME -> RECORDS
# ============================================================

def dataframe_to_records(
    df
):

    if df is None:
        return None

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        return make_json_safe(
            df
        )

    return make_json_safe(
        df.to_dict(
            orient="records"
        )
    )


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_uploaded_file(
    file
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "No file was provided."
            ),
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
                "Currently supported: "
                "CSV, XLSX and XLS."
            ),
        )

    unique_filename = (
        f"{uuid4().hex}{extension}"
    )

    file_path = (
        UPLOAD_DIR
        /
        unique_filename
    )

    try:

        with file_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save uploaded "
                f"file: {error}"
            ),
        ) from error

    finally:

        await file.close()

    return (
        original_filename,
        unique_filename,
        file_path,
    )


# ============================================================
# VALIDATE DATASET ID
# ============================================================

def validate_dataset_id(
    dataset_id: str
) -> str:

    if not isinstance(
        dataset_id,
        str,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset ID must be a string."
            ),
        )

    dataset_id = (
        dataset_id.strip()
    )

    if not dataset_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset ID cannot be empty."
            ),
        )

    return dataset_id


# ============================================================
# VALIDATE QUESTION
# ============================================================

def validate_question(
    question: str
) -> str:

    if not isinstance(
        question,
        str,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Question must be a string."
            ),
        )

    question = (
        question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot be empty."
            ),
        )

    return question


# ============================================================
# BUILD PREPARATION RESPONSE
# ============================================================

def build_preparation_response(
    *,
    result,
    original_filename,
    stored_filename,
):
    """
    Normalize AnalyticsWorkflow.prepare_dataset()
    into the API response contract.
    """

    quality = result.get(
        "quality",
        {},
    )

    if not isinstance(
        quality,
        dict,
    ):

        quality = {}

    cleaning = result.get(
        "cleaning",
        {},
    )

    if not isinstance(
        cleaning,
        dict,
    ):

        cleaning = {}

    anomalies = result.get(
        "anomalies",
        {},
    )

    if not isinstance(
        anomalies,
        dict,
    ):

        anomalies = {}

    schema = result.get(
        "schema",
        {},
    )

    if not isinstance(
        schema,
        dict,
    ):

        schema = {}

    return {

        "success":
            True,

        "dataset_id":
            result.get(
                "dataset_id"
            ),

        "file": {

            "original_name":
                original_filename,

            "stored_name":
                stored_filename,
        },

        "dataset": {

            "rows":
                result.get(
                    "rows"
                ),

            "columns":
                result.get(
                    "columns",
                    [],
                ),

            "column_count":
                result.get(
                    "column_count"
                ),
        },

        "schema":
            schema,

        "quality":
            quality,

        "anomalies":
            anomalies,

        "cleaning":
            cleaning,

        "eda":
            result.get(
                "eda_results",
                {},
            ),

        "visualizations":
            result.get(
                "eda_charts",
                [],
            ),

        "chart_count":
            result.get(
                "chart_count",
                0,
            ),

        "statistical_findings":
            result.get(
                "statistical_findings",
                [],
            ),

        "statistical_finding_count":
            result.get(
                "statistical_finding_count",
                0,
            ),
    }


# ============================================================
# BUILD AGENT RESPONSE
# ============================================================

def build_agent_response(
    result
):
    """
    Normalize AgentService output for the frontend/API.

    The full orchestration metadata is intentionally
    retained during development so planner/executor/
    recovery behavior can be inspected.
    """

    return {

        "success":
            result.get(
                "success",
                False,
            ),

        "completed":
            result.get(
                "completed",
                False,
            ),

        "dataset_id":
            result.get(
                "dataset_id"
            ),

        "question":
            result.get(
                "question"
            ),

        # ====================================================
        # PLANNING
        # ====================================================

        "agent": {

            "intent":
                result.get(
                    "intent"
                ),

            "plan":
                result.get(
                    "plan",
                    [],
                ),

            "plan_reasoning":
                result.get(
                    "plan_reasoning"
                ),

            "planner_source":
                result.get(
                    "planner_source"
                ),

            "planner_error":
                result.get(
                    "planner_error"
                ),

            "executed_tools":
                result.get(
                    "executed_tools",
                    [],
                ),
        },

        # ====================================================
        # ANALYSIS
        # ====================================================

        "analysis": {

            "generated_sql":
                result.get(
                    "generated_sql"
                ),

            "sql_result":
                dataframe_to_records(
                    result.get(
                        "sql_result"
                    )
                ),

            "tool_outputs":
                result.get(
                    "tool_outputs",
                    {},
                ),
        },

        # ====================================================
        # VISUALIZATION
        # ====================================================

        "visualization":
            result.get(
                "visualization"
            ),

        # ====================================================
        # INSIGHT
        # ====================================================

        "insight":
            result.get(
                "insight"
            ),

        "statistical_findings":
            result.get(
                "statistical_findings",
                [],
            ),

        # ====================================================
        # EXECUTION / RECOVERY
        # ====================================================

        "execution": {

            "tool_results":
                result.get(
                    "tool_results",
                    {},
                ),

            "retry_count":
                result.get(
                    "retry_count",
                    0,
                ),

            "max_retries":
                result.get(
                    "max_retries",
                    2,
                ),

            "replan_count":
                result.get(
                    "replan_count",
                    0,
                ),

            "max_replans":
                result.get(
                    "max_replans",
                    2,
                ),

            "failed_tool":
                result.get(
                    "failed_tool"
                ),

            "last_tool_error":
                result.get(
                    "last_tool_error"
                ),
        },

        "trace":
            result.get(
                "trace",
                [],
            ),

        "error":
            result.get(
                "error"
            ),
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health"
)
def api_health():

    return {

        "status":
            "healthy",

        "service":
            "InsightFlow Analytics API",

        "active_datasets":
            dataset_manager.count(),

        "agentic_layer":
            "enabled",
    }


# ============================================================
# PREPARE DATASET
# ============================================================

@router.post(
    "/datasets"
)
async def prepare_dataset(
    file: UploadFile = File(...)
):

    (
        original_filename,
        stored_filename,
        file_path,
    ) = await save_uploaded_file(
        file
    )

    try:

        # ----------------------------------------------------
        # Dataset preparation remains deterministic.
        # ----------------------------------------------------

        workflow = (
            AnalyticsWorkflow()
        )

        result = (
            workflow.prepare_dataset(
                file_path=str(
                    file_path
                ),
                original_filename=(
                    original_filename
                ),
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Dataset preparation returned "
                    "an invalid response."
                ),
            )

        if not result.get(
            "success",
            False,
        ):

            raise HTTPException(
                status_code=500,
                detail={

                    "message":
                        "Dataset preparation failed.",

                    "error":
                        result.get(
                            "error"
                        ),
                },
            )

        dataset_id = (
            result.get(
                "dataset_id"
            )
        )

        if not dataset_id:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Dataset preparation completed "
                    "without creating a dataset session."
                ),
            )

        response = (
            build_preparation_response(
                result=result,
                original_filename=(
                    original_filename
                ),
                stored_filename=(
                    stored_filename
                ),
            )
        )

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
                    (
                        "Unexpected dataset "
                        "preparation error."
                    ),

                "error":
                    str(
                        error
                    ),
            },
        ) from error


# ============================================================
# GET DATASET
# ============================================================

@router.get(
    "/datasets/{dataset_id}"
)
def get_dataset(
    dataset_id: str
):

    dataset_id = (
        validate_dataset_id(
            dataset_id
        )
    )

    try:

        dataset_info = (
            dataset_manager
            .get_dataset_info(
                dataset_id
            )
        )

        if dataset_info is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Dataset session not found."
                ),
            )

        return make_json_safe({

            "success":
                True,

            "dataset":
                dataset_info,
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={

                "message":
                    (
                        "Could not retrieve "
                        "dataset session."
                    ),

                "error":
                    str(
                        error
                    ),
            },
        ) from error


# ============================================================
# ASK DATASET THROUGH AGENT GRAPH
# ============================================================

@router.post(
    "/datasets/{dataset_id}/ask"
)
def ask_dataset(
    dataset_id: str,
    question: str = Form(...),
):
    """
    Run a natural-language analytics request through
    the autonomous InsightFlow agent graph.

    Dataset preparation is NOT repeated.
    """

    dataset_id = (
        validate_dataset_id(
            dataset_id
        )
    )

    question = (
        validate_question(
            question
        )
    )

    # --------------------------------------------------------
    # Check before invoking agent
    # --------------------------------------------------------

    if not dataset_manager.exists(
        dataset_id
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset session not found."
            ),
        )

    try:

        result = (
            agent_service.run(
                dataset_id=dataset_id,
                question=question,
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Agent service returned "
                    "an invalid response."
                ),
            )

        # ----------------------------------------------------
        # Agent completed unsuccessfully
        # ----------------------------------------------------

        if not result.get(
            "success",
            False,
        ):

            error = (
                result.get(
                    "error"
                )
                or
                result.get(
                    "last_tool_error"
                )
                or
                "Agent execution failed."
            )

            if (
                error
                ==
                "Dataset session not found."
            ):

                raise HTTPException(
                    status_code=404,
                    detail=error,
                )

            raise HTTPException(
                status_code=500,
                detail=make_json_safe({

                    "message":
                        (
                            "Agent analytics "
                            "request failed."
                        ),

                    "error":
                        error,

                    "failed_tool":
                        result.get(
                            "failed_tool"
                        ),

                    "plan":
                        result.get(
                            "plan",
                            [],
                        ),

                    "executed_tools":
                        result.get(
                            "executed_tools",
                            [],
                        ),

                    "replan_count":
                        result.get(
                            "replan_count",
                            0,
                        ),

                    "trace":
                        result.get(
                            "trace",
                            [],
                        ),
                }),
            )

        response = (
            build_agent_response(
                result
            )
        )

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except (
        TypeError,
        ValueError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={

                "message":
                    (
                        "Unexpected agent "
                        "execution error."
                    ),

                "error":
                    str(
                        error
                    ),
            },
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

    dataset_id = (
        validate_dataset_id(
            dataset_id
        )
    )

    try:

        deleted = (
            dataset_manager
            .delete_dataset(
                dataset_id
            )
        )

        if not deleted:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Dataset session not found."
                ),
            )

        return {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "message":
                (
                    "Dataset session deleted."
                ),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={

                "message":
                    (
                        "Could not delete "
                        "dataset session."
                    ),

                "error":
                    str(
                        error
                    ),
            },
        ) from error


# ============================================================
# LEGACY ONE-SHOT ANALYZE
# ============================================================

@router.post(
    "/analyze"
)
async def analyze_dataset(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Backward-compatible one-shot endpoint.

    Flow:

        upload
          ↓
        deterministic preparation
          ↓
        DatasetManager
          ↓
        autonomous agent graph
          ↓
        response
    """

    question = (
        validate_question(
            question
        )
    )

    (
        original_filename,
        stored_filename,
        file_path,
    ) = await save_uploaded_file(
        file
    )

    try:

        # ====================================================
        # 1. PREPARE DATASET
        # ====================================================

        workflow = (
            AnalyticsWorkflow()
        )

        preparation_result = (
            workflow.prepare_dataset(
                file_path=str(
                    file_path
                ),
                original_filename=(
                    original_filename
                ),
            )
        )

        if not isinstance(
            preparation_result,
            dict,
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Dataset preparation returned "
                    "an invalid response."
                ),
            )

        if not preparation_result.get(
            "success",
            False,
        ):

            raise HTTPException(
                status_code=500,
                detail={

                    "message":
                        (
                            "Dataset preparation failed."
                        ),

                    "error":
                        preparation_result.get(
                            "error"
                        ),
                },
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
                ),
            )

        # ====================================================
        # 2. RUN AGENT
        # ====================================================

        agent_result = (
            agent_service.run(
                dataset_id=dataset_id,
                question=question,
            )
        )

        if not isinstance(
            agent_result,
            dict,
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Agent service returned "
                    "an invalid response."
                ),
            )

        if not agent_result.get(
            "success",
            False,
        ):

            raise HTTPException(
                status_code=500,
                detail=make_json_safe({

                    "message":
                        (
                            "Agent analytics "
                            "request failed."
                        ),

                    "error":
                        (
                            agent_result.get(
                                "error"
                            )
                            or
                            agent_result.get(
                                "last_tool_error"
                            )
                        ),

                    "failed_tool":
                        agent_result.get(
                            "failed_tool"
                        ),

                    "plan":
                        agent_result.get(
                            "plan",
                            [],
                        ),

                    "executed_tools":
                        agent_result.get(
                            "executed_tools",
                            [],
                        ),

                    "replan_count":
                        agent_result.get(
                            "replan_count",
                            0,
                        ),
                }),
            )

        # ====================================================
        # 3. DATASET INFORMATION
        # ====================================================

        dataset_info = (
            dataset_manager
            .get_dataset_info(
                dataset_id
            )
        )

        # ====================================================
        # 4. PREPARATION RESPONSE
        # ====================================================

        preparation_response = (
            build_preparation_response(
                result=preparation_result,
                original_filename=(
                    original_filename
                ),
                stored_filename=(
                    stored_filename
                ),
            )
        )

        # ====================================================
        # 5. AGENT RESPONSE
        # ====================================================

        analysis_response = (
            build_agent_response(
                agent_result
            )
        )

        # ====================================================
        # FINAL ONE-SHOT RESPONSE
        # ====================================================

        response = {

            "success":
                True,

            "dataset_id":
                dataset_id,

            "file":
                preparation_response.get(
                    "file"
                ),

            "question":
                question,

            "dataset":
                preparation_response.get(
                    "dataset"
                ),

            "dataset_info":
                dataset_info,

            "schema":
                preparation_response.get(
                    "schema",
                    {},
                ),

            "quality":
                preparation_response.get(
                    "quality",
                    {},
                ),

            "anomalies":
                preparation_response.get(
                    "anomalies",
                    {},
                ),

            "cleaning":
                preparation_response.get(
                    "cleaning",
                    {},
                ),

            "eda":
                preparation_response.get(
                    "eda",
                    {},
                ),

            "visualizations":
                preparation_response.get(
                    "visualizations",
                    [],
                ),

            "statistical_findings":
                preparation_response.get(
                    "statistical_findings",
                    [],
                ),

            # =================================================
            # AGENTIC ANALYSIS
            # =================================================

            "agent":
                analysis_response.get(
                    "agent",
                    {},
                ),

            "analysis":
                analysis_response.get(
                    "analysis",
                    {},
                ),

            "visualization":
                analysis_response.get(
                    "visualization"
                ),

            "insight":
                analysis_response.get(
                    "insight"
                ),

            "execution":
                analysis_response.get(
                    "execution",
                    {},
                ),

            "trace":
                analysis_response.get(
                    "trace",
                    [],
                ),
        }

        return make_json_safe(
            response
        )

    except HTTPException:
        raise

    except (
        TypeError,
        ValueError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={

                "message":
                    (
                        "Unexpected analytics "
                        "workflow error."
                    ),

                "error":
                    str(
                        error
                    ),
            },
        ) from error