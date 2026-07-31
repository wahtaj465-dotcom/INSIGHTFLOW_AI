import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from frontend.api_client import api

from frontend.components.overview import render_overview
from frontend.components.quality import render_quality
from frontend.components.charts import render_charts_section
from frontend.components.question import render_question_section



import streamlit as st

from frontend.api_client import api

from frontend.components.overview import (
    render_overview,
)

from frontend.components.quality import (
    render_quality,
)

from frontend.components.charts import (
    render_charts_section,
)

from frontend.components.question import (
    render_question_section,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {

    # Dataset currently registered with backend.
    "dataset_id": None,

    # Complete prepared dataset metadata.
    "dataset_info": None,

    # Name of uploaded file associated with dataset_id.
    "uploaded_filename": None,

    # Last response from the agent graph.
    "analysis_response": None,

    # Last natural-language question.
    "last_question": "",
}


def initialize_session_state():
    """
    Initialize frontend session state.
    """

    for (
        key,
        default_value,
    ) in DEFAULT_SESSION_STATE.items():

        if key not in st.session_state:

            st.session_state[
                key
            ] = default_value


initialize_session_state()


# ============================================================
# RESET DATASET STATE
# ============================================================

def reset_dataset_state():
    """
    Clear all frontend state associated with the
    active dataset and agent conversation.
    """

    st.session_state.dataset_id = None

    st.session_state.dataset_info = None

    st.session_state.uploaded_filename = None

    st.session_state.analysis_response = None

    st.session_state.last_question = ""


# ============================================================
# DELETE ACTIVE DATASET
# ============================================================

def delete_active_dataset():
    """
    Attempt to remove the active dataset from the backend.

    Frontend state is cleared even if backend deletion
    fails because the user explicitly requested the
    dataset to be removed from the current session.
    """

    dataset_id = (
        st.session_state.dataset_id
    )

    if dataset_id:

        try:

            api.delete_dataset(
                dataset_id
            )

        except Exception:
            pass

    reset_dataset_state()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 InsightFlow AI"
)

st.caption(
    "Generative AI powered analytics platform for "
    "automated preprocessing, exploratory data analysis, "
    "visualization and autonomous natural-language "
    "analytics."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "InsightFlow AI"
    )

    st.write(
        "Upload a dataset and explore it using "
        "automated analytics and an autonomous "
        "AI analytics agent."
    )

    st.divider()

    # ========================================================
    # BACKEND STATUS
    # ========================================================

    health = (
        api.health_check()
    )

    backend_available = (
        isinstance(
            health,
            dict,
        )
        and
        health.get(
            "status"
        )
        ==
        "healthy"
    )

    if backend_available:

        st.success(
            "Backend connected"
        )

    else:

        st.error(
            "Backend unavailable"
        )

        if isinstance(
            health,
            dict,
        ):

            backend_error = (
                health.get(
                    "error"
                )
            )

            if backend_error:

                st.caption(
                    backend_error
                )

    st.divider()

    # ========================================================
    # ACTIVE DATASET
    # ========================================================

    if st.session_state.dataset_id:

        st.caption(
            "Active Dataset"
        )

        if st.session_state.uploaded_filename:

            st.write(
                f"**"
                f"{st.session_state.uploaded_filename}"
                f"**"
            )

        st.caption(
            "Dataset ID"
        )

        st.code(
            st.session_state.dataset_id
        )

        st.divider()

        # ====================================================
        # CLEAR DATASET
        # ====================================================

        if st.button(
            "Clear Dataset",
            width="stretch",
            key="clear_active_dataset",
        ):

            delete_active_dataset()

            st.rerun()


# ============================================================
# BACKEND AVAILABILITY
# ============================================================

if not backend_available:

    st.warning(
        "InsightFlow cannot process datasets until "
        "the FastAPI backend is available."
    )


# ============================================================
# 1. DATASET UPLOAD
# ============================================================

st.header(
    "1. Upload Dataset"
)

st.write(
    "Upload a CSV or Excel dataset. InsightFlow will "
    "prepare the data and generate the information "
    "required by the analytics workflow."
)


uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Supported formats: CSV, XLSX and XLS."
    ),
    key="dataset_file_uploader",
)


# ============================================================
# SELECTED FILE
# ============================================================

if uploaded_file is not None:

    st.write(
        f"Selected file: "
        f"**{uploaded_file.name}**"
    )


# ============================================================
# PROCESS DATASET
# ============================================================

if uploaded_file is not None:

    analyze_button = st.button(
        "Analyze Dataset",
        type="primary",
        width="stretch",
        disabled=(
            not backend_available
        ),
        key="analyze_dataset_button",
    )

    if analyze_button:

        try:

            # =================================================
            # REMOVE PREVIOUS DATASET
            # =================================================
            #
            # If another dataset is already active, remove
            # its backend session before registering the new
            # dataset.
            # =================================================

            previous_dataset_id = (
                st.session_state.dataset_id
            )

            if previous_dataset_id:

                try:

                    api.delete_dataset(
                        previous_dataset_id
                    )

                except Exception:
                    pass

            # =================================================
            # CLEAR OLD FRONTEND STATE
            # =================================================

            reset_dataset_state()

            # =================================================
            # UPLOAD + PREPARE DATASET
            # =================================================

            with st.spinner(
                "Preparing dataset and generating "
                "dataset analysis..."
            ):

                preparation = (
                    api.upload_dataset(
                        uploaded_file
                    )
                )

            # =================================================
            # VALIDATE RESPONSE
            # =================================================

            if not isinstance(
                preparation,
                dict,
            ):

                raise RuntimeError(
                    "Backend returned an invalid "
                    "dataset preparation response."
                )

            # =================================================
            # EXTRACT DATASET ID
            # ============================================================
            #
            # Supports:
            #
            # {
            #     "dataset_id": "..."
            # }
            #
            # and:
            #
            # {
            #     "dataset": {
            #         "dataset_id": "..."
            #     }
            # }
            #
            # =================================================

            dataset_id = (
                preparation.get(
                    "dataset_id"
                )
            )

            if not dataset_id:

                preparation_dataset = (
                    preparation.get(
                        "dataset"
                    )
                )

                if isinstance(
                    preparation_dataset,
                    dict,
                ):

                    dataset_id = (
                        preparation_dataset.get(
                            "dataset_id"
                        )
                    )

            # =================================================
            # DATASET ID REQUIRED
            # =================================================

            if not dataset_id:

                st.error(
                    "The backend processed the dataset "
                    "but did not return a dataset ID."
                )

                with st.expander(
                    "Backend Preparation Response"
                ):

                    st.json(
                        preparation
                    )

                st.stop()

            # =================================================
            # STORE BASIC DATASET STATE
            # =================================================

            st.session_state.dataset_id = (
                dataset_id
            )

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )

            # =================================================
            # LOAD COMPLETE DATASET METADATA
            # =================================================

            with st.spinner(
                "Loading prepared dataset..."
            ):

                dataset_info = (
                    api.get_dataset(
                        dataset_id
                    )
                )

            # =================================================
            # VALIDATE METADATA
            # =================================================

            if not isinstance(
                dataset_info,
                dict,
            ):

                raise RuntimeError(
                    "Backend returned invalid "
                    "dataset metadata."
                )

            # =================================================
            # STORE DATASET
            # =================================================

            st.session_state.dataset_info = (
                dataset_info
            )

            # =================================================
            # CLEAR AGENT STATE
            # =================================================

            st.session_state.analysis_response = None

            st.session_state.last_question = ""

            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "Dataset processed successfully."
            )

            st.rerun()

        except Exception as error:

            # Avoid retaining partially initialized state.

            reset_dataset_state()

            st.error(
                f"Dataset processing failed: "
                f"{error}"
            )


# ============================================================
# WAIT FOR ACTIVE DATASET
# ============================================================

if not st.session_state.dataset_id:

    st.info(
        "Upload and analyze a dataset to begin."
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

dataset = (
    st.session_state.dataset_info
)


# ============================================================
# RECOVER METADATA AFTER RERUN
# ============================================================
#
# Streamlit may rerun while dataset_id still exists but the
# cached metadata is unavailable. Reload it from FastAPI.
# ============================================================

if not isinstance(
    dataset,
    dict,
):

    try:

        with st.spinner(
            "Loading dataset analysis..."
        ):

            dataset = (
                api.get_dataset(
                    st.session_state.dataset_id
                )
            )

        if not isinstance(
            dataset,
            dict,
        ):

            raise RuntimeError(
                "Backend returned invalid "
                "dataset metadata."
            )

        st.session_state.dataset_info = (
            dataset
        )

    except Exception as error:

        st.error(
            f"Could not load dataset: "
            f"{error}"
        )

        st.info(
            "Clear the active dataset and upload "
            "the file again."
        )

        st.stop()


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

if not isinstance(
    dataset,
    dict,
):

    st.error(
        "Invalid dataset information received "
        "from the backend."
    )

    st.stop()


# ============================================================
# LEGACY FAILURE RESPONSE SUPPORT
# ============================================================
#
# api.get_dataset() currently unwraps the dataset object.
# This check keeps the UI compatible if an older backend
# response reaches this component.
# ============================================================

if dataset.get(
    "success"
) is False:

    st.error(
        "Backend reported that dataset "
        "processing failed."
    )

    backend_error = (
        dataset.get(
            "error"
        )
    )

    if backend_error:

        st.write(
            backend_error
        )

    with st.expander(
        "Backend Response"
    ):

        st.json(
            dataset
        )

    st.stop()


# ============================================================
# ACTIVE DATASET HEADING
# ============================================================

if st.session_state.uploaded_filename:

    st.caption(
        f"Analyzing: "
        f"{st.session_state.uploaded_filename}"
    )


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

try:

    render_overview(
        dataset
    )

except Exception as error:

    st.error(
        f"Could not render dataset overview: "
        f"{error}"
    )


# ============================================================
# 3. DATA QUALITY
# ============================================================

try:

    render_quality(
        dataset
    )

except Exception as error:

    st.error(
        f"Could not render data quality analysis: "
        f"{error}"
    )


# ============================================================
# 4. AUTOMATED EDA & VISUALIZATIONS
# ============================================================

try:

    render_charts_section(
        dataset
    )

except Exception as error:

    st.error(
        f"Could not render automated EDA: "
        f"{error}"
    )


# ============================================================
# 5. AGENTIC NATURAL-LANGUAGE ANALYTICS
# ============================================================

try:

    render_question_section(
        st.session_state.dataset_id
    )

except Exception as error:

    st.error(
        f"Could not load the InsightFlow "
        f"agent interface: {error}"
    )


# ============================================================
# DEVELOPER INFORMATION
# ============================================================

st.divider()


with st.expander(
    "Developer: Raw Dataset Metadata"
):

    st.json(
        dataset
    )