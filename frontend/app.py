import streamlit as st

from frontend.api_client import api

from frontend.components.overview import render_overview
from frontend.components.quality import render_quality
from frontend.components.charts import render_charts_section
from frontend.components.question import render_question_section


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_SESSION_STATE = {
    "dataset_id": None,
    "dataset_info": None,
    "uploaded_filename": None,
    "analysis_response": None,
    "last_question": ""
}


for key, default_value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# HELPER: RESET DATASET STATE
# ============================================================

def reset_dataset_state():
    """
    Clear all frontend state associated with the current
    dataset and previous analytical question.
    """

    st.session_state.dataset_id = None
    st.session_state.dataset_info = None
    st.session_state.uploaded_filename = None

    st.session_state.analysis_response = None
    st.session_state.last_question = ""


# ============================================================
# HEADER
# ============================================================

st.title("📊 InsightFlow AI")

st.caption(
    "Generative AI powered analytics platform for "
    "automated preprocessing, EDA, visualization, "
    "natural-language SQL and insight generation."
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
        "automated analytics and natural language."
    )

    st.divider()


    # ========================================================
    # BACKEND STATUS
    # ========================================================

    health = (
        api.health_check()
    )


    if (
        isinstance(health, dict)
        and
        health.get("status") == "healthy"
    ):

        st.success(
            "Backend connected"
        )

    else:

        st.error(
            "Backend unavailable"
        )

        if isinstance(
            health,
            dict
        ):

            error = (
                health.get(
                    "error"
                )
            )

            if error:

                st.caption(
                    error
                )


    st.divider()


    # ========================================================
    # ACTIVE DATASET
    # ========================================================

    if st.session_state.dataset_id:

        st.caption(
            "Active Dataset"
        )


        if (
            st.session_state.uploaded_filename
        ):

            st.write(
                f"**{st.session_state.uploaded_filename}**"
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
            width="stretch"
        ):

            dataset_id = (
                st.session_state.dataset_id
            )


            try:

                api.delete_dataset(
                    dataset_id
                )

            except Exception:

                # Even if backend deletion fails,
                # clear frontend state.
                pass


            reset_dataset_state()

            st.rerun()


# ============================================================
# 1. DATASET UPLOAD
# ============================================================

st.header(
    "1. Upload Dataset"
)


uploaded_file = (
    st.file_uploader(
        "Choose a dataset",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        help=(
            "Supported formats: CSV, XLSX and XLS."
        )
    )
)


# ============================================================
# PROCESS DATASET
# ============================================================

if uploaded_file is not None:

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )


    if st.button(
        "Analyze Dataset",
        type="primary",
        width="stretch"
    ):

        try:

            # =================================================
            # UPLOAD + BACKEND PREPARATION
            # =================================================

            with st.spinner(
                "Preparing dataset, analyzing data quality, "
                "cleaning data and generating EDA..."
            ):

                preparation = (
                    api.upload_dataset(
                        uploaded_file
                    )
                )


            # =================================================
            # VALIDATE UPLOAD RESPONSE
            # =================================================

            if not isinstance(
                preparation,
                dict
            ):

                raise RuntimeError(
                    "Backend returned an invalid "
                    "upload response."
                )


            # =================================================
            # EXTRACT DATASET ID
            # =================================================
            #
            # Support both:
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
                    dict
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
                    "Backend processed the dataset but "
                    "did not return a dataset ID."
                )


                with st.expander(
                    "Backend Upload Response"
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
            # CLEAR PREVIOUS QUESTION STATE
            # =================================================

            st.session_state.analysis_response = None
            st.session_state.last_question = ""


            # =================================================
            # FETCH COMPLETE DATASET INFORMATION
            # =================================================
            #
            # IMPORTANT:
            #
            # api.get_dataset() now returns the INNER
            # dataset object directly.
            #
            # Therefore DO NOT do:
            #
            # response.get("dataset")
            #
            # =================================================

            with st.spinner(
                "Loading complete dataset analysis..."
            ):

                dataset_info = (
                    api.get_dataset(
                        dataset_id
                    )
                )


            # =================================================
            # VALIDATE DATASET RESPONSE
            # =================================================

            if not isinstance(
                dataset_info,
                dict
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


            st.success(
                "Dataset processed successfully."
            )


            # =================================================
            # RERUN UI
            # =================================================

            st.rerun()


        except Exception as error:

            st.error(
                f"Dataset processing failed: {error}"
            )


# ============================================================
# WAIT UNTIL DATASET EXISTS
# ============================================================

if not st.session_state.dataset_id:

    st.info(
        "Upload a dataset to begin analysis."
    )

    st.stop()


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

dataset = (
    st.session_state.dataset_info
)


# If Streamlit reruns and metadata is missing,
# retrieve it again from FastAPI.

if not dataset:

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
            dict
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
            f"Could not load dataset: {error}"
        )

        st.stop()


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

if not isinstance(
    dataset,
    dict
):

    st.error(
        "Invalid dataset response received "
        "from backend."
    )

    st.stop()


# ============================================================
# OPTIONAL BACKEND FAILURE CHECK
# ============================================================
#
# api.get_dataset() normally returns the inner dataset,
# so this should usually not exist.
#
# Keeping it makes the frontend tolerant of older backend
# response formats.
# ============================================================

if dataset.get("success") is False:

    st.error(
        "Backend reported that dataset "
        "processing failed."
    )


    with st.expander(
        "Backend Response"
    ):

        st.json(
            dataset
        )


    st.stop()


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

render_overview(
    dataset
)


# ============================================================
# 3. DATA QUALITY
# ============================================================

render_quality(
    dataset
)


# ============================================================
# 4. AUTOMATED EDA & VISUALIZATIONS
# ============================================================

render_charts_section(
    dataset
)


# ============================================================
# 5. NATURAL LANGUAGE ANALYTICS
# ============================================================

render_question_section(
    st.session_state.dataset_id
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