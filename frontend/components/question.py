import pandas as pd
import streamlit as st

from frontend.api_client import api
from frontend.utils.chart_renderer import render_chart


# ============================================================
# SESSION STATE
# ============================================================

def _initialize_question_state():

    if "analysis_response" not in st.session_state:
        st.session_state.analysis_response = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


# ============================================================
# QUERY RESULT TABLE
# ============================================================

def _render_query_result(result):

    if not result:
        st.info(
            "The query returned no rows."
        )

        return

    try:

        dataframe = pd.DataFrame(
            result
        )

    except Exception:

        st.warning(
            "Could not convert query result "
            "into a table."
        )

        st.json(result)

        return

    if dataframe.empty:

        st.info(
            "The query returned no rows."
        )

        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True
    )


# ============================================================
# INSIGHT RENDERER
# ============================================================

def _render_insight(insight):

    if not insight:
        return

    st.subheader(
        "AI Generated Insight"
    )

    # Plain string

    if isinstance(
        insight,
        str
    ):

        st.markdown(insight)

        return

    # Structured dictionary

    if isinstance(
        insight,
        dict
    ):

        summary = insight.get(
            "summary"
        )

        if summary:

            st.markdown(summary)

        chart_specifications = insight.get(
            "chart_specifications"
        )

        if chart_specifications:

            st.markdown(
                "### Chart Specifications"
            )

            if isinstance(
                chart_specifications,
                str
            ):

                st.markdown(
                    chart_specifications
                )

            elif isinstance(
                chart_specifications,
                list
            ):

                for item in chart_specifications:

                    if isinstance(
                        item,
                        str
                    ):

                        st.markdown(
                            f"- {item}"
                        )

                    else:

                        st.json(item)

            elif isinstance(
                chart_specifications,
                dict
            ):

                for key, value in (
                    chart_specifications.items()
                ):

                    st.markdown(
                        f"**{key}:** {value}"
                    )

        key_insights = insight.get(
            "key_insights"
        )

        if key_insights:

            st.divider()

            st.markdown(
                "### Key Analytical Insights"
            )

            if isinstance(
                key_insights,
                list
            ):

                for item in key_insights:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.markdown(
                    str(key_insights)
                )

        limitations = (
            insight.get("limitations")
            or insight.get(
                "data_quality_limitations"
            )
        )

        if limitations:

            st.divider()

            st.markdown(
                "### Data Quality & Limitations"
            )

            if isinstance(
                limitations,
                list
            ):

                for item in limitations:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.markdown(
                    str(limitations)
                )

        return

    # Other formats

    st.write(insight)


# ============================================================
# ANALYSIS RESPONSE
# ============================================================

def _render_analysis_response(
    response
):

    if not response:

        return

    st.divider()

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = (
        response.get("question")
        or st.session_state.last_question
    )

    if question:

        st.subheader(
            "Question"
        )

        st.write(question)

    # --------------------------------------------------------
    # GENERATED SQL
    # --------------------------------------------------------

    sql = (
        response.get("sql")
        or response.get(
            "generated_sql"
        )
    )

    if sql:

        st.subheader(
            "Generated SQL"
        )

        st.code(
            sql,
            language="sql"
        )

    # --------------------------------------------------------
    # QUERY RESULT
    # --------------------------------------------------------

    result = (
        response.get("result")
        or response.get(
            "query_result"
        )
        or response.get(
            "results"
        )
    )

    if result is not None:

        st.subheader(
            "Query Result"
        )

        _render_query_result(
            result
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    chart = (
        response.get("chart")
        or response.get(
            "visualization"
        )
    )

    if chart:

        st.subheader(
            "Visualization"
        )

        render_chart(
            chart
        )

    else:

        st.caption(
            "No visualization was generated "
            "for this result."
        )

    # --------------------------------------------------------
    # AI INSIGHT
    # --------------------------------------------------------

    insight = (
        response.get("insight")
        or response.get(
            "analysis"
        )
        or response.get(
            "generated_insight"
        )
    )

    if insight:

        st.divider()

        _render_insight(
            insight
        )

    # --------------------------------------------------------
    # RELEVANT COLUMNS
    # --------------------------------------------------------

    relevant_columns = response.get(
        "relevant_columns"
    )

    if relevant_columns:

        with st.expander(
            "Relevant Columns"
        ):

            if isinstance(
                relevant_columns,
                list
            ):

                st.write(
                    ", ".join(
                        map(
                            str,
                            relevant_columns
                        )
                    )
                )

            else:

                st.write(
                    relevant_columns
                )

    # --------------------------------------------------------
    # SQL AGENT DETAILS
    # --------------------------------------------------------

    sql_details = (
        response.get(
            "sql_agent"
        )
        or response.get(
            "sql_details"
        )
    )

    if sql_details:

        with st.expander(
            "SQL Agent Execution Details"
        ):

            st.json(
                sql_details
            )

    # --------------------------------------------------------
    # RAW RESPONSE
    # --------------------------------------------------------

    with st.expander(
        "View Raw Analysis Response"
    ):

        st.json(
            response
        )


# ============================================================
# MAIN QUESTION COMPONENT
# ============================================================

def render_question_section(
    dataset_id
):
    """
    Render the natural-language analytics interface.

    Pipeline:

        User question
              ↓
        FastAPI
              ↓
        SQL Agent
              ↓
        DuckDB
              ↓
        Visualization Service
              ↓
        Insight Generator
              ↓
        Streamlit
    """

    _initialize_question_state()

    st.divider()

    st.header(
        "5. Ask InsightFlow"
    )

    st.write(
        "Ask a natural-language question about your dataset. "
        "InsightFlow will generate SQL, execute the query, "
        "visualize the result and generate analytical insight."
    )

    question = st.text_area(
        "Ask a question about your dataset",
        placeholder=(
            "Example: Compare average distance from home "
            "across risk levels."
        ),
        height=100
    )

    ask_button = st.button(
        "Ask InsightFlow",
        type="primary",
        width="stretch"
    )

    # --------------------------------------------------------
    # ASK BACKEND
    # --------------------------------------------------------

    if ask_button:

        cleaned_question = (
            question.strip()
        )

        if not cleaned_question:

            st.warning(
                "Enter a question first."
            )

        else:

            try:

                with st.spinner(
                    "InsightFlow is analyzing your question..."
                ):

                    response = (
                        api.ask_dataset(
                            dataset_id,
                            cleaned_question
                        )
                    )

                st.session_state.analysis_response = (
                    response
                )

                st.session_state.last_question = (
                    cleaned_question
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

    # --------------------------------------------------------
    # SHOW EXISTING RESPONSE
    # --------------------------------------------------------

    if st.session_state.analysis_response:

        _render_analysis_response(
            st.session_state.analysis_response
        )