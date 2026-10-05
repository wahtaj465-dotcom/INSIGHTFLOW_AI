import json

import pandas as pd
import requests
import streamlit as st

from frontend.api_client import api
from frontend.utils.chart_renderer import (
    render_chart,
)

from frontend.utils.executive_report import build_executive_report


# ============================================================
# SESSION STATE
# ============================================================

def _initialize_question_state():
    """
    Initialize state used by the interactive
    agent question interface.
    """

    if "analysis_response" not in st.session_state:
        st.session_state.analysis_response = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    if "business_report" not in st.session_state:
        st.session_state.business_report = None

    if "business_report_pdf" not in st.session_state:
        st.session_state.business_report_pdf = None

    if "business_report_path" not in st.session_state:
        st.session_state.business_report_path = None


# ============================================================
# QUERY RESULT TABLE
# ============================================================

def _render_query_result(
    result,
):
    """
    Render SQL/analytical results safely.
    """

    if result is None:
        return

    if isinstance(result, pd.DataFrame):

        if result.empty:
            st.info(
                "The query returned no rows."
            )
            return

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        return

    if isinstance(result, dict):

        try:
            dataframe = pd.DataFrame(
                [result]
            )

        except Exception:
            st.json(result)
            return

    elif isinstance(result, list):

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
            st.json(result)
            return

    else:

        st.write(result)
        return

    if dataframe.empty:

        st.info(
            "The query returned no rows."
        )

        return

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# INSIGHT RENDERER
# ============================================================

def _render_insight(
    insight,
):
    """
    Render either plain-text or structured
    analytical insight.
    """

    if insight is None:
        return

    if isinstance(
        insight,
        str,
    ):

        if insight.strip():
            st.markdown(
                insight
            )

        return

    if isinstance(
        insight,
        dict,
    ):

        summary = insight.get(
            "summary"
        )

        if summary:
            st.markdown(
                str(summary)
            )

        key_insights = (
            insight.get(
                "key_insights"
            )
            or
            insight.get(
                "insights"
            )
        )

        if key_insights:

            st.markdown(
                "### Key Analytical Insights"
            )

            if isinstance(
                key_insights,
                list,
            ):

                for item in key_insights:
                    st.markdown(
                        f"- {item}"
                    )

            else:
                st.markdown(
                    str(key_insights)
                )

        chart_specifications = (
            insight.get(
                "chart_specifications"
            )
        )

        if chart_specifications:

            with st.expander(
                "Chart Specifications"
            ):

                if isinstance(
                    chart_specifications,
                    dict,
                ):

                    st.json(
                        chart_specifications
                    )

                elif isinstance(
                    chart_specifications,
                    list,
                ):

                    for item in chart_specifications:

                        if isinstance(
                            item,
                            str,
                        ):
                            st.markdown(
                                f"- {item}"
                            )

                        else:
                            st.json(
                                item
                            )

                else:
                    st.write(
                        chart_specifications
                    )

        limitations = (
            insight.get(
                "limitations"
            )
            or
            insight.get(
                "data_quality_limitations"
            )
        )

        if limitations:

            st.markdown(
                "### Data Quality & Limitations"
            )

            if isinstance(
                limitations,
                list,
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

    if isinstance(
        insight,
        list,
    ):

        for item in insight:
            st.markdown(
                f"- {item}"
            )

        return

    st.write(
        insight
    )


# ============================================================
# WORKFLOW STATUS
# ============================================================

def _render_workflow_status(
    response,
):
    """
    Show a compact summary of the autonomous
    agent execution.
    """

    completed = response.get(
        "completed"
    )

    success = response.get(
        "success"
    )

    failed_tool = response.get(
        "failed_tool"
    )

    error = response.get(
        "error"
    )

    if success is True:

        st.success(
            "Agent workflow completed successfully."
        )

    elif failed_tool or error:

        st.error(
            "The agent could not complete the "
            "analysis successfully."
        )

    elif completed is True:

        st.success(
            "Analysis completed."
        )

    elif completed is False:

        st.info(
            "Agent workflow did not reach completion."
        )


# ============================================================
# AGENT PLAN
# ============================================================

def _render_agent_plan(
    response,
):
    """
    Display planner and execution information
    without cluttering the main analytical result.
    """

    plan = response.get(
        "plan"
    ) or []

    executed_tools = response.get(
        "executed_tools"
    ) or []

    intent = response.get(
        "intent"
    )

    planner_source = response.get(
        "planner_source"
    )

    plan_reasoning = response.get(
        "plan_reasoning"
    )

    if not (
        plan
        or executed_tools
        or intent
        or planner_source
        or plan_reasoning
    ):
        return

    with st.expander(
        "Agent Workflow"
    ):

        if intent:

            st.markdown(
                f"**Intent:** `{intent}`"
            )

        if planner_source:

            st.markdown(
                f"**Planner:** `{planner_source}`"
            )

        if plan:

            st.markdown(
                "**Execution Plan**"
            )

            st.write(
                " → ".join(
                    str(tool)
                    for tool in plan
                )
            )

        if executed_tools:

            st.markdown(
                "**Executed Tools**"
            )

            for tool in executed_tools:
                st.markdown(
                    f"- ✓ `{tool}`"
                )

        if plan_reasoning:

            st.markdown(
                "**Planning Reasoning**"
            )

            st.write(
                plan_reasoning
            )


# ============================================================
# RECOVERY DETAILS
# ============================================================

def _render_recovery_details(
    response,
):
    """
    Show recovery/replanning information when
    the workflow needed autonomous recovery.
    """

    replan_count = response.get(
        "replan_count",
        0,
    )

    retry_count = response.get(
        "retry_count",
        0,
    )

    failed_tool = response.get(
        "failed_tool"
    )

    last_tool_error = response.get(
        "last_tool_error"
    )

    if not (
        replan_count
        or retry_count
        or failed_tool
        or last_tool_error
    ):
        return

    with st.expander(
        "Recovery Details"
    ):

        st.write(
            f"Replans: {replan_count}"
        )

        st.write(
            f"Retries: {retry_count}"
        )

        if failed_tool:

            st.write(
                f"Failed tool: {failed_tool}"
            )

        if last_tool_error:

            st.write(
                f"Last tool error: "
                f"{last_tool_error}"
            )


# ============================================================
# TOOL OUTPUTS
# ============================================================

def _render_tool_outputs(
    response,
):
    """
    Render outputs from future/dynamically
    registered tools.

    Known outputs such as sql_result,
    visualization and insight are rendered in
    their dedicated sections instead.
    """

    tool_outputs = response.get(
        "tool_outputs"
    )

    if not isinstance(
        tool_outputs,
        dict,
    ):

        return

    if not tool_outputs:
        return

    known_outputs = {
        "generated_sql",
        "sql_result",
        "visualization",
        "insight",
    }

    dynamic_outputs = {
        key: value
        for key, value
        in tool_outputs.items()
        if key not in known_outputs
    }

    if not dynamic_outputs:
        return

    st.subheader(
        "Additional Agent Outputs"
    )

    for (
        output_name,
        output_value,
    ) in dynamic_outputs.items():

        st.markdown(
            f"**{output_name.replace('_', ' ').title()}**"
        )

        if isinstance(
            output_value,
            (dict, list),
        ):

            st.json(
                output_value
            )

        else:

            st.write(
                output_value
            )


# ============================================================
# TRACE
# ============================================================

def _render_trace(
    response,
):
    """
    Show orchestration trace for debugging and
    explainability.
    """

    trace = response.get(
        "trace"
    )

    if not trace:
        return

    with st.expander(
        "Agent Execution Trace"
    ):

        st.json(
            trace
        )


# ============================================================
# BUSINESS REPORT REQUEST
# ============================================================

def _generate_business_report(
    dataset_id,
    question,
    analysis_response,
):
    """
    Send the already completed analysis response
    to the backend business-report endpoint.

    IMPORTANT:

    The agent is NOT executed again.

    The exact response already displayed to the user
    is reused to construct the report.
    """

    if not dataset_id:

        raise ValueError(
            "Dataset ID is required."
        )

    if not question:

        raise ValueError(
            "Question is required."
        )

    if not isinstance(
        analysis_response,
        dict,
    ):

        raise ValueError(
            "Analysis response is invalid."
        )

    # --------------------------------------------------------
    # Backend URL
    # --------------------------------------------------------

    base_url = getattr(
        api,
        "base_url",
        None,
    )

    if not base_url:

        raise RuntimeError(
            "API client does not expose base_url."
        )

    base_url = base_url.rstrip(
        "/"
    )

    url = (
        f"{base_url}"
        f"/datasets/{dataset_id}/report"
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = requests.post(

        url,

        data={
            "question":
                question,

            "analysis_response":
                json.dumps(
                    analysis_response,
                    default=str,
                ),
        },

        timeout=120,
    )

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_data = (
                response.json()
            )

            raise RuntimeError(
                str(
                    error_data
                )
            )

        except ValueError:

            raise RuntimeError(
                response.text
            )

    try:

        result = (
            response.json()
        )

    except ValueError as error:

        raise RuntimeError(
            "Backend returned invalid JSON."
        ) from error

    if not isinstance(
        result,
        dict,
    ):

        raise RuntimeError(
            "Invalid business report response."
        )

    if not result.get(
        "success",
        False,
    ):

        raise RuntimeError(
            result.get(
                "error",
                "Business report generation failed.",
            )
        )

    return result


# ============================================================
# REPORT TEXT EXTRACTION
# ============================================================

def _extract_report_text(
    report_response,
):
    """
    Support multiple possible report-service response
    field names without changing the backend contract.
    """

    if not isinstance(
        report_response,
        dict,
    ):

        return ""

    possible_fields = [

        "report",

        "markdown",

        "markdown_report",

        "report_markdown",

        "content",

        "text",
    ]

    for field in possible_fields:

        value = report_response.get(
            field
        )

        if isinstance(
            value,
            str,
        ) and value.strip():

            return value

    return ""


# ============================================================
# BUSINESS REPORT UI
# ============================================================

def _render_business_report_section(
    dataset_id,
):
    """
    Render the business stakeholder report section.
    """

    response = (
        st.session_state.get(
            "analysis_response"
        )
    )

    if not response:
        return

    st.divider()

    st.subheader(
        "Business Stakeholder Report"
    )

    st.write(
        "Create a human-readable executive report "
        "from the analysis above. The existing "
        "analytical response is reused, so the "
        "agent is not executed again."
    )

    generate_report = st.button(

        "Generate Executive Report",

        type="secondary",

        width="stretch",

        key="generate_executive_report",
    )

    if generate_report:

        question = (
            st.session_state.get(
                "last_question"
            )
            or
            response.get(
                "question",
                "",
            )
        )

        if not question:

            st.error(
                "The original question could not be found."
            )

            return

        try:

            with st.spinner(
                "Preparing your executive business report..."
            ):

                report_response = (
                    _generate_business_report(

                        dataset_id=
                            dataset_id,

                        question=
                            question,

                        analysis_response=
                            response,
                    )
                )

            st.session_state.business_report = (
                report_response
            )

            st.success(
                "Executive report generated successfully."
            )

        except Exception as error:

            st.session_state.business_report = (
                None
            )

            st.error(
                f"Could not generate report: {error}"
            )

    # --------------------------------------------------------
    # DISPLAY EXISTING REPORT
    # --------------------------------------------------------

    business_report = (
        st.session_state.get(
            "business_report"
        )
    )

    if not business_report:
        return

    markdown_report = (
        _extract_report_text(
            business_report
        )
    )

    if not markdown_report:

        st.warning(
            "The report service returned no report content."
        )

        with st.expander(
            "Report Response"
        ):

            st.json(
                business_report
            )

        return

    # --------------------------------------------------------
    # REPORT PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "Preview Executive Report",
        expanded=True,
    ):

        st.markdown(
            markdown_report
        )

    # --------------------------------------------------------
    # STAKEHOLDER-READY PDF
    # --------------------------------------------------------

    try:

        pdf_bytes, saved_pdf_path = (
            build_executive_report(
                question=question,
                analysis_response=response,
                report_markdown=markdown_report,
                dataset_id=dataset_id,
                output_dir="reports",
            )
        )

        st.session_state.business_report_pdf = (
            pdf_bytes
        )

        st.session_state.business_report_path = (
            saved_pdf_path
        )

        st.success(
            "Stakeholder-ready executive PDF created successfully."
        )

        st.caption(
            f"Saved to: {saved_pdf_path}"
        )

        pdf_file_name = (
            saved_pdf_path
            .replace("\\", "/")
            .split("/")[-1]
        )

        st.download_button(

            label=
                "Download Executive Report PDF",

            data=
                pdf_bytes,

            file_name=
                pdf_file_name,

            mime=
                "application/pdf",

            type="primary",

            width="stretch",

            key=
                "download_executive_report_pdf",
        )

    except Exception as error:

        st.error(
            "PDF generation failed: "
            f"{error}"
        )

    # --------------------------------------------------------
    # MARKDOWN DOWNLOAD
    # --------------------------------------------------------

    st.download_button(

        label=
            "Download Markdown Report",

        data=
            markdown_report,

        file_name=(
            "InsightFlow_"
            "Executive_Report.md"
        ),

        mime=
            "text/markdown",

        width="stretch",

        key=
            "download_executive_report_md",
    )


# ============================================================
# ANALYSIS RESPONSE
# ============================================================

def _render_analysis_response(
    response,
):
    """
    Render the complete response returned by
    AgentService/FastAPI.
    """

    if not isinstance(
        response,
        dict,
    ):

        st.error(
            "InsightFlow returned an invalid response."
        )

        return

    st.divider()

    # --------------------------------------------------------
    # WORKFLOW STATUS
    # --------------------------------------------------------

    _render_workflow_status(
        response
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = (
        response.get(
            "question"
        )
        or
        st.session_state.last_question
    )

    if question:

        st.subheader(
            "Question"
        )

        st.write(
            question
        )

    # --------------------------------------------------------
    # GENERATED SQL
    # --------------------------------------------------------

    generated_sql = (
        response.get(
            "generated_sql"
        )
        or
        response.get(
            "sql"
        )
    )

    # Support nested response shape used by the
    # updated API.

    if not generated_sql:

        analysis = response.get(
            "analysis"
        )

        if isinstance(
            analysis,
            dict,
        ):

            generated_sql = (
                analysis.get(
                    "generated_sql"
                )
            )

    if generated_sql:

        st.subheader(
            "Generated SQL"
        )

        st.code(
            generated_sql,
            language="sql",
        )

    # --------------------------------------------------------
    # SQL / ANALYTICAL RESULT
    # --------------------------------------------------------

    result = response.get(
        "sql_result"
    )

    if result is None:
        result = response.get(
            "result"
        )

    if result is None:
        result = response.get(
            "query_result"
        )

    if result is None:
        result = response.get(
            "results"
        )

    # Support nested API response.

    if result is None:

        analysis = response.get(
            "analysis"
        )

        if isinstance(
            analysis,
            dict,
        ):

            result = (
                analysis.get(
                    "sql_result"
                )
            )

    if result is not None:

        st.subheader(
            "Analysis Result"
        )

        _render_query_result(
            result
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    visualization = response.get(
        "visualization"
    )

    if visualization is None:
        visualization = response.get(
            "chart"
        )

    if visualization:

        st.subheader(
            "Visualization"
        )

        try:

            render_chart(
                visualization
            )

        except Exception as error:

            st.warning(
                "The visualization could not be rendered."
            )

            with st.expander(
                "Visualization Data"
            ):

                st.json(
                    visualization
                )

                st.caption(
                    str(error)
                )

    # --------------------------------------------------------
    # INSIGHT
    # --------------------------------------------------------

    insight = response.get(
        "insight"
    )

    if insight is None:
        insight = response.get(
            "generated_insight"
        )

    if insight:

        st.subheader(
            "AI Generated Insight"
        )

        _render_insight(
            insight
        )

    # --------------------------------------------------------
    # STATISTICAL FINDINGS
    # --------------------------------------------------------

    statistical_findings = response.get(
        "statistical_findings"
    )

    if statistical_findings:

        st.subheader(
            "Statistical Findings"
        )

        if isinstance(
            statistical_findings,
            list,
        ):

            for finding in statistical_findings:

                if isinstance(
                    finding,
                    str,
                ):
                    st.markdown(
                        f"- {finding}"
                    )

                else:
                    st.write(
                        finding
                    )

        else:

            st.write(
                statistical_findings
            )

    # --------------------------------------------------------
    # FUTURE TOOL OUTPUTS
    # --------------------------------------------------------

    _render_tool_outputs(
        response
    )

    # --------------------------------------------------------
    # AGENT WORKFLOW
    # --------------------------------------------------------

    _render_agent_plan(
        response
    )

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    _render_recovery_details(
        response
    )

    # --------------------------------------------------------
    # TRACE
    # --------------------------------------------------------

    _render_trace(
        response
    )

    # --------------------------------------------------------
    # RAW RESPONSE
    # --------------------------------------------------------

    with st.expander(
        "Raw Agent Response"
    ):

        st.json(
            response
        )


# ============================================================
# MAIN QUESTION COMPONENT
# ============================================================

def render_question_section(
    dataset_id,
):
    """
    Render the natural-language interface for the
    completed InsightFlow agentic analytics workflow.

    Runtime:

        User Question
              ↓
        frontend.api_client
              ↓
        FastAPI
              ↓
        AgentService
              ↓
        Agent Graph
              ↓
        Planner
              ↓
        Dependency Resolution
              ↓
        Tool Execution
              ↓
        Observer / Recovery
              ↓
        Final Analytical Response
              ↓
        Streamlit
              ↓
        Business Report
              ↓
        PDF / Markdown
    """

    _initialize_question_state()

    st.divider()

    st.header(
        "5. Ask InsightFlow"
    )

    st.write(
        "Ask a natural-language question about your "
        "dataset. InsightFlow will autonomously select "
        "the required analytical tools, resolve their "
        "dependencies, execute the workflow, recover "
        "from supported failures, and return the result."
    )

    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    if not dataset_id:

        st.warning(
            "Prepare a dataset before asking questions."
        )

        return

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_area(

        "Ask a question about your dataset",

        value="",

        placeholder=(
            "Example: Compare average sales by region "
            "and explain the main differences."
        ),

        height=100,

        key="insightflow_question_input",
    )

    ask_button = st.button(

        "Ask InsightFlow",

        type="primary",

        width="stretch",

        key="insightflow_ask_button",
    )

    # --------------------------------------------------------
    # ASK AGENT
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
                    "InsightFlow agent is analyzing "
                    "your question..."
                ):

                    response = (
                        api.ask_dataset(
                            dataset_id,
                            cleaned_question,
                        )
                    )

                st.session_state.analysis_response = (
                    response
                )

                st.session_state.last_question = (
                    cleaned_question
                )

                # Clear an old report whenever a
                # completely new analysis is performed.

                st.session_state.business_report = (
                    None
                )

                st.session_state.business_report_pdf = (
                    None
                )

                st.session_state.business_report_path = (
                    None
                )

                st.session_state.business_report_pdf = (
                    None
                )

                st.session_state.business_report_path = (
                    None
                )

            except Exception as error:

                st.session_state.analysis_response = (
                    None
                )

                st.session_state.business_report = (
                    None
                )

                st.error(
                    f"Analysis failed: {error}"
                )

    # --------------------------------------------------------
    # EXISTING RESPONSE
    # --------------------------------------------------------

    response = (
        st.session_state.analysis_response
    )

    if response:

        _render_analysis_response(
            response
        )

        # ----------------------------------------------------
        # BUSINESS REPORT
        # ----------------------------------------------------

        _render_business_report_section(
            dataset_id
        )