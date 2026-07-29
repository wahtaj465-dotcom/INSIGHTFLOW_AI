import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# HELPERS
# ============================================================

def _to_dataframe(data):
    """
    Safely convert visualization data into a DataFrame.
    """

    if data is None:

        return pd.DataFrame()


    if isinstance(
        data,
        pd.DataFrame
    ):

        return data.copy()


    if isinstance(
        data,
        dict
    ):

        try:

            return pd.DataFrame(
                data
            )

        except ValueError:

            try:

                return pd.DataFrame(
                    [data]
                )

            except Exception:

                return pd.DataFrame()


    if isinstance(
        data,
        (
            list,
            tuple
        )
    ):

        try:

            return pd.DataFrame(
                data
            )

        except Exception:

            return pd.DataFrame()


    return pd.DataFrame()


def _pretty_name(name):

    if name is None:

        return ""


    return (
        str(name)
        .replace("_", " ")
        .strip()
        .title()
    )


def _render_reason(chart):

    reason = chart.get(
        "reason"
    )


    if reason:

        st.caption(
            reason
        )


def _get_chart_type(chart):

    chart_type = (
        chart.get(
            "chart_type"
        )
        or
        chart.get(
            "type"
        )
        or
        ""
    )


    return (
        str(chart_type)
        .strip()
        .lower()
        .replace(
            "-",
            "_"
        )
        .replace(
            " ",
            "_"
        )
    )


# ============================================================
# KPI
# ============================================================

def _render_kpi(chart):

    title = chart.get(
        "title",
        "Result"
    )


    value = chart.get(
        "value",
        "N/A"
    )


    delta = chart.get(
        "delta"
    )


    if delta is not None:

        st.metric(
            label=title,
            value=value,
            delta=delta
        )

    else:

        st.metric(
            label=title,
            value=value
        )


# ============================================================
# BAR CHART
# ============================================================

def _render_bar(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No data available for this chart."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
        )
    )


    if (
        x not in df.columns
        or
        y not in df.columns
    ):

        st.warning(
            "Bar chart specification "
            "is missing valid axes."
        )

        return


    kwargs = {

        "data_frame":
            df,

        "x":
            x,

        "y":
            y,

        "title":
            chart.get(
                "title"
            ),

        "labels": {

            x:
                _pretty_name(
                    x
                ),

            y:
                _pretty_name(
                    y
                )
        }
    }


    if (
        group
        and
        group in df.columns
    ):

        kwargs[
            "color"
        ] = group

        kwargs[
            "barmode"
        ] = "group"


    fig = px.bar(
        **kwargs
    )


    fig.update_layout(

        xaxis_title=
            _pretty_name(
                x
            ),

        yaxis_title=
            _pretty_name(
                y
            ),

        legend_title=(
            _pretty_name(
                group
            )
            if group
            else None
        )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# HISTOGRAM
# ============================================================

def _render_histogram(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No histogram data available."
        )

        return


    # --------------------------------------------------------
    # Backend pre-binned histogram
    # --------------------------------------------------------

    if (
        "bin" in df.columns
        and
        "frequency" in df.columns
    ):

        fig = px.bar(

            df,

            x="bin",

            y="frequency",

            title=chart.get(
                "title"
            ),

            labels={

                "bin":
                    _pretty_name(
                        chart.get(
                            "x",
                            "Value"
                        )
                    ),

                "frequency":
                    "Frequency"
            }
        )


        fig.update_layout(

            xaxis_title=
                _pretty_name(
                    chart.get(
                        "x",
                        "Value"
                    )
                ),

            yaxis_title=
                "Frequency"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )

        return


    # --------------------------------------------------------
    # Raw numerical values
    # --------------------------------------------------------

    x = chart.get(
        "x"
    )


    if (
        x
        and
        x in df.columns
    ):

        fig = px.histogram(

            df,

            x=x,

            title=chart.get(
                "title"
            ),

            labels={
                x:
                    _pretty_name(
                        x
                    )
            }
        )


        fig.update_layout(

            xaxis_title=
                _pretty_name(
                    x
                ),

            yaxis_title=
                "Frequency"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )

        return


    st.warning(
        "Histogram specification is invalid."
    )


# ============================================================
# LINE CHART
# ============================================================

def _render_line(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No data available for this chart."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
        )
    )


    if (
        x not in df.columns
        or
        y not in df.columns
    ):

        st.warning(
            "Line chart specification "
            "is missing valid axes."
        )

        return


    kwargs = {

        "data_frame":
            df,

        "x":
            x,

        "y":
            y,

        "title":
            chart.get(
                "title"
            ),

        "markers":
            True,

        "labels": {

            x:
                _pretty_name(
                    x
                ),

            y:
                _pretty_name(
                    y
                )
        }
    }


    if (
        group
        and
        group in df.columns
    ):

        kwargs[
            "color"
        ] = group


    fig = px.line(
        **kwargs
    )


    fig.update_layout(

        xaxis_title=
            _pretty_name(
                x
            ),

        yaxis_title=
            _pretty_name(
                y
            )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# SCATTER
# ============================================================

def _render_scatter(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No scatter data available."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
        )
    )


    if (
        x not in df.columns
        or
        y not in df.columns
    ):

        st.warning(
            "Scatter chart specification "
            "is missing valid axes."
        )

        return


    kwargs = {

        "data_frame":
            df,

        "x":
            x,

        "y":
            y,

        "title":
            chart.get(
                "title"
            ),

        "labels": {

            x:
                _pretty_name(
                    x
                ),

            y:
                _pretty_name(
                    y
                )
        }
    }


    if (
        group
        and
        group in df.columns
    ):

        kwargs[
            "color"
        ] = group


    fig = px.scatter(
        **kwargs
    )


    fig.update_layout(

        xaxis_title=
            _pretty_name(
                x
            ),

        yaxis_title=
            _pretty_name(
                y
            )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# BOX PLOT
# ============================================================

def _render_box(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No box plot data available."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
        )
    )


    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    if (
        y
        and
        y in df.columns
        and
        len(df) > 1
    ):

        kwargs = {

            "data_frame":
                df,

            "y":
                y,

            "title":
                chart.get(
                    "title"
                ),

            "points":
                "outliers"
        }


        if (
            x
            and
            x in df.columns
        ):

            kwargs[
                "x"
            ] = x


        if (
            group
            and
            group in df.columns
        ):

            kwargs[
                "color"
            ] = group


        fig = px.box(
            **kwargs
        )


        fig.update_layout(

            xaxis_title=(
                _pretty_name(
                    x
                )
                if x
                else None
            ),

            yaxis_title=
                _pretty_name(
                    y
                ),

            legend_title=(
                _pretty_name(
                    group
                )
                if group
                else None
            )
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        return


    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

    required = {

        "min",
        "q1",
        "median",
        "q3",
        "max"
    }


    if required.issubset(
        set(
            df.columns
        )
    ):

        fig = go.Figure()


        for index, row in (
            df.iterrows()
        ):

            if (
                x
                and
                x in df.columns
            ):

                name = str(
                    row[x]
                )

            else:

                name = (
                    chart.get(
                        "title"
                    )
                    or
                    f"Distribution {index + 1}"
                )


            fig.add_trace(

                go.Box(

                    q1=[
                        row[
                            "q1"
                        ]
                    ],

                    median=[
                        row[
                            "median"
                        ]
                    ],

                    q3=[
                        row[
                            "q3"
                        ]
                    ],

                    lowerfence=[
                        row[
                            "min"
                        ]
                    ],

                    upperfence=[
                        row[
                            "max"
                        ]
                    ],

                    name=name
                )
            )


        fig.update_layout(

            title=
                chart.get(
                    "title"
                ),

            yaxis_title=
                _pretty_name(
                    y
                )
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        return


    st.warning(
        "Box plot data format "
        "is not supported."
    )


# ============================================================
# VIOLIN
# ============================================================

def _render_violin(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No violin plot data available."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
        )
    )


    if (
        not y
        or
        y not in df.columns
    ):

        st.warning(
            "Violin plot requires "
            "a numerical Y column."
        )

        return


    kwargs = {

        "data_frame":
            df,

        "y":
            y,

        "title":
            chart.get(
                "title"
            ),

        "box":
            True,

        "points":
            "outliers"
    }


    if (
        x
        and
        x in df.columns
    ):

        kwargs[
            "x"
        ] = x


    if (
        group
        and
        group in df.columns
    ):

        kwargs[
            "color"
        ] = group


    fig = px.violin(
        **kwargs
    )


    fig.update_layout(

        xaxis_title=(
            _pretty_name(
                x
            )
            if x
            else None
        ),

        yaxis_title=
            _pretty_name(
                y
            ),

        legend_title=(
            _pretty_name(
                group
            )
            if group
            else None
        )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# PIE
# ============================================================

def _render_pie(chart):

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )


    if df.empty:

        st.info(
            "No pie chart data available."
        )

        return


    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )


    if (
        x not in df.columns
        or
        y not in df.columns
    ):

        st.warning(
            "Pie chart specification "
            "is invalid."
        )

        return


    fig = px.pie(

        df,

        names=x,

        values=y,

        title=chart.get(
            "title"
        )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# MAIN RENDERER
# ============================================================

def render_chart(chart):
    """
    Render an InsightFlow visualization specification.

    Supported chart types:

        kpi
        bar
        histogram
        line
        scatter
        box
        boxplot
        violin
        pie
    """


    if not chart:

        st.info(
            "No visualization was generated."
        )

        return


    if not isinstance(
        chart,
        dict
    ):

        st.warning(
            "Invalid visualization specification."
        )

        return


    chart_type = (
        _get_chart_type(
            chart
        )
    )


    _render_reason(
        chart
    )


    renderers = {

        "kpi":
            _render_kpi,

        "metric":
            _render_kpi,

        "bar":
            _render_bar,

        "bar_chart":
            _render_bar,

        "histogram":
            _render_histogram,

        "hist":
            _render_histogram,

        "line":
            _render_line,

        "line_chart":
            _render_line,

        "scatter":
            _render_scatter,

        "scatter_plot":
            _render_scatter,

        "box":
            _render_box,

        "boxplot":
            _render_box,

        "box_plot":
            _render_box,

        "violin":
            _render_violin,

        "violin_plot":
            _render_violin,

        "pie":
            _render_pie,

        "pie_chart":
            _render_pie
    }


    renderer = (
        renderers.get(
            chart_type
        )
    )


    if renderer is None:

        st.info(
            f"Visualization type "
            f"'{chart_type or 'unknown'}' "
            f"is not supported by the "
            f"current frontend."
        )


        with st.expander(
            "Visualization specification"
        ):

            st.json(
                chart
            )


        return


    try:

        renderer(
            chart
        )


    except Exception as error:

        st.error(
            "Could not render visualization: "
            f"{error}"
        )


        with st.expander(
            "Visualization specification"
        ):

            st.json(
                chart
            )