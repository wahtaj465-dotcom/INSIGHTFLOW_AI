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
        .replace(
            "_",
            " "
        )
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


def _get_group(chart):

    return (
        chart.get(
            "group"
        )
        or
        chart.get(
            "color"
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

    group = _get_group(
        chart
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
# STACKED BAR
# ============================================================

def _render_stacked_bar(chart):
    """
    Render categorical composition.

    Expected backend metadata:

        chart_type = stacked_bar
        x = category
        y = count/value
        color = second category
    """

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )

    if df.empty:

        st.info(
            "No stacked bar data available."
        )

        return

    x = chart.get(
        "x"
    )

    y = chart.get(
        "y"
    )

    group = _get_group(
        chart
    )

    if (
        not x
        or
        x not in df.columns
    ):

        st.warning(
            "Stacked bar chart requires "
            "a valid X column."
        )

        return

    if (
        not y
        or
        y not in df.columns
    ):

        st.warning(
            "Stacked bar chart requires "
            "a valid Y column."
        )

        return

    if (
        not group
        or
        group not in df.columns
    ):

        st.warning(
            "Stacked bar chart requires "
            "a valid grouping column."
        )

        return

    fig = px.bar(

        df,

        x=x,

        y=y,

        color=group,

        barmode="stack",

        title=chart.get(
            "title"
        ),

        labels={

            x:
                _pretty_name(
                    x
                ),

            y:
                _pretty_name(
                    y
                ),

            group:
                _pretty_name(
                    group
                )
        }
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

        legend_title=
            _pretty_name(
                group
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
    # BACKEND PRE-BINNED HISTOGRAM
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
                    "Bin",

                "frequency":
                    "Frequency"
            }
        )

        fig.update_layout(

            xaxis_title=
                "Bin",

            yaxis_title=
                "Frequency"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        return

    # --------------------------------------------------------
    # RAW NUMERICAL VALUES
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

    group = _get_group(
        chart
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

    # --------------------------------------------------------
    # Attempt datetime ordering
    # --------------------------------------------------------

    plot_df = df.copy()

    if not pd.api.types.is_numeric_dtype(
        plot_df[x]
    ):

        try:

            converted = pd.to_datetime(
                plot_df[x],
                errors="coerce"
            )

            if (
                converted.notna().mean()
                >= 0.8
            ):

                plot_df[x] = converted

                plot_df = (
                    plot_df.sort_values(
                        x
                    )
                )

        except Exception:

            pass

    kwargs = {

        "data_frame":
            plot_df,

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
        group in plot_df.columns
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

    group = _get_group(
        chart
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

    group = _get_group(
        chart
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
            and
            x != y
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
                if (
                    x
                    and
                    x != y
                )
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
                "column"
                in df.columns
            ):

                name = str(
                    row[
                        "column"
                    ]
                )

            elif (
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

    group = _get_group(
        chart
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
# HEATMAP
# ============================================================

def _render_heatmap(chart):
    """
    Render correlation heatmap.

    Expected backend format:

        [
            {
                "x": "column_a",
                "y": "column_b",
                "correlation": 0.82
            },
            ...
        ]
    """

    df = _to_dataframe(
        chart.get(
            "data"
        )
    )

    if df.empty:

        st.info(
            "No heatmap data available."
        )

        return

    x = chart.get(
        "x",
        "x"
    )

    y = chart.get(
        "y",
        "y"
    )

    metadata = (
        chart.get(
            "metadata"
        )
        or
        {}
    )

    value_column = (
        metadata.get(
            "value"
        )
        or
        chart.get(
            "value"
        )
        or
        "correlation"
    )

    if (
        x not in df.columns
        or
        y not in df.columns
        or
        value_column not in df.columns
    ):

        st.warning(
            "Heatmap specification "
            "is missing required fields."
        )

        return

    try:

        matrix = (
            df.pivot(
                index=y,
                columns=x,
                values=value_column
            )
        )

    except Exception as error:

        st.warning(
            "Could not construct heatmap matrix: "
            f"{error}"
        )

        return

    if matrix.empty:

        st.info(
            "No heatmap matrix could be generated."
        )

        return

    fig = px.imshow(

        matrix,

        text_auto=".2f",

        aspect="auto",

        title=chart.get(
            "title"
        ),

        labels={

            "x":
                "Variable",

            "y":
                "Variable",

            "color":
                _pretty_name(
                    value_column
                )
        }
    )

    fig.update_layout(

        xaxis_title=
            "Variable",

        yaxis_title=
            "Variable"
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
        metric

        bar
        stacked_bar

        histogram

        line

        scatter

        box
        boxplot

        violin

        heatmap

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

        # -----------------------------------------
        # KPI
        # -----------------------------------------

        "kpi":
            _render_kpi,

        "metric":
            _render_kpi,

        # -----------------------------------------
        # BAR
        # -----------------------------------------

        "bar":
            _render_bar,

        "bar_chart":
            _render_bar,

        # -----------------------------------------
        # STACKED BAR
        # -----------------------------------------

        "stacked_bar":
            _render_stacked_bar,

        "stacked_bar_chart":
            _render_stacked_bar,

        "stacked":
            _render_stacked_bar,

        # -----------------------------------------
        # HISTOGRAM
        # -----------------------------------------

        "histogram":
            _render_histogram,

        "hist":
            _render_histogram,

        # -----------------------------------------
        # LINE
        # -----------------------------------------

        "line":
            _render_line,

        "line_chart":
            _render_line,

        # -----------------------------------------
        # SCATTER
        # -----------------------------------------

        "scatter":
            _render_scatter,

        "scatter_plot":
            _render_scatter,

        # -----------------------------------------
        # BOX
        # -----------------------------------------

        "box":
            _render_box,

        "boxplot":
            _render_box,

        "box_plot":
            _render_box,

        # -----------------------------------------
        # VIOLIN
        # -----------------------------------------

        "violin":
            _render_violin,

        "violin_plot":
            _render_violin,

        # -----------------------------------------
        # HEATMAP
        # -----------------------------------------

        "heatmap":
            _render_heatmap,

        "heat_map":
            _render_heatmap,

        "correlation_heatmap":
            _render_heatmap,

        # -----------------------------------------
        # PIE
        # -----------------------------------------

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