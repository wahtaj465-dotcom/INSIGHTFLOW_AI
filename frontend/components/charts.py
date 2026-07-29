import streamlit as st

from frontend.utils.chart_renderer import (
    render_chart
)


# ============================================================
# NORMALIZE CHART COLLECTION
# ============================================================

def _get_eda_charts(dataset):
    """
    Extract automated EDA charts from dataset metadata.

    Current backend contract:

        dataset["eda_charts"]

    Older versions of InsightFlow used:

        dataset["charts"]
        dataset["visualizations"]

    We keep those as compatibility fallbacks so old API
    responses do not immediately break the frontend.
    """

    if not isinstance(
        dataset,
        dict
    ):

        return []


    # --------------------------------------------------------
    # Current contract
    # --------------------------------------------------------

    charts = dataset.get(
        "eda_charts"
    )


    if isinstance(
        charts,
        list
    ):

        return charts


    # --------------------------------------------------------
    # Compatibility: old "charts" key
    # --------------------------------------------------------

    charts = dataset.get(
        "charts"
    )


    if isinstance(
        charts,
        list
    ):

        return charts


    # --------------------------------------------------------
    # Compatibility: upload endpoint may call them
    # "visualizations"
    # --------------------------------------------------------

    charts = dataset.get(
        "visualizations"
    )


    if isinstance(
        charts,
        list
    ):

        return charts


    return []


# ============================================================
# RENDER SINGLE CHART CARD
# ============================================================

def _render_chart_card(
    chart,
    index
):
    """
    Render one automated EDA chart.
    """

    if not isinstance(
        chart,
        dict
    ):

        st.warning(
            f"Visualization {index + 1} "
            f"has an invalid specification."
        )

        return


    title = chart.get(
        "title"
    )


    if not title:

        title = (
            f"Visualization {index + 1}"
        )


    st.subheader(
        title
    )


    try:

        render_chart(
            chart
        )

    except Exception as error:

        st.error(
            "Could not render this "
            f"visualization: {error}"
        )


        with st.expander(
            "Visualization specification"
        ):

            st.json(
                chart
            )


# ============================================================
# AUTOMATED EDA SECTION
# ============================================================

def render_charts_section(
    dataset
):
    """
    Render automated EDA visualizations returned by
    the InsightFlow backend.

    Expected structure:

        dataset = {
            ...
            "eda_charts": [
                {
                    "chart_type": "bar",
                    "title": "...",
                    "x": "...",
                    "y": "...",
                    "data": [...]
                }
            ]
        }
    """


    st.divider()


    st.header(
        "4. Automated EDA & Visualizations"
    )


    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if not isinstance(
        dataset,
        dict
    ):

        st.warning(
            "Dataset metadata is unavailable."
        )

        return


    # --------------------------------------------------------
    # Get charts
    # --------------------------------------------------------

    charts = (
        _get_eda_charts(
            dataset
        )
    )


    # --------------------------------------------------------
    # No charts
    # --------------------------------------------------------

    if not charts:

        st.info(
            "No automated visualizations were generated "
            "for this dataset."
        )


        # Helpful while developing InsightFlow

        with st.expander(
            "Visualization debug information"
        ):

            st.write(
                "Available dataset keys:"
            )

            st.code(
                ", ".join(
                    map(
                        str,
                        dataset.keys()
                    )
                )
            )


            st.write(
                "Backend reported chart count:"
            )

            st.write(
                dataset.get(
                    "eda_chart_count",
                    "Not provided"
                )
            )


        return


    # --------------------------------------------------------
    # Chart count
    # --------------------------------------------------------

    chart_count = len(
        charts
    )


    st.caption(
        f"Generated {chart_count} visualization"
        f"{'s' if chart_count != 1 else ''} "
        f"from the prepared dataset."
    )


    # --------------------------------------------------------
    # TWO-COLUMN GRID
    # --------------------------------------------------------

    for index in range(
        0,
        chart_count,
        2
    ):

        left_column, right_column = (
            st.columns(
                2,
                gap="large"
            )
        )


        # ----------------------------------------------------
        # LEFT CHART
        # ----------------------------------------------------

        with left_column:

            _render_chart_card(
                charts[index],
                index
            )


        # ----------------------------------------------------
        # RIGHT CHART
        # ----------------------------------------------------

        if (
            index + 1
            <
            chart_count
        ):

            with right_column:

                _render_chart_card(
                    charts[
                        index + 1
                    ],
                    index + 1
                )