import streamlit as st


def render_quality(dataset):

    st.header("3. Data Quality")

    dataset_data = dataset.get(
        "dataset",
        dataset
    )

    quality = dataset_data.get(
        "quality",
        dataset.get("quality", {})
    )

    before_score = quality.get(
        "before_score"
    )

    after_score = quality.get(
        "after_score"
    )

    improvement = quality.get(
        "improvement"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Before Cleaning",
        (
            f"{before_score:.2f}"
            if isinstance(before_score, (int, float))
            else "N/A"
        )
    )

    col2.metric(
        "After Cleaning",
        (
            f"{after_score:.2f}"
            if isinstance(after_score, (int, float))
            else "N/A"
        )
    )

    col3.metric(
        "Improvement",
        (
            f"+{improvement:.2f}"
            if isinstance(improvement, (int, float))
            else "N/A"
        )
    )

    st.subheader(
        "Cleaning Operations"
    )

    cleaning_log = quality.get(
        "cleaning_log",
        []
    )

    if cleaning_log:

        for operation in cleaning_log:
            st.write(
                f"✓ {operation}"
            )

    else:

        st.info(
            "No cleaning operations were required."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    quality_report = quality.get(
        "quality_report",
        {}
    )

    missing_values = quality_report.get(
        "missing_values",
        {}
    )

    if missing_values:

        st.subheader(
            "Remaining Missing Values"
        )

        remaining = {
            column: count
            for column, count
            in missing_values.items()
            if count
        }

        if remaining:

            st.json(remaining)

        else:

            st.success(
                "No missing values remain after cleaning."
            )

    # --------------------------------------------------------
    # Anomalies
    # --------------------------------------------------------

    anomalies = quality.get(
        "anomalies",
        {}
    )

    if anomalies:

        with st.expander(
            "Detected Anomalies"
        ):

            st.json(
                anomalies
            )