import streamlit as st


def render_overview(dataset):

    st.header("2. Dataset Overview")

    dataset_data = dataset.get(
        "dataset",
        dataset
    )

    filename = dataset_data.get(
        "original_filename",
        "Unknown"
    )

    rows = dataset_data.get(
        "rows",
        0
    )

    columns = dataset_data.get(
        "columns",
        []
    )

    column_count = dataset_data.get(
        "column_count",
        len(columns)
    )

    quality = dataset_data.get(
        "quality",
        dataset.get("quality", {})
    )

    after_score = quality.get(
        "after_score"
    )

    improvement = quality.get(
        "improvement"
    )

    st.write(
        f"**Dataset:** {filename}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{rows:,}"
    )

    col2.metric(
        "Columns",
        column_count
    )

    col3.metric(
        "Quality Score",
        (
            f"{after_score:.2f}/100"
            if isinstance(after_score, (int, float))
            else "N/A"
        )
    )

    col4.metric(
        "Quality Improvement",
        (
            f"+{improvement:.2f}"
            if isinstance(improvement, (int, float))
            else "N/A"
        )
    )

    st.subheader(
        "Dataset Columns"
    )

    if columns:

        st.write(
            ", ".join(columns)
        )

    else:

        st.info(
            "No column information available."
        )