import math
import re

import pandas as pd


class VisualizationService:

    def __init__(
        self,
        max_categories=15,
        max_charts=10
    ):

        self.max_categories = (
            max_categories
        )

        self.max_charts = (
            max_charts
        )


    # ========================================================
    # JSON SAFE VALUE
    # ========================================================

    def _json_safe_value(
        self,
        value
    ):

        if value is None:

            return None


        try:

            if pd.isna(
                value
            ):

                return None

        except (
            TypeError,
            ValueError
        ):

            pass


        if isinstance(
            value,
            pd.Timestamp
        ):

            return (
                value.isoformat()
            )


        if hasattr(
            value,
            "item"
        ):

            try:

                return (
                    value.item()
                )

            except (
                ValueError,
                TypeError,
                AttributeError
            ):

                pass


        return value


    # ========================================================
    # DATAFRAME -> RECORDS
    # ========================================================

    def dataframe_to_records(
        self,
        df
    ):

        if df is None:

            return []


        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a Pandas DataFrame."
            )


        records = []


        for record in (
            df.to_dict(
                orient="records"
            )
        ):

            safe_record = {}

            for key, value in (
                record.items()
            ):

                safe_record[
                    str(key)
                ] = (
                    self._json_safe_value(
                        value
                    )
                )


            records.append(
                safe_record
            )


        return records


    # ========================================================
    # CREATE CHART
    # ========================================================

    def _create_chart(
        self,
        chart_type,
        title,
        data,
        x=None,
        y=None,
        color=None,
        reason=None,
        metadata=None
    ):

        chart = {

            "chart_type":
                chart_type,

            "title":
                title,

            "x":
                x,

            "y":
                y,

            "data":
                data,

            "reason":
                reason
        }


        if color is not None:

            chart[
                "color"
            ] = color


        if metadata:

            chart[
                "metadata"
            ] = metadata


        return chart


    # ========================================================
    # TYPE CHECKS
    # ========================================================

    def _is_numeric(
        self,
        series
    ):

        return (
            pd.api.types
            .is_numeric_dtype(
                series
            )
        )


    def _is_datetime(
        self,
        series
    ):

        return (
            pd.api.types
            .is_datetime64_any_dtype(
                series
            )
        )


    # ========================================================
    # DETECT REQUESTED CHART TYPE
    # ========================================================

    def _detect_requested_chart_type(
        self,
        question
    ):

        if not isinstance(
            question,
            str
        ):

            return None


        text = (
            question
            .strip()
            .lower()
        )


        patterns = [

            (
                "box",
                [
                    r"\bbox\s*plot\b",
                    r"\bboxplot\b",
                    r"\bbox\s*chart\b"
                ]
            ),

            (
                "violin",
                [
                    r"\bviolin\s*plot\b",
                    r"\bviolin\b"
                ]
            ),

            (
                "histogram",
                [
                    r"\bhistogram\b"
                ]
            ),

            (
                "scatter",
                [
                    r"\bscatter\s*plot\b",
                    r"\bscatterplot\b",
                    r"\bscatter\b"
                ]
            ),

            (
                "line",
                [
                    r"\bline\s*chart\b",
                    r"\bline\s*plot\b",
                    r"\btime\s*series\b"
                ]
            ),

            (
                "bar",
                [
                    r"\bbar\s*chart\b",
                    r"\bbar\s*plot\b"
                ]
            )
        ]


        for chart_type, expressions in (
            patterns
        ):

            for expression in expressions:

                if re.search(
                    expression,
                    text
                ):

                    return chart_type


        return None


    # ========================================================
    # GENERATE EDA CHARTS
    # ========================================================

    def generate_eda_charts(
        self,
        df,
        schema
    ):

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "df must be a Pandas DataFrame."
            )


        if df.empty:

            return []


        if not isinstance(
            schema,
            dict
        ):

            raise TypeError(
                "schema must be a dictionary."
            )


        charts = []

        numerical_columns = []
        categorical_columns = []
        datetime_columns = []


        for column, info in (
            schema.items()
        ):

            if column not in df.columns:

                continue


            detected_type = (
                info.get(
                    "detected_type"
                )
            )


            if detected_type == "Numerical":

                numerical_columns.append(
                    column
                )

            elif detected_type == "Categorical":

                categorical_columns.append(
                    column
                )

            elif detected_type == "Datetime":

                datetime_columns.append(
                    column
                )


        # ----------------------------------------------------
        # CATEGORICAL DISTRIBUTIONS
        # ----------------------------------------------------

        for column in categorical_columns:

            counts = (
                df[column]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(
                    self.max_categories
                )
                .reset_index()
            )


            counts.columns = [

                column,

                "count"
            ]


            charts.append(

                self._create_chart(

                    chart_type=
                        "bar",

                    title=(
                        f"{self._pretty_name(column)} "
                        "Distribution"
                    ),

                    x=
                        column,

                    y=
                        "count",

                    data=
                        self.dataframe_to_records(
                            counts
                        ),

                    reason=(
                        "Compare frequency across "
                        f"{column} categories."
                    )
                )
            )


            if (
                len(charts)
                >=
                self.max_charts
            ):

                return charts


        # ----------------------------------------------------
        # NUMERIC HISTOGRAMS
        # ----------------------------------------------------

        for column in numerical_columns:

            histogram = (
                self._create_histogram_data(

                    df[
                        column
                    ],

                    column
                )
            )


            if histogram:

                charts.append(

                    self._create_chart(

                        chart_type=
                            "histogram",

                        title=(
                            f"{self._pretty_name(column)} "
                            "Distribution"
                        ),

                        x=
                            "bin",

                        y=
                            "frequency",

                        data=
                            histogram,

                        reason=(
                            "Analyze the distribution "
                            f"of {column}."
                        )
                    )
                )


            if (
                len(charts)
                >=
                self.max_charts
            ):

                return charts


        # ----------------------------------------------------
        # BOX PLOTS
        # ----------------------------------------------------

        for column in numerical_columns:

            box_data = (
                self._create_boxplot_data(

                    df[
                        column
                    ],

                    column
                )
            )


            if box_data:

                charts.append(

                    self._create_chart(

                        chart_type=
                            "box",

                        title=(
                            f"{self._pretty_name(column)} "
                            "Spread"
                        ),

                        x=
                            column,

                        y=
                            column,

                        data=
                            box_data,

                        reason=(
                            "Inspect spread and potential "
                            f"outliers in {column}."
                        )
                    )
                )


            if (
                len(charts)
                >=
                self.max_charts
            ):

                return charts


        # ----------------------------------------------------
        # SCATTER
        # ----------------------------------------------------

        for i in range(
            len(
                numerical_columns
            )
        ):

            for j in range(
                i + 1,
                len(
                    numerical_columns
                )
            ):

                x_column = (
                    numerical_columns[
                        i
                    ]
                )

                y_column = (
                    numerical_columns[
                        j
                    ]
                )


                scatter_df = (
                    df[
                        [
                            x_column,
                            y_column
                        ]
                    ]
                    .copy()
                )


                scatter_df[
                    x_column
                ] = pd.to_numeric(

                    scatter_df[
                        x_column
                    ],

                    errors=
                        "coerce"
                )


                scatter_df[
                    y_column
                ] = pd.to_numeric(

                    scatter_df[
                        y_column
                    ],

                    errors=
                        "coerce"
                )


                scatter_df = (
                    scatter_df
                    .dropna()
                    .head(
                        500
                    )
                )


                if scatter_df.empty:

                    continue


                charts.append(

                    self._create_chart(

                        chart_type=
                            "scatter",

                        title=(
                            f"{self._pretty_name(y_column)} "
                            "vs "
                            f"{self._pretty_name(x_column)}"
                        ),

                        x=
                            x_column,

                        y=
                            y_column,

                        data=
                            self.dataframe_to_records(
                                scatter_df
                            ),

                        reason=(
                            "Explore the relationship "
                            f"between {x_column} "
                            f"and {y_column}."
                        )
                    )
                )


                if (
                    len(charts)
                    >=
                    self.max_charts
                ):

                    return charts


        # ----------------------------------------------------
        # TIME SERIES
        # ----------------------------------------------------

        for date_column in datetime_columns:

            for numeric_column in numerical_columns:

                line_df = (
                    df[
                        [
                            date_column,
                            numeric_column
                        ]
                    ]
                    .copy()
                )


                line_df[
                    date_column
                ] = pd.to_datetime(

                    line_df[
                        date_column
                    ],

                    errors=
                        "coerce"
                )


                line_df[
                    numeric_column
                ] = pd.to_numeric(

                    line_df[
                        numeric_column
                    ],

                    errors=
                        "coerce"
                )


                line_df = (
                    line_df
                    .dropna()
                    .sort_values(
                        date_column
                    )
                )


                if line_df.empty:

                    continue


                charts.append(

                    self._create_chart(

                        chart_type=
                            "line",

                        title=(
                            f"{self._pretty_name(numeric_column)} "
                            "Over Time"
                        ),

                        x=
                            date_column,

                        y=
                            numeric_column,

                        data=
                            self.dataframe_to_records(
                                line_df
                            ),

                        reason=(
                            f"Analyze how "
                            f"{numeric_column} "
                            "changes over time."
                        )
                    )
                )


                if (
                    len(charts)
                    >=
                    self.max_charts
                ):

                    return charts


        return charts


    # ========================================================
    # RESULT CHART
    # ========================================================

    def generate_result_chart(
        self,
        df,
        question=None
    ):

        if df is None:

            return None


        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "SQL result must be a "
                "Pandas DataFrame."
            )


        if df.empty:

            return None


        columns = list(
            df.columns
        )


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        if (
            len(df) == 1
            and
            len(columns) == 1
        ):

            return {

                "chart_type":
                    "kpi",

                "title":
                    self._pretty_name(
                        columns[
                            0
                        ]
                    ),

                "value":
                    self._json_safe_value(
                        df.iloc[
                            0,
                            0
                        ]
                    ),

                "data":
                    self.dataframe_to_records(
                        df
                    ),

                "reason":
                    (
                        "Single-value query result "
                        "is best displayed as a KPI."
                    )
            }


        numeric_columns = []

        datetime_columns = []

        categorical_columns = []


        for column in columns:

            series = (
                df[
                    column
                ]
            )


            if self._is_numeric(
                series
            ):

                numeric_columns.append(
                    column
                )

            elif self._is_datetime(
                series
            ):

                datetime_columns.append(
                    column
                )

            else:

                categorical_columns.append(
                    column
                )


        requested_type = (
            self._detect_requested_chart_type(
                question
            )
        )


        # ====================================================
        # EXPLICIT BOX / VIOLIN REQUEST
        # ====================================================

        if (
            requested_type
            in {
                "box",
                "violin"
            }
            and
            numeric_columns
        ):

            numeric_column = (
                numeric_columns[
                    -1
                ]
            )


            if categorical_columns:

                category_column = (
                    categorical_columns[
                        0
                    ]
                )

                color_column = None


                if (
                    len(
                        categorical_columns
                    )
                    >= 2
                ):

                    color_column = (
                        categorical_columns[
                            1
                        ]
                    )


                selected_columns = [

                    category_column,

                    numeric_column
                ]


                if color_column:

                    selected_columns.insert(
                        1,
                        color_column
                    )


                chart_df = (
                    df[
                        selected_columns
                    ]
                    .copy()
                )


                chart_df[
                    numeric_column
                ] = pd.to_numeric(

                    chart_df[
                        numeric_column
                    ],

                    errors=
                        "coerce"
                )


                chart_df[
                    category_column
                ] = (
                    chart_df[
                        category_column
                    ]
                    .fillna(
                        "Missing"
                    )
                    .astype(
                        str
                    )
                )


                if color_column:

                    chart_df[
                        color_column
                    ] = (
                        chart_df[
                            color_column
                        ]
                        .fillna(
                            "Missing"
                        )
                        .astype(
                            str
                        )
                    )


                chart_df = (
                    chart_df.dropna(
                        subset=[
                            numeric_column
                        ]
                    )
                )


                if not chart_df.empty:

                    return (
                        self._create_chart(

                            chart_type=
                                requested_type,

                            title=
                                self._question_title(

                                    question,

                                    fallback=(
                                        f"{self._pretty_name(numeric_column)} "
                                        "by "
                                        f"{self._pretty_name(category_column)}"
                                    )
                                ),

                            x=
                                category_column,

                            y=
                                numeric_column,

                            color=
                                color_column,

                            data=
                                self.dataframe_to_records(
                                    chart_df
                                ),

                            reason=(
                                "The user explicitly "
                                f"requested a {requested_type} "
                                "plot."
                            ),

                            metadata={

                                "grouped":
                                    True,

                                "category":
                                    category_column,

                                "metric":
                                    numeric_column,

                                "color_group":
                                    color_column
                            }
                        )
                    )


            # Numeric-only box plot

            box_data = (
                self._create_boxplot_data(

                    df[
                        numeric_column
                    ],

                    numeric_column
                )
            )


            return (
                self._create_chart(

                    chart_type=
                        "box",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(numeric_column)} "
                                "Distribution"
                            )
                        ),

                    x=
                        numeric_column,

                    y=
                        numeric_column,

                    data=
                        box_data,

                    reason=(
                        "The user explicitly "
                        "requested a box plot."
                    )
                )
            )


        # ====================================================
        # EXPLICIT HISTOGRAM
        # ====================================================

        if (
            requested_type
            ==
            "histogram"
            and
            numeric_columns
        ):

            column = (
                numeric_columns[
                    0
                ]
            )


            histogram = (
                self._create_histogram_data(

                    df[
                        column
                    ],

                    column
                )
            )


            return (
                self._create_chart(

                    chart_type=
                        "histogram",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(column)} "
                                "Distribution"
                            )
                        ),

                    x=
                        "bin",

                    y=
                        "frequency",

                    data=
                        histogram,

                    reason=(
                        "The user explicitly "
                        "requested a histogram."
                    )
                )
            )


        # ====================================================
        # EXPLICIT SCATTER
        # ====================================================

        if (
            requested_type
            ==
            "scatter"
            and
            len(
                numeric_columns
            )
            >= 2
        ):

            x_column = (
                numeric_columns[
                    0
                ]
            )

            y_column = (
                numeric_columns[
                    1
                ]
            )


            chart_df = (
                df[
                    [
                        x_column,
                        y_column
                    ]
                ]
                .dropna()
                .head(
                    500
                )
            )


            return (
                self._create_chart(

                    chart_type=
                        "scatter",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(y_column)} "
                                "vs "
                                f"{self._pretty_name(x_column)}"
                            )
                        ),

                    x=
                        x_column,

                    y=
                        y_column,

                    data=
                        self.dataframe_to_records(
                            chart_df
                        ),

                    reason=(
                        "The user explicitly "
                        "requested a scatter plot."
                    )
                )
            )


        # ====================================================
        # EXPLICIT LINE
        # ====================================================

        if (
            requested_type
            ==
            "line"
            and
            numeric_columns
        ):

            if datetime_columns:

                x_column = (
                    datetime_columns[
                        0
                    ]
                )

            elif categorical_columns:

                x_column = (
                    categorical_columns[
                        0
                    ]
                )

            else:

                x_column = None


            if x_column:

                y_column = (
                    numeric_columns[
                        0
                    ]
                )


                chart_df = (
                    df[
                        [
                            x_column,
                            y_column
                        ]
                    ]
                    .dropna()
                    .copy()
                )


                return (
                    self._create_chart(

                        chart_type=
                            "line",

                        title=
                            self._question_title(

                                question,

                                fallback=(
                                    f"{self._pretty_name(y_column)} "
                                    "by "
                                    f"{self._pretty_name(x_column)}"
                                )
                            ),

                        x=
                            x_column,

                        y=
                            y_column,

                        data=
                            self.dataframe_to_records(
                                chart_df
                            ),

                        reason=(
                            "The user explicitly "
                            "requested a line chart."
                        )
                    )
                )


        # ====================================================
        # EXPLICIT BAR
        # ====================================================

        if (
            requested_type
            ==
            "bar"
            and
            categorical_columns
            and
            numeric_columns
        ):

            return (
                self._category_numeric_bar(

                    df,

                    categorical_columns[
                        0
                    ],

                    numeric_columns[
                        0
                    ],

                    question
                )
            )


        # ====================================================
        # AUTOMATIC DATETIME + NUMERIC
        # ====================================================

        if (
            datetime_columns
            and
            numeric_columns
        ):

            x_column = (
                datetime_columns[
                    0
                ]
            )

            y_column = (
                numeric_columns[
                    0
                ]
            )


            chart_df = (
                df[
                    [
                        x_column,
                        y_column
                    ]
                ]
                .dropna()
                .sort_values(
                    x_column
                )
            )


            return (
                self._create_chart(

                    chart_type=
                        "line",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(y_column)} "
                                "Over Time"
                            )
                        ),

                    x=
                        x_column,

                    y=
                        y_column,

                    data=
                        self.dataframe_to_records(
                            chart_df
                        ),

                    reason=(
                        "Datetime and numerical "
                        "results are suitable "
                        "for a line chart."
                    )
                )
            )


        # ====================================================
        # AUTOMATIC CATEGORY + NUMERIC
        # ====================================================

        if (
            categorical_columns
            and
            numeric_columns
        ):

            return (
                self._category_numeric_bar(

                    df,

                    categorical_columns[
                        0
                    ],

                    numeric_columns[
                        0
                    ],

                    question
                )
            )


        # ====================================================
        # AUTOMATIC NUMERIC + NUMERIC
        # ====================================================

        if (
            len(
                numeric_columns
            )
            >= 2
        ):

            x_column = (
                numeric_columns[
                    0
                ]
            )

            y_column = (
                numeric_columns[
                    1
                ]
            )


            chart_df = (
                df[
                    [
                        x_column,
                        y_column
                    ]
                ]
                .dropna()
                .head(
                    500
                )
            )


            return (
                self._create_chart(

                    chart_type=
                        "scatter",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(y_column)} "
                                "vs "
                                f"{self._pretty_name(x_column)}"
                            )
                        ),

                    x=
                        x_column,

                    y=
                        y_column,

                    data=
                        self.dataframe_to_records(
                            chart_df
                        ),

                    reason=(
                        "Two numerical columns "
                        "are suitable for a "
                        "scatter plot."
                    )
                )
            )


        # ====================================================
        # SINGLE CATEGORY
        # ====================================================

        if (
            len(columns) == 1
            and
            categorical_columns
        ):

            column = (
                categorical_columns[
                    0
                ]
            )


            counts = (
                df[
                    column
                ]
                .fillna(
                    "Missing"
                )
                .astype(
                    str
                )
                .value_counts()
                .head(
                    self.max_categories
                )
                .reset_index()
            )


            counts.columns = [

                column,

                "count"
            ]


            return (
                self._create_chart(

                    chart_type=
                        "bar",

                    title=
                        self._question_title(

                            question,

                            fallback=(
                                f"{self._pretty_name(column)} "
                                "Distribution"
                            )
                        ),

                    x=
                        column,

                    y=
                        "count",

                    data=
                        self.dataframe_to_records(
                            counts
                        ),

                    reason=(
                        "Categorical result is "
                        "suitable for a frequency "
                        "bar chart."
                    )
                )
            )


        return None


    # ========================================================
    # CATEGORY NUMERIC BAR
    # ========================================================

    def _category_numeric_bar(
        self,
        df,
        x_column,
        y_column,
        question
    ):

        chart_df = (
            df[
                [
                    x_column,
                    y_column
                ]
            ]
            .dropna()
            .head(
                self.max_categories
            )
        )


        return (
            self._create_chart(

                chart_type=
                    "bar",

                title=
                    self._question_title(

                        question,

                        fallback=(
                            f"{self._pretty_name(y_column)} "
                            "by "
                            f"{self._pretty_name(x_column)}"
                        )
                    ),

                x=
                    x_column,

                y=
                    y_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "Categorical and numerical "
                    "results are suitable for "
                    "a bar chart."
                )
            )
        )


    # ========================================================
    # HISTOGRAM DATA
    # ========================================================

    def _create_histogram_data(
        self,
        series,
        column
    ):

        numeric_series = (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .dropna()
        )


        if numeric_series.empty:

            return []


        minimum = float(
            numeric_series.min()
        )

        maximum = float(
            numeric_series.max()
        )


        if minimum == maximum:

            return [

                {

                    "bin":
                        str(
                            round(
                                minimum,
                                2
                            )
                        ),

                    "frequency":
                        int(
                            len(
                                numeric_series
                            )
                        )
                }
            ]


        number_of_bins = min(

            10,

            max(

                3,

                int(
                    math.sqrt(
                        len(
                            numeric_series
                        )
                    )
                )
            )
        )


        bins = pd.cut(

            numeric_series,

            bins=
                number_of_bins,

            duplicates=
                "drop"
        )


        counts = (
            bins.value_counts(
                sort=False
            )
        )


        data = []


        for interval, count in (
            counts.items()
        ):

            data.append({

                "bin":
                    (
                        f"{interval.left:.2f} - "
                        f"{interval.right:.2f}"
                    ),

                "frequency":
                    int(
                        count
                    )
            })


        return data


    # ========================================================
    # BOX PLOT DATA
    # ========================================================

    def _create_boxplot_data(
        self,
        series,
        column
    ):

        numeric_series = (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .dropna()
        )


        if numeric_series.empty:

            return []


        q1 = float(
            numeric_series.quantile(
                0.25
            )
        )

        median = float(
            numeric_series.median()
        )

        q3 = float(
            numeric_series.quantile(
                0.75
            )
        )


        iqr = (
            q3 - q1
        )


        lower_bound = (
            q1
            -
            1.5 * iqr
        )

        upper_bound = (
            q3
            +
            1.5 * iqr
        )


        outliers = (

            numeric_series[
                (
                    numeric_series
                    <
                    lower_bound
                )
                |
                (
                    numeric_series
                    >
                    upper_bound
                )
            ]
        )


        return [

            {

                "column":
                    column,

                "min":
                    float(
                        numeric_series.min()
                    ),

                "q1":
                    q1,

                "median":
                    median,

                "q3":
                    q3,

                "max":
                    float(
                        numeric_series.max()
                    ),

                "lower_bound":
                    float(
                        lower_bound
                    ),

                "upper_bound":
                    float(
                        upper_bound
                    ),

                "outliers":
                    [

                        self._json_safe_value(
                            value
                        )

                        for value in (
                            outliers.tolist()
                        )
                    ]
            }
        ]


    # ========================================================
    # PRETTY NAME
    # ========================================================

    def _pretty_name(
        self,
        name
    ):

        return (
            str(name)
            .replace(
                "_",
                " "
            )
            .strip()
            .title()
        )


    # ========================================================
    # QUESTION TITLE
    # ========================================================

    def _question_title(
        self,
        question,
        fallback
    ):

        if (
            isinstance(
                question,
                str
            )
            and
            question.strip()
        ):

            title = (
                question
                .strip()
                .rstrip(
                    "?"
                )
            )


            # Don't let giant NL prompts become
            # giant chart titles.

            if len(title) <= 100:

                return title


        return fallback