import math
import re

import pandas as pd


class VisualizationService:
    """
    Deterministic visualization intelligence for InsightFlow.

    Responsibilities:
    1. Generate useful automated EDA visualizations.
    2. Use semantic schema information when available.
    3. Use EDA statistics when available.
    4. Rank candidate charts by analytical usefulness.
    5. Avoid identifiers, unsuitable text and redundant charts.
    6. Generate visualization metadata for SQL/query results.
    7. Require no LLM.
    """

    def __init__(
        self,
        max_categories=15,
        max_charts=10,
        max_scatter_points=500,
        correlation_threshold=0.30
    ):
        self.max_categories = max_categories
        self.max_charts = max_charts
        self.max_scatter_points = max_scatter_points
        self.correlation_threshold = correlation_threshold

    # ========================================================
    # JSON SAFE VALUE
    # ========================================================

    def _json_safe_value(self, value):

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError, AttributeError):
                pass

        return value

    # ========================================================
    # DATAFRAME -> RECORDS
    # ========================================================

    def dataframe_to_records(self, df):

        if df is None:
            return []

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a Pandas DataFrame."
            )

        records = []

        for record in df.to_dict(orient="records"):

            safe_record = {}

            for key, value in record.items():

                safe_record[str(key)] = (
                    self._json_safe_value(value)
                )

            records.append(safe_record)

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
            "chart_type": chart_type,
            "title": title,
            "x": x,
            "y": y,
            "data": data,
            "reason": reason
        }

        if color is not None:
            chart["color"] = color

        if metadata:
            chart["metadata"] = metadata

        return chart

    # ========================================================
    # TYPE HELPERS
    # ========================================================
    
    def _is_numeric(self, series):

        
        """
        Detect true numeric columns AND object/string columns
        whose values are mostly numeric.
        """

        if pd.api.types.is_numeric_dtype(series):
            return True

        non_null = series.dropna()

        if non_null.empty:
            return False

        converted = pd.to_numeric(
            non_null,
            errors="coerce"
        )

        numeric_ratio = converted.notna().mean()

        return bool(numeric_ratio >= 0.60)
    
    

    def _is_datetime(self, series):

        return pd.api.types.is_datetime64_any_dtype(
            series
        )

    def _semantic_type(
        self,
        schema,
        column,
        series
    ):
        """
        Read semantic type from Schema Intelligence.

        Falls back to Pandas dtype if semantic information
        is unavailable.
        """

        info = schema.get(column, {})

        detected_type = (
            info.get("detected_type")
            or info.get("semantic_type")
            or info.get("type")
        )

        if detected_type:

            normalized = (
                str(detected_type)
                .strip()
                .lower()
            )

            aliases = {
                "numeric": "Numerical",
                "number": "Numerical",
                "numerical": "Numerical",
                "integer": "Numerical",
                "float": "Numerical",

                "category": "Categorical",
                "categorical": "Categorical",

                "datetime": "Datetime",
                "date": "Datetime",
                "timestamp": "Datetime",

                "identifier": "Identifier",
                "id": "Identifier",

                "boolean": "Boolean",
                "bool": "Boolean",

                "text": "Text",
                "string": "Text"
            }

            if normalized in aliases:
                return aliases[normalized]

            return str(detected_type)

        if self._is_datetime(series):
            return "Datetime"

        if self._is_numeric(series):
            return "Numerical"

        return "Text"

    # ========================================================
    # COLUMN COLLECTION
    # ========================================================

    def _collect_columns(
        self,
        df,
        schema
    ):

        groups = {
            "Numerical": [],
            "Categorical": [],
            "Datetime": [],
            "Boolean": [],
            "Identifier": [],
            "Text": []
        }

        for column in df.columns:

            semantic_type = self._semantic_type(
                schema,
                column,
                df[column]
            )

            if semantic_type not in groups:
                semantic_type = "Text"

            groups[semantic_type].append(
                column
            )

        return groups

    # ========================================================
    # COLUMN QUALITY HELPERS
    # ========================================================

    def _valid_numeric_series(
        self,
        series
    ):

        return (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .replace(
                [float("inf"), float("-inf")],
                pd.NA
            )
            .dropna()
        )

    def _usable_category(
        self,
        series
    ):
        """
        Decide whether a column is useful as a grouping
        variable for visualization.
        """

        non_null = series.dropna()

        if non_null.empty:
            return False

        unique_count = non_null.nunique()

        if unique_count < 2:
            return False

        if unique_count > self.max_categories:
            return False

        return True

    def _numeric_has_variation(
        self,
        series
    ):

        numeric = self._valid_numeric_series(
            series
        )

        if len(numeric) < 2:
            return False

        return numeric.nunique() > 1

    # ========================================================
    # CHART CANDIDATE
    # ========================================================

    def _candidate(
        self,
        chart,
        score,
        signature
    ):

        return {
            "chart": chart,
            "score": float(score),
            "signature": signature
        }

    # ========================================================
    # ADD CANDIDATE SAFELY
    # ========================================================

    def _add_candidate(
        self,
        candidates,
        chart,
        score,
        signature
    ):

        if chart is None:
            return

        data = chart.get("data")

        if data is None:
            return

        if isinstance(data, list) and not data:
            return

        candidates.append(
            self._candidate(
                chart,
                score,
                signature
            )
        )

    # ========================================================
    # RANK AND DEDUPLICATE
    # ========================================================

    def _select_best_candidates(
        self,
        candidates
    ):

        ordered = sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True
        )

        selected = []
        seen = set()

        for candidate in ordered:

            signature = candidate[
                "signature"
            ]

            if signature in seen:
                continue

            chart = candidate[
                "chart"
            ]

            metadata = chart.setdefault(
                "metadata",
                {}
            )

            metadata["priority_score"] = round(
                candidate["score"],
                3
            )

            selected.append(chart)

            seen.add(signature)

            if len(selected) >= self.max_charts:
                break

        return selected

    # ========================================================
    # REQUESTED CHART TYPE
    # ========================================================

    def _detect_requested_chart_type(
        self,
        question
    ):

        if not isinstance(question, str):
            return None

        text = question.strip().lower()

        patterns = [
            (
                "stacked_bar",
                [
                    r"\bstacked\s*bar\b",
                    r"\bstacked\s*bar\s*chart\b"
                ]
            ),
            (
                "heatmap",
                [
                    r"\bheat\s*map\b",
                    r"\bheatmap\b",
                    r"\bcorrelation\s*matrix\b"
                ]
            ),
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
                    r"\btime\s*series\b",
                    r"\btrend\b"
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

        for chart_type, expressions in patterns:

            for expression in expressions:

                if re.search(
                    expression,
                    text
                ):
                    return chart_type

        return None

    # ========================================================
    # EDA VISUALIZATION INTELLIGENCE
    # ========================================================

    def generate_eda_charts(
        self,
        df,
        schema,
        eda_results=None
    ):
        """
        Generate ranked deterministic EDA visualizations.

        eda_results is optional so this remains backward
        compatible with existing workflow calls.

        Candidate families:
        - categorical distributions
        - numeric distributions
        - outlier/spread plots
        - numeric relationships
        - correlation heatmap
        - category/numeric comparison
        - categorical composition
        - temporal trends
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "df must be a Pandas DataFrame."
            )

        if df.empty:
            return []

        if not isinstance(schema, dict):

            raise TypeError(
                "schema must be a dictionary."
            )

        if eda_results is None:
            eda_results = {}

        if not isinstance(eda_results, dict):

            raise TypeError(
                "eda_results must be a dictionary."
            )

        groups = self._collect_columns(
            df,
            schema
        )

        numerical_columns = groups[
            "Numerical"
        ]

        categorical_columns = (
            groups["Categorical"]
            +
            groups["Boolean"]
        )

        datetime_columns = groups[
            "Datetime"
        ]

        # IDs and free text intentionally excluded
        # from automatic EDA charts.

        candidates = []

        # ====================================================
        # 1. CATEGORY DISTRIBUTIONS
        # ====================================================

        for column in categorical_columns:

            if not self._usable_category(
                df[column]
            ):
                continue

            counts = (
                df[column]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(self.max_categories)
                .reset_index()
            )

            counts.columns = [
                column,
                "count"
            ]

            unique_count = (
                df[column]
                .nunique(
                    dropna=True
                )
            )

            score = 58

            if unique_count <= 8:
                score += 8

            chart = self._create_chart(
                chart_type="bar",
                title=(
                    f"{self._pretty_name(column)} "
                    "Distribution"
                ),
                x=column,
                y="count",
                data=self.dataframe_to_records(
                    counts
                ),
                reason=(
                    "Shows the frequency and dominance "
                    f"of {column} categories."
                ),
                metadata={
                    "intent": "distribution",
                    "semantic_type": "Categorical",
                    "category_count": int(
                        unique_count
                    )
                }
            )

            self._add_candidate(
                candidates,
                chart,
                score,
                (
                    "category_distribution",
                    column
                )
            )

        # ====================================================
        # 2. NUMERIC DISTRIBUTIONS
        # ====================================================

        for column in numerical_columns:

            if not self._numeric_has_variation(
                df[column]
            ):
                continue

            histogram = (
                self._create_histogram_data(
                    df[column],
                    column
                )
            )

            chart = self._create_chart(
                chart_type="histogram",
                title=(
                    f"{self._pretty_name(column)} "
                    "Distribution"
                ),
                x="bin",
                y="frequency",
                data=histogram,
                reason=(
                    "Shows the shape, concentration and "
                    f"spread of {column}."
                ),
                metadata={
                    "intent": "distribution",
                    "metric": column
                }
            )

            self._add_candidate(
                candidates,
                chart,
                52,
                (
                    "numeric_distribution",
                    column
                )
            )

        # ====================================================
        # 3. NUMERIC OUTLIER / SPREAD
        # ====================================================

        for column in numerical_columns:

            if not self._numeric_has_variation(
                df[column]
            ):
                continue

            box_data = (
                self._create_boxplot_data(
                    df[column],
                    column
                )
            )

            if not box_data:
                continue

            outlier_count = len(
                box_data[0].get(
                    "outliers",
                    []
                )
            )

            score = 45

            if outlier_count:
                score += min(
                    20,
                    outlier_count / 5
                )

            chart = self._create_chart(
                chart_type="box",
                title=(
                    f"{self._pretty_name(column)} "
                    "Spread & Outliers"
                ),
                x=column,
                y=column,
                data=box_data,
                reason=(
                    "Shows quartiles, spread and "
                    f"potential outliers in {column}."
                ),
                metadata={
                    "intent": "outlier",
                    "metric": column,
                    "outlier_count": outlier_count
                }
            )

            self._add_candidate(
                candidates,
                chart,
                score,
                (
                    "numeric_box",
                    column
                )
            )

        # ====================================================
        # 4. CORRELATION MATRIX
        # ====================================================

        correlation_df = (
            self._prepare_numeric_dataframe(
                df,
                numerical_columns
            )
        )

        if correlation_df.shape[1] >= 2:

            correlation_matrix = (
                correlation_df.corr()
            )

            heatmap_data = (
                self._correlation_records(
                    correlation_matrix
                )
            )

            strongest = (
                self._strongest_correlation(
                    correlation_matrix
                )
            )

            score = 72

            if strongest is not None:

                score += (
                    abs(
                        strongest["correlation"]
                    )
                    * 15
                )

            chart = self._create_chart(
                chart_type="heatmap",
                title="Numerical Correlation Heatmap",
                x="x",
                y="y",
                data=heatmap_data,
                reason=(
                    "Compares relationships across "
                    "numerical variables."
                ),
                metadata={
                    "intent": "correlation",
                    "value": "correlation",
                    "strongest_relationship": strongest
                }
            )

            self._add_candidate(
                candidates,
                chart,
                score,
                ("correlation_heatmap",)
            )

        # ====================================================
        # 5. NUMERIC RELATIONSHIPS
        # ====================================================

        numeric_pairs = []

        for i in range(
            len(numerical_columns)
        ):

            for j in range(
                i + 1,
                len(numerical_columns)
            ):

                x_column = (
                    numerical_columns[i]
                )

                y_column = (
                    numerical_columns[j]
                )

                pair_df = (
                    df[
                        [
                            x_column,
                            y_column
                        ]
                    ]
                    .copy()
                )

                pair_df[x_column] = (
                    pd.to_numeric(
                        pair_df[x_column],
                        errors="coerce"
                    )
                )

                pair_df[y_column] = (
                    pd.to_numeric(
                        pair_df[y_column],
                        errors="coerce"
                    )
                )

                pair_df = (
                    pair_df
                    .dropna()
                )

                if len(pair_df) < 3:
                    continue

                if (
                    pair_df[x_column].nunique() < 2
                    or
                    pair_df[y_column].nunique() < 2
                ):
                    continue

                correlation = (
                    pair_df[
                        [
                            x_column,
                            y_column
                        ]
                    ]
                    .corr()
                    .iloc[0, 1]
                )

                if pd.isna(correlation):
                    continue

                numeric_pairs.append(
                    (
                        abs(
                            float(correlation)
                        ),
                        float(correlation),
                        x_column,
                        y_column,
                        pair_df
                    )
                )

        numeric_pairs.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # Only generate strongest relationships.
        # This prevents scatter plots from dominating.

        for (
            absolute_correlation,
            correlation,
            x_column,
            y_column,
            pair_df
        ) in numeric_pairs[:3]:

            if (
                absolute_correlation
                <
                self.correlation_threshold
            ):
                continue

            scatter_df = (
                pair_df
                .head(
                    self.max_scatter_points
                )
            )

            score = (
                65
                +
                absolute_correlation * 25
            )

            chart = self._create_chart(
                chart_type="scatter",
                title=(
                    f"{self._pretty_name(y_column)} "
                    "vs "
                    f"{self._pretty_name(x_column)}"
                ),
                x=x_column,
                y=y_column,
                data=self.dataframe_to_records(
                    scatter_df
                ),
                reason=(
                    "Shows a numerical relationship "
                    f"with correlation {correlation:.2f}."
                ),
                metadata={
                    "intent": "relationship",
                    "correlation": round(
                        correlation,
                        4
                    ),
                    "sample_size": len(
                        scatter_df
                    )
                }
            )

            self._add_candidate(
                candidates,
                chart,
                score,
                (
                    "scatter",
                    x_column,
                    y_column
                )
            )

        # ====================================================
        # 6. CATEGORY + NUMERIC RELATIONSHIPS
        # ====================================================

        for category_column in (
            categorical_columns
        ):

            if not self._usable_category(
                df[category_column]
            ):
                continue

            for numeric_column in (
                numerical_columns
            ):

                relationship_df = (
                    df[
                        [
                            category_column,
                            numeric_column
                        ]
                    ]
                    .copy()
                )

                relationship_df[
                    numeric_column
                ] = pd.to_numeric(
                    relationship_df[
                        numeric_column
                    ],
                    errors="coerce"
                )

                relationship_df = (
                    relationship_df
                    .dropna(
                        subset=[
                            numeric_column
                        ]
                    )
                )

                if relationship_df.empty:
                    continue

                relationship_df[
                    category_column
                ] = (
                    relationship_df[
                        category_column
                    ]
                    .fillna("Missing")
                    .astype(str)
                )

                grouped = (
                    relationship_df
                    .groupby(
                        category_column,
                        dropna=False
                    )[
                        numeric_column
                    ]
                    .agg(
                        mean="mean",
                        median="median",
                        count="count"
                    )
                    .reset_index()
                )

                if len(grouped) < 2:
                    continue

                if len(grouped) > self.max_categories:
                    continue

                mean_variation = (
                    self._group_mean_variation(
                        grouped["mean"]
                    )
                )

                score = (
                    68
                    +
                    min(
                        15,
                        mean_variation * 15
                    )
                )

                chart = self._create_chart(
                    chart_type="bar",
                    title=(
                        f"Average "
                        f"{self._pretty_name(numeric_column)} "
                        "by "
                        f"{self._pretty_name(category_column)}"
                    ),
                    x=category_column,
                    y="mean",
                    data=self.dataframe_to_records(
                        grouped
                    ),
                    reason=(
                        "Compares the average "
                        f"{numeric_column} across "
                        f"{category_column} groups."
                    ),
                    metadata={
                        "intent": "comparison",
                        "aggregation": "mean",
                        "metric": numeric_column,
                        "category": category_column,
                        "mean_variation": round(
                            mean_variation,
                            4
                        )
                    }
                )

                self._add_candidate(
                    candidates,
                    chart,
                    score,
                    (
                        "category_numeric",
                        category_column,
                        numeric_column
                    )
                )

        # ====================================================
        # 7. GROUPED BOX PLOTS
        # ====================================================

        for category_column in (
            categorical_columns
        ):

            if not self._usable_category(
                df[category_column]
            ):
                continue

            for numeric_column in (
                numerical_columns
            ):

                chart_df = (
                    df[
                        [
                            category_column,
                            numeric_column
                        ]
                    ]
                    .copy()
                )

                chart_df[
                    numeric_column
                ] = pd.to_numeric(
                    chart_df[
                        numeric_column
                    ],
                    errors="coerce"
                )

                chart_df = (
                    chart_df.dropna(
                        subset=[
                            numeric_column
                        ]
                    )
                )

                if chart_df.empty:
                    continue

                chart_df[
                    category_column
                ] = (
                    chart_df[
                        category_column
                    ]
                    .fillna("Missing")
                    .astype(str)
                )

                if (
                    chart_df[
                        category_column
                    ]
                    .nunique()
                    < 2
                ):
                    continue

                chart = self._create_chart(
                    chart_type="box",
                    title=(
                        f"{self._pretty_name(numeric_column)} "
                        "by "
                        f"{self._pretty_name(category_column)}"
                    ),
                    x=category_column,
                    y=numeric_column,
                    data=self.dataframe_to_records(
                        chart_df
                    ),
                    reason=(
                        "Compares numerical distributions "
                        f"of {numeric_column} across "
                        f"{category_column} groups."
                    ),
                    metadata={
                        "intent": "group_distribution",
                        "grouped": True,
                        "category": category_column,
                        "metric": numeric_column
                    }
                )

                self._add_candidate(
                    candidates,
                    chart,
                    64,
                    (
                        "grouped_box",
                        category_column,
                        numeric_column
                    )
                )

        # ====================================================
        # 8. CATEGORY + CATEGORY COMPOSITION
        # ====================================================

        usable_categories = [

            column

            for column in categorical_columns

            if self._usable_category(
                df[column]
            )
        ]

        for i in range(
            len(usable_categories)
        ):

            for j in range(
                i + 1,
                len(usable_categories)
            ):

                x_column = (
                    usable_categories[i]
                )

                color_column = (
                    usable_categories[j]
                )

                cross = (
                    df[
                        [
                            x_column,
                            color_column
                        ]
                    ]
                    .copy()
                )

                cross[x_column] = (
                    cross[x_column]
                    .fillna("Missing")
                    .astype(str)
                )

                cross[color_column] = (
                    cross[color_column]
                    .fillna("Missing")
                    .astype(str)
                )

                cross = (
                    cross
                    .groupby(
                        [
                            x_column,
                            color_column
                        ]
                    )
                    .size()
                    .reset_index(
                        name="count"
                    )
                )

                if cross.empty:
                    continue

                chart = self._create_chart(
                    chart_type="stacked_bar",
                    title=(
                        f"{self._pretty_name(color_column)} "
                        "by "
                        f"{self._pretty_name(x_column)}"
                    ),
                    x=x_column,
                    y="count",
                    color=color_column,
                    data=self.dataframe_to_records(
                        cross
                    ),
                    reason=(
                        "Shows how "
                        f"{color_column} composition varies "
                        f"across {x_column} categories."
                    ),
                    metadata={
                        "intent": "composition",
                        "aggregation": "count",
                        "category": x_column,
                        "color_group": color_column
                    }
                )

                self._add_candidate(
                    candidates,
                    chart,
                    66,
                    (
                        "stacked_bar",
                        x_column,
                        color_column
                    )
                )

        # ====================================================
        # 9. TEMPORAL TRENDS
        # ====================================================

        for date_column in datetime_columns:

            converted_dates = (
                self._safe_datetime_series(
                    df[date_column]
                )
            )

            if converted_dates.notna().sum() < 2:
                continue

            for numeric_column in (
                numerical_columns
            ):

                line_df = pd.DataFrame({
                    date_column:
                        converted_dates,

                    numeric_column:
                        pd.to_numeric(
                            df[numeric_column],
                            errors="coerce"
                        )
                })

                line_df = (
                    line_df
                    .dropna()
                )

                if line_df.empty:
                    continue

                if (
                    line_df[
                        date_column
                    ]
                    .nunique()
                    < 2
                ):
                    continue

                # Aggregate duplicate timestamps rather than
                # plotting hundreds of overlapping points.

                line_df = (
                    line_df
                    .groupby(
                        date_column,
                        as_index=False
                    )[
                        numeric_column
                    ]
                    .mean()
                    .sort_values(
                        date_column
                    )
                )

                chart = self._create_chart(
                    chart_type="line",
                    title=(
                        f"{self._pretty_name(numeric_column)} "
                        "Over Time"
                    ),
                    x=date_column,
                    y=numeric_column,
                    data=self.dataframe_to_records(
                        line_df
                    ),
                    reason=(
                        "Shows how "
                        f"{numeric_column} changes across "
                        f"{date_column}."
                    ),
                    metadata={
                        "intent": "temporal",
                        "aggregation": "mean",
                        "date_column": date_column,
                        "metric": numeric_column
                    }
                )

                self._add_candidate(
                    candidates,
                    chart,
                    82,
                    (
                        "time_series",
                        date_column,
                        numeric_column
                    )
                )

        # ====================================================
        # FINAL RANKING
        # ====================================================

        return self._select_best_candidates(
            candidates
        )

    # ========================================================
    # SAFE DATETIME CONVERSION
    # ========================================================

    def _safe_datetime_series(
        self,
        series
    ):
        """
        Convert date values without blindly asking Pandas to
        infer every arbitrary object column.

        The Schema Agent should already have identified this
        column as Datetime.
        """

        if self._is_datetime(series):

            return pd.to_datetime(
                series,
                errors="coerce"
            )

        text = (
            series
            .astype("string")
            .str.strip()
        )

        return pd.to_datetime(
            text,
            errors="coerce"
        )

    # ========================================================
    # PREPARE NUMERIC DATAFRAME
    # ========================================================

    def _prepare_numeric_dataframe(
        self,
        df,
        columns
    ):

        numeric_df = pd.DataFrame(
            index=df.index
        )

        for column in columns:

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if converted.notna().sum() < 2:
                continue

            if converted.nunique(
                dropna=True
            ) < 2:
                continue

            numeric_df[column] = (
                converted
            )

        return numeric_df

    # ========================================================
    # CORRELATION RECORDS
    # ========================================================

    def _correlation_records(
        self,
        matrix
    ):

        records = []

        for row_column in matrix.index:

            for column in matrix.columns:

                value = matrix.loc[
                    row_column,
                    column
                ]

                if pd.isna(value):
                    continue

                records.append({
                    "x": str(column),
                    "y": str(row_column),
                    "correlation": round(
                        float(value),
                        4
                    )
                })

        return records

    # ========================================================
    # STRONGEST CORRELATION
    # ========================================================

    def _strongest_correlation(
        self,
        matrix
    ):

        if matrix.shape[1] < 2:
            return None

        best = None

        columns = list(
            matrix.columns
        )

        for i in range(
            len(columns)
        ):

            for j in range(
                i + 1,
                len(columns)
            ):

                value = matrix.loc[
                    columns[i],
                    columns[j]
                ]

                if pd.isna(value):
                    continue

                value = float(value)

                if (
                    best is None
                    or
                    abs(value)
                    >
                    abs(
                        best[
                            "correlation"
                        ]
                    )
                ):

                    best = {
                        "x": columns[i],
                        "y": columns[j],
                        "correlation": round(
                            value,
                            4
                        )
                    }

        return best

    # ========================================================
    # GROUP MEAN VARIATION
    # ========================================================

    def _group_mean_variation(
        self,
        means
    ):

        numeric = (
            pd.to_numeric(
                means,
                errors="coerce"
            )
            .dropna()
        )

        if len(numeric) < 2:
            return 0.0

        average = abs(
            float(
                numeric.mean()
            )
        )

        spread = float(
            numeric.max()
            -
            numeric.min()
        )

        if average <= 1e-12:
            return min(
                1.0,
                abs(spread)
            )

        return min(
            1.0,
            abs(spread)
            /
            average
        )
    # ========================================================
    # QUESTION COLUMN DETECTION
# ========================================================

    def _find_question_columns(
        self,
        question,
        df,
        schema=None
    ):
        """
        Detect dataset columns explicitly mentioned in the
        user's natural-language question.

        Returns columns in the order they appear in the question.
        """

        if not isinstance(question, str):
            return []

        if not isinstance(df, pd.DataFrame):
            return []

        question_text = question.lower()

        matches = []

        for column in df.columns:

            column_text = str(column).lower()

            variants = {
                column_text,
                column_text.replace("_", " "),
                column_text.replace("-", " "),
            }

            positions = []

            for variant in variants:

                pattern = (
                    r"(?<![a-z0-9_])"
                    + re.escape(variant)
                    + r"(?![a-z0-9_])"
                )

                match = re.search(
                    pattern,
                    question_text
                )

                if match:
                    positions.append(
                        match.start()
                    )

            if positions:

                matches.append(
                    (
                        min(positions),
                        column
                    )
                )

        matches.sort(
            key=lambda item: item[0]
        )

        return [
            column
            for _, column in matches
        ]

    # ========================================================
    # VISUALIZATION SOURCE SELECTION
# ========================================================

    def _select_visualization_source(
        self,
        result_df,
        source_df,
        requested_type
    ):
        """
        Choose whether visualization should use the SQL result
        or the full cleaned dataset.

        Distribution and relationship charts generally require
        row-level observations, while aggregated comparison
        charts generally benefit from SQL results.
        """

        row_level_types = {
            "box",
            "violin",
            "histogram",
            "scatter",
            "heatmap"
        }

        if (
            requested_type in row_level_types
            and isinstance(source_df, pd.DataFrame)
            and not source_df.empty
        ):
            return source_df, "source_dataset"

        return result_df, "sql_result"
    



    # ========================================================
    # RESULT CHART
    # ========================================================



    def generate_result_chart(
        self,
        df,
        question=None,
        source_df=None,
        schema=None
    ):
        """
        Generate deterministic visualization metadata for an
        analytical query.

        df:
            SQL/query result.

        source_df:
            Full cleaned row-level dataset.

        schema:
            Semantic schema for source_df.

        Strategy:
        - Explicit visualization requests take priority.
        - Distribution/relationship charts use row-level data.
        - Aggregated comparisons prefer SQL results.
        - Columns explicitly mentioned in the question are
        preferred over positional guessing.
        """

        if df is None:
            return None

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "SQL result must be a Pandas DataFrame."
            )

        if df.empty:
            return None

        if schema is None:
            schema = {}

        requested_type = (
            self._detect_requested_chart_type(
                question
            )
        )

        # ====================================================
        # KPI
        # ====================================================

        if (
            requested_type is None
            and
            len(df) == 1
            and
            len(df.columns) == 1
        ):

            column = df.columns[0]

            return {
                "chart_type": "kpi",

                "title":
                    self._pretty_name(
                        column
                    ),

                "value":
                    self._json_safe_value(
                        df.iloc[0, 0]
                    ),

                "data":
                    self.dataframe_to_records(
                        df
                    ),

                "reason":
                    (
                        "Single-value query result is best "
                        "displayed as a KPI."
                    ),

                "metadata": {
                    "intent": "summary",
                    "data_source": "sql_result"
                }
            }

        # ====================================================
        # SELECT VISUALIZATION DATA SOURCE
        # ====================================================

        working_df, data_source = (
            self._select_visualization_source(
                result_df=df,
                source_df=source_df,
                requested_type=requested_type
            )
        )

        # ====================================================
        # CLASSIFY WORKING DATASET COLUMNS
        # ====================================================
        
        numeric_columns = []
        datetime_columns = []
        categorical_columns = []

        for column in working_df.columns:

            series = working_df[column]

            semantic_type = self._semantic_type(
                schema,
                column,
                series
            )

            # Datetime first
            if (
                semantic_type == "Datetime"
                or self._is_datetime(series)
            ):
                datetime_columns.append(column)
                continue

            # Identifier should not accidentally become a metric
            if semantic_type == "Identifier":
                continue

            # Important:
            # actual data can override incorrect Text/Categorical
            # schema classification when values are mostly numeric.
            if (
                semantic_type == "Numerical"
                or self._is_numeric(series)
            ):
                numeric_columns.append(column)
                continue

            if semantic_type in {
                "Categorical",
                "Boolean"
            }:
                categorical_columns.append(column)
                continue

            # Text with reasonable cardinality may still be useful
            # as a grouping variable.
            if self._usable_category(series):
                categorical_columns.append(column)
        
        # ====================================================
        # FIND COLUMNS MENTIONED BY USER
        # ====================================================

        mentioned_columns = (
            self._find_question_columns(
                question,
                working_df,
                schema
            )
        )

        mentioned_numeric = [
            column
            for column in mentioned_columns
            if column in numeric_columns
        ]

        mentioned_categories = [
            column
            for column in mentioned_columns
            if column in categorical_columns
        ]

        mentioned_datetimes = [
            column
            for column in mentioned_columns
            if column in datetime_columns
        ]

        # ====================================================
        # EXPLICIT BOX / VIOLIN
        # ====================================================

        if requested_type in {
            "box",
            "violin"
        }:

            if not numeric_columns:
                return None

            numeric_column = (
                mentioned_numeric[0]
                if mentioned_numeric
                else numeric_columns[0]
            )

            category_column = (
                mentioned_categories[0]
                if mentioned_categories
                else None
            )

            color_column = None

            if len(mentioned_categories) >= 2:

                color_column = (
                    mentioned_categories[1]
                )

            

            selected_columns = [
                numeric_column
            ]

            if category_column:
                selected_columns.insert(
                    0,
                    category_column
                )

            if (
                color_column
                and
                color_column not in selected_columns
            ):
                selected_columns.append(
                    color_column
                )

            chart_df = (
                working_df[
                    selected_columns
                ]
                .copy()
            )

            chart_df[numeric_column] = (
                pd.to_numeric(
                    chart_df[numeric_column],
                    errors="coerce"
                )
            )

            chart_df = (
                chart_df.dropna(
                    subset=[
                        numeric_column
                    ]
                )
            )

            if chart_df.empty:
                return None

            if category_column:

                chart_df[category_column] = (
                    chart_df[category_column]
                    .fillna("Missing")
                    .astype(str)
                )

            if color_column:

                chart_df[color_column] = (
                    chart_df[color_column]
                    .fillna("Missing")
                    .astype(str)
                )

            if category_column:

                fallback_title = (
                    f"{self._pretty_name(numeric_column)} "
                    "by "
                    f"{self._pretty_name(category_column)}"
                )

            else:

                fallback_title = (
                    f"{self._pretty_name(numeric_column)} "
                    "Distribution"
                )

            return self._create_chart(
                chart_type=requested_type,

                title=
                    self._question_title(
                        question,
                        fallback=fallback_title
                    ),

                x=category_column,

                y=numeric_column,

                color=color_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "The user explicitly requested "
                    f"a {requested_type} plot."
                ),

                metadata={
                    "intent":
                        (
                            "group_distribution"
                            if category_column
                            else
                            "distribution"
                        ),

                    "metric":
                        numeric_column,

                    "category":
                        category_column,

                    "color_group":
                        color_column,

                    "data_source":
                        data_source,

                    "row_count":
                        len(chart_df)
                }
            )

        # ====================================================
        # EXPLICIT HISTOGRAM
        # ====================================================

        if requested_type == "histogram":

            if not numeric_columns:
                return None

            column = (
                mentioned_numeric[0]
                if mentioned_numeric
                else numeric_columns[0]
            )

            histogram = (
                self._create_histogram_data(
                    working_df[column],
                    column
                )
            )

            if not histogram:
                return None

            return self._create_chart(
                chart_type="histogram",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(column)} "
                            "Distribution"
                        )
                    ),

                x="bin",

                y="frequency",

                data=histogram,

                reason=(
                    "The user explicitly requested "
                    "a histogram."
                ),

                metadata={
                    "intent": "distribution",
                    "metric": column,
                    "data_source": data_source
                }
            )

        # ====================================================
        # EXPLICIT SCATTER
        # ====================================================

        if requested_type == "scatter":

            if len(numeric_columns) < 2:
                return None

            if len(mentioned_numeric) >= 2:

                x_column = mentioned_numeric[0]
                y_column = mentioned_numeric[1]

            else:

                x_column = numeric_columns[0]
                y_column = numeric_columns[1]

            selected_columns = [
                x_column,
                y_column
            ]

            color_column = None

            if mentioned_categories:

                color_column = (
                    mentioned_categories[0]
                )

                selected_columns.append(
                    color_column
                )

            chart_df = (
                working_df[
                    selected_columns
                ]
                .copy()
            )

            chart_df[x_column] = (
                pd.to_numeric(
                    chart_df[x_column],
                    errors="coerce"
                )
            )

            chart_df[y_column] = (
                pd.to_numeric(
                    chart_df[y_column],
                    errors="coerce"
                )
            )

            chart_df = (
                chart_df
                .dropna(
                    subset=[
                        x_column,
                        y_column
                    ]
                )
                .head(
                    self.max_scatter_points
                )
            )

            if chart_df.empty:
                return None

            if color_column:

                chart_df[color_column] = (
                    chart_df[color_column]
                    .fillna("Missing")
                    .astype(str)
                )

            return self._create_chart(
                chart_type="scatter",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(y_column)} "
                            "vs "
                            f"{self._pretty_name(x_column)}"
                        )
                    ),

                x=x_column,

                y=y_column,

                color=color_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "The user explicitly requested "
                    "a scatter plot."
                ),

                metadata={
                    "intent": "relationship",
                    "x_metric": x_column,
                    "y_metric": y_column,
                    "color_group": color_column,
                    "data_source": data_source,
                    "sample_size": len(chart_df)
                }
            )

        # ====================================================
        # EXPLICIT HEATMAP
        # ====================================================

        if requested_type == "heatmap":

            selected_numeric = (
                mentioned_numeric
                if len(mentioned_numeric) >= 2
                else numeric_columns
            )

            numeric_df = (
                self._prepare_numeric_dataframe(
                    working_df,
                    selected_numeric
                )
            )

            if numeric_df.shape[1] < 2:
                return None

            matrix = (
                numeric_df.corr()
            )

            return self._create_chart(
                chart_type="heatmap",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            "Numerical Correlation Heatmap"
                        )
                    ),

                x="x",

                y="y",

                data=
                    self._correlation_records(
                        matrix
                    ),

                reason=(
                    "The user explicitly requested "
                    "a heatmap."
                ),

                metadata={
                    "intent": "correlation",
                    "value": "correlation",
                    "data_source": data_source,
                    "columns": list(
                        numeric_df.columns
                    )
                }
            )

        # ====================================================
        # EXPLICIT STACKED BAR
        # ====================================================

        if requested_type == "stacked_bar":

            # Prefer SQL result for aggregated output.

            bar_df = df

            result_categories = [
                column
                for column in bar_df.columns
                if not self._is_numeric(
                    bar_df[column]
                )
                and not self._is_datetime(
                    bar_df[column]
                )
            ]

            result_numeric = [
                column
                for column in bar_df.columns
                if self._is_numeric(
                    bar_df[column]
                )
            ]

            mentioned_result = (
                self._find_question_columns(
                    question,
                    bar_df
                )
            )

            mentioned_result_categories = [
                column
                for column in mentioned_result
                if column in result_categories
            ]

            if len(mentioned_result_categories) >= 2:

                x_column = (
                    mentioned_result_categories[0]
                )

                color_column = (
                    mentioned_result_categories[1]
                )

            elif len(result_categories) >= 2:

                x_column = result_categories[0]
                color_column = result_categories[1]

            else:

                return None

            if result_numeric:

                y_column = (
                    result_numeric[-1]
                )

                chart_df = (
                    bar_df[
                        [
                            x_column,
                            color_column,
                            y_column
                        ]
                    ]
                    .copy()
                )

            else:

                chart_df = (
                    bar_df[
                        [
                            x_column,
                            color_column
                        ]
                    ]
                    .copy()
                )

                chart_df = (
                    chart_df
                    .groupby(
                        [
                            x_column,
                            color_column
                        ]
                    )
                    .size()
                    .reset_index(
                        name="count"
                    )
                )

                y_column = "count"

            return self._create_chart(
                chart_type="stacked_bar",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(color_column)} "
                            "by "
                            f"{self._pretty_name(x_column)}"
                        )
                    ),

                x=x_column,

                y=y_column,

                color=color_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "The user explicitly requested "
                    "a stacked bar chart."
                ),

                metadata={
                    "intent": "composition",
                    "data_source": "sql_result"
                }
            )

        # ====================================================
        # EXPLICIT LINE
        # ====================================================

        if requested_type == "line":

            line_df = df

            result_numeric = [
                column
                for column in line_df.columns
                if self._is_numeric(
                    line_df[column]
                )
            ]

            result_datetime = [
                column
                for column in line_df.columns
                if self._is_datetime(
                    line_df[column]
                )
            ]

            result_categories = [
                column
                for column in line_df.columns
                if column not in result_numeric
                and column not in result_datetime
            ]

            if not result_numeric:
                return None

            mentioned_result = (
                self._find_question_columns(
                    question,
                    line_df
                )
            )

            y_candidates = [
                column
                for column in mentioned_result
                if column in result_numeric
            ]

            x_candidates = [
                column
                for column in mentioned_result
                if (
                    column in result_datetime
                    or column in result_categories
                )
            ]

            y_column = (
                y_candidates[0]
                if y_candidates
                else result_numeric[0]
            )

            if x_candidates:

                x_column = x_candidates[0]

            elif result_datetime:

                x_column = result_datetime[0]

            elif result_categories:

                x_column = result_categories[0]

            else:

                return None

            chart_df = (
                line_df[
                    [
                        x_column,
                        y_column
                    ]
                ]
                .dropna()
                .copy()
            )

            if x_column in result_datetime:

                chart_df = (
                    chart_df.sort_values(
                        x_column
                    )
                )

            return self._create_chart(
                chart_type="line",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(y_column)} "
                            "by "
                            f"{self._pretty_name(x_column)}"
                        )
                    ),

                x=x_column,

                y=y_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "The user explicitly requested "
                    "a line chart."
                ),

                metadata={
                    "intent": "trend",
                    "data_source": "sql_result"
                }
            )

        # ====================================================
        # EXPLICIT BAR
        # ====================================================

        if requested_type == "bar":

            result_numeric = [
                column
                for column in df.columns
                if self._is_numeric(
                    df[column]
                )
            ]

            result_categories = [
                column
                for column in df.columns
                if column not in result_numeric
                and not self._is_datetime(
                    df[column]
                )
            ]

            mentioned_result = (
                self._find_question_columns(
                    question,
                    df
                )
            )

            mentioned_result_numeric = [
                column
                for column in mentioned_result
                if column in result_numeric
            ]

            mentioned_result_categories = [
                column
                for column in mentioned_result
                if column in result_categories
            ]

            if not result_numeric:
                return None

            if not result_categories:
                return None

            x_column = (
                mentioned_result_categories[0]
                if mentioned_result_categories
                else result_categories[0]
            )

            y_column = (
                mentioned_result_numeric[0]
                if mentioned_result_numeric
                else result_numeric[-1]
            )

            return self._category_numeric_bar(
                df,
                x_column,
                y_column,
                question
            )

        # ====================================================
        # AUTOMATIC DATETIME + NUMERIC
        # ====================================================

        result_numeric = [
            column
            for column in df.columns
            if self._is_numeric(
                df[column]
            )
        ]

        result_datetime = [
            column
            for column in df.columns
            if self._is_datetime(
                df[column]
            )
        ]

        result_categories = [
            column
            for column in df.columns
            if column not in result_numeric
            and column not in result_datetime
        ]

        if (
            result_datetime
            and
            result_numeric
        ):

            x_column = result_datetime[0]
            y_column = result_numeric[0]

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

            return self._create_chart(
                chart_type="line",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(y_column)} "
                            "Over Time"
                        )
                    ),

                x=x_column,
                y=y_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "Datetime and numerical results "
                    "are suitable for a line chart."
                ),

                metadata={
                    "intent": "temporal",
                    "data_source": "sql_result"
                }
            )

        # ====================================================
        # AUTOMATIC CATEGORY + NUMERIC
        # ====================================================

        if (
            result_categories
            and
            result_numeric
        ):

            return self._category_numeric_bar(
                df,
                result_categories[0],
                result_numeric[0],
                question
            )

        # ====================================================
        # AUTOMATIC NUMERIC + NUMERIC
        # ====================================================

        if len(result_numeric) >= 2:

            x_column = result_numeric[0]
            y_column = result_numeric[1]

            chart_df = (
                df[
                    [
                        x_column,
                        y_column
                    ]
                ]
                .dropna()
                .head(
                    self.max_scatter_points
                )
            )

            return self._create_chart(
                chart_type="scatter",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(y_column)} "
                            "vs "
                            f"{self._pretty_name(x_column)}"
                        )
                    ),

                x=x_column,
                y=y_column,

                data=
                    self.dataframe_to_records(
                        chart_df
                    ),

                reason=(
                    "Two numerical query-result columns "
                    "are suitable for a scatter plot."
                ),

                metadata={
                    "intent": "relationship",
                    "data_source": "sql_result"
                }
            )

        # ====================================================
        # SINGLE CATEGORY
        # ====================================================

        if (
            len(df.columns) == 1
            and
            result_categories
        ):

            column = result_categories[0]

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

            return self._create_chart(
                chart_type="bar",

                title=
                    self._question_title(
                        question,
                        fallback=(
                            f"{self._pretty_name(column)} "
                            "Distribution"
                        )
                    ),

                x=column,
                y="count",

                data=
                    self.dataframe_to_records(
                        counts
                    ),

                reason=(
                    "Categorical result is suitable "
                    "for a frequency bar chart."
                ),

                metadata={
                    "intent": "distribution",
                    "data_source": "sql_result"
                }
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
            .copy()
        )

        chart_df[
            y_column
        ] = pd.to_numeric(
            chart_df[
                y_column
            ],
            errors="coerce"
        )

        chart_df = (
            chart_df
            .dropna(
                subset=[
                    y_column
                ]
            )
        )

        if chart_df.empty:
            return None

        chart_df[
            x_column
        ] = (
            chart_df[
                x_column
            ]
            .fillna("Missing")
            .astype(str)
        )

        # SQL result may already be aggregated.
        # Preserve it when categories are unique.

        if (
            chart_df[x_column]
            .nunique()
            ==
            len(chart_df)
        ):

            chart_df = (
                chart_df.head(
                    self.max_categories
                )
            )

        else:

            chart_df = (
                chart_df
                .groupby(
                    x_column,
                    as_index=False
                )[
                    y_column
                ]
                .mean()
                .sort_values(
                    y_column,
                    ascending=False
                )
                .head(
                    self.max_categories
                )
            )

        return self._create_chart(
            chart_type="bar",

            title=
                self._question_title(
                    question,
                    fallback=(
                        f"{self._pretty_name(y_column)} "
                        "by "
                        f"{self._pretty_name(x_column)}"
                    )
                ),

            x=x_column,

            y=y_column,

            data=
                self.dataframe_to_records(
                    chart_df
                ),

            reason=(
                "Categorical and numerical results "
                "are suitable for a bar chart."
            ),

            metadata={
                "intent": "comparison"
            }
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

        # Square-root rule with a practical cap.

        number_of_bins = min(
            20,
            max(
                5,
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
            bins=number_of_bins,
            duplicates="drop"
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
                    int(count)
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

        iqr = q3 - q1

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

        # Do not send thousands of outlier values to
        # the frontend.

        outlier_values = (
            outliers
            .head(100)
            .tolist()
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
                        for value
                        in outlier_values
                    ],

                "outlier_count":
                    int(
                        len(outliers)
                    )
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
                .rstrip("?")
            )

            if len(title) <= 100:
                return title

        return fallback