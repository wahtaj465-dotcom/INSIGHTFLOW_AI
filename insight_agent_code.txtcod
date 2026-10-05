import re

import pandas as pd


class InsightAgent:
    """
    InsightFlow Insight Agent V4.

    Strategy:

        SQL result
            ↓
        Try LLM insight
            ↓
        Success?
         /   \
       yes    no
       ↓       ↓
      LLM    Local fallback

    The LLM is an enrichment layer.

    A temporary LLM failure, quota error, timeout,
    connection problem, or other generation failure
    must NOT destroy a valid SQL analysis.
    """

    def __init__(
        self,
        llm_service,
        max_rows=30
    ):
        self.llm_service = llm_service
        self.max_rows = max_rows


    # ========================================================
    # 1. VALIDATE INPUTS
    # ========================================================

    def validate_inputs(
        self,
        question,
        sql,
        result
    ):

        if not isinstance(question, str):

            raise TypeError(
                "Question must be a string."
            )

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        if not isinstance(sql, str):

            raise TypeError(
                "SQL must be a string."
            )

        if not sql.strip():

            raise ValueError(
                "SQL cannot be empty."
            )

        if not isinstance(
            result,
            pd.DataFrame
        ):

            raise TypeError(
                "SQL result must be a Pandas DataFrame."
            )


    # ========================================================
    # 2. PREPARE SQL RESULT
    # ========================================================

    def prepare_result_context(
        self,
        result
    ):

        if result.empty:

            return (
                "The SQL query returned zero rows."
            )

        limited_result = (
            result.head(
                self.max_rows
            )
        )

        result_text = (
            limited_result.to_string(
                index=False
            )
        )

        if len(result) > self.max_rows:

            result_text += (
                f"\n\nOnly the first "
                f"{self.max_rows} of "
                f"{len(result)} rows are shown."
            )

        return result_text


    # ========================================================
    # 3. IDENTIFY RELEVANT COLUMNS
    # ========================================================

    def identify_relevant_columns(
        self,
        question,
        sql,
        eda_results
    ):

        if not eda_results:

            return []

        available_columns = set()

        for section in [
            "numerical",
            "categorical",
            "distributions",
            "datetime"
        ]:

            section_data = (
                eda_results.get(
                    section,
                    {}
                )
            )

            if isinstance(
                section_data,
                dict
            ):

                available_columns.update(
                    section_data.keys()
                )


        correlations = (
            eda_results.get(
                "correlations",
                {}
            )
        )

        if isinstance(
            correlations,
            dict
        ):

            for pair in correlations.keys():

                if (
                    isinstance(pair, str)
                    and
                    " vs " in pair
                ):

                    column_1, column_2 = (
                        pair.split(
                            " vs ",
                            1
                        )
                    )

                    available_columns.add(
                        column_1
                    )

                    available_columns.add(
                        column_2
                    )


        search_text = (
            f"{question or ''} "
            f"{sql or ''}"
        ).lower()


        relevant_columns = []


        for column in available_columns:

            column_string = str(
                column
            )

            variants = {

                column_string.lower(),

                column_string
                .lower()
                .replace(
                    "_",
                    " "
                )
            }


            for variant in variants:

                pattern = (
                    r"(?<!\w)"
                    +
                    re.escape(
                        variant
                    )
                    +
                    r"(?!\w)"
                )

                if re.search(
                    pattern,
                    search_text
                ):

                    relevant_columns.append(
                        column
                    )

                    break


        return sorted(
            relevant_columns
        )


    # ========================================================
    # 4. PREPARE QUALITY CONTEXT
    # ========================================================

    def prepare_quality_context(
        self,
        quality_report,
        relevant_columns=None
    ):

        if not quality_report:

            return (
                "No data-quality information "
                "was provided."
            )


        relevant_columns = set(
            relevant_columns or []
        )

        lines = []


        duplicate_rows = (
            quality_report.get(
                "duplicate_rows",
                0
            )
        )

        if duplicate_rows:

            lines.append(
                f"Duplicate rows: "
                f"{duplicate_rows}"
            )


        missing_values = (
            quality_report.get(
                "missing_values",
                {}
            )
        )

        if isinstance(
            missing_values,
            dict
        ):

            relevant_missing = {}

            for column, count in (
                missing_values.items()
            ):

                if (
                    not relevant_columns
                    or
                    column in relevant_columns
                ):

                    if count:

                        relevant_missing[
                            column
                        ] = count


            if relevant_missing:

                lines.append(
                    "Relevant missing values:"
                )

                for column, count in (
                    relevant_missing.items()
                ):

                    lines.append(
                        f"- {column}: {count}"
                    )


        invalid_dates = (
            quality_report.get(
                "invalid_dates",
                {}
            )
        )

        if isinstance(
            invalid_dates,
            dict
        ):

            relevant_invalid = {}

            for column, count in (
                invalid_dates.items()
            ):

                if (
                    not relevant_columns
                    or
                    column in relevant_columns
                ):

                    if count:

                        relevant_invalid[
                            column
                        ] = count


            if relevant_invalid:

                lines.append(
                    "Relevant invalid dates:"
                )

                for column, count in (
                    relevant_invalid.items()
                ):

                    lines.append(
                        f"- {column}: {count}"
                    )


        if not lines:

            return (
                "No relevant data-quality issues "
                "were detected for this analysis."
            )


        return "\n".join(
            lines
        )


    # ========================================================
    # 5. PREPARE ANOMALY CONTEXT
    # ========================================================

    def prepare_anomaly_context(
        self,
        anomalies,
        relevant_columns=None
    ):

        if not anomalies:

            return (
                "No numerical anomalies "
                "were detected."
            )


        relevant_columns = set(
            relevant_columns or []
        )

        lines = []


        for column, info in (
            anomalies.items()
        ):

            if (
                relevant_columns
                and
                column not in relevant_columns
            ):

                continue


            if not isinstance(
                info,
                dict
            ):

                continue


            negative_values = (
                info.get(
                    "negative_values",
                    []
                )
            )

            outliers = (
                info.get(
                    "outliers",
                    []
                )
            )

            lower_bound = (
                info.get(
                    "lower_bound"
                )
            )

            upper_bound = (
                info.get(
                    "upper_bound"
                )
            )


            if not (
                negative_values
                or
                outliers
            ):

                continue


            lines.append(
                f"Column: {column}"
            )


            if negative_values:

                lines.append(
                    "- Flagged negative values: "
                    f"{negative_values}"
                )


            if outliers:

                lines.append(
                    "- IQR outliers: "
                    f"{outliers}"
                )


            if (
                lower_bound is not None
                and
                upper_bound is not None
            ):

                try:

                    lines.append(
                        "- IQR expected range: "
                        f"{float(lower_bound):.2f} "
                        "to "
                        f"{float(upper_bound):.2f}"
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass


        if not lines:

            return (
                "No relevant numerical anomalies "
                "were detected for this analysis."
            )


        return "\n".join(
            lines
        )


    # ========================================================
    # 6. PREPARE RELEVANT EDA CONTEXT
    # ========================================================

    def prepare_eda_context(
        self,
        eda_results,
        relevant_columns
    ):

        if not eda_results:

            return (
                "No EDA information was provided."
            )


        if not relevant_columns:

            return (
                "No directly relevant EDA columns "
                "were identified."
            )


        relevant_columns = set(
            relevant_columns
        )

        lines = []


        # ----------------------------------------------------
        # NUMERICAL
        # ----------------------------------------------------

        numerical = (
            eda_results.get(
                "numerical",
                {}
            )
        )

        if isinstance(
            numerical,
            dict
        ):

            for column in relevant_columns:

                stats = numerical.get(
                    column
                )

                if not isinstance(
                    stats,
                    dict
                ):

                    continue


                lines.append(
                    f"Numerical column: {column}"
                )


                for key, label in [

                    (
                        "count",
                        "Count"
                    ),

                    (
                        "mean",
                        "Mean"
                    ),

                    (
                        "median",
                        "Median"
                    ),

                    (
                        "min",
                        "Minimum"
                    ),

                    (
                        "max",
                        "Maximum"
                    ),

                    (
                        "std",
                        "Standard deviation"
                    )
                ]:

                    value = stats.get(
                        key
                    )

                    if value is None:

                        continue


                    if isinstance(
                        value,
                        (int, float)
                    ):

                        lines.append(
                            f"- {label}: "
                            f"{value:.4f}"
                        )

                    else:

                        lines.append(
                            f"- {label}: "
                            f"{value}"
                        )


        # ----------------------------------------------------
        # CATEGORICAL
        # ----------------------------------------------------

        categorical = (
            eda_results.get(
                "categorical",
                {}
            )
        )

        if isinstance(
            categorical,
            dict
        ):

            for column in relevant_columns:

                stats = categorical.get(
                    column
                )

                if not isinstance(
                    stats,
                    dict
                ):

                    continue


                lines.append(
                    f"Categorical column: {column}"
                )


                for key, label in [

                    (
                        "unique_values",
                        "Unique values"
                    ),

                    (
                        "most_common",
                        "Most common value"
                    ),

                    (
                        "most_common_count",
                        "Most common count"
                    ),

                    (
                        "distribution",
                        "Distribution"
                    )
                ]:

                    if key in stats:

                        lines.append(
                            f"- {label}: "
                            f"{stats[key]}"
                        )


        # ----------------------------------------------------
        # DISTRIBUTIONS
        # ----------------------------------------------------

        distributions = (
            eda_results.get(
                "distributions",
                {}
            )
        )

        if isinstance(
            distributions,
            dict
        ):

            for column in relevant_columns:

                distribution = (
                    distributions.get(
                        column
                    )
                )

                if not isinstance(
                    distribution,
                    dict
                ):

                    continue


                lines.append(
                    f"Distribution for "
                    f"{column}:"
                )


                skewness = (
                    distribution.get(
                        "skewness"
                    )
                )

                shape = (
                    distribution.get(
                        "shape"
                    )
                )


                if skewness is not None:

                    try:

                        lines.append(
                            "- Skewness: "
                            f"{float(skewness):.4f}"
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        pass


                if shape is not None:

                    lines.append(
                        f"- Shape: {shape}"
                    )


        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        datetime_results = (
            eda_results.get(
                "datetime",
                {}
            )
        )

        if isinstance(
            datetime_results,
            dict
        ):

            for column in relevant_columns:

                info = (
                    datetime_results.get(
                        column
                    )
                )

                if not isinstance(
                    info,
                    dict
                ):

                    continue


                lines.append(
                    f"Datetime column: {column}"
                )


                for key, label in [

                    (
                        "valid_dates",
                        "Valid dates"
                    ),

                    (
                        "missing_dates",
                        "Missing dates"
                    ),

                    (
                        "earliest",
                        "Earliest"
                    ),

                    (
                        "latest",
                        "Latest"
                    ),

                    (
                        "range_days",
                        "Range in days"
                    )
                ]:

                    if key in info:

                        lines.append(
                            f"- {label}: "
                            f"{info[key]}"
                        )


        # ----------------------------------------------------
        # CORRELATIONS
        # ----------------------------------------------------

        correlations = (
            eda_results.get(
                "correlations",
                {}
            )
        )

        if isinstance(
            correlations,
            dict
        ):

            for pair, value in (
                correlations.items()
            ):

                if (
                    not isinstance(
                        pair,
                        str
                    )
                    or
                    " vs " not in pair
                ):

                    continue


                column_1, column_2 = (
                    pair.split(
                        " vs ",
                        1
                    )
                )


                if (
                    column_1
                    not in relevant_columns
                    and
                    column_2
                    not in relevant_columns
                ):

                    continue


                try:

                    numeric_value = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue


                absolute_value = abs(
                    numeric_value
                )


                if absolute_value >= 0.7:

                    strength = "strong"

                elif absolute_value >= 0.4:

                    strength = "moderate"

                else:

                    strength = "weak"


                if numeric_value > 0:

                    direction = "positive"

                elif numeric_value < 0:

                    direction = "negative"

                else:

                    direction = "no"


                lines.append(
                    f"Correlation: {pair}"
                )

                lines.append(
                    "- Value: "
                    f"{numeric_value:.4f}"
                )

                lines.append(
                    "- Interpretation: "
                    f"{strength} "
                    f"{direction} correlation"
                )


        if not lines:

            return (
                "No relevant EDA information "
                "was available."
            )


        return "\n".join(
            lines
        )


    # ========================================================
    # 7. BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        question,
        sql,
        result_context,
        quality_context,
        anomaly_context,
        eda_context
    ):

        return f"""
You are the Insight Agent inside an automated
analytics platform called InsightFlow.

Generate an accurate, concise, evidence-grounded
analytical answer.

==================================================
ORIGINAL USER QUESTION
==================================================

{question}

==================================================
SQL USED
==================================================

{sql}

==================================================
SQL RESULT
==================================================

{result_context}

==================================================
RELEVANT DATA QUALITY INFORMATION
==================================================

{quality_context}

==================================================
RELEVANT ANOMALIES
==================================================

{anomaly_context}

==================================================
RELEVANT EDA
==================================================

{eda_context}

==================================================
INSTRUCTIONS
==================================================

1. Answer the original question directly.

2. Treat the SQL result as the primary evidence.

3. Use EDA only as supporting analytical context.

4. Never invent numbers, categories, percentages,
   causes, units, currencies, dates, or facts.

5. Never claim causation from correlation.

6. Statistical outliers are flagged observations,
   not automatically errors.

7. Missing data should be mentioned only when
   relevant to the requested analysis.

8. Do not assume currencies or measurement units.

9. If the SQL result is empty, clearly state that
   no matching records were returned.

10. If evidence is insufficient, say so.

11. Keep the answer concise and analytical.

12. Do not output the SQL query.

13. Do not mention Gemini, LLMs, prompts,
    agents, or internal architecture.

Generate the final analytical insight:
"""


    # ========================================================
    # 8. GENERATE LLM INSIGHT
    # ========================================================

    def generate_insight(
        self,
        question,
        sql,
        result,
        quality_report=None,
        anomalies=None,
        eda_results=None
    ):

        self.validate_inputs(
            question,
            sql,
            result
        )


        relevant_columns = (
            self.identify_relevant_columns(

                question=
                    question,

                sql=
                    sql,

                eda_results=
                    eda_results
            )
        )


        result_context = (
            self.prepare_result_context(
                result
            )
        )


        quality_context = (
            self.prepare_quality_context(

                quality_report,

                relevant_columns
            )
        )


        anomaly_context = (
            self.prepare_anomaly_context(

                anomalies,

                relevant_columns
            )
        )


        eda_context = (
            self.prepare_eda_context(

                eda_results,

                relevant_columns
            )
        )


        prompt = self.build_prompt(

            question=
                question,

            sql=
                sql,

            result_context=
                result_context,

            quality_context=
                quality_context,

            anomaly_context=
                anomaly_context,

            eda_context=
                eda_context
        )


        insight = (
            self.llm_service.generate(
                prompt
            )
        )


        if not isinstance(
            insight,
            str
        ):

            insight = str(
                insight
            )


        return insight.strip()


    # ========================================================
    # 9. FORMAT LOCAL VALUE
    # ========================================================

    def _format_value(
        self,
        value
    ):

        if value is None:

            return "missing"


        try:

            if pd.isna(
                value
            ):

                return "missing"

        except (
            TypeError,
            ValueError
        ):

            pass


        if isinstance(
            value,
            float
        ):

            if value.is_integer():

                return (
                    f"{int(value):,}"
                )

            return (
                f"{value:,.2f}"
            )


        if isinstance(
            value,
            int
        ):

            return (
                f"{value:,}"
            )


        return str(
            value
        )


    # ========================================================
    # 10. LOCAL FALLBACK INSIGHT
    # ========================================================

    def generate_fallback_insight(
        self,
        question,
        result,
        relevant_columns=None,
        quality_report=None
    ):
        """
        Generate a deterministic analytical summary without
        calling an external model.

        This is intentionally conservative. It describes only
        what can be established directly from the SQL result.
        """

        if not isinstance(
            result,
            pd.DataFrame
        ):

            return (
                "The query completed, but its result "
                "could not be summarized."
            )


        if result.empty:

            return (
                "The query completed successfully, "
                "but no matching records were returned."
            )


        rows = len(
            result
        )

        columns = list(
            result.columns
        )


        # ----------------------------------------------------
        # SINGLE KPI
        # ----------------------------------------------------

        if (
            rows == 1
            and
            len(columns) == 1
        ):

            column = columns[0]

            value = (
                result.iloc[
                    0,
                    0
                ]
            )

            return (
                f"The result for "
                f"{self._pretty_name(column)} "
                f"is {self._format_value(value)}."
            )


        numeric_columns = list(
            result.select_dtypes(
                include="number"
            ).columns
        )

        non_numeric_columns = [

            column

            for column in columns

            if column
            not in numeric_columns
        ]


        lines = [

            (
                "The query completed successfully "
                f"and returned {rows:,} "
                f"record{'s' if rows != 1 else ''}."
            )
        ]


        # ----------------------------------------------------
        # CATEGORY + NUMERIC
        # ----------------------------------------------------

        if (
            non_numeric_columns
            and
            numeric_columns
        ):

            category = (
                non_numeric_columns[0]
            )

            metric = (
                numeric_columns[0]
            )


            valid = (
                result[
                    [
                        category,
                        metric
                    ]
                ]
                .dropna()
            )


            if not valid.empty:

                maximum_index = (
                    valid[
                        metric
                    ].idxmax()
                )

                minimum_index = (
                    valid[
                        metric
                    ].idxmin()
                )


                maximum_category = (
                    valid.loc[
                        maximum_index,
                        category
                    ]
                )

                maximum_value = (
                    valid.loc[
                        maximum_index,
                        metric
                    ]
                )


                minimum_category = (
                    valid.loc[
                        minimum_index,
                        category
                    ]
                )

                minimum_value = (
                    valid.loc[
                        minimum_index,
                        metric
                    ]
                )


                lines.append(

                    f"The highest "
                    f"{self._pretty_name(metric)} "
                    f"in the returned result is "
                    f"{self._format_value(maximum_value)} "
                    f"for "
                    f"{self._format_value(maximum_category)}."
                )


                if len(valid) > 1:

                    lines.append(

                        f"The lowest is "
                        f"{self._format_value(minimum_value)} "
                        f"for "
                        f"{self._format_value(minimum_category)}."
                    )


        # ----------------------------------------------------
        # NUMERIC RESULT
        # ----------------------------------------------------

        elif numeric_columns:

            metric = (
                numeric_columns[0]
            )

            values = (
                pd.to_numeric(
                    result[metric],
                    errors="coerce"
                )
                .dropna()
            )


            if not values.empty:

                lines.append(

                    f"{self._pretty_name(metric)} "
                    f"ranges from "
                    f"{self._format_value(values.min())} "
                    f"to "
                    f"{self._format_value(values.max())}, "
                    f"with an average of "
                    f"{self._format_value(values.mean())}."
                )


        # ----------------------------------------------------
        # CATEGORICAL RESULT
        # ----------------------------------------------------

        elif non_numeric_columns:

            category = (
                non_numeric_columns[0]
            )

            counts = (
                result[
                    category
                ]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )


            if not counts.empty:

                most_common = (
                    counts.index[0]
                )

                count = (
                    counts.iloc[0]
                )


                lines.append(

                    f"The most frequent "
                    f"{self._pretty_name(category)} "
                    f"is {most_common}, appearing "
                    f"{int(count):,} time"
                    f"{'s' if int(count) != 1 else ''}."
                )


        # ----------------------------------------------------
        # RELEVANT MISSING VALUES
        # ----------------------------------------------------

        if quality_report:

            missing_values = (
                quality_report.get(
                    "missing_values",
                    {}
                )
            )

            relevant_columns = (
                relevant_columns or []
            )


            relevant_missing = {

                column: count

                for column, count
                in missing_values.items()

                if (
                    count
                    and
                    (
                        not relevant_columns
                        or
                        column in relevant_columns
                    )
                )
            }


            if relevant_missing:

                missing_text = ", ".join(

                    f"{column}: {count}"

                    for column, count
                    in relevant_missing.items()
                )


                lines.append(

                    "Interpret this result with care "
                    "because relevant missing values "
                    f"remain ({missing_text})."
                )


        return " ".join(
            lines
        )


    # ========================================================
    # 11. PRETTY NAME
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
    # 12. ANALYZE SQL RESPONSE
    # ========================================================

    def analyze(
        self,
        sql_response,
        quality_report=None,
        anomalies=None,
        eda_results=None
    ):

        if not isinstance(
            sql_response,
            dict
        ):

            raise TypeError(
                "SQL response must be a dictionary."
            )


        # ----------------------------------------------------
        # SQL FAILED
        # ----------------------------------------------------

        if not sql_response.get(
            "success",
            False
        ):

            return {

                "success":
                    False,

                "llm_success":
                    False,

                "source":
                    None,

                "question":
                    sql_response.get(
                        "question"
                    ),

                "sql":
                    sql_response.get(
                        "sql"
                    ),

                "result":
                    None,

                "insight":
                    None,

                "relevant_columns":
                    [],

                "error":
                    (
                        "Cannot generate insights because "
                        "the SQL Agent did not complete "
                        "successfully."
                    )
            }


        question = (
            sql_response.get(
                "question"
            )
        )

        sql = (
            sql_response.get(
                "sql"
            )
        )

        result = (
            sql_response.get(
                "result"
            )
        )


        # ----------------------------------------------------
        # IDENTIFY COLUMNS BEFORE LLM CALL
        # ----------------------------------------------------

        try:

            relevant_columns = (
                self.identify_relevant_columns(

                    question=
                        question,

                    sql=
                        sql,

                    eda_results=
                        eda_results
                )
            )

        except Exception:

            relevant_columns = []


        # ----------------------------------------------------
        # TRY LLM
        # ----------------------------------------------------

        try:

            insight = (
                self.generate_insight(

                    question=
                        question,

                    sql=
                        sql,

                    result=
                        result,

                    quality_report=
                        quality_report,

                    anomalies=
                        anomalies,

                    eda_results=
                        eda_results
                )
            )


            if insight:

                return {

                    "success":
                        True,

                    "llm_success":
                        True,

                    "source":
                        "llm",

                    "question":
                        question,

                    "sql":
                        sql,

                    "result":
                        result,

                    "insight":
                        insight,

                    "relevant_columns":
                        relevant_columns,

                    "error":
                        None
                }


        except Exception as error:

            llm_error = str(
                error
            )

            print(
                "Insight LLM unavailable. "
                "Using local fallback. "
                f"Reason: {llm_error}"
            )

        else:

            llm_error = (
                "LLM returned an empty insight."
            )


        # ----------------------------------------------------
        # LOCAL FALLBACK
        # ----------------------------------------------------

        try:

            fallback = (
                self.generate_fallback_insight(

                    question=
                        question,

                    result=
                        result,

                    relevant_columns=
                        relevant_columns,

                    quality_report=
                        quality_report
                )
            )


            return {

                "success":
                    True,

                "llm_success":
                    False,

                "source":
                    "fallback",

                "question":
                    question,

                "sql":
                    sql,

                "result":
                    result,

                "insight":
                    fallback,

                "relevant_columns":
                    relevant_columns,

                "error":
                    None,

                "llm_error":
                    llm_error
            }


        except Exception as fallback_error:

            return {

                # SQL is still valid, so insight failure
                # should not invalidate the analysis.

                "success":
                    True,

                "llm_success":
                    False,

                "source":
                    "unavailable",

                "question":
                    question,

                "sql":
                    sql,

                "result":
                    result,

                "insight":
                    (
                        "The query completed successfully, "
                        "but an analytical summary could "
                        "not be generated."
                    ),

                "relevant_columns":
                    relevant_columns,

                "error":
                    None,

                "llm_error":
                    llm_error,

                "fallback_error":
                    str(
                        fallback_error
                    )
            }