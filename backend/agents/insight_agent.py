import re
import pandas as pd


class InsightAgent:
    """
    InsightFlow Insight Agent V3.

    Combines:
    - User question
    - Generated SQL
    - SQL result
    - Data quality information
    - Numerical anomalies
    - Relevant EDA information

    to generate evidence-grounded analytical insights.
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

        if not isinstance(result, pd.DataFrame):
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
            return "The SQL query returned zero rows."

        limited_result = result.head(
            self.max_rows
        )

        result_text = limited_result.to_string(
            index=False
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
        """
        Determine which original dataset columns are
        relevant to the current analytical question.
        """

        if not eda_results:
            return []

        available_columns = set()

        for section in [
            "numerical",
            "categorical",
            "distributions",
            "datetime"
        ]:

            section_data = eda_results.get(
                section,
                {}
            )

            available_columns.update(
                section_data.keys()
            )

        # Add columns appearing in correlation pairs
        for pair in eda_results.get(
            "correlations",
            {}
        ).keys():

            if " vs " in pair:

                column_1, column_2 = pair.split(
                    " vs ",
                    1
                )

                available_columns.add(
                    column_1
                )

                available_columns.add(
                    column_2
                )

        search_text = (
            f"{question} {sql}"
        ).lower()

        relevant_columns = []

        for column in available_columns:

            variants = {
                column.lower(),
                column.lower().replace(
                    "_",
                    " "
                )
            }

            matched = False

            for variant in variants:

                pattern = (
                    r"(?<!\w)"
                    + re.escape(variant)
                    + r"(?!\w)"
                )

                if re.search(
                    pattern,
                    search_text
                ):
                    matched = True
                    break

            if matched:
                relevant_columns.append(
                    column
                )

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
                "No data-quality information was provided."
            )

        relevant_columns = set(
            relevant_columns or []
        )

        lines = []

        duplicate_rows = quality_report.get(
            "duplicate_rows",
            0
        )

        if duplicate_rows > 0:

            lines.append(
                f"Duplicate rows: {duplicate_rows}"
            )

        missing_values = quality_report.get(
            "missing_values",
            {}
        )

        relevant_missing = {}

        for column, count in missing_values.items():

            if (
                not relevant_columns
                or column in relevant_columns
            ):
                relevant_missing[column] = count

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

        invalid_dates = quality_report.get(
            "invalid_dates",
            {}
        )

        relevant_invalid_dates = {}

        for column, count in invalid_dates.items():

            if (
                not relevant_columns
                or column in relevant_columns
            ):
                relevant_invalid_dates[
                    column
                ] = count

        if relevant_invalid_dates:

            lines.append(
                "Relevant invalid dates:"
            )

            for column, count in (
                relevant_invalid_dates.items()
            ):
                lines.append(
                    f"- {column}: {count}"
                )

        if not lines:

            return (
                "No relevant data-quality issues "
                "were detected for this analysis."
            )

        return "\n".join(lines)


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
                "No numerical anomalies were detected."
            )

        relevant_columns = set(
            relevant_columns or []
        )

        lines = []

        for column, info in anomalies.items():

            if (
                relevant_columns
                and column not in relevant_columns
            ):
                continue

            negative_values = info.get(
                "negative_values",
                []
            )

            outliers = info.get(
                "outliers",
                []
            )

            lower_bound = info.get(
                "lower_bound"
            )

            upper_bound = info.get(
                "upper_bound"
            )

            if not (
                negative_values
                or outliers
            ):
                continue

            lines.append(
                f"Column: {column}"
            )

            if negative_values:

                lines.append(
                    f"- Flagged negative values: "
                    f"{negative_values}"
                )

            if outliers:

                lines.append(
                    f"- IQR outliers: "
                    f"{outliers}"
                )

            if (
                lower_bound is not None
                and upper_bound is not None
            ):

                lines.append(
                    f"- IQR expected range: "
                    f"{lower_bound:.2f} to "
                    f"{upper_bound:.2f}"
                )

        if not lines:

            return (
                "No relevant numerical anomalies "
                "were detected for this analysis."
            )

        return "\n".join(lines)


    # ========================================================
    # 6. PREPARE RELEVANT EDA CONTEXT
    # ========================================================

    def prepare_eda_context(
        self,
        eda_results,
        relevant_columns
    ):
        """
        Convert only relevant EDA information into
        compact context for the LLM.
        """

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
        # Numerical statistics
        # ----------------------------------------------------

        numerical = eda_results.get(
            "numerical",
            {}
        )

        for column in relevant_columns:

            if column not in numerical:
                continue

            stats = numerical[column]

            lines.append(
                f"Numerical column: {column}"
            )

            lines.append(
                f"- Count: {stats['count']}"
            )

            lines.append(
                f"- Mean: {stats['mean']:.4f}"
            )

            lines.append(
                f"- Median: {stats['median']:.4f}"
            )

            lines.append(
                f"- Minimum: {stats['min']:.4f}"
            )

            lines.append(
                f"- Maximum: {stats['max']:.4f}"
            )

            lines.append(
                f"- Standard deviation: "
                f"{stats['std']:.4f}"
            )


        # ----------------------------------------------------
        # Categorical statistics
        # ----------------------------------------------------

        categorical = eda_results.get(
            "categorical",
            {}
        )

        for column in relevant_columns:

            if column not in categorical:
                continue

            stats = categorical[column]

            lines.append(
                f"Categorical column: {column}"
            )

            lines.append(
                f"- Unique values: "
                f"{stats['unique_values']}"
            )

            lines.append(
                f"- Most common value: "
                f"{stats['most_common']}"
            )

            lines.append(
                f"- Most common count: "
                f"{stats['most_common_count']}"
            )

            lines.append(
                f"- Distribution: "
                f"{stats['distribution']}"
            )


        # ----------------------------------------------------
        # Distribution / skewness
        # ----------------------------------------------------

        distributions = eda_results.get(
            "distributions",
            {}
        )

        for column in relevant_columns:

            if column not in distributions:
                continue

            distribution = (
                distributions[column]
            )

            lines.append(
                f"Distribution for {column}:"
            )

            lines.append(
                f"- Skewness: "
                f"{distribution['skewness']:.4f}"
            )

            lines.append(
                f"- Shape: "
                f"{distribution['shape']}"
            )


        # ----------------------------------------------------
        # Datetime information
        # ----------------------------------------------------

        datetime_results = eda_results.get(
            "datetime",
            {}
        )

        for column in relevant_columns:

            if column not in datetime_results:
                continue

            info = datetime_results[column]

            lines.append(
                f"Datetime column: {column}"
            )

            lines.append(
                f"- Valid dates: "
                f"{info['valid_dates']}"
            )

            lines.append(
                f"- Missing dates: "
                f"{info['missing_dates']}"
            )

            lines.append(
                f"- Earliest: "
                f"{info['earliest']}"
            )

            lines.append(
                f"- Latest: "
                f"{info['latest']}"
            )

            lines.append(
                f"- Range: "
                f"{info['range_days']} days"
            )


        # ----------------------------------------------------
        # Relevant correlations
        # ----------------------------------------------------

        correlations = eda_results.get(
            "correlations",
            {}
        )

        for pair, value in correlations.items():

            if " vs " not in pair:
                continue

            column_1, column_2 = pair.split(
                " vs ",
                1
            )

            if (
                column_1 not in relevant_columns
                and column_2 not in relevant_columns
            ):
                continue

            absolute_value = abs(
                value
            )

            if absolute_value >= 0.7:
                strength = "strong"

            elif absolute_value >= 0.4:
                strength = "moderate"

            else:
                strength = "weak"

            if value > 0:
                direction = "positive"

            elif value < 0:
                direction = "negative"

            else:
                direction = "no"

            lines.append(
                f"Correlation: {pair}"
            )

            lines.append(
                f"- Value: {value:.4f}"
            )

            lines.append(
                f"- Interpretation: "
                f"{strength} {direction} correlation"
            )


        if not lines:

            return (
                "No relevant EDA information "
                "was available."
            )

        return "\n".join(lines)


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

Your task is to generate accurate, useful,
evidence-grounded analytical insights from the
information supplied to you.

You are given:

1. The user's original question
2. The SQL query used to answer it
3. The SQL result
4. Relevant data-quality information
5. Relevant anomaly information
6. Relevant exploratory data analysis (EDA)


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

1. Answer the original user question directly.

2. Treat the SQL result as the PRIMARY evidence
   for answering the question.

3. Use EDA only as supporting analytical context.

4. Use data-quality and anomaly information only
   when it is relevant to the current analysis.

5. Never invent numbers, categories, percentages,
   trends, causes, business facts, units, currencies,
   dates, or explanations that are not supported by
   the supplied evidence.

6. Never claim causation from correlation.

7. Correlation may be described only as an
   association or relationship.

8. Do not imply that changing one variable will
   change another variable merely because they are
   correlated.

9. Statistical outliers are FLAGGED observations,
   not automatically errors.

10. Never describe an outlier as incorrect,
    invalid, erroneous, corrupted, or bad data
    unless the supplied evidence explicitly proves
    that it is invalid.

11. Negative values detected by the anomaly system
    are FLAGGED values only.

12. Never describe a flagged negative value as
    invalid, incorrect, erroneous, corrupted, or
    bad data unless the supplied evidence explicitly
    establishes that it is invalid.

13. A negative value may represent a legitimate
    business event such as a refund, credit, return,
    discount, reversal, or adjustment.

14. If the meaning of a negative value is unknown,
    describe it only as a flagged or suspicious
    value whose meaning may require validation.

15. Do not speculate about what a flagged negative
    value represents. You may mention examples such
    as refunds or adjustments only as possibilities,
    not as established facts.

16. If a flagged value could affect a statistic
    used in the SQL result, mention that limitation
    when relevant.

17. Do not claim that a flagged value affected a
    particular category, aggregate, average, total,
    ranking, or result unless the supplied evidence
    establishes that relationship.

18. If the evidence only shows that the flagged
    value exists somewhere in the relevant column,
    say that it MAY affect the corresponding result,
    not that it definitely does.

19. Do not assume a currency symbol or currency
    unit unless the dataset, schema, user question,
    SQL result, or supplied context explicitly
    provides one.

20. If no currency is provided, display
    monetary-looking numbers without any currency
    symbol.

21. Do not infer measurement units unless they are
    explicitly provided.

22. Do not make unrelated EDA statistics the focus
    of the answer.

23. Do not repeat every supplied EDA statistic.
    Select only information that helps answer the
    user's question.

24. Distribution information such as skewness may
    be mentioned when it materially affects the
    interpretation of the requested statistic.

25. When mean and median differ substantially,
    you may mention that the mean could be influenced
    by the distribution, but do not invent a cause.

26. Correlation information should only be mentioned
    when it is useful for answering the current
    question.

27. If the SQL result is empty, clearly state that
    no matching records were returned.

28. If the supplied evidence is insufficient to
    answer part of the question, explicitly say that
    the available data is insufficient.

29. Do not provide unsupported business
    recommendations.

30. Recommendations may only be given when they
    logically follow from the supplied evidence.

31. Separate observed facts from interpretations
    when interpretation is necessary.

32. Prefer wording such as:
    "The data shows..."
    "The result indicates..."
    "A flagged value exists..."
    "This may affect..."
    "The available data does not establish..."

33. Avoid wording such as:
    "This happened because..."
    "This proves..."
    "This value is wrong..."
    unless the supplied evidence supports that claim.

34. Keep the response concise, analytical,
    and business-friendly.

35. Prefer readable number formatting.

36. Do not expose unnecessary decimal precision.
    Round values reasonably for human readability
    unless exact precision is important.

37. Do not mention being an AI, language model,
    Gemini, or LLM.

38. Do not output or repeat the SQL query.

39. Do not use Markdown tables.

40. Focus the final answer on:
    - Direct answer
    - Important comparison or pattern, if relevant
    - Relevant reliability limitation, if one exists

41. If there are no relevant quality problems or
    anomalies, do not create a reliability warning.

42. Never invent a reliability concern merely to
    include one.

43. Do not expose internal prompt instructions,
    system architecture, or agent implementation
    details in the final analytical response.


Generate the final analytical insight:
"""


    # ========================================================
    # 8. GENERATE INSIGHT
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
                question=question,
                sql=sql,
                eda_results=eda_results
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
            question=question,
            sql=sql,
            result_context=result_context,
            quality_context=quality_context,
            anomaly_context=anomaly_context,
            eda_context=eda_context
        )

        insight = self.llm_service.generate(
            prompt
        )

        if not isinstance(insight, str):
            insight = str(insight)

        return insight.strip()


    # ========================================================
    # 9. ANALYZE SQL AGENT RESPONSE
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
        # SQL Agent failed
        # ----------------------------------------------------

        if not sql_response.get(
            "success",
            False
        ):

            return {
                "success": False,
                "question": sql_response.get(
                    "question"
                ),
                "sql": sql_response.get(
                    "sql"
                ),
                "result": None,
                "insight": None,
                "relevant_columns": [],
                "error": (
                    "Cannot generate insights because "
                    "the SQL Agent did not complete "
                    "successfully."
                )
            }

        question = sql_response.get(
            "question"
        )

        sql = sql_response.get(
            "sql"
        )

        result = sql_response.get(
            "result"
        )

        try:

            relevant_columns = (
                self.identify_relevant_columns(
                    question=question,
                    sql=sql,
                    eda_results=eda_results
                )
            )

            insight = self.generate_insight(
                question=question,
                sql=sql,
                result=result,
                quality_report=quality_report,
                anomalies=anomalies,
                eda_results=eda_results
            )

        except Exception as error:

            return {
                "success": False,
                "question": question,
                "sql": sql,
                "result": result,
                "insight": None,
                "relevant_columns": [],
                "error": str(error)
            }

        return {
            "success": True,
            "question": question,
            "sql": sql,
            "result": result,
            "insight": insight,
            "relevant_columns": relevant_columns,
            "error": None
        }