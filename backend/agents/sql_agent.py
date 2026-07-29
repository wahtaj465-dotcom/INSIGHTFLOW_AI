import re


class SQLAgent:
    """
    Hybrid SQL Agent for InsightFlow.

    Primary mode:
        Natural language
            -> Gemini
            -> SQL
            -> DuckDB

    Fallback mode:
        Natural language
            -> deterministic local parser
            -> SQL
            -> DuckDB

    The fallback allows basic analytics to continue when
    Gemini quota/rate limits are reached.
    """


    def __init__(
        self,
        llm_service,
        sql_engine,
        table_name="sales",
        max_retries=2
    ):

        self.llm_service = llm_service
        self.sql_engine = sql_engine
        self.table_name = table_name
        self.max_retries = max_retries


    # ========================================================
    # GET SCHEMA DATAFRAME
    # ========================================================

    def get_schema_df(
        self
    ):

        schema_df = (
            self.sql_engine.get_table_schema(
                self.table_name
            )
        )


        if (
            schema_df is None
            or schema_df.empty
        ):

            raise ValueError(
                "Could not retrieve database schema."
            )


        return schema_df


    # ========================================================
    # GET COLUMN NAMES
    # ========================================================

    def get_columns(
        self
    ):

        schema_df = (
            self.get_schema_df()
        )


        return (
            schema_df[
                "column_name"
            ]
            .astype(str)
            .tolist()
        )


    # ========================================================
    # GET SCHEMA CONTEXT
    # ========================================================

    def get_schema_context(
        self
    ):

        schema_df = (
            self.get_schema_df()
        )


        schema_lines = []


        for _, row in schema_df.iterrows():

            column_name = str(
                row["column_name"]
            )

            column_type = str(
                row["column_type"]
            )


            schema_lines.append(
                f"- {column_name}: "
                f"{column_type}"
            )


            sample_query = f'''
                SELECT DISTINCT "{column_name}"
                FROM "{self.table_name}"
                WHERE "{column_name}" IS NOT NULL
                LIMIT 5
            '''


            try:

                sample_df = (
                    self.sql_engine.execute_query(
                        sample_query
                    )
                )


                if (
                    sample_df is not None
                    and not sample_df.empty
                ):

                    sample_values = (
                        sample_df[
                            column_name
                        ]
                        .astype(str)
                        .tolist()
                    )


                    schema_lines.append(
                        "  Sample values: "
                        + ", ".join(
                            sample_values
                        )
                    )


            except Exception:

                pass


        return "\n".join(
            schema_lines
        )


    # ========================================================
    # BUILD LLM PROMPT
    # ========================================================

    def build_prompt(
        self,
        question,
        schema_context
    ):

        return f"""
You are the SQL generation agent inside InsightFlow,
an automated data analytics platform.

Convert the user's analytics request into ONE valid
DuckDB SQL query.


DATABASE

Table:
{self.table_name}

Schema:

{schema_context}


RULES

1. Generate DuckDB-compatible SQL.

2. Use ONLY the table and columns provided above.

3. Never invent tables or columns.

4. Only SELECT or WITH queries are allowed.

5. Never generate INSERT, UPDATE, DELETE, DROP,
   ALTER, CREATE, REPLACE, TRUNCATE, ATTACH,
   DETACH, COPY, EXPORT, IMPORT, INSTALL, LOAD,
   CALL or PRAGMA.

6. Use aggregation only when the question actually
   requires aggregation.

7. Use GROUP BY for category-level aggregations.

8. Use ORDER BY for ranking.

9. Use LIMIT when the user explicitly asks for a
   number of rows/results.

10. IMPORTANT FOR VISUALIZATIONS:

    If the user asks for a distribution chart,
    histogram, box plot, violin plot, scatter plot,
    grouped plot, correlation visualization or other
    chart that requires raw observations, return the
    relevant raw columns rather than unnecessarily
    aggregating them.

    Example:

    Request:
    Box plot of distance_from_home_m by
    behavior_profile colored by risk_level

    Appropriate query:

    SELECT
        behavior_profile,
        risk_level,
        distance_from_home_m
    FROM {self.table_name}
    WHERE distance_from_home_m IS NOT NULL

11. Return ONLY SQL.

12. No Markdown.

13. No code fences.

14. No explanation.


USER QUESTION

{question}


SQL:
"""


    # ========================================================
    # BUILD CORRECTION PROMPT
    # ========================================================

    def build_correction_prompt(
        self,
        question,
        schema_context,
        failed_sql,
        error_message
    ):

        return f"""
Correct the failed DuckDB query.

DATABASE

Table:
{self.table_name}

Schema:

{schema_context}


USER QUESTION

{question}


FAILED SQL

{failed_sql}


ERROR

{error_message}


RULES

Use only the provided table and columns.

Generate DuckDB SQL.

Only SELECT or WITH.

Do not perform modifications or administrative
operations.

Correct the actual cause of the error.

For visualization requests requiring distributions,
box plots, violin plots or scatter plots, preserve
the raw observations needed by the visualization.

Return ONLY SQL.

No Markdown.

No explanation.


CORRECTED SQL:
"""


    # ========================================================
    # CLEAN SQL
    # ========================================================

    @staticmethod
    def clean_generated_sql(
        generated_sql
    ):

        if not isinstance(
            generated_sql,
            str
        ):

            raise TypeError(
                "Generated SQL must be a string."
            )


        sql = generated_sql.strip()


        sql = re.sub(
            r"^```(?:sql)?\s*",
            "",
            sql,
            flags=re.IGNORECASE
        )


        sql = re.sub(
            r"\s*```$",
            "",
            sql
        )


        sql = re.sub(
            r"^sql\s*:\s*",
            "",
            sql,
            flags=re.IGNORECASE
        )


        return sql.strip()


    # ========================================================
    # GENERATE USING LLM
    # ========================================================

    def generate_sql(
        self,
        question,
        schema_context=None
    ):

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "Question must be a string."
            )


        question = question.strip()


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        if schema_context is None:

            schema_context = (
                self.get_schema_context()
            )


        prompt = (
            self.build_prompt(
                question,
                schema_context
            )
        )


        generated_sql = (
            self.llm_service.generate(
                prompt
            )
        )


        return (
            self.clean_generated_sql(
                generated_sql
            )
        )


    # ========================================================
    # CORRECT USING LLM
    # ========================================================

    def correct_sql(
        self,
        question,
        schema_context,
        failed_sql,
        error_message
    ):

        prompt = (
            self.build_correction_prompt(
                question,
                schema_context,
                failed_sql,
                error_message
            )
        )


        corrected_sql = (
            self.llm_service.generate(
                prompt
            )
        )


        return (
            self.clean_generated_sql(
                corrected_sql
            )
        )


    # ========================================================
    # FIND COLUMNS MENTIONED IN QUESTION
    # ========================================================

    def find_mentioned_columns(
        self,
        question
    ):

        question_lower = (
            question.lower()
        )


        columns = (
            self.get_columns()
        )


        matches = []


        # Exact database column names first.

        for column in columns:

            if (
                column.lower()
                in question_lower
            ):

                matches.append(
                    column
                )


        # Also support human-friendly versions:
        #
        # distance from home
        # ->
        # distance_from_home_m

        for column in columns:

            normalized_column = (
                column.lower()
                .replace("_", " ")
            )


            if (
                normalized_column
                in question_lower
                and column not in matches
            ):

                matches.append(
                    column
                )


        return matches


    # ========================================================
    # QUOTE COLUMN
    # ========================================================

    @staticmethod
    def quote_column(
        column
    ):

        return (
            '"'
            + column.replace(
                '"',
                '""'
            )
            + '"'
        )


    # ========================================================
    # LOCAL SQL FALLBACK
    # ========================================================

    def generate_fallback_sql(
        self,
        question
    ):
        """
        Deterministic SQL generator used when Gemini is
        unavailable.

        This is NOT intended to replace the LLM.

        It handles common analytical requests so development
        can continue during Gemini quota exhaustion.
        """

        question_lower = (
            question.lower()
        )


        columns = (
            self.get_columns()
        )


        mentioned = (
            self.find_mentioned_columns(
                question
            )
        )


        table = (
            f'"{self.table_name}"'
        )


        # ====================================================
        # 1. VISUALIZATION / DISTRIBUTION REQUEST
        # ====================================================

        visualization_words = [

            "chart",
            "plot",
            "graph",
            "visualize",
            "visualization",

            "box plot",
            "boxplot",

            "violin",

            "scatter",

            "distribution",

            "histogram"
        ]


        if any(
            word in question_lower
            for word in visualization_words
        ):

            if mentioned:

                selected = ", ".join(
                    self.quote_column(
                        column
                    )
                    for column in mentioned
                )


                return (
                    f"SELECT {selected} "
                    f"FROM {table}"
                )


        # ====================================================
        # 2. COUNT ROWS
        # ====================================================

        count_phrases = [

            "how many rows",
            "number of rows",
            "total rows",
            "row count",
            "total records",
            "number of records"
        ]


        if any(
            phrase in question_lower
            for phrase in count_phrases
        ):

            return (
                f"SELECT COUNT(*) AS total_rows "
                f"FROM {table}"
            )


        # ====================================================
        # 3. DISTINCT VALUES
        # ====================================================

        if (
            "distinct" in question_lower
            or "unique values" in question_lower
            or "unique categories" in question_lower
        ):

            if mentioned:

                column = (
                    mentioned[0]
                )


                quoted = (
                    self.quote_column(
                        column
                    )
                )


                return (
                    f"SELECT DISTINCT {quoted} "
                    f"FROM {table} "
                    f"ORDER BY {quoted}"
                )


        # ====================================================
        # 4. COUNT BY CATEGORY
        # ====================================================

        if (
            "count" in question_lower
            and mentioned
        ):

            category = (
                mentioned[0]
            )


            quoted = (
                self.quote_column(
                    category
                )
            )


            return (
                f"SELECT {quoted}, "
                f"COUNT(*) AS count "
                f"FROM {table} "
                f"GROUP BY {quoted} "
                f"ORDER BY count DESC"
            )


        # ====================================================
        # 5. AVERAGE
        # ====================================================

        average_words = [

            "average",
            "avg",
            "mean"
        ]


        if any(
            word in question_lower
            for word in average_words
        ):

            if mentioned:

                # If multiple columns were mentioned,
                # assume the final column is the metric
                # and the first is the grouping column.

                if len(
                    mentioned
                ) >= 2:

                    group_column = (
                        mentioned[0]
                    )

                    metric_column = (
                        mentioned[-1]
                    )


                    group_q = (
                        self.quote_column(
                            group_column
                        )
                    )


                    metric_q = (
                        self.quote_column(
                            metric_column
                        )
                    )


                    return (
                        f"SELECT {group_q}, "
                        f"AVG({metric_q}) "
                        f"AS average_{metric_column} "
                        f"FROM {table} "
                        f"GROUP BY {group_q} "
                        f"ORDER BY average_{metric_column} "
                        f"DESC"
                    )


                metric_column = (
                    mentioned[0]
                )


                metric_q = (
                    self.quote_column(
                        metric_column
                    )
                )


                return (
                    f"SELECT AVG({metric_q}) "
                    f"AS average_{metric_column} "
                    f"FROM {table}"
                )


        # ====================================================
        # 6. MAXIMUM
        # ====================================================

        maximum_words = [

            "maximum",
            "highest",
            "max"
        ]


        if any(
            word in question_lower
            for word in maximum_words
        ):

            if mentioned:

                column = (
                    mentioned[-1]
                )


                quoted = (
                    self.quote_column(
                        column
                    )
                )


                return (
                    f"SELECT MAX({quoted}) "
                    f"AS maximum_{column} "
                    f"FROM {table}"
                )


        # ====================================================
        # 7. MINIMUM
        # ====================================================

        minimum_words = [

            "minimum",
            "lowest",
            "min"
        ]


        if any(
            word in question_lower
            for word in minimum_words
        ):

            if mentioned:

                column = (
                    mentioned[-1]
                )


                quoted = (
                    self.quote_column(
                        column
                    )
                )


                return (
                    f"SELECT MIN({quoted}) "
                    f"AS minimum_{column} "
                    f"FROM {table}"
                )


        # ====================================================
        # 8. SHOW SPECIFIC COLUMNS
        # ====================================================

        if mentioned:

            selected = ", ".join(
                self.quote_column(
                    column
                )
                for column in mentioned
            )


            return (
                f"SELECT {selected} "
                f"FROM {table} "
                f"LIMIT 5000"
            )


        # ====================================================
        # 9. GENERIC PREVIEW
        # ====================================================

        return (
            f"SELECT * "
            f"FROM {table} "
            f"LIMIT 100"
        )


    # ========================================================
    # EXECUTE SQL
    # ========================================================

    def execute_sql(
        self,
        sql,
        question,
        source,
        attempts
    ):

        is_valid, validation_error = (
            self.sql_engine.validate_query(
                sql
            )
        )


        if not is_valid:

            attempts.append({

                "attempt":
                    len(attempts) + 1,

                "sql":
                    sql,

                "stage":
                    "validation",

                "source":
                    source,

                "error":
                    validation_error
            })


            return (
                False,
                None,
                validation_error
            )


        try:

            result = (
                self.sql_engine.execute_query(
                    sql
                )
            )


        except Exception as error:

            error_message = (
                str(error)
            )


            attempts.append({

                "attempt":
                    len(attempts) + 1,

                "sql":
                    sql,

                "stage":
                    "execution",

                "source":
                    source,

                "error":
                    error_message
            })


            return (
                False,
                None,
                error_message
            )


        if result is None:

            error_message = (
                "SQL engine returned no result."
            )


            attempts.append({

                "attempt":
                    len(attempts) + 1,

                "sql":
                    sql,

                "stage":
                    "execution",

                "source":
                    source,

                "error":
                    error_message
            })


            return (
                False,
                None,
                error_message
            )


        attempts.append({

            "attempt":
                len(attempts) + 1,

            "sql":
                sql,

            "stage":
                "success",

            "source":
                source,

            "error":
                None
        })


        return (
            True,
            result,
            None
        )


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question
    ):

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "Question must be a string."
            )


        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        attempts = []


        schema_context = (
            self.get_schema_context()
        )


        # ====================================================
        # TRY GEMINI FIRST
        # ====================================================

        llm_error = None


        try:

            sql = (
                self.generate_sql(
                    question,
                    schema_context
                )
            )


            success, result, error = (
                self.execute_sql(
                    sql=sql,
                    question=question,
                    source="llm",
                    attempts=attempts
                )
            )


            if success:

                return {

                    "success":
                        True,

                    "question":
                        question,

                    "sql":
                        sql,

                    "generated_sql":
                        sql,

                    "sql_source":
                        "llm",

                    "fallback_used":
                        False,

                    "llm_error":
                        None,

                    "error":
                        None,

                    "result":
                        result,

                    "attempts":
                        attempts
                }


            # =================================================
            # LLM CORRECTION
            # =================================================

            failed_sql = sql
            correction_error = error


            for _ in range(
                self.max_retries
            ):

                try:

                    corrected_sql = (
                        self.correct_sql(
                            question=question,
                            schema_context=schema_context,
                            failed_sql=failed_sql,
                            error_message=correction_error
                        )
                    )


                except Exception as error:

                    llm_error = str(
                        error
                    )

                    break


                success, result, error = (
                    self.execute_sql(
                        sql=corrected_sql,
                        question=question,
                        source="llm_correction",
                        attempts=attempts
                    )
                )


                if success:

                    return {

                        "success":
                            True,

                        "question":
                            question,

                        "sql":
                            corrected_sql,

                        "generated_sql":
                            corrected_sql,

                        "sql_source":
                            "llm",

                        "fallback_used":
                            False,

                        "llm_error":
                            None,

                        "error":
                            None,

                        "result":
                            result,

                        "attempts":
                            attempts
                    }


                failed_sql = (
                    corrected_sql
                )

                correction_error = (
                    error
                )


        except Exception as error:

            llm_error = (
                str(error)
            )


        # ====================================================
        # LOCAL FALLBACK
        # ====================================================

        try:

            fallback_sql = (
                self.generate_fallback_sql(
                    question
                )
            )


            success, result, fallback_error = (
                self.execute_sql(
                    sql=fallback_sql,
                    question=question,
                    source="local_fallback",
                    attempts=attempts
                )
            )


            if success:

                return {

                    "success":
                        True,

                    "question":
                        question,

                    "sql":
                        fallback_sql,

                    "generated_sql":
                        fallback_sql,

                    "sql_source":
                        "local_fallback",

                    "fallback_used":
                        True,

                    "llm_error":
                        llm_error,

                    "error":
                        None,

                    "result":
                        result,

                    "attempts":
                        attempts
                }


            return {

                "success":
                    False,

                "question":
                    question,

                "sql":
                    fallback_sql,

                "generated_sql":
                    fallback_sql,

                "sql_source":
                    "local_fallback",

                "fallback_used":
                    True,

                "llm_error":
                    llm_error,

                "error":
                    fallback_error,

                "result":
                    None,

                "attempts":
                    attempts
            }


        except Exception as fallback_error:

            return {

                "success":
                    False,

                "question":
                    question,

                "sql":
                    None,

                "generated_sql":
                    None,

                "sql_source":
                    None,

                "fallback_used":
                    True,

                "llm_error":
                    llm_error,

                "error":
                    str(
                        fallback_error
                    ),

                "result":
                    None,

                "attempts":
                    attempts
            }