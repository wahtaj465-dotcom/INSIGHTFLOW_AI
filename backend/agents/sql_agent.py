import re


class SQLAgent:
    """
    LLM-powered SQL Agent for InsightFlow.

    Responsibilities:
    1. Receive a natural-language question
    2. Inspect the database schema
    3. Retrieve useful sample values
    4. Ask the LLM to generate DuckDB SQL
    5. Clean the generated SQL
    6. Validate the SQL
    7. Execute the SQL
    8. Retry with LLM correction if validation/execution fails
    9. Return SQL, result, and attempt history
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
    # GET DATABASE SCHEMA
    # ========================================================

    def get_schema_context(self):
        """
        Retrieve the DuckDB table schema and create richer
        context for the LLM.

        Example:

        - region: VARCHAR
          Sample values: North, South, East, West

        - price: DOUBLE
          Sample values: 55000, 800, 1500
        """

        schema_df = self.sql_engine.get_table_schema(
            self.table_name
        )

        if schema_df is None or schema_df.empty:

            raise ValueError(
                "Could not retrieve database schema."
            )

        schema_lines = []

        for _, row in schema_df.iterrows():

            column_name = row["column_name"]
            column_type = row["column_type"]

            schema_lines.append(
                f"- {column_name}: {column_type}"
            )

            # ------------------------------------------------
            # Retrieve sample values
            # ------------------------------------------------

            sample_query = f"""
            SELECT DISTINCT "{column_name}"
            FROM "{self.table_name}"
            WHERE "{column_name}" IS NOT NULL
            LIMIT 5
            """

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
                        sample_df[column_name]
                        .astype(str)
                        .tolist()
                    )

                    sample_text = ", ".join(
                        sample_values
                    )

                    schema_lines.append(
                        f"  Sample values: {sample_text}"
                    )

            except Exception:

                # Sample values are helpful,
                # but not required for SQL generation.
                pass

        return "\n".join(schema_lines)


    # ========================================================
    # BUILD SQL GENERATION PROMPT
    # ========================================================

    def build_prompt(
        self,
        question,
        schema_context
    ):
        """
        Build the initial prompt used to generate SQL.
        """

        prompt = f"""
You are the SQL generation agent inside an automated
data analytics platform called InsightFlow.

Your job is to convert the user's analytics question
into one valid DuckDB SQL query.


DATABASE INFORMATION

Table name:
{self.table_name}

Schema and sample values:

{schema_context}


SQL GENERATION RULES

1. Generate DuckDB-compatible SQL.

2. Use ONLY the table and columns provided in the
   database information.

3. Never invent columns or tables.

4. Generate read-only analytical queries.

5. Only SELECT or WITH queries are allowed.

6. Never generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   REPLACE
   TRUNCATE
   ATTACH
   DETACH
   COPY
   EXPORT
   IMPORT
   INSTALL
   LOAD
   CALL
   PRAGMA

7. Use appropriate SQL aggregation functions when
   required, such as:
   COUNT
   SUM
   AVG
   MIN
   MAX

8. Use GROUP BY when aggregation is required across
   categories.

9. Use ORDER BY when ranking or comparison is
   requested.

10. Use LIMIT when the user explicitly asks for a
    specific number of results such as:
    top 5
    bottom 10
    first 3

11. For questions asking for the highest, lowest,
    maximum, or minimum category, preserve ties when
    practical instead of arbitrarily returning one row.

12. Use sample values only to understand the data.
    Do not assume values that are not shown.

13. Do not assume business meaning beyond the
    available column names, types, and values.

14. If the question can be answered directly from
    the available data, generate the simplest correct
    query.

15. Return ONLY SQL.

16. Do not return Markdown.

17. Do not use ```sql code fences.

18. Do not explain your answer.


USER QUESTION

{question}


SQL:
"""

        return prompt


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
        """
        Build a prompt asking the LLM to repair a failed SQL
        query.

        The LLM receives:
        - original question
        - database schema
        - failed SQL
        - validation/execution error
        """

        prompt = f"""
You are correcting a failed DuckDB SQL query inside
an automated analytics platform.

The previous SQL query failed.

Analyze the error and generate a corrected query.


DATABASE INFORMATION

Table name:
{self.table_name}

Schema and sample values:

{schema_context}


ORIGINAL USER QUESTION

{question}


FAILED SQL

{failed_sql}


ERROR

{error_message}


CORRECTION RULES

1. Fix the SQL so it correctly answers the original
   user question.

2. Generate DuckDB-compatible SQL.

3. Use ONLY the table and columns provided above.

4. Never invent columns.

5. Never invent tables.

6. Only SELECT or WITH queries are allowed.

7. Never generate data-modifying or administrative SQL.

8. Do not repeat the same mistake that caused the
   previous query to fail.

9. Preserve ties for highest/lowest questions when
   practical.

10. Return ONLY the corrected SQL.

11. Do not use Markdown.

12. Do not use ```sql code fences.

13. Do not explain the correction.


CORRECTED SQL:
"""

        return prompt


    # ========================================================
    # CLEAN GENERATED SQL
    # ========================================================

    def clean_generated_sql(
        self,
        generated_sql
    ):
        """
        Remove common formatting added by an LLM.

        Example:

        ```sql
        SELECT * FROM sales;
        ```

        becomes:

        SELECT * FROM sales;
        """

        if not isinstance(
            generated_sql,
            str
        ):

            raise TypeError(
                "Generated SQL must be a string."
            )

        sql = generated_sql.strip()

        # ----------------------------------------------------
        # Remove opening Markdown code fence
        # ----------------------------------------------------

        sql = re.sub(
            r"^```(?:sql)?\s*",
            "",
            sql,
            flags=re.IGNORECASE
        )

        # ----------------------------------------------------
        # Remove closing Markdown code fence
        # ----------------------------------------------------

        sql = re.sub(
            r"\s*```$",
            "",
            sql
        )

        # ----------------------------------------------------
        # Remove accidental SQL: prefix
        # ----------------------------------------------------

        sql = re.sub(
            r"^sql\s*:\s*",
            "",
            sql,
            flags=re.IGNORECASE
        )

        return sql.strip()


    # ========================================================
    # GENERATE INITIAL SQL
    # ========================================================

    def generate_sql(
        self,
        question,
        schema_context=None
    ):
        """
        Generate the initial SQL query from a natural-language
        question.
        """

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "Question must be a string."
            )

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        if schema_context is None:

            schema_context = (
                self.get_schema_context()
            )

        prompt = self.build_prompt(
            question,
            schema_context
        )

        generated_sql = (
            self.llm_service.generate(
                prompt
            )
        )

        return self.clean_generated_sql(
            generated_sql
        )


    # ========================================================
    # CORRECT FAILED SQL
    # ========================================================

    def correct_sql(
        self,
        question,
        schema_context,
        failed_sql,
        error_message
    ):
        """
        Ask Gemini to correct SQL that failed validation
        or execution.
        """

        correction_prompt = (
            self.build_correction_prompt(
                question=question,
                schema_context=schema_context,
                failed_sql=failed_sql,
                error_message=error_message
            )
        )

        corrected_sql = (
            self.llm_service.generate(
                correction_prompt
            )
        )

        return self.clean_generated_sql(
            corrected_sql
        )


    # ========================================================
    # ASK SQL AGENT
    # ========================================================

    def ask(self, question):
        """
        Complete SQL Agent workflow.

        Question
            ↓
        Schema Context
            ↓
        Generate SQL
            ↓
        Validate
            ↓
        Execute
            ↓
        Success

        If failure:

        Error
            ↓
        LLM Correction
            ↓
        Validate Again
            ↓
        Execute Again
        """

        # ----------------------------------------------------
        # Validate user question
        # ----------------------------------------------------

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "Question must be a string."
            )

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        # ----------------------------------------------------
        # Get schema once
        # ----------------------------------------------------

        schema_context = (
            self.get_schema_context()
        )

        # ----------------------------------------------------
        # Generate initial SQL
        # ----------------------------------------------------

        sql = self.generate_sql(
            question,
            schema_context
        )

        # Keep history for debugging and future LangGraph use
        attempts = []

        # Initial attempt + retry attempts
        total_attempts = (
            self.max_retries + 1
        )

        for attempt_number in range(
            1,
            total_attempts + 1
        ):

            # =================================================
            # VALIDATE SQL
            # =================================================

            is_valid, validation_error = (
                self.sql_engine.validate_query(
                    sql
                )
            )

            if not is_valid:

                attempts.append(
                    {
                        "attempt": attempt_number,
                        "sql": sql,
                        "stage": "validation",
                        "error": validation_error
                    }
                )

                # No retries remaining
                if attempt_number >= total_attempts:

                    return {
                        "success": False,
                        "question": question,
                        "sql": sql,
                        "error": validation_error,
                        "result": None,
                        "attempts": attempts
                    }

                # Ask LLM to repair query
                sql = self.correct_sql(
                    question=question,
                    schema_context=schema_context,
                    failed_sql=sql,
                    error_message=validation_error
                )

                continue

            # =================================================
            # EXECUTE SQL
            # =================================================

            try:

                result = (
                    self.sql_engine.execute_query(
                        sql
                    )
                )

            except Exception as error:

                execution_error = str(
                    error
                )

                attempts.append(
                    {
                        "attempt": attempt_number,
                        "sql": sql,
                        "stage": "execution",
                        "error": execution_error
                    }
                )

                if attempt_number >= total_attempts:

                    return {
                        "success": False,
                        "question": question,
                        "sql": sql,
                        "error": execution_error,
                        "result": None,
                        "attempts": attempts
                    }

                sql = self.correct_sql(
                    question=question,
                    schema_context=schema_context,
                    failed_sql=sql,
                    error_message=execution_error
                )

                continue

            # =================================================
            # HANDLE ENGINE RETURNING NONE
            # =================================================

            if result is None:

                execution_error = (
                    "SQL execution failed. "
                    "The SQL engine returned no result."
                )

                attempts.append(
                    {
                        "attempt": attempt_number,
                        "sql": sql,
                        "stage": "execution",
                        "error": execution_error
                    }
                )

                if attempt_number >= total_attempts:

                    return {
                        "success": False,
                        "question": question,
                        "sql": sql,
                        "error": execution_error,
                        "result": None,
                        "attempts": attempts
                    }

                sql = self.correct_sql(
                    question=question,
                    schema_context=schema_context,
                    failed_sql=sql,
                    error_message=execution_error
                )

                continue

            # =================================================
            # SUCCESS
            # =================================================

            attempts.append(
                {
                    "attempt": attempt_number,
                    "sql": sql,
                    "stage": "success",
                    "error": None
                }
            )

            return {
                "success": True,
                "question": question,
                "sql": sql,
                "error": None,
                "result": result,
                "attempts": attempts
            }

        # ----------------------------------------------------
        # Defensive fallback
        # ----------------------------------------------------

        return {
            "success": False,
            "question": question,
            "sql": sql,
            "error": (
                "SQL Agent failed after "
                "all retry attempts."
            ),
            "result": None,
            "attempts": attempts
        }