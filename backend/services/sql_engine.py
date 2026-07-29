import duckdb
import pandas as pd
import re


class SQLEngine:
    """
    DuckDB-based SQL engine for InsightFlow.

    Responsibilities:
    1. Create a DuckDB connection
    2. Register Pandas DataFrames
    3. Validate SQL queries
    4. Execute safe read-only SQL
    5. Return results as Pandas DataFrames
    6. Propagate execution errors to calling agents
    """

    def __init__(self):
        """
        Create an in-memory DuckDB database.
        """

        self.connection = duckdb.connect(
            database=":memory:"
        )

        self.registered_tables = set()


    # ========================================================
    # 1. REGISTER DATAFRAME
    # ========================================================

    def register_dataframe(
        self,
        df,
        table_name="dataset"
    ):
        """
        Register a Pandas DataFrame with DuckDB.
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input must be a Pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "Cannot register an empty DataFrame."
            )

        # ----------------------------------------------------
        # Validate table name
        # ----------------------------------------------------

        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*",
            table_name
        ):

            raise ValueError(
                "Invalid table name."
            )

        self.connection.register(
            table_name,
            df
        )

        self.registered_tables.add(
            table_name.lower()
        )

        print(
            f"\nDataset registered in DuckDB "
            f"as table: {table_name}"
        )


    # ========================================================
    # 2. VALIDATE QUERY
    # ========================================================

    def validate_query(self, query):
        """
        Validate SQL before execution.

        Safety rules:

        1. Query must be a string
        2. Query cannot be empty
        3. Only SELECT or WITH queries are allowed
        4. Multiple SQL statements are rejected
        5. Dangerous SQL operations are rejected

        Returns:

            (True, None)

        or:

            (False, reason)
        """

        # ----------------------------------------------------
        # Query must be a string
        # ----------------------------------------------------

        if not isinstance(query, str):

            return (
                False,
                "SQL query must be a string."
            )

        cleaned_query = query.strip()

        # ----------------------------------------------------
        # Query cannot be empty
        # ----------------------------------------------------

        if not cleaned_query:

            return (
                False,
                "SQL query cannot be empty."
            )

        # ----------------------------------------------------
        # Remove optional trailing semicolon
        # ----------------------------------------------------

        if cleaned_query.endswith(";"):

            query_without_semicolon = (
                cleaned_query[:-1].strip()
            )

        else:

            query_without_semicolon = (
                cleaned_query
            )

        # ----------------------------------------------------
        # Reject multiple SQL statements
        # ----------------------------------------------------

        if ";" in query_without_semicolon:

            return (
                False,
                "Multiple SQL statements are not allowed."
            )

        # ----------------------------------------------------
        # Normalize query
        # ----------------------------------------------------

        normalized_query = (
            query_without_semicolon.lower()
        )

        # ----------------------------------------------------
        # Only SELECT / WITH allowed
        # ----------------------------------------------------

        allowed_query = re.match(
            r"^(select|with)\s",
            normalized_query
        )

        if not allowed_query:

            return (
                False,
                "Only SELECT and WITH queries are allowed."
            )

        # ----------------------------------------------------
        # Block dangerous operations
        # ----------------------------------------------------

        forbidden_keywords = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "replace",
            "truncate",
            "attach",
            "detach",
            "copy",
            "export",
            "import",
            "install",
            "load",
            "call",
            "pragma"
        ]

        for keyword in forbidden_keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                normalized_query
            ):

                return (
                    False,
                    f"Forbidden SQL operation detected: "
                    f"{keyword.upper()}"
                )

        return True, None


    # ========================================================
    # 3. EXECUTE QUERY
    # ========================================================

    def execute_query(self, query):
        """
        Validate and execute a read-only SQL query.

        Returns:
            Pandas DataFrame when successful.

        Raises:
            ValueError:
                If the SQL fails our safety validation.

            RuntimeError:
                If DuckDB cannot execute the query.

        Important:
            Execution errors are intentionally propagated
            so AI agents can inspect the real database error
            and attempt to repair the SQL.
        """

        # ----------------------------------------------------
        # Safety validation
        # ----------------------------------------------------

        is_valid, error_message = (
            self.validate_query(query)
        )

        if not is_valid:

            raise ValueError(
                f"SQL query rejected: "
                f"{error_message}"
            )

        # ----------------------------------------------------
        # Execute query
        # ----------------------------------------------------

        try:

            result = (
                self.connection
                .execute(query)
                .fetchdf()
            )

            return result

        except Exception as error:

            # Preserve DuckDB's actual error message
            # for SQLAgent correction/retry.

            raise RuntimeError(
                f"DuckDB execution error: {error}"
            ) from error


    # ========================================================
    # 4. GET TABLE SCHEMA
    # ========================================================

    def get_table_schema(
        self,
        table_name="dataset"
    ):
        """
        Return information about columns in a
        registered DuckDB table.
        """

        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*",
            table_name
        ):

            raise ValueError(
                "Invalid table name."
            )

        # ----------------------------------------------------
        # Make sure the requested table belongs to our engine
        # ----------------------------------------------------

        if (
            table_name.lower()
            not in self.registered_tables
        ):

            raise ValueError(
                f"Table '{table_name}' "
                f"is not registered."
            )

        try:

            result = (
                self.connection
                .execute(
                    f'DESCRIBE "{table_name}"'
                )
                .fetchdf()
            )

            return result

        except Exception as error:

            raise RuntimeError(
                f"Could not retrieve schema for "
                f"table '{table_name}': {error}"
            ) from error


    # ========================================================
    # 5. CHECK REGISTERED TABLE
    # ========================================================

    def is_table_registered(
        self,
        table_name
    ):
        """
        Check whether a table has been registered
        with this SQL engine.
        """

        if not isinstance(
            table_name,
            str
        ):

            return False

        return (
            table_name.lower()
            in self.registered_tables
        )


    # ========================================================
    # 6. CLOSE CONNECTION
    # ========================================================

    def close(self):
        """
        Close the DuckDB connection.
        """

        if self.connection is not None:

            self.connection.close()

            self.connection = None