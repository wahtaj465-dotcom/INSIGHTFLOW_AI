import re

import duckdb
import pandas as pd


class SQLEngine:
    """
    DuckDB-based SQL engine for InsightFlow.

    Responsibilities:
    1. Create an in-memory DuckDB connection
    2. Register Pandas DataFrames
    3. Validate SQL queries
    4. Execute safe read-only SQL
    5. Return Pandas DataFrames
    6. Retrieve table schemas
    """


    def __init__(
        self
    ):

        self.connection = (
            duckdb.connect(
                database=":memory:"
            )
        )

        self.registered_tables = set()


    # ========================================================
    # VALIDATE IDENTIFIER
    # ========================================================

    @staticmethod
    def _validate_identifier(
        identifier
    ):

        if not isinstance(
            identifier,
            str
        ):

            raise TypeError(
                "SQL identifier must be a string."
            )


        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*",
            identifier
        ):

            raise ValueError(
                f"Invalid SQL identifier: "
                f"{identifier}"
            )


        return identifier


    # ========================================================
    # REGISTER DATAFRAME
    # ========================================================

    def register_dataframe(
        self,
        df,
        table_name="dataset"
    ):

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a Pandas DataFrame."
            )


        if df.empty:

            raise ValueError(
                "Cannot register an empty DataFrame."
            )


        self._validate_identifier(
            table_name
        )


        # Remove previous registration if the same table
        # name is being reused.

        if (
            table_name.lower()
            in self.registered_tables
        ):

            try:

                self.connection.unregister(
                    table_name
                )

            except Exception:

                pass


        self.connection.register(
            table_name,
            df
        )


        self.registered_tables.add(
            table_name.lower()
        )


        print(
            "\nDataset registered in DuckDB "
            f"as table: {table_name}"
        )


    # ========================================================
    # VALIDATE QUERY
    # ========================================================

    def validate_query(
        self,
        query
    ):

        if not isinstance(
            query,
            str
        ):

            return (
                False,
                "SQL query must be a string."
            )


        cleaned_query = (
            query.strip()
        )


        if not cleaned_query:

            return (
                False,
                "SQL query cannot be empty."
            )


        # ----------------------------------------------------
        # Remove trailing semicolon
        # ----------------------------------------------------

        if cleaned_query.endswith(
            ";"
        ):

            query_without_semicolon = (
                cleaned_query[:-1].strip()
            )

        else:

            query_without_semicolon = (
                cleaned_query
            )


        # ----------------------------------------------------
        # Multiple statements
        # ----------------------------------------------------

        if ";" in query_without_semicolon:

            return (
                False,
                "Multiple SQL statements are not allowed."
            )


        normalized_query = (
            query_without_semicolon.lower()
        )


        # ----------------------------------------------------
        # SELECT / WITH only
        # ----------------------------------------------------

        if not re.match(
            r"^(select|with)\b",
            normalized_query
        ):

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
                + re.escape(
                    keyword
                )
                + r"\b"
            )


            if re.search(
                pattern,
                normalized_query
            ):

                return (
                    False,
                    "Forbidden SQL operation detected: "
                    f"{keyword.upper()}"
                )


        return (
            True,
            None
        )


    # ========================================================
    # EXECUTE QUERY
    # ========================================================

    def execute_query(
        self,
        query
    ):

        if self.connection is None:

            raise RuntimeError(
                "DuckDB connection is closed."
            )


        is_valid, error_message = (
            self.validate_query(
                query
            )
        )


        if not is_valid:

            raise ValueError(
                "SQL query rejected: "
                f"{error_message}"
            )


        try:

            return (
                self.connection
                .execute(
                    query
                )
                .fetchdf()
            )


        except Exception as error:

            raise RuntimeError(
                "DuckDB execution error: "
                f"{error}"
            ) from error


    # ========================================================
    # GET TABLE SCHEMA
    # ========================================================

    def get_table_schema(
        self,
        table_name="dataset"
    ):

        if self.connection is None:

            raise RuntimeError(
                "DuckDB connection is closed."
            )


        self._validate_identifier(
            table_name
        )


        if (
            table_name.lower()
            not in self.registered_tables
        ):

            raise ValueError(
                f"Table '{table_name}' "
                "is not registered."
            )


        try:

            return (
                self.connection
                .execute(
                    f'DESCRIBE "{table_name}"'
                )
                .fetchdf()
            )


        except Exception as error:

            raise RuntimeError(
                "Could not retrieve schema for "
                f"table '{table_name}': "
                f"{error}"
            ) from error


    # ========================================================
    # CHECK TABLE
    # ========================================================

    def is_table_registered(
        self,
        table_name
    ):

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
    # CLOSE
    # ========================================================

    def close(
        self
    ):

        if self.connection is not None:

            self.connection.close()

            self.connection = None

            self.registered_tables.clear()