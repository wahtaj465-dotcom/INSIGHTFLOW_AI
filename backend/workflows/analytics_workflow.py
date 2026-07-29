from typing import TypedDict, Any
from pathlib import Path

import pandas as pd

from langgraph.graph import StateGraph, START, END

from backend.agents.ingestion_agent import load_dataset
from backend.agents.schema_agent import analyze_schema

from backend.agents.cleaning_agent import (
    generate_quality_report,
    clean_dataset,
    detect_numeric_anomalies,
    calculate_quality_score,
)

from backend.agents.eda_agent import perform_eda
from backend.agents.sql_agent import SQLAgent
from backend.agents.insight_agent import InsightAgent

from backend.services.sql_engine import SQLEngine
from backend.services.llm_service import LLMService

from backend.services.dataset_manager import (
    dataset_manager
)


# ============================================================
# WORKFLOW STATE
# ============================================================

class AnalyticsState(TypedDict, total=False):

    # User / dataset input
    file_path: str
    original_filename: str
    dataset_id: str
    question: str
    table_name: str

    # Original dataset
    raw_df: pd.DataFrame
    original_schema: dict
    original_quality_report: dict
    original_anomalies: dict
    before_score: float

    # Cleaned dataset
    cleaned_df: pd.DataFrame
    cleaned_schema: dict
    cleaned_quality_report: dict
    cleaned_anomalies: dict
    after_score: float
    cleaning_log: list

    # EDA
    eda_results: dict

    # SQL
    sql_response: dict
    generated_sql: str
    sql_result: Any
    sql_attempts: list

    # Insight
    insight_response: dict
    insight: str
    relevant_columns: list

    # Final status
    success: bool
    error: Any


# ============================================================
# ANALYTICS WORKFLOW
# ============================================================

class AnalyticsWorkflow:
    """
    InsightFlow orchestration layer.

    Supports:

    1. prepare_dataset()
       Preprocess a dataset once and store it.

    2. ask_dataset()
       Ask multiple questions against an already
       prepared dataset.

    3. run()
       Backwards-compatible one-shot pipeline used
       by the existing /api/analyze endpoint.
    """

    def __init__(
        self,
        table_name="sales",
        max_sql_retries=2
    ):

        self.table_name = table_name
        self.max_sql_retries = max_sql_retries

        # Shared LLM service
        self.llm_service = LLMService()

        # Used by the legacy one-shot workflow.
        self.sql_engine = SQLEngine()

        self.sql_agent = SQLAgent(
            llm_service=self.llm_service,
            sql_engine=self.sql_engine,
            table_name=self.table_name,
            max_retries=self.max_sql_retries
        )

        self.insight_agent = InsightAgent(
            llm_service=self.llm_service
        )

        # Three graphs:
        #
        # preparation_graph:
        # upload -> preprocessing -> EDA
        #
        # question_graph:
        # prepared dataframe -> SQL -> insight
        #
        # graph:
        # backwards-compatible complete workflow

        self.preparation_graph = (
            self._build_preparation_graph()
        )

        self.question_graph = (
            self._build_question_graph()
        )

        self.graph = (
            self._build_full_graph()
        )


    # ========================================================
    # NODE 1 — INGESTION
    # ========================================================

    def ingestion_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[1/7] Loading dataset..."
        )

        file_path = state[
            "file_path"
        ]

        df = load_dataset(
            file_path
        )

        print(
            f"Dataset loaded: "
            f"{len(df)} rows, "
            f"{len(df.columns)} columns"
        )

        return {
            "raw_df": df,
            "table_name": self.table_name
        }


    # ========================================================
    # NODE 2 — ORIGINAL SCHEMA
    # ========================================================

    def original_schema_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[2/7] Analyzing original schema..."
        )

        df = state[
            "raw_df"
        ]

        schema = analyze_schema(
            df
        )

        return {
            "original_schema": schema
        }


    # ========================================================
    # NODE 3 — ORIGINAL QUALITY
    # ========================================================

    def original_quality_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[3/7] Analyzing original data quality..."
        )

        df = state[
            "raw_df"
        ]

        schema = state[
            "original_schema"
        ]

        quality_report = (
            generate_quality_report(
                df,
                schema
            )
        )

        return {
            "original_quality_report":
                quality_report
        }


    # ========================================================
    # NODE 4 — ORIGINAL ANOMALIES
    # ========================================================

    def original_anomaly_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[4/7] Detecting original anomalies..."
        )

        df = state[
            "raw_df"
        ]

        schema = state[
            "original_schema"
        ]

        quality_report = state[
            "original_quality_report"
        ]

        anomalies = (
            detect_numeric_anomalies(
                df,
                schema
            )
        )

        score = (
            calculate_quality_score(
                df,
                quality_report,
                anomalies
            )
        )

        print(
            f"Original quality score: "
            f"{score}/100"
        )

        return {
            "original_anomalies":
                anomalies,

            "before_score":
                score
        }


    # ========================================================
    # NODE 5 — CLEANING
    # ========================================================

    def cleaning_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[5/7] Cleaning dataset..."
        )

        df = state[
            "raw_df"
        ]

        schema = state[
            "original_schema"
        ]

        cleaned_df, cleaning_log = (
            clean_dataset(
                df,
                schema
            )
        )

        print(
            f"Cleaning completed with "
            f"{len(cleaning_log)} operation(s)."
        )

        return {
            "cleaned_df":
                cleaned_df,

            "cleaning_log":
                cleaning_log
        }


    # ========================================================
    # NODE 6 — CLEANED ANALYSIS
    # ========================================================

    def cleaned_analysis_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[6/7] Re-analyzing cleaned dataset..."
        )

        cleaned_df = state[
            "cleaned_df"
        ]

        cleaned_schema = (
            analyze_schema(
                cleaned_df
            )
        )

        cleaned_quality_report = (
            generate_quality_report(
                cleaned_df,
                cleaned_schema
            )
        )

        cleaned_anomalies = (
            detect_numeric_anomalies(
                cleaned_df,
                cleaned_schema
            )
        )

        after_score = (
            calculate_quality_score(
                cleaned_df,
                cleaned_quality_report,
                cleaned_anomalies
            )
        )

        print(
            f"Cleaned quality score: "
            f"{after_score}/100"
        )

        return {
            "cleaned_schema":
                cleaned_schema,

            "cleaned_quality_report":
                cleaned_quality_report,

            "cleaned_anomalies":
                cleaned_anomalies,

            "after_score":
                after_score
        }


    # ========================================================
    # NODE 7 — EDA
    # ========================================================

    def eda_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[7/7] Running automated EDA..."
        )

        cleaned_df = state[
            "cleaned_df"
        ]

        cleaned_schema = state[
            "cleaned_schema"
        ]

        eda_results = (
            perform_eda(
                cleaned_df,
                cleaned_schema
            )
        )

        print(
            "EDA completed."
        )

        return {
            "eda_results":
                eda_results
        }


    # ========================================================
    # QUESTION NODE 1 — DATABASE
    # ========================================================

    def database_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[1/3] Registering prepared dataset "
            "in DuckDB..."
        )

        cleaned_df = state[
            "cleaned_df"
        ]

        table_name = state.get(
            "table_name",
            self.table_name
        )

        self.sql_engine.register_dataframe(
            cleaned_df,
            table_name=table_name
        )

        return {}


    # ========================================================
    # QUESTION NODE 2 — SQL AGENT
    # ========================================================

    def sql_agent_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[2/3] Running SQL Agent..."
        )

        question = state[
            "question"
        ]

        sql_response = (
            self.sql_agent.ask(
                question
            )
        )

        if not sql_response.get(
            "success",
            False
        ):

            error = (
                sql_response.get(
                    "error"
                )
                or
                "SQL Agent failed."
            )

            print(
                f"SQL Agent failed: {error}"
            )

            return {
                "sql_response":
                    sql_response,

                "generated_sql":
                    sql_response.get(
                        "sql"
                    ),

                "sql_result":
                    None,

                "sql_attempts":
                    sql_response.get(
                        "attempts",
                        []
                    ),

                "success":
                    False,

                "error":
                    error
            }

        print(
            "SQL Agent completed successfully."
        )

        return {
            "sql_response":
                sql_response,

            "generated_sql":
                sql_response.get(
                    "sql"
                ),

            "sql_result":
                sql_response.get(
                    "result"
                ),

            "sql_attempts":
                sql_response.get(
                    "attempts",
                    []
                ),

            "error":
                None
        }


    # ========================================================
    # ROUTE AFTER SQL
    # ========================================================

    def route_after_sql(
        self,
        state: AnalyticsState
    ):

        sql_response = state.get(
            "sql_response",
            {}
        )

        if sql_response.get(
            "success",
            False
        ):

            return "insight"

        return "end"


    # ========================================================
    # QUESTION NODE 3 — INSIGHT
    # ========================================================

    def insight_agent_node(
        self,
        state: AnalyticsState
    ):

        print(
            "\n[3/3] Generating analytical insight..."
        )

        sql_response = state[
            "sql_response"
        ]

        insight_response = (
            self.insight_agent.analyze(

                sql_response=
                    sql_response,

                quality_report=
                    state.get(
                        "cleaned_quality_report"
                    ),

                anomalies=
                    state.get(
                        "cleaned_anomalies"
                    ),

                eda_results=
                    state.get(
                        "eda_results"
                    )
            )
        )

        if not insight_response.get(
            "success",
            False
        ):

            error = (
                insight_response.get(
                    "error"
                )
                or
                "Insight Agent failed."
            )

            print(
                f"Insight Agent failed: "
                f"{error}"
            )

            return {
                "insight_response":
                    insight_response,

                "insight":
                    None,

                "relevant_columns":
                    [],

                "success":
                    False,

                "error":
                    error
            }

        print(
            "Insight generated successfully."
        )

        return {
            "insight_response":
                insight_response,

            "insight":
                insight_response.get(
                    "insight"
                ),

            "relevant_columns":
                insight_response.get(
                    "relevant_columns",
                    []
                ),

            "success":
                True,

            "error":
                None
        }


    # ========================================================
    # BUILD PREPARATION GRAPH
    # ========================================================

    def _build_preparation_graph(self):

        workflow = StateGraph(
            AnalyticsState
        )

        workflow.add_node(
            "ingestion",
            self.ingestion_node
        )

        workflow.add_node(
            "original_schema",
            self.original_schema_node
        )

        workflow.add_node(
            "original_quality",
            self.original_quality_node
        )

        workflow.add_node(
            "original_anomalies",
            self.original_anomaly_node
        )

        workflow.add_node(
            "cleaning",
            self.cleaning_node
        )

        workflow.add_node(
            "cleaned_analysis",
            self.cleaned_analysis_node
        )

        workflow.add_node(
            "eda",
            self.eda_node
        )

        workflow.add_edge(
            START,
            "ingestion"
        )

        workflow.add_edge(
            "ingestion",
            "original_schema"
        )

        workflow.add_edge(
            "original_schema",
            "original_quality"
        )

        workflow.add_edge(
            "original_quality",
            "original_anomalies"
        )

        workflow.add_edge(
            "original_anomalies",
            "cleaning"
        )

        workflow.add_edge(
            "cleaning",
            "cleaned_analysis"
        )

        workflow.add_edge(
            "cleaned_analysis",
            "eda"
        )

        workflow.add_edge(
            "eda",
            END
        )

        return workflow.compile()


    # ========================================================
    # BUILD QUESTION GRAPH
    # ========================================================

    def _build_question_graph(self):

        workflow = StateGraph(
            AnalyticsState
        )

        workflow.add_node(
            "database",
            self.database_node
        )

        workflow.add_node(
            "sql_agent",
            self.sql_agent_node
        )

        workflow.add_node(
            "insight_agent",
            self.insight_agent_node
        )

        workflow.add_edge(
            START,
            "database"
        )

        workflow.add_edge(
            "database",
            "sql_agent"
        )

        workflow.add_conditional_edges(
            "sql_agent",

            self.route_after_sql,

            {
                "insight":
                    "insight_agent",

                "end":
                    END
            }
        )

        workflow.add_edge(
            "insight_agent",
            END
        )

        return workflow.compile()


    # ========================================================
    # BUILD FULL GRAPH
    # ========================================================

    def _build_full_graph(self):
        """
        Existing one-shot workflow.

        Maintained so /api/analyze and existing tests
        continue to work.
        """

        workflow = StateGraph(
            AnalyticsState
        )

        workflow.add_node(
            "ingestion",
            self.ingestion_node
        )

        workflow.add_node(
            "original_schema",
            self.original_schema_node
        )

        workflow.add_node(
            "original_quality",
            self.original_quality_node
        )

        workflow.add_node(
            "original_anomalies",
            self.original_anomaly_node
        )

        workflow.add_node(
            "cleaning",
            self.cleaning_node
        )

        workflow.add_node(
            "cleaned_analysis",
            self.cleaned_analysis_node
        )

        workflow.add_node(
            "eda",
            self.eda_node
        )

        workflow.add_node(
            "database",
            self.database_node
        )

        workflow.add_node(
            "sql_agent",
            self.sql_agent_node
        )

        workflow.add_node(
            "insight_agent",
            self.insight_agent_node
        )

        workflow.add_edge(
            START,
            "ingestion"
        )

        workflow.add_edge(
            "ingestion",
            "original_schema"
        )

        workflow.add_edge(
            "original_schema",
            "original_quality"
        )

        workflow.add_edge(
            "original_quality",
            "original_anomalies"
        )

        workflow.add_edge(
            "original_anomalies",
            "cleaning"
        )

        workflow.add_edge(
            "cleaning",
            "cleaned_analysis"
        )

        workflow.add_edge(
            "cleaned_analysis",
            "eda"
        )

        workflow.add_edge(
            "eda",
            "database"
        )

        workflow.add_edge(
            "database",
            "sql_agent"
        )

        workflow.add_conditional_edges(
            "sql_agent",

            self.route_after_sql,

            {
                "insight":
                    "insight_agent",

                "end":
                    END
            }
        )

        workflow.add_edge(
            "insight_agent",
            END
        )

        return workflow.compile()


    # ========================================================
    # PREPARE DATASET
    # ========================================================

    def prepare_dataset(
        self,
        file_path,
        original_filename=None
    ):
        """
        Run preprocessing once and store the prepared
        dataset inside DatasetManager.
        """

        self._validate_file_path(
            file_path
        )

        if original_filename is None:

            original_filename = (
                Path(file_path).name
            )

        initial_state = {
            "file_path":
                file_path,

            "original_filename":
                original_filename,

            "table_name":
                self.table_name,

            "success":
                False,

            "error":
                None
        }

        try:

            prepared_state = (
                self.preparation_graph.invoke(
                    initial_state
                )
            )

        except Exception as error:

            return {
                "success":
                    False,

                "error":
                    (
                        "Dataset preparation failed: "
                        f"{error}"
                    )
            }

        try:

            dataset_id = (
                dataset_manager.create_dataset(

                    cleaned_df=
                        prepared_state[
                            "cleaned_df"
                        ],

                    original_filename=
                        original_filename,

                    original_schema=
                        prepared_state.get(
                            "original_schema"
                        ),

                    original_quality_report=
                        prepared_state.get(
                            "original_quality_report"
                        ),

                    original_anomalies=
                        prepared_state.get(
                            "original_anomalies"
                        ),

                    before_score=
                        prepared_state.get(
                            "before_score"
                        ),

                    cleaned_schema=
                        prepared_state.get(
                            "cleaned_schema"
                        ),

                    cleaned_quality_report=
                        prepared_state.get(
                            "cleaned_quality_report"
                        ),

                    cleaned_anomalies=
                        prepared_state.get(
                            "cleaned_anomalies"
                        ),

                    after_score=
                        prepared_state.get(
                            "after_score"
                        ),

                    cleaning_log=
                        prepared_state.get(
                            "cleaning_log"
                        ),

                    eda_results=
                        prepared_state.get(
                            "eda_results"
                        )
                )
            )

        except Exception as error:

            return {
                "success":
                    False,

                "error":
                    (
                        "Could not create dataset "
                        f"session: {error}"
                    )
            }

        cleaned_df = prepared_state[
            "cleaned_df"
        ]

        return {
            "success":
                True,

            "dataset_id":
                dataset_id,

            "original_filename":
                original_filename,

            "original_rows":
                len(
                    prepared_state[
                        "raw_df"
                    ]
                ),

            "cleaned_rows":
                len(
                    cleaned_df
                ),

            "columns":
                list(
                    cleaned_df.columns
                ),

            "before_score":
                prepared_state.get(
                    "before_score"
                ),

            "after_score":
                prepared_state.get(
                    "after_score"
                ),

            "cleaning_log":
                prepared_state.get(
                    "cleaning_log",
                    []
                ),

            "quality_report":
                prepared_state.get(
                    "cleaned_quality_report",
                    {}
                ),

            "anomalies":
                prepared_state.get(
                    "cleaned_anomalies",
                    {}
                ),

            "eda_results":
                prepared_state.get(
                    "eda_results",
                    {}
                )
        }


    # ========================================================
    # ASK PREPARED DATASET
    # ========================================================

    def ask_dataset(
        self,
        dataset_id,
        question
    ):
        """
        Ask a question against an already-prepared dataset.

        Preprocessing and EDA are NOT repeated.
        """

        self._validate_question(
            question
        )

        if not isinstance(
            dataset_id,
            str
        ):

            raise TypeError(
                "dataset_id must be a string."
            )

        dataset_id = (
            dataset_id.strip()
        )

        if not dataset_id:

            raise ValueError(
                "dataset_id cannot be empty."
            )

        dataset = (
            dataset_manager.get_dataset(
                dataset_id
            )
        )

        if dataset is None:

            return {
                "success":
                    False,

                "dataset_id":
                    dataset_id,

                "question":
                    question,

                "error":
                    "Dataset session not found."
            }

        # ----------------------------------------------------
        # Important:
        #
        # Give every question its own DuckDB engine.
        #
        # This prevents one dataset/query from interfering
        # with another dataset.
        # ----------------------------------------------------

        question_engine = SQLEngine()

        question_sql_agent = SQLAgent(
            llm_service=self.llm_service,
            sql_engine=question_engine,
            table_name=self.table_name,
            max_retries=self.max_sql_retries
        )

        # Temporarily use these services for the
        # question graph nodes.

        previous_engine = (
            self.sql_engine
        )

        previous_agent = (
            self.sql_agent
        )

        self.sql_engine = (
            question_engine
        )

        self.sql_agent = (
            question_sql_agent
        )

        initial_state = {

            "dataset_id":
                dataset_id,

            "question":
                question,

            "table_name":
                self.table_name,

            "cleaned_df":
                dataset[
                    "cleaned_df"
                ],

            "cleaned_schema":
                dataset.get(
                    "cleaned_schema",
                    {}
                ),

            "cleaned_quality_report":
                dataset.get(
                    "cleaned_quality_report",
                    {}
                ),

            "cleaned_anomalies":
                dataset.get(
                    "cleaned_anomalies",
                    {}
                ),

            "eda_results":
                dataset.get(
                    "eda_results",
                    {}
                ),

            "success":
                False,

            "error":
                None
        }

        try:

            final_state = (
                self.question_graph.invoke(
                    initial_state
                )
            )

        except Exception as error:

            return {
                "success":
                    False,

                "dataset_id":
                    dataset_id,

                "question":
                    question,

                "error":
                    (
                        "Question workflow failed: "
                        f"{error}"
                    )
            }

        finally:

            try:
                question_engine.close()

            except Exception:
                pass

            self.sql_engine = (
                previous_engine
            )

            self.sql_agent = (
                previous_agent
            )

        final_state[
            "dataset_id"
        ] = dataset_id

        return final_state


    # ========================================================
    # GET DATASET INFO
    # ========================================================

    def get_dataset_info(
        self,
        dataset_id
    ):

        return (
            dataset_manager.get_dataset_info(
                dataset_id
            )
        )


    # ========================================================
    # DELETE DATASET
    # ========================================================

    def delete_dataset(
        self,
        dataset_id
    ):

        return (
            dataset_manager.delete_dataset(
                dataset_id
            )
        )


    # ========================================================
    # LEGACY ONE-SHOT RUN
    # ========================================================

    def run(
        self,
        file_path,
        question
    ):
        """
        Run the original complete pipeline.

        Used by the existing /api/analyze endpoint.
        """

        self._validate_file_path(
            file_path
        )

        self._validate_question(
            question
        )

        initial_state = {

            "file_path":
                file_path,

            "question":
                question,

            "table_name":
                self.table_name,

            "success":
                False,

            "error":
                None
        }

        try:

            final_state = (
                self.graph.invoke(
                    initial_state
                )
            )

        except Exception as error:

            return {
                "success":
                    False,

                "error":
                    (
                        "Workflow execution failed: "
                        f"{error}"
                    ),

                "file_path":
                    file_path,

                "question":
                    question
            }

        return final_state


    # ========================================================
    # VALIDATE FILE PATH
    # ========================================================

    def _validate_file_path(
        self,
        file_path
    ):

        if not isinstance(
            file_path,
            str
        ):

            raise TypeError(
                "file_path must be a string."
            )

        file_path = (
            file_path.strip()
        )

        if not file_path:

            raise ValueError(
                "file_path cannot be empty."
            )


    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    def _validate_question(
        self,
        question
    ):

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "question must be a string."
            )

        if not question.strip():

            raise ValueError(
                "question cannot be empty."
            )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.sql_engine is not None:

            try:
                self.sql_engine.close()

            except Exception:
                pass