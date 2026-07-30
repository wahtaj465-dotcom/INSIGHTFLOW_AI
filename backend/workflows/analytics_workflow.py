"""
InsightFlow AI
Analytics Preparation Workflow

Deterministic analytics pipeline executed whenever
a user uploads a dataset.

Pipeline:

    Dataset
        ↓
    Ingestion
        ↓
    Schema Intelligence
        ↓
    Quality Intelligence - BEFORE
        ↓
    Anomaly Intelligence - BEFORE
        ↓
    Cleaning Intelligence
        ↓
    Schema Intelligence - AFTER
        ↓
    Quality Intelligence - AFTER
        ↓
    Anomaly Intelligence - AFTER
        ↓
    EDA Intelligence
        ↓
    Visualization Intelligence
        ↓
    Statistical Insight Engine
        ↓
    DatasetManager

The workflow itself does not require an LLM.

The deterministic analytics layer produces reliable
metadata and statistics first. These results can later
be consumed by an LLM reasoning layer.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# INGESTION
# ============================================================

from backend.agents.ingestion_agent import (
    load_dataset
)


# ============================================================
# ANALYTICS AGENTS
# ============================================================

from backend.agents.schema_agent import (
    analyze_schema
)

from backend.agents.quality_agent import (
    analyze_data_quality,
    calculate_quality_score
)

from backend.agents.anomaly_agent import (
    detect_anomalies
)

from backend.agents.cleaning_agent import (
    clean_dataset
)

from backend.agents.eda_agent import (
    run_eda
)


# ============================================================
# QUESTION / ANALYTICAL AGENTS
# ============================================================

from backend.agents.sql_agent import (
    SQLAgent
)

from backend.agents.insight_agent import (
    InsightAgent
)


# ============================================================
# SERVICES
# ============================================================

from backend.services.dataset_manager import (
    dataset_manager
)

from backend.services.visualization_service import (
    VisualizationService
)

from backend.services.sql_engine import (
    SQLEngine
)

from backend.services.llm_service import (
    LLMService
)

# ============================================================
# ANALYTICS WORKFLOW
# ============================================================

class AnalyticsWorkflow:
    
    """
    Central deterministic analytics workflow.

    Responsibilities:

    1. Load dataset
    2. Analyze original schema
    3. Analyze original quality
    4. Detect original anomalies
    5. Clean dataset
    6. Re-analyze cleaned schema
    7. Re-analyze cleaned quality
    8. Detect remaining anomalies
    9. Perform EDA
    10. Generate visualization metadata
    11. Generate statistical findings
    12. Store prepared dataset session
    """
    
    def __init__(
    self,
    max_charts=10
):
        

        self.visualization_service = (
        VisualizationService(
            max_charts=max_charts
        )
    )

    # --------------------------------------------------------
    # LLM service is initialized lazily.
    #
    # Dataset upload / preprocessing must NOT depend on
    # Gemini being available.
    # --------------------------------------------------------

        self.llm_service = None
    


    # ========================================================
    # PREPARE DATASET
    # ========================================================

    def prepare_dataset(
        self,
        file_path,
        original_filename=None
    ):
        """
        Run the complete deterministic analytics pipeline.
        """

        try:

            # =================================================
            # 1. INGESTION
            # =================================================

            print(
                "\n[1/10] Loading dataset..."
            )

            df = load_dataset(
                file_path
            )

            if not isinstance(
                df,
                pd.DataFrame
            ):
                raise TypeError(
                    "Dataset loader did not return "
                    "a Pandas DataFrame."
                )

            if df.empty:
                raise ValueError(
                    "The uploaded dataset is empty."
                )

            print(
                f"Dataset loaded: "
                f"{len(df)} rows, "
                f"{len(df.columns)} columns"
            )


            # =================================================
            # RESOLVE ORIGINAL FILENAME
            # =================================================

            if original_filename is None:

                try:

                    original_filename = (
                        Path(file_path).name
                    )

                except Exception:

                    original_filename = None


            # =================================================
            # 2. ORIGINAL SCHEMA
            # =================================================

            print(
                "\n[2/10] Analyzing original schema..."
            )

            original_schema = (
                analyze_schema(
                    df
                )
            )

            if not isinstance(
                original_schema,
                dict
            ):
                original_schema = {}


            # =================================================
            # 3. ORIGINAL QUALITY
            # =================================================

            print(
                "\n[3/10] Analyzing original data quality..."
            )

            original_quality_report = (
                analyze_data_quality(
                    df,
                    schema=original_schema
                )
            )

            if not isinstance(
                original_quality_report,
                dict
            ):
                original_quality_report = {}


            original_quality_components = (
                self._extract_quality_components(
                    original_quality_report
                )
            )


            before_score = (
                self._resolve_quality_score(
                    original_quality_report
                )
            )


            if before_score is not None:

                print(
                    "Original quality score: "
                    f"{before_score:.2f}/100"
                )

            else:

                print(
                    "Original quality score: N/A"
                )


            # =================================================
            # 4. ORIGINAL ANOMALIES
            # =================================================

            print(
                "\n[4/10] Detecting original anomalies..."
            )

            original_anomalies = (
                detect_anomalies(
                    df,
                    schema=original_schema
                )
            )

            if not isinstance(
                original_anomalies,
                dict
            ):
                original_anomalies = {}


            # =================================================
            # 5. CLEANING
            # =================================================

            print(
                "\n[5/10] Cleaning dataset..."
            )

            cleaning_result = (
                clean_dataset(
                    df,
                    schema=original_schema,
                    quality_report=(
                        original_quality_report
                    ),
                    anomalies=(
                        original_anomalies
                    )
                )
            )


            (
                cleaned_df,
                cleaning_log,
                cleaning_summary
            ) = self._normalize_cleaning_result(
                cleaning_result
            )


            if cleaned_df is None:

                raise RuntimeError(
                    "Cleaning agent did not return "
                    "a DataFrame."
                )


            if not isinstance(
                cleaned_df,
                pd.DataFrame
            ):

                raise TypeError(
                    "Cleaning agent must return "
                    "a Pandas DataFrame."
                )


            if cleaned_df.empty:

                raise ValueError(
                    "Cleaning removed all rows "
                    "from the dataset."
                )


            print(
                "Cleaning completed with "
                f"{len(cleaning_log)} operation(s)."
            )


            # =================================================
            # 6. CLEANED SCHEMA
            # =================================================

            print(
                "\n[6/10] Re-analyzing cleaned schema..."
            )

            cleaned_schema = (
                analyze_schema(
                    cleaned_df
                )
            )

            if not isinstance(
                cleaned_schema,
                dict
            ):
                cleaned_schema = {}


            # =================================================
            # 7. CLEANED QUALITY
            # =================================================

            print(
                "\n[7/10] Re-analyzing cleaned data quality..."
            )

            cleaned_quality_report = (
                analyze_data_quality(
                    cleaned_df,
                    schema=cleaned_schema
                )
            )

            if not isinstance(
                cleaned_quality_report,
                dict
            ):
                cleaned_quality_report = {}


            cleaned_quality_components = (
                self._extract_quality_components(
                    cleaned_quality_report
                )
            )


            after_score = (
                self._resolve_quality_score(
                    cleaned_quality_report
                )
            )


            if after_score is not None:

                print(
                    "Cleaned quality score: "
                    f"{after_score:.2f}/100"
                )

            else:

                print(
                    "Cleaned quality score: N/A"
                )


            # =================================================
            # 8. CLEANED ANOMALIES
            # =================================================

            print(
                "\n[8/10] Detecting cleaned anomalies..."
            )

            cleaned_anomalies = (
                detect_anomalies(
                    cleaned_df,
                    schema=cleaned_schema
                )
            )

            if not isinstance(
                cleaned_anomalies,
                dict
            ):
                cleaned_anomalies = {}


            # =================================================
            # 9. EDA
            # =================================================

            print(
                "\n[9/10] Running automated EDA..."
            )

            eda_results = (
                run_eda(
                    df=cleaned_df,
                    schema=cleaned_schema
                )
            )

            if not isinstance(
                eda_results,
                dict
            ):
                raise TypeError(
                    "EDA agent must return "
                    "a dictionary."
                )


            print(
                "EDA completed."
            )


            # =================================================
            # VISUALIZATION INTELLIGENCE
            # =================================================

            print(
                "\nGenerating visualization metadata..."
            )

            eda_charts = (
                self.visualization_service
                .generate_eda_charts(
                    df=cleaned_df,
                    schema=cleaned_schema,
                    eda_results=eda_results
                )
            )


            if not isinstance(
                eda_charts,
                list
            ):
                eda_charts = []


            print(
                f"Generated "
                f"{len(eda_charts)} chart(s)."
            )


            # =================================================
            # 10. STATISTICAL INSIGHT ENGINE
            # =================================================

            print(
                "\n[10/10] Generating statistical findings..."
            )

            statistical_findings = (
                self._generate_statistical_findings(
                    df=cleaned_df,
                    schema=cleaned_schema,
                    eda_results=eda_results,
                    quality_report=(
                        cleaned_quality_report
                    ),
                    anomalies=(
                        cleaned_anomalies
                    )
                )
            )


            print(
                "Generated "
                f"{len(statistical_findings)} "
                "statistical finding(s)."
            )


            # =================================================
            # QUALITY IMPROVEMENT
            # =================================================

            quality_improvement = None


            if (
                before_score is not None
                and
                after_score is not None
            ):

                quality_improvement = round(
                    after_score
                    -
                    before_score,
                    2
                )


            # =================================================
            # STORE DATASET
            # =================================================

            print(
                "\nStoring prepared dataset session..."
            )


            dataset_id = (
                dataset_manager.create_dataset(

                    cleaned_df=cleaned_df,

                    original_filename=(
                        original_filename
                    ),


                    # -----------------------------------------
                    # ORIGINAL
                    # -----------------------------------------

                    original_schema=(
                        original_schema
                    ),

                    original_quality_report=(
                        original_quality_report
                    ),

                    original_quality_components=(
                        original_quality_components
                    ),

                    original_anomalies=(
                        original_anomalies
                    ),

                    before_score=(
                        before_score
                    ),


                    # -----------------------------------------
                    # CLEANED
                    # -----------------------------------------

                    cleaned_schema=(
                        cleaned_schema
                    ),

                    cleaned_quality_report=(
                        cleaned_quality_report
                    ),

                    cleaned_quality_components=(
                        cleaned_quality_components
                    ),

                    cleaned_anomalies=(
                        cleaned_anomalies
                    ),

                    after_score=(
                        after_score
                    ),


                    # -----------------------------------------
                    # CLEANING
                    # -----------------------------------------

                    cleaning_log=(
                        cleaning_log
                    ),

                    cleaning_summary=(
                        cleaning_summary
                    ),


                    # -----------------------------------------
                    # EDA
                    # -----------------------------------------

                    eda_results=(
                        eda_results
                    ),


                    # -----------------------------------------
                    # CHARTS
                    # -----------------------------------------

                    eda_charts=(
                        eda_charts
                    ),


                    # -----------------------------------------
                    # STATISTICAL INTELLIGENCE
                    # -----------------------------------------

                    statistical_findings=(
                        statistical_findings
                    )
                )
            )


            if dataset_id is None:

                raise RuntimeError(
                    "DatasetManager did not return "
                    "a dataset ID."
                )


            print(
                "Dataset preparation completed: "
                f"{dataset_id}"
            )


            # =================================================
            # RETURN
            # =================================================

            return {

                "success":
                    True,

                "dataset_id":
                    dataset_id,

                "original_filename":
                    original_filename,

                "rows":
                    int(
                        len(
                            cleaned_df
                        )
                    ),

                "columns":
                    [
                        str(column)
                        for column
                        in cleaned_df.columns
                    ],

                "column_count":
                    int(
                        len(
                            cleaned_df.columns
                        )
                    ),


                # ---------------------------------------------
                # SCHEMA
                # ---------------------------------------------

                "schema": {

                    "original":
                        original_schema,

                    "cleaned":
                        cleaned_schema
                },


                # ---------------------------------------------
                # QUALITY
                # ---------------------------------------------

                "quality": {

                    "before_score":
                        before_score,

                    "after_score":
                        after_score,

                    "improvement":
                        quality_improvement,

                    "before_components":
                        original_quality_components,

                    "after_components":
                        cleaned_quality_components,

                    "before_report":
                        original_quality_report,

                    "after_report":
                        cleaned_quality_report
                },


                # ---------------------------------------------
                # ANOMALIES
                # ---------------------------------------------

                "anomalies": {

                    "before":
                        original_anomalies,

                    "after":
                        cleaned_anomalies
                },


                # ---------------------------------------------
                # CLEANING
                # ---------------------------------------------

                "cleaning": {

                    "operations":
                        cleaning_log,

                    "summary":
                        cleaning_summary
                },


                # ---------------------------------------------
                # EDA
                # ---------------------------------------------

                "eda_results":
                    eda_results,


                # ---------------------------------------------
                # VISUALIZATIONS
                # ---------------------------------------------

                "eda_charts":
                    eda_charts,

                "chart_count":
                    len(
                        eda_charts
                    ),


                # ---------------------------------------------
                # STATISTICAL FINDINGS
                # ---------------------------------------------

                "statistical_findings":
                    statistical_findings,

                "statistical_finding_count":
                    len(
                        statistical_findings
                    )
            }


        except Exception as error:

            print(
                "\nDataset preparation failed:"
            )

            print(
                str(error)
            )

            raise RuntimeError(
                "Analytics workflow failed: "
                f"{error}"
            ) from error


    # ========================================================
    # LOCAL STATISTICAL INSIGHT ENGINE
    # ========================================================
    
    # ========================================================
# ASK DATASET
# ========================================================

    def ask_dataset(
        self,
        dataset_id,
        question
    ):
        """
        Analyze a natural-language question against a
        previously prepared dataset.

        Pipeline:

            dataset_id
                ↓
            DatasetManager
                ↓
            cleaned DataFrame
                ↓
            DuckDB
                ↓
            SQL Agent
                ↓
            SQL execution
                ↓
            Visualization Service
                ↓
            Insight Agent
                ↓
            Gemini or deterministic fallback

        The prepared dataset is reused. The dataset does NOT
        need to be cleaned or analyzed again.
        """

        # ====================================================
        # VALIDATE DATASET ID
        # ====================================================

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


        # ====================================================
        # VALIDATE QUESTION
        # ====================================================

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "question must be a string."
            )


        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "question cannot be empty."
            )


        # ====================================================
        # 1. RETRIEVE PREPARED DATASET
        # ====================================================

        print(
            "\n[1/4] Retrieving prepared dataset..."
        )


        dataset = (
            dataset_manager.get_dataset(
                dataset_id
            )
        )


        if dataset is None:

            raise ValueError(
                f"Dataset '{dataset_id}' "
                "does not exist or has expired."
            )


        cleaned_df = (
            dataset.get(
                "cleaned_df"
            )
        )


        if cleaned_df is None:

            raise RuntimeError(
                "Prepared dataset does not contain "
                "a cleaned DataFrame."
            )


        if not isinstance(
            cleaned_df,
            pd.DataFrame
        ):

            raise TypeError(
                "Stored cleaned dataset is not "
                "a Pandas DataFrame."
            )


        if cleaned_df.empty:

            raise ValueError(
                "Prepared dataset is empty."
            )


        # ====================================================
        # GET ANALYTICAL CONTEXT
        # ====================================================

        cleaned_schema = (
            dataset.get(
                "cleaned_schema",
                {}
            )
        )


        quality_report = (
            dataset.get(
                "cleaned_quality_report",
                {}
            )
        )


        anomalies = (
            dataset.get(
                "cleaned_anomalies",
                {}
            )
        )


        eda_results = (
            dataset.get(
                "eda_results",
                {}
            )
        )


        statistical_findings = (
            dataset.get(
                "statistical_findings",
                []
            )
        )


        # ====================================================
        # 2. REGISTER DATASET IN DUCKDB
        # ====================================================

        print(
            "\n[2/4] Registering prepared dataset "
            "in DuckDB..."
        )


        sql_engine = (
            SQLEngine()
        )


        try:

            table_name = "dataset"


            sql_engine.register_dataframe(
                cleaned_df,
                table_name=table_name
            )


            # =================================================
            # CREATE LLM SERVICE
            # =================================================

            llm_service = None


            try:

                if self.llm_service is None:

                    self.llm_service = (
                        LLMService()
                    )


                llm_service = (
                    self.llm_service
                )


            except Exception as error:

                print(
                    "LLM service unavailable. "
                    "Question analysis will use "
                    "deterministic fallbacks."
                )

                print(
                    f"Reason: {error}"
                )


            # =================================================
            # FALLBACK LLM ADAPTER
            # =================================================

            if llm_service is None:

                llm_service = (
                    _UnavailableLLMService()
                )


            # =================================================
            # 3. RUN SQL AGENT
            # =================================================

            print(
                "\n[3/4] Running SQL Agent..."
            )


            sql_agent = (
                SQLAgent(
                    llm_service=llm_service,
                    sql_engine=sql_engine,
                    table_name=table_name
                )
            )


            sql_response = (
                sql_agent.ask(
                    question
                )
            )


            if not isinstance(
                sql_response,
                dict
            ):

                raise RuntimeError(
                    "SQL Agent returned an invalid response."
                )


            if not sql_response.get(
                "success",
                False
            ):

                return {

                    "success":
                        False,

                    "dataset_id":
                        dataset_id,

                    "question":
                        question,

                    "sql":
                        sql_response.get(
                            "sql"
                        ),

                    "generated_sql":
                        sql_response.get(
                            "generated_sql"
                        ),

                    "sql_source":
                        sql_response.get(
                            "sql_source"
                        ),

                    "fallback_used":
                        sql_response.get(
                            "fallback_used",
                            False
                        ),

                    "result":
                        [],

                    "result_count":
                        0,

                    "chart":
                        None,

                    "visualization":
                        None,

                    "insight":
                        None,

                    "insight_source":
                        None,

                    "llm_success":
                        False,

                    "statistical_findings":
                        statistical_findings,

                    "error":
                        sql_response.get(
                            "error"
                        ),

                    "llm_error":
                        sql_response.get(
                            "llm_error"
                        ),

                    "attempts":
                        sql_response.get(
                            "attempts",
                            []
                        )
                }


            print(
                "SQL Agent completed successfully."
            )


            result_df = (
                sql_response.get(
                    "result"
                )
            )


            if result_df is None:

                result_df = (
                    pd.DataFrame()
                )


            if not isinstance(
                result_df,
                pd.DataFrame
            ):

                raise TypeError(
                    "SQL Agent result must be "
                    "a Pandas DataFrame."
                )


            # =================================================
            # GENERATE RESULT VISUALIZATION
            # =================================================

            chart = None


            try:

                chart = (
                    self.visualization_service
                    .generate_result_chart(
                        df=result_df,
                        question=question
                    )
                )


            except Exception as error:

                print(
                    "Result visualization could not "
                    f"be generated: {error}"
                )

                chart = None


            # =================================================
            # 4. GENERATE ANALYTICAL INSIGHT
            # =================================================

            print(
                "\n[4/4] Generating analytical insight..."
            )


            insight_agent = (
                InsightAgent(
                    llm_service=llm_service
                )
            )


            insight_response = (
                insight_agent.analyze(

                    sql_response=sql_response,

                    quality_report=(
                        quality_report
                    ),

                    anomalies=(
                        anomalies
                    ),

                    eda_results=(
                        eda_results
                    )
                )
            )


            if not isinstance(
                insight_response,
                dict
            ):

                insight_response = {}


            insight = (
                insight_response.get(
                    "insight"
                )
            )


            insight_source = (
                insight_response.get(
                    "source"
                )
            )


            llm_success = (
                insight_response.get(
                    "llm_success",
                    False
                )
            )


            print(
                "Insight generation completed."
            )


            # =================================================
            # CONVERT SQL RESULT TO JSON-SAFE RECORDS
            # =================================================

            result_records = (
                self.visualization_service
                .dataframe_to_records(
                    result_df
                )
            )

                        # =================================================
            # AI / LLM STATUS DEBUG
            # =================================================

            print("\n========== AI STATUS ==========")

            print(
                "SQL Source:",
                sql_response.get("sql_source")
            )

            print(
                "SQL Fallback:",
                sql_response.get(
                    "fallback_used",
                    False
                )
            )

            print(
                "Insight Source:",
                insight_response.get("source")
            )

            print(
                "LLM Success:",
                insight_response.get(
                    "llm_success",
                    False
                )
            )

            print(
                "LLM Error:",
                (
                    insight_response.get("llm_error")
                    or
                    sql_response.get("llm_error")
                )
            )

            print("================================\n")



            # =================================================
            # FINAL RESPONSE
            # =================================================

            return {

                "success":
                    True,

                "dataset_id":
                    dataset_id,

                "question":
                    question,


                # ---------------------------------------------
                # SQL
                # ---------------------------------------------

                "sql":
                    sql_response.get(
                        "sql"
                    ),

                "generated_sql":
                    sql_response.get(
                        "generated_sql"
                    ),

                "sql_source":
                    sql_response.get(
                        "sql_source"
                    ),

                "fallback_used":
                    sql_response.get(
                        "fallback_used",
                        False
                    ),


                # ---------------------------------------------
                # RESULT
                # ---------------------------------------------

                "result":
                    result_records,

                "result_count":
                    int(
                        len(
                            result_df
                        )
                    ),

                "result_columns":
                    [
                        str(column)
                        for column
                        in result_df.columns
                    ],


                # ---------------------------------------------
                # VISUALIZATION
                # ---------------------------------------------

                "chart":
                    chart,

                # Keep this alias for frontend compatibility.
                "visualization":
                    chart,


                # ---------------------------------------------
                # INSIGHT
                # ---------------------------------------------

                "insight":
                    insight,

                "insight_source":
                    insight_source,

                "llm_success":
                    llm_success,


                # ---------------------------------------------
                # PRECOMPUTED ANALYTICS
                # ---------------------------------------------

                "statistical_findings":
                    statistical_findings,


                # ---------------------------------------------
                # DEBUG / DEVELOPMENT METADATA
                # ---------------------------------------------

                "sql_attempts":
                    sql_response.get(
                        "attempts",
                        []
                    ),

                "llm_error":
                    (
                        insight_response.get(
                            "llm_error"
                        )
                        or
                        sql_response.get(
                            "llm_error"
                        )
                    ),

                "error":
                    None
            }


        finally:

            # =================================================
            # ALWAYS CLOSE QUERY-SPECIFIC DUCKDB CONNECTION
            # =================================================

            try:

                sql_engine.close()

            except Exception:

                pass


    @staticmethod
    def _generate_statistical_findings(
        df,
        schema=None,
        eda_results=None,
        quality_report=None,
        anomalies=None
    ):
        """
        Generate deterministic statistical findings.

        No LLM is used here.

        Later this can be moved into:

            backend/agents/statistical_insight_agent.py
        """

        findings = []


        if not isinstance(
            df,
            pd.DataFrame
        ):
            return findings


        if df.empty:
            return findings


        # ====================================================
        # DATASET SIZE
        # ====================================================

        findings.append({

            "type":
                "dataset_summary",

            "importance":
                "info",

            "message":
                (
                    f"The prepared dataset contains "
                    f"{len(df):,} rows and "
                    f"{len(df.columns)} columns."
                )
        })


        # ====================================================
        # MISSING VALUES
        # ====================================================

        total_cells = (
            len(df)
            *
            len(df.columns)
        )


        total_missing = int(
            df.isna()
            .sum()
            .sum()
        )


        if total_cells > 0:

            missing_percentage = (
                total_missing
                /
                total_cells
                *
                100
            )

        else:

            missing_percentage = 0.0


        if total_missing > 0:

            findings.append({

                "type":
                    "missing_values",

                "importance":
                    (
                        "high"
                        if missing_percentage >= 20
                        else "medium"
                    ),

                "value":
                    round(
                        missing_percentage,
                        2
                    ),

                "message":
                    (
                        f"{total_missing:,} values are "
                        f"missing across the prepared dataset "
                        f"({missing_percentage:.2f}% of cells)."
                    )
            })

        else:

            findings.append({

                "type":
                    "missing_values",

                "importance":
                    "low",

                "value":
                    0.0,

                "message":
                    (
                        "No missing values remain in the "
                        "prepared dataset."
                    )
            })


        # ====================================================
        # DUPLICATES
        # ====================================================

        duplicate_count = int(
            df.duplicated().sum()
        )


        if duplicate_count > 0:

            duplicate_percentage = (
                duplicate_count
                /
                len(df)
                *
                100
            )


            findings.append({

                "type":
                    "duplicates",

                "importance":
                    "medium",

                "value":
                    duplicate_count,

                "message":
                    (
                        f"{duplicate_count:,} duplicate rows "
                        f"remain "
                        f"({duplicate_percentage:.2f}% "
                        f"of records)."
                    )
            })


        # ====================================================
        # NUMERIC COLUMNS
        # ====================================================

        numeric_columns = (
            df.select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )


        for column in numeric_columns[:10]:

            series = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                .dropna()
            )


            if series.empty:
                continue


            mean_value = (
                series.mean()
            )

            median_value = (
                series.median()
            )

            minimum = (
                series.min()
            )

            maximum = (
                series.max()
            )


            findings.append({

                "type":
                    "numeric_summary",

                "importance":
                    "info",

                "column":
                    str(column),

                "statistics": {

                    "mean":
                        round(
                            float(mean_value),
                            4
                        ),

                    "median":
                        round(
                            float(median_value),
                            4
                        ),

                    "minimum":
                        round(
                            float(minimum),
                            4
                        ),

                    "maximum":
                        round(
                            float(maximum),
                            4
                        )
                },

                "message":
                    (
                        f"{column} ranges from "
                        f"{minimum:.2f} to "
                        f"{maximum:.2f}, with a mean "
                        f"of {mean_value:.2f} and median "
                        f"of {median_value:.2f}."
                    )
            })


            # ------------------------------------------------
            # SKEWNESS
            # ------------------------------------------------

            if len(
                series
            ) >= 3:

                skew = (
                    series.skew()
                )


                if pd.notna(
                    skew
                ):

                    if abs(
                        skew
                    ) >= 1:

                        direction = (
                            "right"
                            if skew > 0
                            else "left"
                        )


                        findings.append({

                            "type":
                                "skewness",

                            "importance":
                                "medium",

                            "column":
                                str(column),

                            "value":
                                round(
                                    float(skew),
                                    3
                                ),

                            "message":
                                (
                                    f"{column} is strongly "
                                    f"{direction}-skewed "
                                    f"(skewness "
                                    f"{skew:.2f})."
                                )
                        })


        # ====================================================
        # CATEGORICAL DOMINANCE
        # ====================================================

        categorical_columns = (
            df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool"
                ]
            )
            .columns
            .tolist()
        )


        for column in categorical_columns[:10]:

            series = (
                df[column]
                .dropna()
            )


            if series.empty:
                continue


            counts = (
                series
                .value_counts()
            )


            if counts.empty:
                continue


            dominant_value = (
                counts.index[0]
            )

            dominant_count = int(
                counts.iloc[0]
            )


            dominant_percentage = (
                dominant_count
                /
                len(series)
                *
                100
            )


            findings.append({

                "type":
                    "category_distribution",

                "importance":
                    (
                        "medium"
                        if dominant_percentage >= 70
                        else "info"
                    ),

                "column":
                    str(column),

                "dominant_value":
                    str(dominant_value),

                "percentage":
                    round(
                        dominant_percentage,
                        2
                    ),

                "message":
                    (
                        f"{dominant_value} is the most common "
                        f"value in {column}, representing "
                        f"{dominant_percentage:.2f}% "
                        f"of non-missing records."
                    )
            })


        # ====================================================
        # CORRELATIONS
        # ====================================================

        if len(
            numeric_columns
        ) >= 2:

            try:

                correlation_matrix = (
                    df[
                        numeric_columns
                    ]
                    .corr(
                        numeric_only=True
                    )
                )


                correlation_pairs = []


                for i, column_a in enumerate(
                    correlation_matrix.columns
                ):

                    for column_b in (
                        correlation_matrix.columns[
                            i + 1:
                        ]
                    ):

                        value = (
                            correlation_matrix
                            .loc[
                                column_a,
                                column_b
                            ]
                        )


                        if pd.isna(
                            value
                        ):
                            continue


                        correlation_pairs.append(
                            (
                                abs(
                                    float(value)
                                ),
                                float(value),
                                str(column_a),
                                str(column_b)
                            )
                        )


                correlation_pairs.sort(
                    reverse=True
                )


                for (
                    absolute_correlation,
                    correlation,
                    column_a,
                    column_b
                ) in correlation_pairs[:5]:


                    if absolute_correlation < 0.5:
                        continue


                    strength = (

                        "strong"

                        if absolute_correlation >= 0.7

                        else "moderate"
                    )


                    direction = (

                        "positive"

                        if correlation > 0

                        else "negative"
                    )


                    findings.append({

                        "type":
                            "correlation",

                        "importance":
                            (
                                "high"
                                if absolute_correlation >= 0.7
                                else "medium"
                            ),

                        "columns": [
                            column_a,
                            column_b
                        ],

                        "value":
                            round(
                                correlation,
                                3
                            ),

                        "message":
                            (
                                f"{column_a} and {column_b} "
                                f"have a {strength} "
                                f"{direction} correlation "
                                f"(r = {correlation:.2f})."
                            )
                    })


            except Exception:
                pass


        # ====================================================
        # QUALITY
        # ====================================================

        if isinstance(
            quality_report,
            dict
        ):

            score = None


            for key in [
                "overall_score",
                "quality_score",
                "score"
            ]:

                if (
                    quality_report.get(
                        key
                    )
                    is not None
                ):

                    try:

                        score = float(
                            quality_report[
                                key
                            ]
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        score = None

                    break


            if score is not None:

                findings.append({

                    "type":
                        "quality",

                    "importance":
                        (
                            "high"
                            if score < 60
                            else
                            "medium"
                            if score < 80
                            else
                            "low"
                        ),

                    "value":
                        round(
                            score,
                            2
                        ),

                    "message":
                        (
                            f"The prepared dataset has an "
                            f"overall quality score of "
                            f"{score:.2f}/100."
                        )
                })


        return findings


    # ========================================================
    # QUALITY SCORE
    # ========================================================

    @staticmethod
    def _resolve_quality_score(
        quality_report
    ):

        if not isinstance(
            quality_report,
            dict
        ):
            return None


        possible_keys = [

            "overall_score",

            "quality_score",

            "score"
        ]


        # ----------------------------------------------------
        # TOP LEVEL
        # ----------------------------------------------------

        for key in possible_keys:

            value = (
                quality_report.get(
                    key
                )
            )


            if value is not None:

                try:

                    return round(
                        float(
                            value
                        ),
                        2
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    pass


        # ----------------------------------------------------
        # NESTED QUALITY OBJECT
        # ----------------------------------------------------

        quality = (
            quality_report.get(
                "quality"
            )
        )


        if isinstance(
            quality,
            dict
        ):

            for key in possible_keys:

                value = (
                    quality.get(
                        key
                    )
                )


                if value is not None:

                    try:

                        return round(
                            float(
                                value
                            ),
                            2
                        )

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass


        # ----------------------------------------------------
        # LEGACY FALLBACK
        # ----------------------------------------------------

        try:

            score = (
                calculate_quality_score(
                    quality_report
                )
            )


            if score is not None:

                return round(
                    float(
                        score
                    ),
                    2
                )


        except Exception:
            pass


        return None


    # ========================================================
    # QUALITY COMPONENTS
    # ========================================================

    @staticmethod
    def _extract_quality_components(
        quality_report
    ):

        if not isinstance(
            quality_report,
            dict
        ):
            return {}


        components = (
            quality_report.get(
                "components"
            )
        )


        if isinstance(
            components,
            dict
        ):
            return components


        components = (
            quality_report.get(
                "component_scores"
            )
        )


        if isinstance(
            components,
            dict
        ):
            return components


        component_names = [

            "completeness",

            "uniqueness",

            "validity",

            "consistency",

            "anomaly_quality"
        ]


        extracted = {}


        for component in component_names:

            if component in quality_report:

                extracted[
                    component
                ] = (
                    quality_report[
                        component
                    ]
                )


        return extracted


    # ========================================================
    # NORMALIZE CLEANING RESULT
    # ========================================================

    @staticmethod
    def _normalize_cleaning_result(
        cleaning_result
    ):

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        if isinstance(
            cleaning_result,
            pd.DataFrame
        ):

            return (
                cleaning_result,
                [],
                {}
            )


        # ----------------------------------------------------
        # TUPLE
        # ----------------------------------------------------

        if isinstance(
            cleaning_result,
            tuple
        ):


            if len(
                cleaning_result
            ) == 2:

                cleaned_df = (
                    cleaning_result[0]
                )

                cleaning_log = (
                    cleaning_result[1]
                )


                return (

                    cleaned_df,

                    (
                        cleaning_log
                        if isinstance(
                            cleaning_log,
                            list
                        )
                        else []
                    ),

                    {}
                )


            if len(
                cleaning_result
            ) >= 3:

                cleaned_df = (
                    cleaning_result[0]
                )

                cleaning_log = (
                    cleaning_result[1]
                )

                cleaning_summary = (
                    cleaning_result[2]
                )


                return (

                    cleaned_df,

                    (
                        cleaning_log
                        if isinstance(
                            cleaning_log,
                            list
                        )
                        else []
                    ),

                    (
                        cleaning_summary
                        if isinstance(
                            cleaning_summary,
                            dict
                        )
                        else {}
                    )
                )


        # ----------------------------------------------------
        # DICTIONARY
        # ----------------------------------------------------

        if isinstance(
            cleaning_result,
            dict
        ):

            cleaned_df = (
                cleaning_result.get(
                    "cleaned_df"
                )
            )


            if cleaned_df is None:

                cleaned_df = (
                    cleaning_result.get(
                        "dataframe"
                    )
                )


            cleaning_log = (
                cleaning_result.get(
                    "cleaning_log",
                    cleaning_result.get(
                        "operations",
                        []
                    )
                )
            )


            cleaning_summary = (
                cleaning_result.get(
                    "cleaning_summary",
                    cleaning_result.get(
                        "summary",
                        {}
                    )
                )
            )


            return (

                cleaned_df,

                (
                    cleaning_log
                    if isinstance(
                        cleaning_log,
                        list
                    )
                    else []
                ),

                (
                    cleaning_summary
                    if isinstance(
                        cleaning_summary,
                        dict
                    )
                    else {}
                )
            )


        raise TypeError(
            "Unsupported cleaning agent response. "
            "Expected DataFrame, tuple or dictionary."
        )

# ============================================================
# UNAVAILABLE LLM ADAPTER
# ============================================================

class _UnavailableLLMService:
    """
    Small adapter used when Gemini cannot even be initialized.

    SQLAgent and InsightAgent already contain deterministic
    fallback logic. They expect an object with a generate()
    method, so this adapter intentionally raises a predictable
    error and allows those fallbacks to activate.

    This means:

        Gemini available
            -> Gemini SQL + Gemini insight

        Gemini unavailable
            -> local SQL + local insight
    """

    available = False

    last_error = (
        "LLM service could not be initialized."
    )


    def generate(
        self,
        prompt
    ):

        raise RuntimeError(
            "LLM_UNAVAILABLE: External language "
            "model service is unavailable."
        )


    def get_status(
        self
    ):

        return {

            "model":
                None,

            "available":
                False,

            "last_error":
                self.last_error
        }

# ============================================================
# SHARED INSTANCE
# ============================================================

analytics_workflow = (
    AnalyticsWorkflow(
        max_charts=10
    )
)