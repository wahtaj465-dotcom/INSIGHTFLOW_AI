from typing import Any, Dict

from backend.orchestration.graph import (
    agent_graph,
)

from backend.services.dataset_manager import (
    dataset_manager,
)


class AgentService:
    """
    Application service connecting the FastAPI layer
    to the InsightFlow agentic orchestration graph.

    Responsibilities:

    - validate dataset_id
    - validate question
    - verify prepared dataset exists
    - construct initial AgentState
    - invoke LangGraph
    - normalize the final agent response

    Dataset preparation remains the responsibility of
    AnalyticsWorkflow.
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        max_replans: int = 2,
    ):

        self.max_retries = max_retries
        self.max_replans = max_replans

    # ========================================================
    # RUN AGENT
    # ========================================================

    def run(
        self,
        dataset_id: str,
        question: str,
    ) -> Dict[str, Any]:

        dataset_id = self._validate_dataset_id(
            dataset_id
        )

        question = self._validate_question(
            question
        )

        # ----------------------------------------------------
        # Dataset must already have passed through the
        # deterministic preparation workflow.
        # ----------------------------------------------------

        if not dataset_manager.exists(
            dataset_id
        ):

            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "error": "Dataset session not found.",
            }

        # ----------------------------------------------------
        # Initial state
        #
        # graph.initialize_node() initializes the remaining
        # orchestration fields.
        # ----------------------------------------------------

        initial_state = {

            "dataset_id":
                dataset_id,

            "question":
                question,

            "max_retries":
                self.max_retries,

            "max_replans":
                self.max_replans,

            "trace":
                [],
        }

        # ----------------------------------------------------
        # Invoke autonomous agent
        # ----------------------------------------------------

        try:

            result = agent_graph.invoke(
                initial_state
            )

        except Exception as error:

            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "error": str(error),
            }

        # ----------------------------------------------------
        # Validate graph response
        # ----------------------------------------------------

        if not isinstance(
            result,
            dict
        ):

            return {
                "success": False,
                "dataset_id": dataset_id,
                "question": question,
                "error": (
                    "Agent graph returned an invalid response."
                ),
            }

        return self._normalize_result(
            result=result,
            dataset_id=dataset_id,
            question=question,
        )

    # ========================================================
    # NORMALIZE GRAPH RESULT
    # ========================================================

    @staticmethod
    def _normalize_result(
        *,
        result: Dict[str, Any],
        dataset_id: str,
        question: str,
    ) -> Dict[str, Any]:

        final_response = result.get(
            "final_response"
        )

        # finish_node() should produce final_response.
        #
        # Keeping the fallback makes this service robust if
        # execution terminates with state but without a
        # final_response.

        if not isinstance(
            final_response,
            dict
        ):

            final_response = {}

        error = (
            final_response.get("error")
            or
            result.get("error")
        )

        success = final_response.get(
            "success"
        )

        if success is None:

            success = (
                result.get(
                    "completed",
                    False
                )
                and
                error is None
            )

        # ----------------------------------------------------
        # Generic dynamic outputs
        # ----------------------------------------------------

        tool_outputs = (
            final_response.get(
                "tool_outputs"
            )
        )

        if not isinstance(
            tool_outputs,
            dict
        ):

            tool_outputs = result.get(
                "tool_outputs",
                {}
            )

        if not isinstance(
            tool_outputs,
            dict
        ):

            tool_outputs = {}

        # ----------------------------------------------------
        # Raw tool results
        # ----------------------------------------------------

        tool_results = (
            final_response.get(
                "tool_results"
            )
        )

        if not isinstance(
            tool_results,
            dict
        ):

            tool_results = result.get(
                "tool_results",
                {}
            )

        if not isinstance(
            tool_results,
            dict
        ):

            tool_results = {}

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        trace = result.get(
            "trace",
            []
        )

        if not isinstance(
            trace,
            list
        ):

            trace = []

        # ----------------------------------------------------
        # Final normalized service response
        # ----------------------------------------------------

        return {

            "success":
                bool(success),

            "completed":
                bool(
                    result.get(
                        "completed",
                        False
                    )
                ),

            "dataset_id":
                final_response.get(
                    "dataset_id",
                    dataset_id
                ),

            "question":
                final_response.get(
                    "question",
                    question
                ),

            # =================================================
            # PLANNER
            # =================================================

            "intent":
                final_response.get(
                    "intent",
                    result.get("intent")
                ),

            "plan":
                final_response.get(
                    "plan",
                    result.get(
                        "plan",
                        []
                    )
                ),

            "plan_reasoning":
                final_response.get(
                    "plan_reasoning",
                    result.get(
                        "plan_reasoning"
                    )
                ),

            "planner_source":
                final_response.get(
                    "planner_source",
                    result.get(
                        "planner_source"
                    )
                ),

            "planner_error":
                final_response.get(
                    "planner_error",
                    result.get(
                        "planner_error"
                    )
                ),

            # =================================================
            # EXECUTION
            # =================================================

            "executed_tools":
                final_response.get(
                    "executed_tools",
                    result.get(
                        "executed_tools",
                        []
                    )
                ),

            "tool_results":
                tool_results,

            "tool_outputs":
                tool_outputs,

            # =================================================
            # STANDARD ANALYTICS OUTPUTS
            # =================================================

            "generated_sql":
                final_response.get(
                    "generated_sql",
                    result.get(
                        "generated_sql"
                    )
                ),

            "sql_result":
                final_response.get(
                    "sql_result",
                    result.get(
                        "sql_result"
                    )
                ),

            "visualization":
                final_response.get(
                    "visualization",
                    result.get(
                        "visualization"
                    )
                ),

            "statistical_findings":
                final_response.get(
                    "statistical_findings",
                    result.get(
                        "statistical_findings",
                        []
                    )
                ),

            "insight":
                final_response.get(
                    "insight",
                    result.get(
                        "insight"
                    )
                ),

            # =================================================
            # RECOVERY
            # =================================================

            "retry_count":
                final_response.get(
                    "retry_count",
                    result.get(
                        "retry_count",
                        0
                    )
                ),

            "max_retries":
                final_response.get(
                    "max_retries",
                    result.get(
                        "max_retries",
                        2
                    )
                ),

            "replan_count":
                final_response.get(
                    "replan_count",
                    result.get(
                        "replan_count",
                        0
                    )
                ),

            "max_replans":
                final_response.get(
                    "max_replans",
                    result.get(
                        "max_replans",
                        2
                    )
                ),

            "failed_tool":
                final_response.get(
                    "failed_tool",
                    result.get(
                        "failed_tool"
                    )
                ),

            "last_tool_error":
                final_response.get(
                    "last_tool_error",
                    result.get(
                        "last_tool_error"
                    )
                ),

            # =================================================
            # STATUS
            # =================================================

            "error":
                error,

            "trace":
                trace,
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_dataset_id(
        dataset_id: str
    ) -> str:

        if not isinstance(
            dataset_id,
            str
        ):

            raise TypeError(
                "dataset_id must be a string."
            )

        dataset_id = dataset_id.strip()

        if not dataset_id:

            raise ValueError(
                "dataset_id cannot be empty."
            )

        return dataset_id

    @staticmethod
    def _validate_question(
        question: str
    ) -> str:

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "question must be a string."
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "question cannot be empty."
            )

        return question


# ============================================================
# SHARED INSTANCE
# ============================================================

agent_service = AgentService()