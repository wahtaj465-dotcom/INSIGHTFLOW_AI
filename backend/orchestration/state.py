from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between InsightFlow LangGraph nodes.
    """

    # -----------------------------
    # Request
    # -----------------------------

    dataset_id: str
    question: str

    # -----------------------------
    # Understanding / planning
    # -----------------------------

    # Analytical intent identified by planner.
    # Examples:
    # aggregation, visualization,
    # relationship_analysis, dataset_context
    intent: Optional[str]

    # Ordered tools selected by planner.
    # Example:
    # ["sql", "visualization", "insight"]
    plan: List[str]

    # Explanation of why the planner
    # selected the current plan.
    plan_reasoning: Optional[str]

    # Indicates how the plan was produced.
    # Expected values:
    # "llm" or "fallback"
    planner_source: Optional[str]

    # Stores planner-specific failure information
    # without treating it as a workflow failure.
    planner_error: Optional[str]

    # Tool currently being processed.
    current_step: Optional[str]

    # -----------------------------
    # Tool execution
    # -----------------------------

    # Tools already executed during this run.
    executed_tools: List[str]

    # Raw outputs returned by individual tools.
    #
    # Example:
    # {
    #     "sql": {...},
    #     "visualization": {...}
    # }
    tool_results: Dict[str, Any]

    # -----------------------------
    # Analytics outputs
    # -----------------------------

    generated_sql: Optional[str]

    sql_result: Any

    visualization: Any

    statistical_findings: List[Any]

    insight: Optional[str]

    # -----------------------------
    # Agent control
    # -----------------------------

    # Number of retries/replanning attempts.
    retry_count: int

    # Maximum retries allowed.
    max_retries: int

    # Workflow/tool execution error.
    error: Optional[str]

    completed: bool

    replan_count: int
    max_replans: int

    failed_tool: Optional[str]
    last_tool_error: Optional[str]

    # -----------------------------
    # Observability
    # -----------------------------

    # Execution history across planner,
    # tools, observer, retries and finish.
    trace: List[Dict[str, Any]]

    # -----------------------------
    # Final response
    # -----------------------------

    final_response: Optional[Dict[str, Any]]