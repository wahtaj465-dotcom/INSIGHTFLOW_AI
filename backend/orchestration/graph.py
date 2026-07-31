from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from backend.orchestration.state import (
    AgentState,
)

from backend.orchestration.planner import (
    planner_node,
)

from backend.orchestration.replanner import (
    replanner_node,
)

from backend.orchestration.nodes.execution import (
    tool_executor_node,
)

from backend.orchestration.nodes.observer import (
    observer_node,
    route_after_observation,
)


# ============================================================
# INITIALIZE
# ============================================================

def initialize_node(
    state: AgentState
) -> dict:
    """
    Initialize runtime fields before planning begins.

    Dynamic tool outputs are stored inside tool_outputs.
    """

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    trace.append({
        "node":
            "initialize",

        "status":
            "success",
    })

    return {

        # ====================================================
        # PLANNING
        # ====================================================

        "intent":
            None,

        "plan":
            [],

        "plan_reasoning":
            None,

        "planner_source":
            None,

        "planner_error":
            None,

        # ====================================================
        # EXECUTION
        # ====================================================

        "current_step":
            None,

        "executed_tools":
            [],

        "tool_results":
            {},

        # ----------------------------------------------------
        # Dynamic registry-defined outputs
        # ----------------------------------------------------

        "tool_outputs":
            {},

        # ====================================================
        # ANALYTICS OUTPUTS
        # ====================================================

        "generated_sql":
            None,

        "sql_result":
            None,

        "visualization":
            None,

        "statistical_findings":
            [],

        "insight":
            None,

        # ====================================================
        # RETRY CONTROL
        # ====================================================

        "retry_count":
            0,

        "max_retries":
            state.get(
                "max_retries",
                2
            ),

        # ====================================================
        # REPLANNING / RECOVERY CONTROL
        # ====================================================

        "replan_count":
            0,

        "max_replans":
            state.get(
                "max_replans",
                2
            ),

        "failed_tool":
            None,

        "last_tool_error":
            None,

        # ====================================================
        # WORKFLOW STATE
        # ====================================================

        "error":
            None,

        "completed":
            False,

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        "final_response":
            None,

        # ====================================================
        # OBSERVABILITY
        # ====================================================

        "trace":
            trace,
    }


# ============================================================
# FINALIZE
# ============================================================

def finish_node(
    state: AgentState
) -> dict:
    """
    Build the final response returned by the agent.
    """

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    trace.append({
        "node":
            "finish",

        "status": (
            "error"
            if state.get(
                "error"
            )
            else "success"
        ),
    })

    final_response = {

        # ====================================================
        # STATUS
        # ====================================================

        "success": (
            state.get(
                "error"
            )
            is None
        ),

        "error":
            state.get(
                "error"
            ),

        # ====================================================
        # REQUEST
        # ====================================================

        "dataset_id":
            state.get(
                "dataset_id"
            ),

        "question":
            state.get(
                "question"
            ),

        # ====================================================
        # PLANNING
        # ====================================================

        "intent":
            state.get(
                "intent"
            ),

        "plan":
            state.get(
                "plan",
                []
            ),

        "plan_reasoning":
            state.get(
                "plan_reasoning"
            ),

        "planner_source":
            state.get(
                "planner_source"
            ),

        "planner_error":
            state.get(
                "planner_error"
            ),

        # ====================================================
        # TOOL EXECUTION
        # ====================================================

        "executed_tools":
            state.get(
                "executed_tools",
                []
            ),

        "tool_results":
            state.get(
                "tool_results",
                {}
            ),

        # ----------------------------------------------------
        # Generic outputs from all registry-defined tools
        # ----------------------------------------------------

        "tool_outputs":
            state.get(
                "tool_outputs",
                {}
            ),

        # ====================================================
        # EXISTING ANALYTICS OUTPUTS
        # ====================================================

        "generated_sql":
            state.get(
                "generated_sql"
            ),

        "sql_result":
            state.get(
                "sql_result"
            ),

        "visualization":
            state.get(
                "visualization"
            ),

        "statistical_findings":
            state.get(
                "statistical_findings",
                []
            ),

        "insight":
            state.get(
                "insight"
            ),

        # ====================================================
        # RETRY INFORMATION
        # ====================================================

        "retry_count":
            state.get(
                "retry_count",
                0
            ),

        "max_retries":
            state.get(
                "max_retries",
                2
            ),

        # ====================================================
        # REPLANNING INFORMATION
        # ====================================================

        "replan_count":
            state.get(
                "replan_count",
                0
            ),

        "max_replans":
            state.get(
                "max_replans",
                2
            ),

        "failed_tool":
            state.get(
                "failed_tool"
            ),

        "last_tool_error":
            state.get(
                "last_tool_error"
            ),
    }

    return {
        "completed":
            True,

        "final_response":
            final_response,

        "trace":
            trace,
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_agent_graph():
    """
    Build and compile the InsightFlow agent graph.

    Flow:

        START
          |
          v
      initialize
          |
          v
       planner
          |
          v
       executor
          |
          v
       observer
        /  |   \\
       /   |    \\
 execute replan finish
    |       |      |
    |       v      v
    |   replanner finish
    |       |
    +-------+
        |
        v
     executor
    """

    workflow = StateGraph(
        AgentState
    )

    # ========================================================
    # NODES
    # ========================================================

    workflow.add_node(
        "initialize",
        initialize_node
    )

    workflow.add_node(
        "planner",
        planner_node
    )

    workflow.add_node(
        "executor",
        tool_executor_node
    )

    workflow.add_node(
        "observer",
        observer_node
    )

    workflow.add_node(
        "replanner",
        replanner_node
    )

    workflow.add_node(
        "finish",
        finish_node
    )

    # ========================================================
    # ENTRY
    # ========================================================

    workflow.add_edge(
        START,
        "initialize"
    )

    workflow.add_edge(
        "initialize",
        "planner"
    )

    # ========================================================
    # INITIAL PLAN EXECUTION
    # ========================================================

    workflow.add_edge(
        "planner",
        "executor"
    )

    # ========================================================
    # EXECUTION -> OBSERVATION
    # ========================================================

    workflow.add_edge(
        "executor",
        "observer"
    )

    # ========================================================
    # OBSERVER ROUTING
    # ========================================================

    workflow.add_conditional_edges(
        "observer",

        route_after_observation,

        {
            "execute":
                "executor",

            "replan":
                "replanner",

            "finish":
                "finish",
        }
    )

    # ========================================================
    # REPLANNING LOOP
    # ========================================================

    workflow.add_edge(
        "replanner",
        "executor"
    )

    # ========================================================
    # END
    # ========================================================

    workflow.add_edge(
        "finish",
        END
    )

    return workflow.compile()


# ============================================================
# COMPILED AGENT
# ============================================================

agent_graph = (
    build_agent_graph()
)