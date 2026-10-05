from backend.orchestration.tools.dataset_tools import (
    get_dataset_context,
)

from backend.orchestration.tools.sql_tools import (
    run_sql_analysis,
)

from backend.orchestration.tools.visualization_tools import (
    generate_visualization,
)

from backend.orchestration.tools.insight_tools import (
    generate_analytical_insight,
)


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {

    # --------------------------------------------------------
    # DATASET CONTEXT
    # --------------------------------------------------------

    "dataset_context": {

        "tool":
            get_dataset_context,

        "description": (
            "Inspect the uploaded dataset and return useful "
            "context such as columns, schema, data types, "
            "shape, and dataset metadata."
        ),

        "inputs": {
            "dataset_id":
                "dataset_id",
        },

        "outputs": {},

        "dependencies": [],
    },


    # --------------------------------------------------------
    # SQL ANALYSIS
    # --------------------------------------------------------

    "sql": {

        "tool":
            run_sql_analysis,

        "description": (
            "Generate and execute SQL-style analytical queries "
            "against the uploaded dataset. Use this for "
            "filtering, aggregation, grouping, ranking, "
            "comparisons, totals, averages, and analytical "
            "data retrieval."
        ),

        "inputs": {
            "dataset_id":
                "dataset_id",

            "question":
                "question",
        },

        "outputs": {
            "generated_sql":
                "generated_sql",

            "result":
                "sql_result",
        },

        "dependencies": [],
    },


    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    "visualization": {

        "tool":
            generate_visualization,

        "description": (
            "Generate an appropriate visualization for the "
            "user's analytical request, such as bar charts, "
            "scatter plots, line charts, histograms, box "
            "plots, heatmaps, or other supported charts."
        ),

        "inputs": {
            "dataset_id":
                "dataset_id",

            "question":
                "question",

            "sql_result":
                "sql_result",
        },

        "outputs": {
            "chart":
                "visualization",
        },

        "dependencies": [],
    },


    # --------------------------------------------------------
    # ANALYTICAL INSIGHT
    # --------------------------------------------------------

    "insight": {

        "tool":
            generate_analytical_insight,

        "description": (
            "Interpret analytical results and generate a "
            "natural-language explanation of important "
            "patterns, relationships, findings, and business "
            "insights."
        ),

        "inputs": {
            "dataset_id":
                "dataset_id",

            "question":
                "question",

            "sql_result":
                "sql_result",

            "generated_sql":
                "generated_sql",
        },

        "outputs": {
            "insight":
                "insight",
        },

        "dependencies": [],
    },
}


# ============================================================
# GET TOOL
# ============================================================

def get_tool(
    tool_name: str
):
    """
    Return the executable LangChain tool.

    Kept for compatibility with existing code.
    """

    definition = TOOL_REGISTRY.get(
        tool_name
    )

    if definition is None:
        return None

    return definition.get(
        "tool"
    )


# ============================================================
# GET TOOL DEFINITION
# ============================================================

def get_tool_definition(
    tool_name: str
):
    """
    Return the complete metadata definition
    for a registered tool.
    """

    return TOOL_REGISTRY.get(
        tool_name
    )


# ============================================================
# GET AVAILABLE TOOL NAMES
# ============================================================

def get_available_tools():
    """
    Return names of all tools available
    to the agent.
    """

    return list(
        TOOL_REGISTRY.keys()
    )


# ============================================================
# GET TOOL DESCRIPTIONS
# ============================================================

def get_tool_descriptions():
    """
    Return tool descriptions for planner/replanner use.

    The executable tool objects are intentionally excluded.
    """

    descriptions = {}

    for (
        tool_name,
        definition
    ) in TOOL_REGISTRY.items():

        descriptions[
            tool_name
        ] = {
            "description":
                definition.get(
                    "description",
                    ""
                ),

            "inputs":
                definition.get(
                    "inputs",
                    {}
                ),

            "outputs":
                definition.get(
                    "outputs",
                    {}
                ),

            "dependencies":
                definition.get(
                    "dependencies",
                    []
                ),
        }

    return descriptions