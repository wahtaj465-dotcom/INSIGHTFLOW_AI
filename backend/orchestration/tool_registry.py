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

    "dataset_context": {

        "tool":
            get_dataset_context,

        "description": (
            "Inspect the uploaded dataset and return "
            "information about its structure, columns, "
            "data types, shape, and dataset context."
        ),

        "inputs": {
            "dataset_id":
                "dataset_id",
        },

        "outputs": {},
    },


    "sql": {

        "tool":
            run_sql_analysis,

        "description": (
            "Generate and execute SQL for analytical "
            "questions about the uploaded dataset."
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
    },


    "visualization": {

        "tool":
            generate_visualization,

        "description": (
            "Generate an appropriate visualization "
            "for the user's analytical question."
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
    },


    "insight": {

        "tool":
            generate_analytical_insight,

        "description": (
            "Generate a natural-language analytical "
            "insight from the dataset and previous "
            "analysis results."
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
    },
}


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
# GET TOOL
# ============================================================

def get_tool(
    tool_name: str
):
    """
    Return the executable LangChain tool.
    """

    definition = (
        get_tool_definition(
            tool_name
        )
    )

    if definition is None:
        return None

    return definition[
        "tool"
    ]


# ============================================================
# GET AVAILABLE TOOLS
# ============================================================

def get_available_tools():
    """
    Return the names of all tools available
    to the orchestrator.
    """

    return list(
        TOOL_REGISTRY.keys()
    )


# ============================================================
# GET TOOL CATALOG
# ============================================================

def get_tool_catalog():
    """
    Return tool descriptions for planning.

    Executable tool objects are intentionally
    excluded from the catalog.
    """

    catalog = {}

    for (
        tool_name,
        definition
    ) in TOOL_REGISTRY.items():

        catalog[
            tool_name
        ] = {
            "description":
                definition[
                    "description"
                ],

            "inputs":
                list(
                    definition[
                        "inputs"
                    ].keys()
                ),
        }

    return catalog