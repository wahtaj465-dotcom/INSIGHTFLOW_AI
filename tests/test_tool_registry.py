from backend.orchestration.tool_registry import (
    get_available_tools,
    get_tool,
    get_tool_definition,
    get_tool_descriptions,
)


# ============================================================
# TEST AVAILABLE TOOLS
# ============================================================

def test_available_tools():

    tools = get_available_tools()

    assert "dataset_context" in tools
    assert "sql" in tools
    assert "visualization" in tools
    assert "insight" in tools


# ============================================================
# TEST EXECUTABLE TOOL LOOKUP
# ============================================================

def test_get_tool():

    tool = get_tool(
        "sql"
    )

    assert tool is not None

    assert hasattr(
        tool,
        "invoke"
    )


# ============================================================
# TEST UNKNOWN TOOL
# ============================================================

def test_unknown_tool():

    tool = get_tool(
        "does_not_exist"
    )

    assert tool is None


# ============================================================
# TEST COMPLETE TOOL DEFINITION
# ============================================================

def test_tool_definition():

    definition = get_tool_definition(
        "sql"
    )

    assert definition is not None

    assert "tool" in definition
    assert "description" in definition
    assert "inputs" in definition
    assert "outputs" in definition
    assert "dependencies" in definition


# ============================================================
# TEST SQL INPUT / OUTPUT CONTRACT
# ============================================================

def test_sql_tool_contract():

    definition = get_tool_definition(
        "sql"
    )

    assert definition["inputs"] == {
        "dataset_id": "dataset_id",
        "question": "question",
    }

    assert definition["outputs"] == {
        "generated_sql": "generated_sql",
        "result": "sql_result",
    }


# ============================================================
# TEST PLANNER-SAFE DESCRIPTIONS
# ============================================================

def test_tool_descriptions():

    descriptions = (
        get_tool_descriptions()
    )

    assert "sql" in descriptions

    sql = descriptions[
        "sql"
    ]

    assert "description" in sql
    assert "inputs" in sql
    assert "outputs" in sql
    assert "dependencies" in sql

    # Planner metadata must not expose
    # executable Python objects.
    assert "tool" not in sql


# ============================================================
# TEST ALL TOOLS HAVE METADATA
# ============================================================

def test_all_tools_have_metadata():

    descriptions = (
        get_tool_descriptions()
    )

    for (
        tool_name,
        metadata
    ) in descriptions.items():

        assert metadata[
            "description"
        ]

        assert isinstance(
            metadata["inputs"],
            dict
        )

        assert isinstance(
            metadata["outputs"],
            dict
        )

        assert isinstance(
            metadata["dependencies"],
            list
        )