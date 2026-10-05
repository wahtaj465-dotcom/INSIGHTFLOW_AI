from frontend.utils.executive_report import build_executive_report


# ============================================================
# SAMPLE ANALYSIS RESPONSE
# ============================================================

analysis_response = {

    "success": True,

    "sql_result": [

        {
            "product": "Laptop Pro",
            "total_sales": 350000,
        },

        {
            "product": "Phone X",
            "total_sales": 285000,
        },

        {
            "product": "Tablet Air",
            "total_sales": 241000,
        },

        {
            "product": "Monitor 4K",
            "total_sales": 198000,
        },

        {
            "product": "Keyboard Max",
            "total_sales": 166000,
        },
    ],

    "insight": {

        "summary": (
            "Laptop Pro is the strongest-performing product, "
            "generating the highest sales among the products analyzed."
        ),

        "key_insights": [

            "Laptop Pro generated the highest sales.",

            "The top five products represent a significant "
            "share of overall sales.",

            "Laptop Pro substantially outperformed "
            "the fifth-ranked product.",
        ],

        "recommendations": [

            "Maintain sufficient inventory for Laptop Pro.",

            "Investigate the performance gap between "
            "leading and lower-ranked products.",

            "Use targeted promotions to improve "
            "lower-performing products.",
        ],
    },

    "statistical_findings": [

        {
            "type": "dataset_summary",

            "importance": "info",

            "message": (
                "The dataset contains 1187 records "
                "and 7 columns."
            ),
        },

        {
            "type": "missing_values",

            "importance": "medium",

            "message": (
                "3.3% of dataset cells contain "
                "missing values."
            ),
        },
    ],
}


# ============================================================
# SAMPLE REPORT METADATA
# ============================================================

report_markdown = """
## Business Health

| KPI | Value |
|---|---:|
| Records analyzed | 1187 |
| Fields analyzed | 7 |
| Data quality score | 97.83/100 |
| ML-detected anomalies | N/A |
| Anomaly rate | N/A |
"""


# ============================================================
# GENERATE PDF
# ============================================================

pdf_bytes, saved_path = build_executive_report(

    question=(
        "What are the top 5 products by sales?"
    ),

    analysis_response=analysis_response,

    report_markdown=report_markdown,

    dataset_id="local-test-dataset",

    output_dir="reports",
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("INSIGHTFLOW EXECUTIVE REPORT TEST")
print("=" * 60)

print()
print("PDF generated successfully!")

print()
print("Saved to:")
print(saved_path)

print()
print(
    f"PDF size: {len(pdf_bytes):,} bytes"
)

print()
print("=" * 60)