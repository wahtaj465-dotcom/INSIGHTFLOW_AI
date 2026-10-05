"""

InsightFlow AI - Executive Stakeholder Report Generator


Creates a professional PDF from the already completed analytical response.

The analytical agent is NOT executed again.


The generated PDF is:

1. Saved permanently under reports/

2. Returned as bytes for Streamlit download

3. Written for business stakeholders rather than technical users

"""


from __future__ import annotations


import io

import re

from datetime import datetime

from html import escape

from pathlib import Path

from typing import Any, Iterable


import pandas as pd


from reportlab.lib import colors

from reportlab.lib.enums import TA_CENTER, TA_LEFT

from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from reportlab.lib.units import mm

from reportlab.platypus import (

    Image,

    KeepTogether,

    Paragraph,

    SimpleDocTemplate,

    Spacer,

    Table,

    TableStyle,

)


try:

    import matplotlib.pyplot as plt

except ImportError:  # pragma: no cover

    plt = None
# ---------------------------------------------------------------------# Helpers# ---------------------------------------------------------------------
def _first_non_empty(mapping: dict, *keys: str, default=None):
    """Return the first meaningful value without truth-testing DataFrames."""
    for key in keys:
        value = mapping.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, (list, tuple, dict, set)) and not value:
            continue

        # DataFrames and other array-like objects must not be evaluated
        # directly in a boolean expression.
        return value

    return default


def _to_dataframe(result: Any) -> pd.DataFrame | None:

    if result is None:

        return None


    if isinstance(result, pd.DataFrame):

        return result.copy()


    if isinstance(result, list):

        if not result:

            return pd.DataFrame()

        try:

            return pd.DataFrame(result)

        except Exception:

            return None


    if isinstance(result, dict):
# Some APIs return {"rows": [...]}.
        rows = _first_non_empty(result, "rows", "data", "results")

        if isinstance(rows, list):

            try:

                return pd.DataFrame(rows)

            except Exception:

                return None


        try:

            return pd.DataFrame([result])

        except Exception:

            return None


    return None


def _get_result(response: dict) -> Any:

    result = _first_non_empty(

        response,

        "sql_result",

        "result",

        "query_result",

        "results",

    )


    if result is not None:

        return result


    analysis = response.get("analysis")

    if isinstance(analysis, dict):

        return _first_non_empty(

            analysis,

            "sql_result",

            "result",

            "query_result",

            "results",

        )


    return None


def _get_insight(response: dict) -> Any:

    insight = _first_non_empty(

        response,

        "insight",

        "generated_insight",

    )


    if insight is not None:

        return insight


    analysis = response.get("analysis")

    if isinstance(analysis, dict):

        return _first_non_empty(

            analysis,

            "insight",

            "generated_insight",

        )


    return None


def _plain_text(value: Any) -> str:

    if value is None:

        return ""


    if isinstance(value, str):

        return value.strip()


    return str(value).strip()


# ---------------------------------------------------------------------
# LLM insight cleanup / presentation helpers
# ---------------------------------------------------------------------

_MAX_SUMMARY_WORDS = 70
_MAX_FINDING_WORDS = 32
_MAX_RECOMMENDATION_WORDS = 32

# Headings that commonly contain useful executive-level content.
_FINDING_HEADINGS = (
    "key findings",
    "key insights",
    "key business insights",
    "key ml anomaly detection summary",
    "key numerical drivers",
    "key drivers",
    "main findings",
    "analysis findings",
    "important findings",
    "business insights",
)

_RECOMMENDATION_HEADINGS = (
    "recommended actions",
    "recommendations",
    "recommended next steps",
    "next steps",
    "further steps",
    "actions",
)

# These sections are evidence/technical detail, not executive narrative.
_NOISE_HEADINGS = (
    "top flagged records",
    "top flagged records sample",
    "raw records",
    "raw data",
    "detailed records",
    "record sample",
    "sample records",
    "technical details",
    "technical detail",
    "generated sql",
    "sql query",
    "sql",
    "methodology",
)

def _normalize_llm_markdown(text: str) -> str:
    """Turn free-form LLM markdown into plain, predictable text."""
    if not text:
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Remove fenced-code wrappers.
    text = re.sub(r"```(?:markdown|md|text)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Normalize escaped markdown characters sometimes returned by an LLM.
    text = text.replace(r"\**", "**")
    text = text.replace(r"\*", "*")
    text = text.replace(r"\_", "_")

    # Turn explicit inline Markdown headings into real line boundaries.
    text = re.sub(r"\s+(?=#{1,6}\s*)", "\n", text)

    # Remove markdown heading markers, while retaining the heading text.
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text, flags=re.DOTALL)

    # Remove markdown links but preserve their visible label.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Keep table content out of prose. The analytical result already has
    # a proper PDF table when structured result data is available.
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if stripped.startswith("|"):
            continue
        if re.fullmatch(r"[-|:\s]+", stripped):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)

    # Remove repeated whitespace while preserving paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalize_heading(value: str) -> str:
    value = _normalize_llm_markdown(value)
    value = re.sub(r"[:\-–—]+$", "", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def _is_heading(value: str) -> bool:
    normalized = _normalize_heading(value)
    if not normalized:
        return False

    candidates = (
        _FINDING_HEADINGS
        + _RECOMMENDATION_HEADINGS
        + _NOISE_HEADINGS
        + (
            "executive summary",
            "summary",
            "overview",
            "key business metrics",
            "data quality",
        )
    )
    return normalized in candidates


def _split_llm_sections(text: str) -> dict[str, str]:
    """
    Split free-form LLM text into lightweight sections.

    This is intentionally presentation-only: it does not ask the LLM to
    produce a new answer and does not change the underlying analysis.
    """
    if not text:
        return {}

    # Work from the original text so explicit Markdown headings remain
    # detectable. Cleanup of the actual section body happens afterwards.
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"```(?:markdown|md|text)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")
    raw = raw.replace(r"\**", "**").replace(r"\*", "*").replace(r"\_", "_")

    # Treat horizontal separators as section boundaries.
    raw = re.sub(r"\s*---+\s*", "\n", raw)

    # Insert boundaries before common heading phrases even when the LLM
    # omitted Markdown "#" markers.
    heading_names = sorted(
        set(
            _FINDING_HEADINGS
            + _RECOMMENDATION_HEADINGS
            + _NOISE_HEADINGS
            + (
                "executive summary",
                "key business metrics",
                "data quality",
                "methodology",
            )
        ),
        key=len,
        reverse=True,
    )

    for heading in heading_names:
        pattern = re.compile(
            rf"(?i)(?<!^)(?<!\n)\s*(?:#+\s*)?({re.escape(heading)})(?=\s|:|$)"
        )
        raw = pattern.sub(lambda m: "\n" + m.group(1), raw)

    # Explicit Markdown headings are always section boundaries.
    raw = re.sub(r"(?m)^\s*#{1,6}\s*", "", raw)

    lines = raw.splitlines()
    sections: dict[str, list[str]] = {}
    current = "__preamble__"
    sections[current] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        candidate = re.sub(r"[:\-–—]+$", "", line).strip()
        normalized_candidate = _normalize_heading(candidate)

        if _is_heading(normalized_candidate):
            current = normalized_candidate
            sections.setdefault(current, [])
            continue

        # If the line starts with a known heading plus content, split it.
        matched_heading = None
        for heading in heading_names:
            if re.match(
                rf"(?i)^{re.escape(heading)}(?:\s|:|$)",
                line,
            ):
                matched_heading = heading
                remainder = re.sub(
                    rf"(?i)^{re.escape(heading)}\s*:?\s*",
                    "",
                    line,
                ).strip()
                current = heading
                sections.setdefault(current, [])
                if remainder:
                    sections[current].append(remainder)
                break

        if matched_heading:
            continue

        sections.setdefault(current, []).append(line)

    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
        if value
    }


def _sentence_list(text: str) -> list[str]:
    """Convert prose/bullets into short, clean sentences."""
    text = _normalize_llm_markdown(text)
    if not text:
        return []

    # Remove obvious evidence/table remnants.
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (
            line.startswith("|")
            or "|" in line
            or re.fullmatch(r"[-|:\s]+", line)
        ):
            continue

        line = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line)
        if line:
            clean_lines.append(line)

    text = " ".join(clean_lines)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Sentence boundaries plus semicolon boundaries work well for LLM
    # analytical prose without aggressively fragmenting decimal numbers.
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|;\s+(?=[A-Z])", text)

    output = []
    for piece in pieces:
        piece = piece.strip(" -•")
        piece = re.sub(r"\s+\d+\.$", ".", piece)
        piece = re.sub(r":\.", ".", piece)
        if piece:
            output.append(piece)

    return output


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    shortened = " ".join(words[:max_words]).rstrip(" ,;:-")
    if not shortened.endswith((".", "!", "?")):
        shortened += "."
    return shortened


def _dedupe_text(items: Iterable[str], limit: int) -> list[str]:
    output = []
    seen = set()

    for item in items:
        cleaned = _normalize_llm_markdown(item)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -•")
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

        if len(output) >= limit:
            break

    return output


def _extract_summary_from_free_text(text: str) -> str:
    """
    Extract a compact executive summary from an arbitrary LLM response.

    Important: this never invents content. It only selects and shortens
    content that already exists in the LLM response.
    """
    normalized = _normalize_llm_markdown(text)
    if not normalized:
        return ""

    sections = _split_llm_sections(normalized)

    # Prefer an explicit summary/overview section.
    preferred = (
        sections.get("executive summary"),
        sections.get("summary"),
        sections.get("overview"),
        sections.get("__preamble__"),
    )

    source = next((value for value in preferred if value), "")

    # If the response begins directly with a useful sentence and then
    # immediately enters a technical section, the preamble is ideal.
    if not source:
        for key, value in sections.items():
            if key not in _NOISE_HEADINGS and value:
                source = value
                break

    sentences = _sentence_list(source)

    # Drop sentences that are clearly raw-record/evidence dumps.
    useful = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(
            marker in lowered
            for marker in (
                "top flagged records",
                "row index",
                "anomaly score |",
                "restaurant id |",
                "features used:",
            )
        ):
            continue
        useful.append(sentence)

    if not useful:
        return ""

    summary = " ".join(useful[:2])
    return _limit_words(summary, _MAX_SUMMARY_WORDS)


def _extract_findings_from_free_text(text: str) -> list[str]:
    """Extract concise findings while ignoring raw-record sections."""
    normalized = _normalize_llm_markdown(text)
    if not normalized:
        return []

    sections = _split_llm_sections(normalized)
    findings = []

    for heading, content in sections.items():
        if heading == "__preamble__":
            # The preamble is used for the executive summary first. Only use
            # it as a finding when no explicit findings section exists.
            continue

        if heading in _NOISE_HEADINGS:
            continue

        if heading in _RECOMMENDATION_HEADINGS:
            continue

        if heading in (
            "executive summary",
            "summary",
            "overview",
            "key business metrics",
            "data quality",
            "methodology",
        ):
            continue

        if (
            heading in _FINDING_HEADINGS
            or "finding" in heading
            or "insight" in heading
            or "driver" in heading
            or "observation" in heading
        ):
            findings.extend(_sentence_list(content))

    # If no explicit finding section was found, use concise sentences from
    # the non-technical narrative after the preamble.
    if not findings:
        for heading, content in sections.items():
            if heading == "__preamble__" or heading in _NOISE_HEADINGS:
                continue
            if heading in _RECOMMENDATION_HEADINGS:
                continue
            findings.extend(_sentence_list(content))

    filtered_findings = []
    for item in findings:
        lowered = item.lower()
        if any(
            marker in lowered
            for marker in (
                "algorithm:",
                "rows analyzed:",
                "anomalous records flagged:",
                "features used:",
                "randomly selecting features",
                "isolates observations",
                "top flagged records",
                "row index",
                "anomaly score",
            )
        ):
            continue
        if re.fullmatch(r"\d+[.:]?", item.strip()):
            continue
        filtered_findings.append(item)

    findings = [
        _limit_words(item, _MAX_FINDING_WORDS)
        for item in filtered_findings
        if item
    ]

    return _dedupe_text(findings, 6)


def _extract_recommendations_from_free_text(text: str) -> list[str]:
    """Extract recommendations/further steps from free-form LLM text."""
    normalized = _normalize_llm_markdown(text)
    if not normalized:
        return []

    sections = _split_llm_sections(normalized)
    recommendations = []

    for heading, content in sections.items():
        if (
            heading in _RECOMMENDATION_HEADINGS
            or "recommend" in heading
            or heading == "further steps"
            or heading == "next steps"
        ):
            recommendations.extend(_sentence_list(content))

    recommendations = [
        _limit_words(item, _MAX_RECOMMENDATION_WORDS)
        for item in recommendations
        if item
    ]

    return _dedupe_text(recommendations, 5)


def _clean_report_text(value: Any, max_words: int | None = None) -> str:
    """Final safety-net cleaner before text enters ReportLab."""
    text = _normalize_llm_markdown(_plain_text(value))
    if not text:
        return ""

    # Prevent raw Markdown headings from ever reaching the PDF.
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_words:
        text = _limit_words(text, max_words)

    return text


def _extract_structured_insight(insight: Any) -> dict[str, Any]:
    """Normalize dict/string insight into presentation fields."""
    if isinstance(insight, dict):
        summary = _first_non_empty(
            insight,
            "executive_summary",
            "summary",
            "message",
        )

        findings = _extract_list(
            insight.get("key_insights")
            or insight.get("insights")
            or insight.get("key_findings")
        )

        recommendations = _extract_list(
            insight.get("recommendations")
            or insight.get("recommended_actions")
        )

        return {
            "summary": _clean_report_text(summary, _MAX_SUMMARY_WORDS),
            "findings": _dedupe_text(findings, 6),
            "recommendations": _dedupe_text(recommendations, 5),
        }

    if isinstance(insight, str):
        return {
            "summary": _extract_summary_from_free_text(insight),
            "findings": _extract_findings_from_free_text(insight),
            "recommendations": _extract_recommendations_from_free_text(insight),
        }

    return {
        "summary": "",
        "findings": [],
        "recommendations": [],
    }


def _extract_insight_text(insight: Any) -> str:
    """Backward-compatible summary accessor using the presentation parser."""
    return _extract_structured_insight(insight).get("summary", "")


def _extract_list(value: Any) -> list[str]:

    if value is None:

        return []


    if isinstance(value, str):

        return [value.strip()] if value.strip() else []


    if isinstance(value, (tuple, list)):

        output = []

        for item in value:

            if isinstance(item, dict):

                text = _first_non_empty(

                    item,

                    "message",

                    "text",

                    "finding",

                    "recommendation",

                )

                if text:

                    output.append(str(text).strip())

            elif item is not None:

                output.append(str(item).strip())

        return [x for x in output if x]


    return [str(value).strip()]


def _extract_report_value(markdown: str, label: str) -> str | None:

    if not markdown:

        return None


    pattern = re.compile(

        rf"{re.escape(label)}\s*[:|]\s*\*\*([^|\n]+)",

        re.IGNORECASE,

    )


    match = pattern.search(markdown)

    return match.group(1).strip() if match else None


def _extract_quality_info(markdown: str) -> dict[str, str]:

    info: dict[str, str] = {}


    if not markdown:

        return info


    patterns = {

        "records": r"Records analyzed\s*\|\s*\*\*([^|\n]+)",

        "fields": r"Fields analyzed\s*\|\s*\*\*([^|\n]+)",

        "quality": r"Data quality score\s*\|\s*\*\*([^|\n]+)",

        "missing": r"missing[^|]*\|\s*\*\*([^|\n]+)",

        "anomalies": r"ML-detected anomalies\s*\|\s*\*\*([^|\n]+)",

        "anomaly_rate": r"Anomaly rate\s*\|\s*\*\*([^|\n]+)",

    }


    for key, pattern in patterns.items():

        match = re.search(pattern, markdown, re.IGNORECASE)

        if match:

            info[key] = match.group(1).strip()
# Also support prose such as:# "The dataset contains 1187 records across 15 fields."
    if "records" not in info:

        match = re.search(

            r"contains\s+([\d,]+)\s+records\s+across\s+([\d,]+)\s+fields",

            markdown,

            re.IGNORECASE,

        )

        if match:

            info["records"] = match.group(1)

            info["fields"] = match.group(2)


    quality_match = re.search(

        r"overall data quality score[^0-9]*([0-9]+(?:\.[0-9]+)?/100)",

        markdown,

        re.IGNORECASE,

    )

    if quality_match and "quality" not in info:

        info["quality"] = quality_match.group(1)


    return info



def _extract_anomaly_info(text: str) -> dict[str, str]:
    """Extract anomaly count/rate from existing LLM prose when available."""
    info: dict[str, str] = {}
    if not text:
        return info

    normalized = _normalize_llm_markdown(text)

    count_match = re.search(
        r"\b([\d,]+)\s+anomalous\s+records\b",
        normalized,
        re.IGNORECASE,
    )
    if count_match:
        info["anomalies"] = count_match.group(1)

    rate_match = re.search(
        r"\b([\d]+(?:\.[\d]+)?)%\s+of\s+(?:the\s+)?dataset\b",
        normalized,
        re.IGNORECASE,
    )
    if rate_match:
        info["anomaly_rate"] = f"{rate_match.group(1)}%"

    return info


def _extract_recommendations(markdown: str) -> list[str]:

    if not markdown:

        return []


    match = re.search(

        r"##+\s*Recommended Actions\s*(.*?)(?=\n##+|\Z)",

        markdown,

        re.IGNORECASE | re.DOTALL,

    )


    if not match:

        return []


    section = match.group(1)

    recommendations = []


    for line in section.splitlines():

        line = line.strip()

        line = re.sub(r"^\d+\.\s*", "", line)

        line = re.sub(r"^[-*]\s*", "", line)


        if line:

            recommendations.append(line)


    return recommendations[:5]


def _get_findings(response: dict) -> list[str]:

    insight = _get_insight(response)

    findings: list[str] = []


    if isinstance(insight, dict):

        findings.extend(

            _extract_list(

                insight.get("key_insights")

                or insight.get("insights")

            )

        )


    statistical = response.get("statistical_findings")

    if isinstance(statistical, list):

        for item in statistical:

            if isinstance(item, dict):

                message = item.get("message")

                if message:

                    findings.append(str(message))

            elif item:

                findings.append(str(item))
# Remove duplicates while preserving order.
    unique = []

    seen = set()

    for finding in findings:

        normalized = finding.strip()

        if normalized and normalized.lower() not in seen:

            unique.append(normalized)

            seen.add(normalized.lower())


    return unique[:7]


def _format_number(value: Any) -> str:

    try:

        number = float(value)

    except (TypeError, ValueError):

        return str(value)


    if number.is_integer():

        return f"{int(number):,}"


    return f"{number:,.2f}"


def _build_chart(

    dataframe: pd.DataFrame,

    output_dir: Path,

) -> Path | None:

    """

    Create a simple evidence chart when the result is naturally

    visualizable: one categorical column + one numeric column.

    """

    if plt is None or dataframe is None or dataframe.empty:

        return None


    if len(dataframe.columns) < 2:

        return None


    numeric_columns = [

        column

        for column in dataframe.columns

        if pd.api.types.is_numeric_dtype(dataframe[column])

    ]


    categorical_columns = [

        column

        for column in dataframe.columns

        if column not in numeric_columns

    ]


    if not numeric_columns or not categorical_columns:

        return None


    category = categorical_columns[0]

    numeric = numeric_columns[0]


    chart_df = dataframe[[category, numeric]].dropna().head(10)

    if chart_df.empty:

        return None


    path = output_dir / "executive_chart.png"


    figure = plt.figure(figsize=(7.2, 3.5))

    axis = figure.add_subplot(111)


    labels = chart_df[category].astype(str).tolist()

    values = pd.to_numeric(

        chart_df[numeric],

        errors="coerce",

    ).tolist()


    axis.bar(labels, values)

    axis.set_xlabel(str(category).replace("_", " ").title())

    axis.set_ylabel(str(numeric).replace("_", " ").title())

    axis.set_title(

        f"{str(numeric).replace('_', ' ').title()} by "

        f"{str(category).replace('_', ' ').title()}"

    )


    axis.tick_params(axis="x", rotation=25)

    figure.tight_layout()

    figure.savefig(path, dpi=180, bbox_inches="tight")

    plt.close(figure)


    return path
# ---------------------------------------------------------------------# PDF generation# ---------------------------------------------------------------------
def build_executive_report(

    *,

    question: str,

    analysis_response: dict,

    report_markdown: str = "",

    dataset_id: str = "",

    output_dir: str | Path = "reports",

) -> tuple[bytes, str]:

    """

    Generate and permanently save a stakeholder-friendly PDF.


    Returns:

        (pdf_bytes, saved_pdf_path)

    """


    if not isinstance(analysis_response, dict):

        raise ValueError("analysis_response must be a dictionary.")


    if not question.strip():

        raise ValueError("Business question is required.")


    report_dir = Path(output_dir)

    report_dir.mkdir(parents=True, exist_ok=True)


    generated_at = datetime.now()

    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")


    temp_dir = report_dir / f".report_{timestamp}"

    temp_dir.mkdir(parents=True, exist_ok=True)


    saved_pdf = report_dir / (

        f"InsightFlow_Executive_Report_{timestamp}.pdf"

    )


    result = _get_result(analysis_response)

    dataframe = _to_dataframe(result)

    insight = _get_insight(analysis_response)

    # Normalize the LLM output at the presentation boundary. This keeps the
    # analytical agent untouched while preventing free-form Markdown,
    # technical sections, and raw-record dumps from entering the PDF.
    structured_insight = _extract_structured_insight(insight)
    summary = structured_insight["summary"]

    quality = _extract_quality_info(report_markdown)

    # If the report metadata does not already contain anomaly metrics,
    # recover the count/rate from the existing LLM insight text. This is
    # presentation-only and does not perform any new analysis.
    anomaly_info = _extract_anomaly_info(_plain_text(insight))
    for key, value in anomaly_info.items():
        quality.setdefault(key, value)

    if dataframe is not None and not dataframe.empty:

        if "records" not in quality:

            quality["records"] = str(len(dataframe))


    # Prefer structured findings already returned by the analytical
    # workflow. If the insight is free-form text, parse it locally into
    # concise report sections.
    findings = _get_findings(analysis_response)

    if isinstance(insight, str):
        parsed_findings = structured_insight["findings"]
        if parsed_findings:
            findings = parsed_findings

    findings = [
        _clean_report_text(item, _MAX_FINDING_WORDS)
        for item in findings
        if item
    ]
    findings = _dedupe_text(findings, 6)

    if not findings and summary:
        findings = [summary]

    recommendations = structured_insight["recommendations"]

    if not recommendations:
        recommendations = _extract_recommendations(report_markdown)

    recommendations = [
        _clean_report_text(item, _MAX_RECOMMENDATION_WORDS)
        for item in recommendations
        if item
    ]
    recommendations = _dedupe_text(recommendations, 5)


    if not recommendations:

        recommendations = [

            "Use the findings together with business context before making operational decisions."

        ]
# -----------------------------------------------------------------# Styles# -----------------------------------------------------------------
    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "IFTitle",

        parent=styles["Title"],

        fontName="Helvetica-Bold",

        fontSize=23,

        leading=28,

        alignment=TA_CENTER,

        spaceAfter=6,

    )


    subtitle = ParagraphStyle(

        "IFSubtitle",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        spaceAfter=18,

    )


    section = ParagraphStyle(

        "IFSection",

        parent=styles["Heading2"],

        fontName="Helvetica-Bold",

        fontSize=14,

        leading=18,

        spaceBefore=13,

        spaceAfter=8,

    )


    subheading = ParagraphStyle(

        "IFSubheading",

        parent=styles["Heading3"],

        fontName="Helvetica-Bold",

        fontSize=10.5,

        leading=14,

        spaceBefore=7,

        spaceAfter=5,

    )


    body = ParagraphStyle(

        "IFBody",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=9.5,

        leading=14,

        spaceAfter=7,

    )


    small = ParagraphStyle(

        "IFSmall",

        parent=body,

        fontSize=8,

        leading=11,

        textColor=colors.HexColor("#5B6472"),

    )


    bullet = ParagraphStyle(

        "IFBullet",

        parent=body,

        leftIndent=13,

        firstLineIndent=-7,

    )


    kpi_value = ParagraphStyle(

        "IFKpiValue",

        parent=body,

        fontName="Helvetica-Bold",

        fontSize=15,

        leading=18,

        alignment=TA_CENTER,

        spaceAfter=2,

    )


    kpi_label = ParagraphStyle(

        "IFKpiLabel",

        parent=body,

        fontSize=8,

        leading=10,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#5B6472"),

    )


    story = []
# -----------------------------------------------------------------# Cover / header# -----------------------------------------------------------------
    story.append(

        Paragraph(

            "INSIGHTFLOW AI",

            title,

        )

    )


    story.append(

        Paragraph(

            "Executive Business Analytics Report",

            subtitle,

        )

    )


    metadata = [

        [

            Paragraph("<b>Business Question</b>", body),

            Paragraph(escape(question), body),

        ],

        [

            Paragraph("<b>Generated</b>", body),

            Paragraph(

                generated_at.strftime("%d %B %Y, %H:%M"),

                body,

            ),

        ],

    ]


    if dataset_id:

        metadata.append(

            [

                Paragraph("<b>Dataset</b>", body),

                Paragraph(escape(dataset_id), small),

            ]

        )


    metadata_table = Table(

        metadata,

        colWidths=[42 * mm, 125 * mm],

    )


    metadata_table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F4F8")),

                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6DCE5")),

                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E6EC")),

                ("VALIGN", (0, 0), (-1, -1), "TOP"),

                ("LEFTPADDING", (0, 0), (-1, -1), 7),

                ("RIGHTPADDING", (0, 0), (-1, -1), 7),

                ("TOPPADDING", (0, 0), (-1, -1), 6),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),

            ]

        )

    )


    story.append(metadata_table)

    story.append(Spacer(1, 8))
# -----------------------------------------------------------------# Executive summary# -----------------------------------------------------------------
    story.append(Paragraph("Executive Summary", section))


    if summary:
        story.append(
            Paragraph(
                escape(_clean_report_text(summary, _MAX_SUMMARY_WORDS)),
                body,
            )
        )

    else:

        story.append(

            Paragraph(

                "The analysis was completed using the available dataset and analytical evidence.",

                body,

            )

        )
# -----------------------------------------------------------------# KPI cards# -----------------------------------------------------------------
    story.append(Paragraph("Key Business Metrics", section))


    records = quality.get("records", "N/A")

    fields = quality.get("fields", "N/A")

    quality_score = quality.get("quality", "N/A")


    if dataframe is not None and not dataframe.empty:

        result_rows = len(dataframe)

    else:

        result_rows = "N/A"


    kpi_data = [

        [

            Paragraph(str(records), kpi_value),

            Paragraph(str(fields), kpi_value),

            Paragraph(str(quality_score), kpi_value),

            Paragraph(str(result_rows), kpi_value),

        ],

        [

            Paragraph("Records", kpi_label),

            Paragraph("Fields", kpi_label),

            Paragraph("Data quality", kpi_label),

            Paragraph("Result rows", kpi_label),

        ],

    ]


    kpi_table = Table(

        kpi_data,

        colWidths=[42 * mm] * 4,

        rowHeights=[14 * mm, 8 * mm],

    )


    kpi_table.setStyle(

        TableStyle(

            [

                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6DCE5")),

                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("LEFTPADDING", (0, 0), (-1, -1), 5),

                ("RIGHTPADDING", (0, 0), (-1, -1), 5),

            ]

        )

    )


    story.append(kpi_table)
# -----------------------------------------------------------------# Main analytical result# -----------------------------------------------------------------
    if dataframe is not None and not dataframe.empty:

        story.append(Paragraph("Analytical Result", section))


        display_df = dataframe.head(10).copy()
# Format values for stakeholder readability.
        for column in display_df.columns:

            if pd.api.types.is_numeric_dtype(display_df[column]):

                display_df[column] = display_df[column].map(_format_number)


        header = [

            Paragraph(

                escape(str(column).replace("_", " ").title()),

                small,

            )

            for column in display_df.columns

        ]


        rows = [header]


        for _, row in display_df.iterrows():

            rows.append(

                [

                    Paragraph(escape(str(value)), small)

                    for value in row.tolist()

                ]

            )


        available_width = 168 * mm

        column_width = available_width / max(1, len(header))


        result_table = Table(

            rows,

            repeatRows=1,

            colWidths=[column_width] * len(header),

        )


        result_table.setStyle(

            TableStyle(

                [

                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),

                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6DCE5")),

                    ("VALIGN", (0, 0), (-1, -1), "TOP"),

                    ("LEFTPADDING", (0, 0), (-1, -1), 5),

                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),

                    ("TOPPADDING", (0, 0), (-1, -1), 5),

                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),

                ]

            )

        )


        story.append(result_table)
# Evidence chart, only when the result naturally supports one.
        chart_path = _build_chart(dataframe, temp_dir)


        if chart_path and chart_path.exists():

            story.append(Spacer(1, 8))

            story.append(

                Paragraph(

                    "Visual Evidence",

                    subheading,

                )

            )

            story.append(

                Image(

                    str(chart_path),

                    width=160 * mm,

                    height=78 * mm,

                )

            )
# -----------------------------------------------------------------# Key findings# -----------------------------------------------------------------
    story.append(Paragraph("Key Findings", section))


    if findings:

        for finding in findings:

            story.append(

                Paragraph(

                    f"• {escape(_clean_report_text(finding, _MAX_FINDING_WORDS))}",

                    bullet,

                )

            )

    else:

        story.append(

            Paragraph(

                "No additional findings were returned by the analytical workflow.",

                body,

            )

        )
# -----------------------------------------------------------------# Recommendations# -----------------------------------------------------------------
    story.append(Paragraph("Recommended Actions", section))


    for recommendation in recommendations[:5]:

        story.append(

            Paragraph(

                f"• {escape(_clean_report_text(recommendation, _MAX_RECOMMENDATION_WORDS))}",

                bullet,

            )

        )
# -----------------------------------------------------------------# Data quality# -----------------------------------------------------------------
    story.append(Paragraph("Data Quality", section))


    quality_rows = [

        [

            Paragraph("<b>Metric</b>", body),

            Paragraph("<b>Value</b>", body),

        ]

    ]


    quality_items = [

        ("Records analyzed", quality.get("records", "N/A")),

        ("Fields analyzed", quality.get("fields", "N/A")),

        ("Data quality score", quality.get("quality", "N/A")),

        ("Missing values", quality.get("missing", "N/A")),

        ("ML-detected anomalies", quality.get("anomalies", "N/A")),

        ("Anomaly rate", quality.get("anomaly_rate", "N/A")),

    ]


    for label, value in quality_items:

        quality_rows.append(

            [

                Paragraph(escape(label), body),

                Paragraph(escape(str(value)), body),

            ]

        )


    quality_table = Table(

        quality_rows,

        colWidths=[85 * mm, 83 * mm],

    )


    quality_table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),

                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6DCE5")),

                ("VALIGN", (0, 0), (-1, -1), "TOP"),

                ("LEFTPADDING", (0, 0), (-1, -1), 6),

                ("RIGHTPADDING", (0, 0), (-1, -1), 6),

                ("TOPPADDING", (0, 0), (-1, -1), 5),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),

            ]

        )

    )


    story.append(quality_table)
# -----------------------------------------------------------------# Methodology# -----------------------------------------------------------------
    story.append(Paragraph("Methodology", section))

    story.append(

        Paragraph(

            "This report summarizes the completed InsightFlow AI analytical workflow. "

            "The executive report uses the existing analysis response and does not "

            "re-run the analytical agent. Technical execution traces, raw SQL, "

            "planner details, and internal recovery logs are intentionally excluded "

            "from this stakeholder-facing document.",

            body,

        )

    )
# -----------------------------------------------------------------# Footer# -----------------------------------------------------------------
    def draw_footer(canvas, doc):

        canvas.saveState()

        canvas.setFont("Helvetica", 7.5)

        canvas.setFillColor(colors.HexColor("#6B7280"))

        canvas.drawString(

            18 * mm,

            10 * mm,

            "InsightFlow AI • Executive Business Analytics Report",

        )

        canvas.drawRightString(

            192 * mm,

            10 * mm,

            f"Page {doc.page}",

        )

        canvas.restoreState()


    document = SimpleDocTemplate(

        str(saved_pdf),

        pagesize=A4,

        rightMargin=21 * mm,

        leftMargin=21 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title="InsightFlow AI Executive Business Analytics Report",

        author="InsightFlow AI",

    )


    document.build(

        story,

        onFirstPage=draw_footer,

        onLaterPages=draw_footer,

    )


    pdf_bytes = saved_pdf.read_bytes()
# Clean up temporary chart assets.
    for item in temp_dir.iterdir():

        try:

            item.unlink()

        except OSError:

            pass


    try:

        temp_dir.rmdir()

    except OSError:

        pass


    return pdf_bytes, str(saved_pdf)