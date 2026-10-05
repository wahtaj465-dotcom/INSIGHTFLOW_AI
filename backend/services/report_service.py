from datetime import datetime
from typing import Any


class BusinessReportService:
    """
    Converts existing InsightFlow analytics results into
    a business-stakeholder-friendly executive report.

    This service does NOT perform analytics.
    It only formats already-generated analytical results.
    """

    # ========================================================
    # PUBLIC METHOD
    # ========================================================

    def generate_report(
        self,
        *,
        dataset_id: str,
        question: str,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a business-readable Markdown report.

        Existing analytical outputs are reused as-is.
        No existing agent functionality is modified.
        """

        response = response or {}

        analysis = (
            response.get("analysis")
            or {}
        )

        tool_outputs = (
            analysis.get("tool_outputs")
            or {}
        )

        dataset_info = (
            response.get("dataset")
            or {}
        )

        quality = (
            response.get("quality")
            or {}
        )

        anomalies = (
            response.get("anomalies")
            or {}
        )

        statistical_findings = (
            response.get(
                "statistical_findings"
            )
            or []
        )

        insight = (
            response.get("insight")
        )

        ml_anomaly = (
            tool_outputs.get(
                "ml_anomaly_result"
            )
            or {}
        )

        report = self._build_markdown(
            dataset_id=dataset_id,
            question=question,
            response=response,
            dataset_info=dataset_info,
            quality=quality,
            anomalies=anomalies,
            statistical_findings=(
                statistical_findings
            ),
            insight=insight,
            ml_anomaly=ml_anomaly,
        )

        return {
            "success": True,
            "dataset_id": dataset_id,
            "question": question,
            "generated_at": datetime.now().isoformat(),
            "format": "markdown",
            "report": report,
        }

    # ========================================================
    # MARKDOWN REPORT
    # ========================================================

    def _build_markdown(
        self,
        *,
        dataset_id: str,
        question: str,
        response: dict[str, Any],
        dataset_info: dict[str, Any],
        quality: dict[str, Any],
        anomalies: dict[str, Any],
        statistical_findings: list[Any],
        insight: Any,
        ml_anomaly: dict[str, Any],
    ) -> str:

        sections = []

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        sections.append(
            "# InsightFlow AI\n\n"
            "## Executive Business Analytics Report"
        )

        sections.append(
            f"**Generated:** "
            f"{datetime.now().strftime('%d %B %Y, %H:%M')}\n\n"
            f"**Dataset ID:** `{dataset_id}`\n\n"
            f"**Business Question:** {question}"
        )

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        sections.append(
            self._build_executive_summary(
                dataset_info=dataset_info,
                quality=quality,
                ml_anomaly=ml_anomaly,
                insight=insight,
            )
        )

        # ----------------------------------------------------
        # BUSINESS HEALTH
        # ----------------------------------------------------

        sections.append(
            self._build_business_health(
                dataset_info=dataset_info,
                quality=quality,
                ml_anomaly=ml_anomaly,
            )
        )

        # ----------------------------------------------------
        # KEY INSIGHTS
        # ----------------------------------------------------

        sections.append(
            self._build_key_insights(
                insight=insight,
                statistical_findings=(
                    statistical_findings
                ),
            )
        )

        # ----------------------------------------------------
        # ANOMALY / RISK ANALYSIS
        # ----------------------------------------------------

        sections.append(
            self._build_anomaly_section(
                anomalies=anomalies,
                ml_anomaly=ml_anomaly,
            )
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        sections.append(
            self._build_quality_section(
                quality=quality,
            )
        )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        sections.append(
            self._build_recommendations(
                insight=insight,
                ml_anomaly=ml_anomaly,
            )
        )

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        sections.append(
            "---\n\n"
            "*Generated automatically by InsightFlow AI.*"
        )

        return "\n\n".join(
            section
            for section in sections
            if section
        )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    def _build_executive_summary(
        self,
        *,
        dataset_info: dict[str, Any],
        quality: dict[str, Any],
        ml_anomaly: dict[str, Any],
        insight: Any,
    ) -> str:

        rows = (
            dataset_info.get("rows")
            or dataset_info.get("row_count")
            or "N/A"
        )

        columns = (
            dataset_info.get("column_count")
            or len(
                dataset_info.get(
                    "columns",
                    [],
                )
            )
            or "N/A"
        )

        quality_score = self._quality_score(
            quality
        )

        anomaly_count = ml_anomaly.get(
            "anomaly_count"
        )

        anomaly_percentage = ml_anomaly.get(
            "anomaly_percentage"
        )

        summary = (
            "## Executive Summary\n\n"
            f"The dataset contains **{rows} records** "
            f"across **{columns} fields**."
        )

        if quality_score is not None:

            summary += (
                f" The available data-quality assessment "
                f"indicates an overall score of "
                f"**{quality_score:.2f}/100**."
            )

        if anomaly_count is not None:

            summary += (
                f" Machine-learning analysis identified "
                f"**{anomaly_count} potentially anomalous "
                f"records**"
            )

            if anomaly_percentage is not None:

                summary += (
                    f" (**{anomaly_percentage:.2f}%** "
                    f"of the dataset)"
                )

            summary += "."

        if isinstance(
            insight,
            str,
        ) and insight.strip():

            summary += (
                "\n\nThe detailed analytical findings "
                "and business implications are presented "
                "in the sections below."
            )

        return summary

    # ========================================================
    # BUSINESS HEALTH
    # ========================================================

    def _build_business_health(
        self,
        *,
        dataset_info: dict[str, Any],
        quality: dict[str, Any],
        ml_anomaly: dict[str, Any],
    ) -> str:

        rows = (
            dataset_info.get("rows")
            or dataset_info.get("row_count")
            or "N/A"
        )

        columns = (
            dataset_info.get("column_count")
            or len(
                dataset_info.get(
                    "columns",
                    [],
                )
            )
            or "N/A"
        )

        quality_score = self._quality_score(
            quality
        )

        anomaly_count = ml_anomaly.get(
            "anomaly_count",
            "N/A",
        )

        anomaly_percentage = ml_anomaly.get(
            "anomaly_percentage"
        )

        anomaly_rate = (
            f"{anomaly_percentage:.2f}%"
            if isinstance(
                anomaly_percentage,
                (int, float),
            )
            else "N/A"
        )

        return (
            "## Business Health\n\n"
            "| KPI | Value |\n"
            "|---|---:|\n"
            f"| Records analyzed | {rows} |\n"
            f"| Fields analyzed | {columns} |\n"
            f"| Data quality score | "
            f"{self._format_score(quality_score)} |\n"
            f"| ML-detected anomalies | "
            f"{anomaly_count} |\n"
            f"| Anomaly rate | {anomaly_rate} |"
        )

    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    def _build_key_insights(
        self,
        *,
        insight: Any,
        statistical_findings: list[Any],
    ) -> str:

        output = [
            "## Key Findings"
        ]

        if isinstance(
            insight,
            str,
        ) and insight.strip():

            output.append(
                insight.strip()
            )

        elif isinstance(
            insight,
            dict,
        ):

            summary = (
                insight.get("summary")
            )

            if summary:
                output.append(
                    str(summary)
                )

            findings = (
                insight.get(
                    "key_insights"
                )
                or
                insight.get(
                    "insights"
                )
            )

            if isinstance(
                findings,
                list,
            ):

                for finding in findings:

                    output.append(
                        f"- {finding}"
                    )

            elif findings:

                output.append(
                    str(findings)
                )

        if statistical_findings:

            output.append(
                "\n### Statistical Findings"
            )

            for finding in statistical_findings[:10]:

                if isinstance(
                    finding,
                    str,
                ):

                    output.append(
                        f"- {finding}"
                    )

                else:

                    output.append(
                        f"- {finding}"
                    )

        if len(output) == 1:

            output.append(
                "No additional analytical findings "
                "were available for this report."
            )

        return "\n\n".join(output)

    # ========================================================
    # ANOMALY SECTION
    # ========================================================

    def _build_anomaly_section(
        self,
        *,
        anomalies: dict[str, Any],
        ml_anomaly: dict[str, Any],
    ) -> str:

        output = [
            "## Anomaly & Risk Analysis"
        ]

        algorithm = ml_anomaly.get(
            "algorithm"
        )

        anomaly_count = ml_anomaly.get(
            "anomaly_count"
        )

        anomaly_percentage = ml_anomaly.get(
            "anomaly_percentage"
        )

        if anomaly_count is not None:

            text = (
                f"The **{algorithm or 'machine-learning'} "
                f"anomaly detection analysis identified "
                f"**{anomaly_count} records**"
            )

            if anomaly_percentage is not None:

                text += (
                    f" ({anomaly_percentage:.2f}% "
                    f"of the dataset)"
                )

            text += " for further review."

            output.append(
                text
            )

        features = ml_anomaly.get(
            "features_used"
        )

        if features:

            output.append(
                "**Numerical features considered:** "
                + ", ".join(
                    str(feature)
                    for feature in features
                )
            )

        existing_anomalies = (
            anomalies.get(
                "summary"
            )
            if isinstance(
                anomalies,
                dict,
            )
            else None
        )

        if existing_anomalies:

            output.append(
                "### Existing Data Quality Anomalies\n\n"
                + str(existing_anomalies)
            )

        if len(output) == 1:

            output.append(
                "No machine-learning anomaly results "
                "were available for this analysis."
            )

        return "\n\n".join(output)

    # ========================================================
    # QUALITY
    # ========================================================

    def _build_quality_section(
        self,
        *,
        quality: dict[str, Any],
    ) -> str:

        output = [
            "## Data Quality"
        ]

        score = self._quality_score(
            quality
        )

        if score is not None:

            output.append(
                f"**Overall data quality score: "
                f"{score:.2f}/100**"
            )

        if isinstance(
            quality,
            dict,
        ):

            for key in (
                "missing_values",
                "duplicate_rows",
                "outliers",
            ):

                if key in quality:

                    output.append(
                        f"- **{key.replace('_', ' ').title()}:** "
                        f"{quality[key]}"
                    )

        if len(output) == 1:

            output.append(
                "No additional data-quality metrics "
                "were available."
            )

        return "\n\n".join(output)

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _build_recommendations(
        self,
        *,
        insight: Any,
        ml_anomaly: dict[str, Any],
    ) -> str:

        recommendations = []

        anomaly_count = ml_anomaly.get(
            "anomaly_count"
        )

        if anomaly_count:

            recommendations.append(
                "Review the ML-flagged records to "
                "determine whether they represent genuine "
                "business exceptions or data-quality issues."
            )

        if isinstance(
            insight,
            dict,
        ):

            limitations = (
                insight.get(
                    "limitations"
                )
                or
                insight.get(
                    "data_quality_limitations"
                )
            )

            if limitations:

                recommendations.append(
                    "Address the identified data-quality "
                    "limitations before using the analysis "
                    "for high-impact decisions."
                )

        recommendations.append(
            "Use the analytical findings together with "
            "business context before making operational "
            "decisions."
        )

        output = [
            "## Recommended Actions"
        ]

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            output.append(
                f"{index}. {recommendation}"
            )

        return "\n\n".join(output)

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _quality_score(
        quality: dict[str, Any],
    ):

        if not isinstance(
            quality,
            dict,
        ):

            return None

        for key in (
            "score",
            "quality_score",
            "overall_score",
        ):

            value = quality.get(
                key
            )

            if isinstance(
                value,
                (int, float),
            ):

                return float(value)

        return None

    @staticmethod
    def _format_score(
        score,
    ):

        if score is None:
            return "N/A"

        return f"{score:.2f}/100"