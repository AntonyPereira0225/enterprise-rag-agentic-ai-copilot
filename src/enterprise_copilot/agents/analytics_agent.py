from __future__ import annotations

from statistics import fmean
from typing import Any

from enterprise_copilot.agents.state import AgentCitation, AgentExecution

_FILTER_FIELDS = ("region", "product", "issue_type", "priority", "channel", "status")


class AnalyticsAgent:
    """Safe, read-only aggregates over the synthetic support-case dataset."""

    def __init__(self, cases: list[dict[str, Any]], *, source_uri: str) -> None:
        if not cases:
            raise ValueError("AnalyticsAgent requires at least one support case")
        self.cases = cases
        self.source_uri = source_uri
        self.values = {
            field: sorted(
                {str(case[field]) for case in cases}, key=lambda value: (-len(value), value)
            )
            for field in _FILTER_FIELDS
        }

    def _filters(self, question: str) -> dict[str, str]:
        normalised = question.casefold().replace("_", " ")
        filters: dict[str, str] = {}
        for field, values in self.values.items():
            for value in values:
                if value.casefold().replace("_", " ") in normalised:
                    filters[field] = value
                    break
        return filters

    def execute(self, question: str) -> AgentExecution:
        filters = self._filters(question)
        matching = [
            case
            for case in self.cases
            if all(str(case[field]) == value for field, value in filters.items())
        ]
        if not matching:
            return AgentExecution(
                status="refused",
                answer="No synthetic support cases match the requested filters.",
                confidence=1.0,
                citations=(),
                verified=True,
                reason="The filtered dataset is empty.",
                details={
                    "filters": filters,
                    "matching_rows": 0,
                    "generator": "deterministic_analytics",
                    "model": None,
                },
            )

        normalised = question.casefold()
        if "average" in normalised and "resolution" in normalised:
            average = fmean(float(case["resolution_minutes"]) for case in matching)
            result_text = (
                f"The average resolution time is {average:.1f} minutes across "
                f"{len(matching)} matching synthetic support cases."
            )
            metric = "average_resolution_minutes"
            value: float | int = average
        elif any(cue in normalised for cue in ("how many", "number of", "count of")):
            result_text = f"There are {len(matching)} matching synthetic support cases."
            metric = "case_count"
            value = len(matching)
        else:
            return AgentExecution(
                status="refused",
                answer=(
                    "The analytics workflow currently supports case counts and average "
                    "resolution time only."
                ),
                confidence=1.0,
                citations=(),
                verified=True,
                reason="The requested aggregate is not supported.",
                details={
                    "filters": filters,
                    "matching_rows": len(matching),
                    "generator": "deterministic_analytics",
                    "model": None,
                },
            )

        quote = f"Aggregate calculated from {len(matching)} matching synthetic records."
        citation = AgentCitation(
            citation_id="D1",
            source_type="dataset",
            source_id="support_cases",
            title="Synthetic support cases",
            source_uri=self.source_uri,
            quote=quote,
        )
        return AgentExecution(
            status="answered",
            answer=f"{result_text} [D1]",
            confidence=1.0,
            citations=(citation,),
            verified=True,
            details={
                "metric": metric,
                "value": value,
                "filters": filters,
                "matching_rows": len(matching),
                "generator": "deterministic_analytics",
                "model": None,
            },
        )
