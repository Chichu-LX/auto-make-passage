from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.process_logging import ProcessLogStore
from research_agents.providers import JsonModel, ProviderRequestError
from research_agents.project_layout import task_project_slug
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    PaperCodeConsistencyReport,
    PaperManuscriptManifest,
    PaperReviewDraft,
    PaperReviewFinding,
    PaperReviewReport,
    ProcessLogRecord,
    TaskEnvelope,
    WorkflowState,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_MANIFEST_BYTES,
    MAX_REVIEW_CONTEXT_BYTES,
    MAX_REVIEW_REPORT_BYTES,
    deterministic_manuscript_findings,
    file_token,
    repair_route,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.PAPER_REVIEW:
    raise RuntimeError("Paper Review Agent package name mismatch")


@dataclass(frozen=True)
class _ReviewInputs:
    manuscript: PaperManuscriptManifest
    manuscript_ref: ArtifactRef
    consistency: PaperCodeConsistencyReport
    consistency_ref: ArtifactRef
    review_round: int


class PaperReviewAgent:
    """Senior final reviewer that passes only manuscripts with no major issue."""

    name = AgentName.PAPER_REVIEW
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(self, model: JsonModel, artifacts: ArtifactStore) -> None:
        self.model = model
        self.artifacts = artifacts
        self.logs = ProcessLogStore(artifacts.root)

    def _log(
        self,
        task: TaskEnvelope,
        *,
        event: str,
        status: str,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.logs.append(
            ProcessLogRecord(
                run_id=task.run_id,
                agent=self.name,
                event=event,
                status=status,
                workflow_state=WorkflowState.PAPER_REVIEW,
                state_version=int(task.context.get("state_version", 1)),
                task_id=task.task_id,
                attempt=task.attempt,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def _resolve_inputs(self, task: TaskEnvelope) -> _ReviewInputs | None:
        manuscripts: list[tuple[PaperManuscriptManifest, ArtifactRef]] = []
        consistency_reports: list[tuple[PaperCodeConsistencyReport, ArtifactRef]] = []
        prior_rounds: list[int] = []
        slug = task_project_slug(task)
        for artifact in task.inputs:
            try:
                if artifact.kind == "paper_manuscript_manifest":
                    payload = self.artifacts.read_artifact_json(
                        artifact, max_bytes=MAX_MANIFEST_BYTES
                    )
                    value = PaperManuscriptManifest.model_validate(payload)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        manuscripts.append((value, artifact))
                elif artifact.kind == "paper_code_consistency_report":
                    payload = self.artifacts.read_artifact_json(
                        artifact, max_bytes=MAX_MANIFEST_BYTES
                    )
                    value = PaperCodeConsistencyReport.model_validate(payload)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        consistency_reports.append((value, artifact))
                elif artifact.kind == "paper_review_report":
                    payload = self.artifacts.read_artifact_json(
                        artifact, max_bytes=MAX_MANIFEST_BYTES
                    )
                    value = PaperReviewReport.model_validate(payload)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        prior_rounds.append(value.review_round)
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        for manuscript, manuscript_ref in reversed(manuscripts):
            matching = next(
                (
                    item
                    for item in reversed(consistency_reports)
                    if item[0].manuscript_manifest_uri == manuscript_ref.uri
                    and item[0].verdict == "passed"
                ),
                None,
            )
            if matching is not None:
                return _ReviewInputs(
                    manuscript=manuscript,
                    manuscript_ref=manuscript_ref,
                    consistency=matching[0],
                    consistency_ref=matching[1],
                    review_round=max(prior_rounds, default=0) + 1,
                )
        return None

    @staticmethod
    def _prompt(task: TaskEnvelope, inputs: _ReviewInputs) -> str:
        manuscript = inputs.manuscript
        context = {
            "objective": task.objective,
            "review_round": inputs.review_round,
            "review_standard": (
                "Act as a senior authoritative reviewer for NeurIPS/ICML or a leading "
                "Bioinformatics/Briefings in Bioinformatics-style journal. A final pass is "
                "legal only when no blocking or major issue remains. Minor issues may remain."
            ),
            "user_revision": task.context.get("user_revision"),
            "verified_consistency_report": inputs.consistency.model_dump(mode="json"),
            "manuscript": {
                "title": manuscript.title,
                "abstract": manuscript.abstract,
                "keywords": manuscript.keywords,
                "sections": [item.model_dump(mode="json") for item in manuscript.sections],
                "claims": [item.model_dump(mode="json") for item in manuscript.claims],
                "formulas": [item.model_dump(mode="json") for item in manuscript.formulas],
                "experiment_results": [
                    item.model_dump(mode="json") for item in manuscript.experiment_results
                ],
                "citations": [item.model_dump(mode="json") for item in manuscript.citations],
            },
            "required_dimensions": [
                "novelty and significance",
                "methodological soundness",
                "experiment sufficiency and statistical support",
                "reproducibility and code/data traceability",
                "result interpretation and claim calibration",
                "citation coverage",
                "writing and submission readiness",
                "limitations and ethical disclosure",
            ],
            "deterministic_preflight_findings": [
                item.model_dump(mode="json")
                for item in deterministic_manuscript_findings(manuscript)
            ],
            "response_contract": PaperReviewDraft.model_json_schema(),
        }
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_REVIEW_CONTEXT_BYTES:
            raise ValueError("paper review context exceeds bounded policy")
        return encoded

    @staticmethod
    def _normalize(response: dict[str, Any]) -> dict[str, Any]:
        wrapped = response.get("paper_review") or response.get("review")
        payload = wrapped if isinstance(wrapped, dict) else response
        findings = payload.get("findings", [])
        normalized_findings: list[dict[str, Any]] = []
        if isinstance(findings, list):
            for index, item in enumerate(findings[:100], start=1):
                if not isinstance(item, dict):
                    continue
                severity = str(item.get("severity") or "minor").lower().replace("-", "_")
                severity = {
                    "critical": "blocking",
                    "error": "major",
                    "warning": "minor",
                    "info": "note",
                }.get(severity, severity)
                normalized_findings.append(
                    {
                        **item,
                        "finding_id": item.get("finding_id") or f"review-finding-{index}",
                        "severity": severity,
                        "evidence_refs": item.get("evidence_refs") or [],
                        "required_revision": item.get("required_revision")
                        or item.get("recommendation")
                        or "Clarify this issue with evidence-grounded manuscript revisions.",
                    }
                )
        return {
            **payload,
            "reviewer_role": payload.get("reviewer_role")
            or "Senior authoritative computational science peer reviewer",
            "target_venue_profile": payload.get("target_venue_profile")
            or "authoritative_bioinformatics_journal",
            "findings": normalized_findings,
            "strengths": payload.get("strengths") or [],
            "required_revisions": payload.get("required_revisions") or [],
        }

    @staticmethod
    def _markdown(report: PaperReviewReport) -> str:
        lines = [
            "# Final Authoritative Paper Review",
            "",
            f"- Round: `{report.review_round}`",
            f"- Verdict: `{report.verdict}`",
            f"- Recommendation: `{report.recommendation}`",
            f"- Major issues: `{report.major_issue_count}`",
            f"- Minor issues: `{report.minor_issue_count}`",
            f"- Manuscript: `{report.manuscript_manifest_uri}`",
            f"- Consistency report: `{report.consistency_report_uri}`",
            "",
            report.summary,
            "",
            "## Strengths",
            "",
        ]
        lines.extend(f"- {value}" for value in report.strengths)
        lines.extend(["", "## Findings", ""])
        for item in report.findings:
            lines.append(
                f"- `{item.severity}` `{item.category}` `{item.finding_id}`: "
                f"{item.summary} Revision: {item.required_revision}"
            )
        lines.extend(["", "## Required revisions", ""])
        lines.extend(f"- {value}" for value in report.required_revisions)
        lines.append("")
        return "\n".join(lines)

    def _blocked(self, task: TaskEnvelope, reason: str) -> AgentResult:
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="failed",
            summary=reason,
            issues=[
                {
                    "severity": "blocking",
                    "code": "FINAL_PAPER_REVIEW_INPUT_INVALID",
                    "message": reason,
                    "root_cause_key": "G11:FINAL_PAPER_REVIEW_INPUT_INVALID:paper",
                }
            ],
            route_hint=AgentName.CHECK_PAPER_AND_CODE,
        )

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        started = perf_counter()
        self._log(task, event="final_peer_review_started", status="started")
        inputs = self._resolve_inputs(task)
        if inputs is None:
            result = self._blocked(
                task,
                "Final review requires a project-matched manuscript and a passed paper/code consistency report.",
            )
            self._log(
                task,
                event="final_peer_review_completed",
                status="failed",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                details={"issue_code": "FINAL_PAPER_REVIEW_INPUT_INVALID"},
            )
            return result
        try:
            response = self.model.complete_json(
                PACKAGE.system_prompt, self._prompt(task, inputs)
            )
            draft = PaperReviewDraft.model_validate(self._normalize(response))
        except (ProviderRequestError, ValidationError, ValueError, TypeError) as exc:
            reason = f"Final authoritative review failed closed: {type(exc).__name__}."
            result = AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="needs_human",
                summary=reason,
                issues=[
                    {
                        "severity": "blocking",
                        "code": "FINAL_PAPER_REVIEW_CONTRACT_INVALID",
                        "message": reason,
                        "root_cause_key": "G11:FINAL_PAPER_REVIEW_CONTRACT_INVALID:model",
                    }
                ],
                route_hint=AgentName.PAPER_REVIEW,
                needs_human=True,
            )
            self._log(
                task,
                event="final_peer_review_completed",
                status="blocked",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                details={"error_type": type(exc).__name__},
            )
            return result

        deterministic_findings = deterministic_manuscript_findings(inputs.manuscript)
        deterministic_ids = {item.finding_id for item in deterministic_findings}
        merged_findings = [
            *deterministic_findings,
            *[
                item
                for item in draft.findings
                if item.finding_id not in deterministic_ids
            ],
        ][:100]
        major = [
            item
            for item in merged_findings
            if item.severity in {"blocking", "major"}
        ]
        minor = [item for item in merged_findings if item.severity == "minor"]
        route = repair_route(major) if major else None
        required_revisions = list(
            dict.fromkeys(
                [
                    *draft.required_revisions,
                    *(item.required_revision for item in deterministic_findings),
                ]
            )
        )[:100]
        recommendation = draft.recommendation
        if major and recommendation in {"accept", "minor_revision"}:
            recommendation = "major_revision"
        summary = draft.summary
        if deterministic_findings:
            summary = (
                f"{summary} Deterministic scientific preflight added "
                f"{len(deterministic_findings)} evidence-grounded finding(s)."
            )
        summary = summary[:6000]
        report = PaperReviewReport(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            review_round=inputs.review_round,
            manuscript_manifest_uri=inputs.manuscript_ref.uri,
            consistency_report_uri=inputs.consistency_ref.uri,
            verdict="needs_revision" if major else "passed",
            reviewer_role=draft.reviewer_role,
            target_venue_profile=draft.target_venue_profile,
            recommendation=recommendation,
            summary=summary,
            strengths=draft.strengths,
            findings=merged_findings,
            required_revisions=required_revisions,
            major_issue_count=len(major),
            minor_issue_count=len(minor),
            recommended_route=route,
        )
        token = file_token(task.task_id)
        base = f"project/reviews/paper/{task_project_slug(task)}/{token}"
        json_ref = self.artifacts.write_agent_json(
            self.name, f"{base}.json", report.model_dump(mode="json")
        ).model_copy(update={"kind": "paper_review_report"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._markdown(report).encode("utf-8"),
            max_bytes=MAX_REVIEW_REPORT_BYTES,
            kind="paper_review_markdown",
        )
        status = "failed" if major else "succeeded"
        self._log(
            task,
            event="final_peer_review_completed",
            status=status,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            details={
                "review_round": report.review_round,
                "verdict": report.verdict,
                "major_issue_count": len(major),
                "minor_issue_count": len(minor),
                "recommended_route": route.value if route else None,
            },
        )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status=status,
            summary=(
                f"Final authoritative review round {report.review_round}: "
                f"{len(major)} major and {len(minor)} minor issues."
            ),
            artifacts=[json_ref, markdown_ref],
            evidence=[
                {
                    "review_standard": report.target_venue_profile,
                    "consistency_report_verdict": inputs.consistency.verdict,
                    "major_issue_count": len(major),
                    "minor_issue_count": len(minor),
                    "pass_requires_zero_major_issues": True,
                }
            ],
            issues=[
                {
                    "severity": "blocking",
                    "code": "FINAL_PAPER_MAJOR_ISSUE",
                    "message": item.summary,
                    "root_cause_key": f"G11:FINAL_PAPER_MAJOR_ISSUE:{item.finding_id}",
                }
                for item in major
            ],
            route_hint=route,
        )
