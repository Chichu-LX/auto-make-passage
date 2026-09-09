from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.providers import JsonModel, ProviderRequestError
from research_agents.project_layout import task_project_slug
from research_agents.schemas import (
    AgentIssue,
    AgentName,
    AgentResult,
    FeedbackControl,
    FeedbackCoordinationReport,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.FEEDBACK:
    raise RuntimeError("Feedback Agent package name mismatch")


class FeedbackAgent:
    """Coordinate redacted workflow evidence and emit Runtime-validated advice."""

    name = AgentName.FEEDBACK
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(self, model: JsonModel, artifacts: ArtifactStore | None = None) -> None:
        self.model = model
        self.artifacts = artifacts

    @staticmethod
    def _fallback_control(reason: str, log_count: int) -> FeedbackControl:
        return FeedbackControl(
            action="request_human",
            severity="blocking",
            reason=reason,
            issue_codes=["FEEDBACK_CONTRACT_INVALID"],
            observed_log_count=log_count,
        )

    @staticmethod
    def _normalize_control(
        response: dict[str, Any], *, log_count: int
    ) -> dict[str, Any]:
        raw = response.get("control") or response.get("feedback_control")
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
            raw = raw[0]
        if not isinstance(raw, dict) and any(
            key in response for key in ("action", "severity", "reason", "target_agent")
        ):
            raw = response
        if not isinstance(raw, dict):
            raise TypeError("feedback response has no control object")
        action_raw = str(raw.get("action") or "continue").lower().replace("-", "_")
        action = {
            "proceed": "continue",
            "approve": "continue",
            "continue_workflow": "continue",
            "rerun": "retry",
            "escalate": "request_human",
            "stop": "abort",
        }.get(action_raw, action_raw)
        severity_raw = str(raw.get("severity") or "info").lower()
        severity = {
            "none": "info",
            "low": "info",
            "medium": "warning",
            "high": "error",
            "critical": "blocking",
        }.get(severity_raw, severity_raw)
        reason = raw.get("reason") or response.get("summary") or "No monitoring anomaly detected."
        target = raw.get("target_agent") or raw.get("target")
        issue_codes = raw.get("issue_codes")
        if not isinstance(issue_codes, list):
            issue_codes = []
        return {
            "schema_version": "1.0",
            "action": action,
            "severity": severity,
            "reason": str(reason),
            "target_agent": target if action == "retry" else None,
            "issue_codes": [str(value) for value in issue_codes[:30]],
            "observed_log_count": log_count,
        }

    @staticmethod
    def _recover_control_from_source(task: TaskEnvelope, log_count: int) -> FeedbackControl:
        """Fail safely using Runtime-supplied source evidence after model drift."""

        source_status = str(task.context.get("source_status") or "")
        issue_codes = [
            str(value)
            for value in task.context.get("source_issue_codes", [])[:30]
            if str(value)
        ]
        if source_status in {"succeeded", "in_progress"} and not issue_codes:
            return FeedbackControl(
                action="continue",
                severity="warning",
                reason=(
                    "Feedback model output drifted from its JSON contract; Runtime source "
                    "evidence contains no concrete failure, so the workflow may continue."
                ),
                issue_codes=["FEEDBACK_MODEL_CONTRACT_RECOVERED"],
                observed_log_count=log_count,
            )
        return FeedbackControl(
            action="request_human",
            severity="blocking",
            reason=(
                "Feedback model output drifted from its JSON contract; the source task "
                f"reported status={source_status or 'unknown'}"
                + (f" with issues {', '.join(issue_codes)}." if issue_codes else ".")
            ),
            issue_codes=issue_codes or ["FEEDBACK_MODEL_CONTRACT_RECOVERED"],
            observed_log_count=log_count,
        )

    @staticmethod
    def _string_list(value: Any, *, limit: int = 30) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            str(item).strip()[:2000]
            for item in value[:limit]
            if str(item).strip()
        ]

    @classmethod
    def _coordination_payload(
        cls,
        response: dict[str, Any],
        *,
        control: FeedbackControl,
        fallback_summary: str,
    ) -> dict[str, Any]:
        raw = response.get("coordination") or response.get("plan") or {}
        if not isinstance(raw, dict):
            raw = {}
        health = str(raw.get("workflow_health") or "on_track").lower().replace("-", "_")
        health = {
            "healthy": "on_track",
            "warning": "attention",
            "error": "blocked",
            "critical": "blocked",
        }.get(health, health)
        if control.action == "abort":
            health = "aborted"
        elif control.action in {"retry", "request_human"} and health == "on_track":
            health = "blocked"
        return {
            "workflow_health": health,
            "progress_summary": str(
                raw.get("progress_summary") or response.get("summary") or fallback_summary
            ),
            "stage_assessment": str(
                raw.get("stage_assessment")
                or "The latest transition was assessed against redacted logs and typed evidence."
            ),
            "next_actions": cls._string_list(raw.get("next_actions")),
            "risks": cls._string_list(raw.get("risks")),
            "coordination_notes": cls._string_list(raw.get("coordination_notes")),
        }

    @staticmethod
    def _markdown(report: FeedbackCoordinationReport) -> str:
        lines = [
            "# Feedback Coordination Report",
            "",
            f"- Workflow health: `{report.workflow_health}`",
            f"- Control: `{report.control.action}` / `{report.control.severity}`",
            f"- Target: `{report.control.target_agent.value if report.control.target_agent else 'none'}`",
            f"- Observed logs: `{report.observed_log_count}`",
            "",
            report.progress_summary,
            "",
            "## Stage assessment",
            "",
            report.stage_assessment,
            "",
            "## Next actions",
            "",
        ]
        lines.extend(f"- {value}" for value in report.next_actions)
        lines.extend(["", "## Risks", ""])
        lines.extend(f"- {value}" for value in report.risks)
        lines.extend(["", "## Coordination notes", ""])
        lines.extend(f"- {value}" for value in report.coordination_notes)
        lines.append("")
        return "\n".join(lines)

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        monitoring = task.context.get("monitoring_summary", {})
        log_count = (
            int(monitoring.get("record_count", 0))
            if isinstance(monitoring, dict)
            else 0
        )
        try:
            response = self.model.complete_json(
                PACKAGE.system_prompt,
                json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "feedback_control_contract": FeedbackControl.model_json_schema(),
                        "coordination_report_contract": FeedbackCoordinationReport.model_json_schema(),
                        "agent_result_issue_contract": AgentIssue.model_json_schema(),
                    },
                    ensure_ascii=False,
                ),
            )
            if not isinstance(response, dict):
                raise TypeError("feedback response must be an object")
            control_payload = self._normalize_control(response, log_count=log_count)
            control = FeedbackControl.model_validate(control_payload)
            allowed = set(task.context.get("allowed_control_targets", []))
            if control.action == "retry" and (
                control.target_agent is None
                or control.target_agent.value not in allowed
            ):
                raise ValueError("feedback selected a target outside the Runtime allowlist")
            raw_issues = response.get("issues", [])
            issues = []
            if isinstance(raw_issues, list):
                for value in raw_issues[:30]:
                    try:
                        issues.append(AgentIssue.model_validate(value).model_dump(mode="json"))
                    except ValidationError:
                        continue
            raw_evidence = response.get("evidence", [])
            evidence = (
                [value for value in raw_evidence[:100] if isinstance(value, dict)]
                if isinstance(raw_evidence, list)
                else []
            )
            summary = str(response.get("summary", "feedback monitoring completed"))
        except (ProviderRequestError, ValidationError, ValueError, TypeError) as exc:
            control = self._recover_control_from_source(task, log_count)
            summary = control.reason
            response = {}
            evidence = [
                {
                    "check": "feedback-contract",
                    "status": "failed_closed",
                    "observed_log_count": log_count,
                }
            ]
            issues = [
                {
                    "severity": "warning",
                    "code": "FEEDBACK_MODEL_CONTRACT_RECOVERED",
                    "message": (
                        f"Recovered deterministically after {type(exc).__name__} in the "
                        "model response contract."
                    ),
                    "root_cause_key": "G-FEEDBACK:CONTRACT:recovered",
                }
            ]

        coordination = self._coordination_payload(
            response,
            control=control,
            fallback_summary=summary,
        )
        artifacts = []
        if self.artifacts is not None:
            project_slug = task_project_slug(task)
            report = FeedbackCoordinationReport(
                run_id=task.run_id,
                task_id=task.task_id,
                project_slug=project_slug,
                source_task_id=task.parent_task_id or task.task_id,
                observed_log_count=log_count,
                control=control,
                **coordination,
            )
            token = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:12]
            base = f"project/feedback/{project_slug}/{token}"
            json_ref = self.artifacts.write_agent_json(
                self.name,
                f"{base}.json",
                report.model_dump(mode="json"),
            ).model_copy(update={"kind": "feedback_coordination_report"})
            markdown_ref = self.artifacts.write_agent_bytes(
                self.name,
                f"{base}.md",
                self._markdown(report).encode("utf-8"),
                max_bytes=1024 * 1024,
                kind="feedback_coordination_markdown",
            )
            artifacts = [json_ref, markdown_ref]

        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded",
            summary=summary,
            artifacts=artifacts,
            evidence=evidence,
            issues=issues,
            feedback_control=control,
            needs_human=False,
        )
