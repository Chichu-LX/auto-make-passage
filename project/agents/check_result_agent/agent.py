from __future__ import annotations

from datetime import timedelta
import json
import math
from statistics import fmean
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.project_layout import artifact_prefix, scoped_path, task_project_slug
from research_agents.providers import JsonModel
from research_agents.remote_execution import (
    RemoteExecutionBackend,
    RemoteExecutionError,
    SafeSSHBackend,
    load_server_execution_config,
    project_execution,
)
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    JobObservation,
    MetricGateCheck,
    ResearchDesign,
    ResultCheckReport,
    ServerExecutionConfig,
    ServerScheduleManifest,
    TaskEnvelope,
    utc_now,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_INPUT_BYTES,
    MAX_LOG_SNAPSHOT_BYTES,
    REPORT_INTERVAL_SECONDS,
    file_token,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.CHECK_RESULT:
    raise RuntimeError("Check Result Agent package name mismatch")


class CheckResultAgent:
    """Poll managed jobs, persist bounded evidence, and apply deterministic result gates."""

    name = AgentName.CHECK_RESULT
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(
        self,
        model: JsonModel,
        artifacts: ArtifactStore,
        *,
        backend: RemoteExecutionBackend | None = None,
        execution_config: ServerExecutionConfig | None = None,
    ) -> None:
        self.model = model
        self.artifacts = artifacts
        self.backend = backend or SafeSSHBackend()
        self.execution_config = execution_config

    def _read_inputs(
        self, task: TaskEnvelope
    ) -> tuple[
        ResearchDesign | None,
        ArtifactRef | None,
        ServerScheduleManifest | None,
        ArtifactRef | None,
        list[tuple[ResultCheckReport, ArtifactRef]],
    ]:
        slug = task_project_slug(task)
        design = schedule = None
        design_ref = schedule_ref = None
        previous: list[tuple[ResultCheckReport, ArtifactRef]] = []
        schedule_prefix = f"artifact://project/var/runs/{task.run_id}/server_schedule/"
        for artifact in task.inputs:
            try:
                if artifact.kind == "research_design" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "research-designs")
                ):
                    candidate = ResearchDesign.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id:
                        design, design_ref = candidate, artifact
                elif artifact.kind == "server_schedule_manifest" and artifact.uri.startswith(
                    schedule_prefix
                ):
                    candidate = ServerScheduleManifest.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id and candidate.project_slug == slug:
                        schedule, schedule_ref = candidate, artifact
                elif artifact.kind == "result_check_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "results")
                ):
                    candidate = ResultCheckReport.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id and candidate.project_slug == slug:
                        previous.append((candidate, artifact))
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        if schedule_ref is not None:
            previous = [
                item for item in previous if item[0].schedule_manifest_uri == schedule_ref.uri
            ]
        return design, design_ref, schedule, schedule_ref, previous

    def _config(self) -> ServerExecutionConfig:
        return self.execution_config or load_server_execution_config(self.artifacts.root)

    def _write_gap(self, task: TaskEnvelope, reason: str, code: str) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "design",
                task_project_slug(task),
                "reviews",
                "results",
                f"{file_token(task.task_id)}.gap.json",
            ),
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": task_project_slug(task),
                "status": "blocked",
                "reason": reason,
                "issue_code": code,
            },
        ).model_copy(update={"kind": "result_check_gap"})

    def _write_log_snapshot(
        self,
        task: TaskEnvelope,
        sequence: int,
        observation: JobObservation,
    ) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            (
                f"project/var/logs/jobs/{task.run_id}/report-{sequence:04d}/"
                f"{observation.job_id}.json"
            ),
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "job_id": observation.job_id,
                "status": observation.status,
                "exit_code": observation.exit_code,
                "best_metric": (
                    observation.best_metric.model_dump(mode="json")
                    if observation.best_metric
                    else None
                ),
                "best_checkpoint_path": observation.best_checkpoint_path,
                "metric_count": len(observation.metrics),
                "log_tail": observation.log_tail[-16000:],
                "observed_at": observation.observed_at.isoformat(),
            },
        ).model_copy(update={"kind": "job_log_snapshot"})

    @staticmethod
    def _route_for_failed_jobs(observations: list[JobObservation]) -> AgentName:
        text = "\n".join(item.log_tail for item in observations if item.status == "failed").lower()
        if any(value in text for value in ("out of memory", "cuda oom", "nccl", "gpu", "ssh")):
            return AgentName.SERVER_SCHEDULE
        if any(value in text for value in ("dataset", "data file", "filenotfound", "no such file")):
            return AgentName.DOWNLOAD_DATA
        return AgentName.CODE_MAKE

    @staticmethod
    def _markdown(report: ResultCheckReport) -> str:
        lines = [
            "# Experiment Result Check",
            "",
            f"- Verdict: `{report.verdict}`",
            f"- Report sequence: `{report.report_sequence}`",
            f"- Schedule: `{report.schedule_manifest_uri}`",
            f"- Report interval: `{report.report_interval_seconds}` seconds",
            f"- Next report: `{report.next_report_at.isoformat() if report.next_report_at else 'none'}`",
            "",
            report.summary,
            "",
            "## Jobs",
            "",
            "| Job | Status | Exit | Best metric | Best checkpoint |",
            "|---|---|---:|---|---|",
        ]
        for item in report.observations:
            metric = (
                f"{item.best_metric.metric_name}={item.best_metric.value:.8g}"
                if item.best_metric
                else "none"
            )
            checkpoint = (item.best_checkpoint_path or "none").replace("|", "\\|")
            lines.append(
                f"| `{item.job_id}` | `{item.status}` | {item.exit_code} | {metric} | `{checkpoint}` |"
            )
        lines.extend(["", "## Metric gates", ""])
        if report.metric_checks:
            for item in report.metric_checks:
                lines.append(
                    f"- `{item.check_id}`: `{item.status}` — {item.summary}"
                )
        else:
            lines.append("- No metric gate was evaluated yet.")
        if report.issues:
            lines.extend(["", "## Issues", ""])
            lines.extend(f"- {value}" for value in report.issues)
        return "\n".join(lines) + "\n"

    def _write_report(
        self, task: TaskEnvelope, report: ResultCheckReport
    ) -> tuple[ArtifactRef, ArtifactRef]:
        base = scoped_path(
            "design",
            task_project_slug(task),
            "reviews",
            "results",
            file_token(task.task_id),
        )
        json_ref = self.artifacts.write_agent_json(
            self.name,
            f"{base}.json",
            report.model_dump(mode="json"),
        ).model_copy(update={"kind": "result_check_report"})
        markdown = self._markdown(report).encode("utf-8")
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            markdown,
            max_bytes=MAX_LOG_SNAPSHOT_BYTES,
            kind="result_check_markdown",
        )
        return json_ref, markdown_ref

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        design, design_ref, schedule, schedule_ref, previous = self._read_inputs(task)
        if (
            design is None
            or design_ref is None
            or schedule is None
            or schedule_ref is None
            or schedule.status != "launched"
        ):
            reason = "Result checking requires a project-matched ResearchDesign and launched ServerScheduleManifest."
            gap = self._write_gap(task, reason, "RESULT_INPUT_MISSING")
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="blocked",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "RESULT_INPUT_MISSING",
                        "message": reason,
                        "root_cause_key": "G7:RESULT_INPUT_MISSING:inputs",
                    }
                ],
                needs_human=True,
            )

        previous.sort(key=lambda item: item[0].report_sequence)
        now = utc_now()
        if previous:
            latest, latest_ref = previous[-1]
            if latest.verdict == "in_progress" and latest.next_report_at and now < latest.next_report_at:
                return AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=self.name,
                    status="in_progress",
                    summary=(
                        "The next server poll is not due; the existing report remains current until "
                        f"{latest.next_report_at.isoformat()}."
                    ),
                    artifacts=[latest_ref],
                    evidence=[
                        {
                            "check": "poll-rate-limit",
                            "report_sequence": latest.report_sequence,
                            "next_report_at": latest.next_report_at.isoformat(),
                            "report_interval_seconds": REPORT_INTERVAL_SECONDS,
                        }
                    ],
                )

        try:
            config = self._config()
            project = project_execution(config, task_project_slug(task))
        except (RemoteExecutionError, ValidationError) as exc:
            reason = f"Result monitoring configuration is unavailable: {type(exc).__name__}."
            gap = self._write_gap(task, reason, "RESULT_SERVER_CONFIG_INVALID")
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="needs_human",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "RESULT_SERVER_CONFIG_INVALID",
                        "message": reason,
                        "root_cause_key": "G7:RESULT_SERVER_CONFIG_INVALID:config",
                    }
                ],
                needs_human=True,
            )
        profiles = {item.server_id: item for item in config.servers}
        observations: list[JobObservation] = []
        for job in schedule.jobs:
            profile = profiles.get(job.server_id)
            if profile is None:
                observations.append(JobObservation(job_id=job.job_id, status="unknown"))
                continue
            observations.append(self.backend.observe_job(profile, job))

        sequence = (previous[-1][0].report_sequence + 1) if previous else 1
        log_refs = [
            self._write_log_snapshot(task, sequence, observation)
            for observation in observations
        ]
        active = [
            item for item in observations if item.status in {"queued", "running", "unknown"}
        ]
        failed = [item for item in observations if item.status == "failed"]
        completed = [item for item in observations if item.status == "completed"]
        matching_values = [
            item.best_metric.value
            for item in observations
            if item.best_metric is not None
            and item.best_metric.metric_name == project.metric_name
            and math.isfinite(item.best_metric.value)
        ]
        aggregate = fmean(matching_values) if matching_values else None
        metric_checks: list[MetricGateCheck] = []
        issues: list[str] = []
        route: AgentName | None = None
        next_report_at = None

        if failed:
            verdict = "failed"
            route = self._route_for_failed_jobs(failed)
            issues.append(f"{len(failed)} managed jobs failed with persisted log evidence.")
            summary = (
                f"Observed {len(failed)} failed, {len(completed)} completed, and {len(active)} active jobs; "
                f"repair is routed to {route.value}."
            )
        elif active:
            verdict = "in_progress"
            next_report_at = now + timedelta(seconds=REPORT_INTERVAL_SECONDS)
            metric_checks.append(
                MetricGateCheck(
                    check_id="metric-primary-pending",
                    metric_name=project.metric_name,
                    status="pending",
                    observed_value=aggregate,
                    target_value=project.target_value,
                    direction=project.metric_direction,
                    summary=(
                        "Primary metric remains pending until every managed experiment/seed job completes."
                    ),
                )
            )
            summary = (
                f"{len(completed)}/{len(observations)} jobs completed; {len(active)} remain queued, "
                "running, or temporarily unknown. The next user report is due within 12 hours."
            )
        elif len(completed) != len(schedule.jobs):
            verdict = "failed"
            route = AgentName.SERVER_SCHEDULE
            issues.append("Job terminal states do not cover the immutable schedule.")
            summary = "The server schedule and terminal job observations are inconsistent."
        elif len(matching_values) != len(schedule.jobs):
            verdict = "needs_improvement"
            route = AgentName.CODE_MAKE
            issues.append(
                "At least one completed job omitted a finite primary metric with the declared name."
            )
            summary = (
                "All jobs exited successfully, but primary metric coverage is incomplete; "
                "the training/result-writing contract requires Code Make repair."
            )
        elif project.target_value is None:
            verdict = "needs_improvement"
            route = AgentName.CHECK_RESULT
            issues.append(
                "No numeric target_value is configured, so accuracy or quality cannot be accepted objectively."
            )
            metric_checks.append(
                MetricGateCheck(
                    check_id="metric-primary-target",
                    metric_name=project.metric_name,
                    status="pending",
                    observed_value=aggregate,
                    target_value=None,
                    direction=project.metric_direction,
                    summary="A finite aggregate was observed, but the user has not configured a numeric gate.",
                )
            )
            summary = "All jobs completed, but a user-approved numeric metric threshold is required."
        else:
            assert aggregate is not None
            primary_passed = (
                aggregate <= project.target_value
                if project.metric_direction == "lower"
                else aggregate >= project.target_value
            )
            metric_checks.append(
                MetricGateCheck(
                    check_id="metric-primary-target",
                    metric_name=project.metric_name,
                    status="passed" if primary_passed else "failed",
                    observed_value=aggregate,
                    target_value=project.target_value,
                    direction=project.metric_direction,
                    summary=(
                        f"Mean of {len(matching_values)} per-job best values is {aggregate:.8g}; "
                        f"required {project.metric_direction} target is {project.target_value:.8g}."
                    ),
                )
            )
            for index, gate in enumerate(project.additional_metric_gates, start=1):
                per_job_values: list[float] = []
                for observation in observations:
                    candidates = [
                        metric.value
                        for metric in observation.metrics
                        if metric.metric_name == gate.metric_name
                        and math.isfinite(metric.value)
                    ]
                    if candidates:
                        per_job_values.append(
                            min(candidates) if gate.direction == "lower" else max(candidates)
                        )
                observed = fmean(per_job_values) if per_job_values else None
                coverage_complete = len(per_job_values) == len(schedule.jobs)
                gate_passed = bool(
                    coverage_complete
                    and observed is not None
                    and (
                        observed <= gate.target_value
                        if gate.direction == "lower"
                        else observed >= gate.target_value
                    )
                )
                metric_checks.append(
                    MetricGateCheck(
                        check_id=f"metric-additional-{index}",
                        metric_name=gate.metric_name,
                        status="passed" if gate_passed else "failed",
                        observed_value=observed,
                        target_value=gate.target_value,
                        direction=gate.direction,
                        summary=(
                            f"Mean of {len(per_job_values)}/{len(schedule.jobs)} per-job values "
                            f"is {observed:.8g}; required {gate.direction} target is "
                            f"{gate.target_value:.8g}."
                            if observed is not None
                            else "No finite per-job values were written for this configured metric gate."
                        ),
                    )
                )
            all_gates_passed = all(item.status == "passed" for item in metric_checks)
            if all_gates_passed:
                verdict = "passed"
                summary = (
                    f"All {len(completed)} managed jobs completed and all {len(metric_checks)} "
                    f"configured metric gates pass; mean {project.metric_name}={aggregate:.8g}."
                )
            else:
                verdict = "needs_improvement"
                route = AgentName.CODE_MAKE
                issues.append(
                    "The completed experiment misses or omits at least one configured metric target."
                )
                summary = (
                    f"All jobs completed reliably, but {sum(item.status == 'failed' for item in metric_checks)} "
                    f"of {len(metric_checks)} accuracy/calibration/other metric gates failed; route to Code Make."
                )

        report = ResultCheckReport(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            schedule_manifest_uri=schedule_ref.uri,
            report_sequence=sequence,
            verdict=verdict,
            summary=summary,
            observations=observations,
            metric_checks=metric_checks,
            best_log_artifacts=log_refs,
            recommended_route=route,
            next_report_at=next_report_at,
            issues=issues,
        )
        report_ref, markdown_ref = self._write_report(task, report)
        artifacts = [report_ref, markdown_ref, *log_refs]
        evidence: list[dict[str, Any]] = [
            {
                "check": "managed-job-status",
                "total": len(observations),
                "completed": len(completed),
                "active": len(active),
                "failed": len(failed),
            },
            {
                "check": "best-log-persistence",
                "snapshot_count": len(log_refs),
                "finite_primary_metric_count": len(matching_values),
            },
            {
                "check": "monitoring-policy",
                "unbounded_result_cycles": True,
                "report_interval_seconds": REPORT_INTERVAL_SECONDS,
            },
        ]
        if verdict == "in_progress":
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="in_progress",
                summary=summary,
                artifacts=artifacts,
                evidence=evidence,
            )
        if verdict == "passed":
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="succeeded",
                summary=summary,
                artifacts=artifacts,
                evidence=evidence,
            )
        missing_target = project.target_value is None and not failed
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="needs_human" if missing_target else "failed",
            summary=summary,
            artifacts=artifacts,
            evidence=evidence,
            issues=[
                {
                    "severity": "blocking" if missing_target else "error",
                    "code": (
                        "RESULT_TARGET_MISSING"
                        if missing_target
                        else "RESULT_REPAIR_REQUIRED"
                    ),
                    "message": issues[0] if issues else summary,
                    "root_cause_key": (
                        "G7:RESULT_TARGET_MISSING:metric"
                        if missing_target
                        else f"G7:RESULT_REPAIR_REQUIRED:{route.value if route else 'unknown'}"
                    ),
                }
            ],
            route_hint=None if missing_target else route,
            needs_human=missing_target,
        )
