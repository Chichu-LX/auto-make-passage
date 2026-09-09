from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import json
import math
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.project_layout import artifact_prefix, task_project_slug
from research_agents.process_logging import ProcessLogStore
from research_agents.providers import JsonModel
from research_agents.remote_execution import (
    RemoteExecutionBackend,
    RemoteExecutionError,
    SafeSSHBackend,
    available_gpu_keys,
    load_server_execution_config,
    project_execution,
)
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    CodeImplementationManifest,
    CodeReviewReport,
    ExperimentCodeRunManifest,
    ExperimentCodeRunReport,
    ExperimentCompletenessCheck,
    ExperimentCoverageAudit,
    ExperimentCoverageDraft,
    ExperimentMetricAggregate,
    ExperimentPlan,
    ExperimentSeedMetric,
    JobObservation,
    ProcessLogRecord,
    ResearchDesign,
    ResultCheckReport,
    ScheduledExperimentJob,
    ServerDemoRecord,
    ServerExecutionConfig,
    ServerScheduleManifest,
    TaskEnvelope,
    WorkflowState,
    utc_now,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_INPUT_BYTES,
    MAX_JOB_SPEC_BYTES,
    MAX_LOG_SNAPSHOT_BYTES,
    MAX_REPORT_BYTES,
    MAX_SUPPLEMENTARY_JOBS,
    REPORT_INTERVAL_SECONDS,
    file_token,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.EXPERIMENT_CODE_RUN:
    raise RuntimeError("Experiment Code Run Agent package name mismatch")


SUPPLEMENTARY_PRIMARY_METRIC = "supplementary_primary_rmse"


@dataclass
class _ResolvedInputs:
    design: ResearchDesign
    design_ref: ArtifactRef
    code: CodeImplementationManifest
    code_ref: ArtifactRef
    review: CodeReviewReport
    review_ref: ArtifactRef
    schedule: ServerScheduleManifest
    schedule_ref: ArtifactRef
    main_result: ResultCheckReport
    main_result_ref: ArtifactRef
    launch: ExperimentCodeRunManifest | None
    launch_ref: ArtifactRef | None
    reports: list[tuple[ExperimentCodeRunReport, ArtifactRef]]


class ExperimentCodeRunAgent:
    """Audit, launch, monitor, and truthfully aggregate supplementary experiments."""

    name = AgentName.EXPERIMENT_CODE_RUN
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
        self.phase_logs = ProcessLogStore(artifacts.root)

    def _log_phase(
        self,
        task: TaskEnvelope,
        *,
        event: str,
        status: str,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.phase_logs.append(
            ProcessLogRecord(
                run_id=task.run_id,
                agent=self.name,
                event=event,
                status=status,
                workflow_state=WorkflowState.EXPERIMENT_CODE_RUN,
                state_version=max(1, int(task.context.get("state_version", 1))),
                task_id=task.task_id,
                attempt=task.attempt,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def _config(self) -> ServerExecutionConfig:
        return self.execution_config or load_server_execution_config(self.artifacts.root)

    def _read_candidate(self, artifact: ArtifactRef, model_type):
        return model_type.model_validate(
            self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
        )

    def _read_inputs(self, task: TaskEnvelope) -> _ResolvedInputs | None:
        slug = task_project_slug(task)
        design_candidates: list[tuple[ResearchDesign, ArtifactRef]] = []
        code_candidates: list[tuple[CodeImplementationManifest, ArtifactRef]] = []
        review_candidates: list[tuple[CodeReviewReport, ArtifactRef]] = []
        schedule_candidates: list[tuple[ServerScheduleManifest, ArtifactRef]] = []
        main_result_candidates: list[tuple[ResultCheckReport, ArtifactRef]] = []
        launch_candidates: list[tuple[ExperimentCodeRunManifest, ArtifactRef]] = []
        report_candidates: list[tuple[ExperimentCodeRunReport, ArtifactRef]] = []
        for artifact in task.inputs:
            try:
                if artifact.kind == "research_design" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "research-designs")
                ):
                    value = self._read_candidate(artifact, ResearchDesign)
                    if value.run_id == task.run_id:
                        design_candidates.append((value, artifact))
                elif artifact.kind == "code_implementation_manifest" and artifact.uri.startswith(
                    artifact_prefix("code", slug, "manifests")
                ):
                    value = self._read_candidate(artifact, CodeImplementationManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        code_candidates.append((value, artifact))
                elif artifact.kind == "code_review_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "code")
                ):
                    value = self._read_candidate(artifact, CodeReviewReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        review_candidates.append((value, artifact))
                elif artifact.kind == "server_schedule_manifest" and artifact.uri.startswith(
                    f"artifact://project/var/runs/{task.run_id}/server_schedule/"
                ):
                    value = self._read_candidate(artifact, ServerScheduleManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        schedule_candidates.append((value, artifact))
                elif artifact.kind == "result_check_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "results")
                ):
                    value = self._read_candidate(artifact, ResultCheckReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        main_result_candidates.append((value, artifact))
                elif artifact.kind == "experiment_code_run_manifest" and artifact.uri.startswith(
                    f"artifact://project/var/runs/{task.run_id}/experiment_code_run/"
                ):
                    value = self._read_candidate(artifact, ExperimentCodeRunManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        launch_candidates.append((value, artifact))
                elif artifact.kind == "experiment_code_run_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "experiments", "results")
                ):
                    value = self._read_candidate(artifact, ExperimentCodeRunReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        report_candidates.append((value, artifact))
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        if not all(
            (
                design_candidates,
                code_candidates,
                review_candidates,
                schedule_candidates,
                main_result_candidates,
            )
        ):
            return None
        design, design_ref = design_candidates[-1]
        code, code_ref = code_candidates[-1]
        reviews = [
            item
            for item in review_candidates
            if item[0].verdict == "passed" and item[0].code_manifest_uri == code_ref.uri
        ]
        if not reviews:
            return None
        review, review_ref = reviews[-1]
        schedules = [
            item
            for item in schedule_candidates
            if item[0].status == "launched"
            and item[0].research_design_uri == design_ref.uri
            and item[0].code_manifest_uri == code_ref.uri
            and item[0].code_review_uri == review_ref.uri
        ]
        if not schedules:
            return None
        schedule, schedule_ref = schedules[-1]
        main_results = [
            item
            for item in main_result_candidates
            if item[0].verdict == "passed"
            and item[0].schedule_manifest_uri == schedule_ref.uri
        ]
        if not main_results:
            return None
        main_result, main_result_ref = main_results[-1]
        launches = [
            item
            for item in launch_candidates
            if item[0].research_design_uri == design_ref.uri
            and item[0].code_manifest_uri == code_ref.uri
            and item[0].code_review_uri == review_ref.uri
            and item[0].main_schedule_manifest_uri == schedule_ref.uri
            and item[0].main_result_report_uri == main_result_ref.uri
        ]
        launch: ExperimentCodeRunManifest | None = None
        launch_ref: ArtifactRef | None = None
        if launches:
            launch, launch_ref = launches[-1]
        reports = [
            item
            for item in report_candidates
            if launch_ref is not None and item[0].run_manifest_uri == launch_ref.uri
        ]
        reports.sort(key=lambda item: item[0].report_sequence)
        return _ResolvedInputs(
            design=design,
            design_ref=design_ref,
            code=code,
            code_ref=code_ref,
            review=review,
            review_ref=review_ref,
            schedule=schedule,
            schedule_ref=schedule_ref,
            main_result=main_result,
            main_result_ref=main_result_ref,
            launch=launch,
            launch_ref=launch_ref,
            reports=reports,
        )

    def _write_gap(
        self,
        task: TaskEnvelope,
        *,
        reason: str,
        issue_code: str,
    ) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            (
                f"design/{task_project_slug(task)}/experiments/gaps/"
                f"{file_token(task.task_id)}.json"
            ),
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": task_project_slug(task),
                "status": "blocked",
                "issue_code": issue_code,
                "reason": reason,
            },
        ).model_copy(update={"kind": "experiment_code_run_gap"})

    @staticmethod
    def _supplementary_plans(design: ResearchDesign) -> list[ExperimentPlan]:
        return [
            item
            for item in design.experiment_plans
            if item.experiment_type != "main_benchmark"
        ]

    @staticmethod
    def _validate_main_coverage(inputs: _ResolvedInputs) -> None:
        main_plans = [
            item
            for item in inputs.design.experiment_plans
            if item.experiment_type == "main_benchmark"
        ]
        expected = {
            (item.experiment_id, seed)
            for item in main_plans
            for seed in item.random_seeds
        }
        observed = {(item.experiment_id, item.seed) for item in inputs.schedule.jobs}
        if not expected or observed != expected:
            raise RemoteExecutionError(
                "passed main schedule does not exactly cover designed main benchmarks"
            )
        scheduled_job_ids = {item.job_id for item in inputs.schedule.jobs}
        observed_job_ids = {item.job_id for item in inputs.main_result.observations}
        if len(observed_job_ids) != len(inputs.main_result.observations):
            raise RemoteExecutionError("passed main result duplicates job observations")
        if observed_job_ids != scheduled_job_ids:
            raise RemoteExecutionError(
                "passed main result does not exactly observe the launched main jobs"
            )
        if any(item.status != "completed" for item in inputs.main_result.observations):
            raise RemoteExecutionError("passed main result contains a non-completed job")
        if any(item.status != "passed" for item in inputs.main_result.metric_checks):
            raise RemoteExecutionError("passed main result contains a non-passing metric gate")

    def _coverage_audit(
        self,
        task: TaskEnvelope,
        inputs: _ResolvedInputs,
    ) -> ExperimentCoverageAudit:
        plans = self._supplementary_plans(inputs.design)
        payload = {
            "objective": task.objective,
            "method_name": inputs.design.method_name,
            "executive_summary": inputs.design.executive_summary,
            "method_components": [
                item.model_dump(mode="json") for item in inputs.design.method_components
            ],
            "existing_supplementary_experiments": [
                item.model_dump(mode="json") for item in plans
            ],
            "expected_results": [
                item.model_dump(mode="json") for item in inputs.design.expected_results
            ],
            "paper_comparisons": [
                item.model_dump(mode="json") for item in inputs.design.paper_comparisons
            ],
            "claim_boundaries": inputs.design.claim_boundaries,
            "unresolved_questions": inputs.design.unresolved_questions,
            "user_revision": task.context.get("user_revision"),
            "required_output": {
                "summary": "grounded coverage conclusion",
                "missing_requirements": [],
            },
            "constraints": [
                "Only necessary computational experiments may be reported missing.",
                "Do not repeat an experiment type and objective already covered by the design.",
                "Every gap must support an explicit design claim or expected result.",
                "Never propose wet-lab, biological, animal, or clinical work.",
            ],
        }
        response = self.model.complete_json(
            PACKAGE.system_prompt,
            json.dumps(payload, ensure_ascii=False),
        )
        draft = ExperimentCoverageDraft.model_validate(response)
        allowed_evidence_refs = {
            ref
            for component in inputs.design.method_components
            for ref in component.evidence_refs
        }
        allowed_evidence_refs.update(
            ref
            for experiment in inputs.design.experiment_plans
            for ref in experiment.evidence_refs
        )
        allowed_evidence_refs.update(
            ref
            for expected in inputs.design.expected_results
            for ref in expected.evidence_refs
        )
        allowed_evidence_refs.update(
            item.paper_id for item in inputs.design.paper_comparisons
        )
        existing_ids = {item.experiment_id for item in inputs.design.experiment_plans}
        for requirement in draft.missing_requirements:
            if requirement.requirement_id in existing_ids:
                raise RemoteExecutionError(
                    "missing experiment requirement reuses an existing experiment ID"
                )
            if not set(requirement.evidence_refs).issubset(allowed_evidence_refs):
                raise RemoteExecutionError(
                    "missing experiment requirement cites evidence outside the approved design"
                )
        if not plans and not draft.missing_requirements:
            raise RemoteExecutionError(
                "a design with no supplementary experiment requires an explicit design revision"
            )
        designed = [item.experiment_id for item in plans]
        manifest_covered = {
            experiment_id
            for item in inputs.code.traceability
            for experiment_id in item.experiment_ids
        }
        independently_reviewed = set(inputs.review.covered_experiment_ids)
        code_covered = sorted(
            set(designed).intersection(manifest_covered, independently_reviewed)
        )
        missing_code = sorted(set(designed) - set(code_covered))
        if draft.missing_requirements:
            decision = "requires_design_revision"
        elif missing_code:
            decision = "requires_code_revision"
        else:
            decision = "ready"
        return ExperimentCoverageAudit(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            research_design_uri=inputs.design_ref.uri,
            code_manifest_uri=inputs.code_ref.uri,
            main_result_report_uri=inputs.main_result_ref.uri,
            decision=decision,
            designed_experiment_ids=designed,
            code_covered_experiment_ids=code_covered,
            missing_code_experiment_ids=missing_code,
            missing_requirements=draft.missing_requirements,
            summary=draft.summary,
        )

    @staticmethod
    def _coverage_markdown(audit: ExperimentCoverageAudit) -> str:
        lines = [
            "# Supplementary Experiment Coverage Audit",
            "",
            f"- Decision: `{audit.decision}`",
            f"- Designed non-main experiments: `{len(audit.designed_experiment_ids)}`",
            f"- Reviewed-code covered experiments: `{len(audit.code_covered_experiment_ids)}`",
            f"- Missing reviewed-code experiments: `{len(audit.missing_code_experiment_ids)}`",
            f"- Missing design requirements: `{len(audit.missing_requirements)}`",
            "",
            audit.summary,
            "",
            "## Designed execution set",
            "",
        ]
        lines.extend(f"- `{value}`" for value in audit.designed_experiment_ids)
        if audit.missing_code_experiment_ids:
            lines.extend(["", "## Missing reviewed-code coverage", ""])
            lines.extend(f"- `{value}`" for value in audit.missing_code_experiment_ids)
        if audit.missing_requirements:
            lines.extend(["", "## Required design additions", ""])
            for item in audit.missing_requirements:
                lines.append(
                    f"- `{item.requirement_id}` / `{item.experiment_type}`: {item.objective} "
                    f"(reason: {item.rationale})"
                )
        return "\n".join(lines) + "\n"

    def _write_coverage(
        self, task: TaskEnvelope, audit: ExperimentCoverageAudit
    ) -> tuple[ArtifactRef, ArtifactRef]:
        base = (
            f"design/{task_project_slug(task)}/experiments/coverage/"
            f"{file_token(task.task_id)}"
        )
        json_ref = self.artifacts.write_agent_json(
            self.name, f"{base}.json", audit.model_dump(mode="json")
        ).model_copy(update={"kind": "experiment_coverage_audit"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._coverage_markdown(audit).encode("utf-8"),
            max_bytes=MAX_REPORT_BYTES,
            kind="experiment_coverage_markdown",
        )
        return json_ref, markdown_ref

    @staticmethod
    def _expand_argv(
        project,
        *,
        experiment_id: str,
        seed: int,
        workspace: str,
        workdir: str,
        metrics_path: str,
        gpu_count: int,
    ) -> list[str]:
        replacements = {
            "{experiment_id}": experiment_id,
            "{seed}": str(seed),
            "{data_dir}": str(PurePosixPath(workspace, "data")),
            "{model_dir}": str(PurePosixPath(workspace, "models")),
            "{output_dir}": workdir,
            "{metrics_path}": metrics_path,
        }
        argv: list[str] = []
        for raw in project.train_argv:
            value = raw
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            argv.append(value)
        if project.distributed_mode == "torchrun" and argv[0].endswith("torchrun"):
            if not any(value.startswith("--nproc_per_node") for value in argv):
                argv.insert(1, f"--nproc_per_node={gpu_count}")
        return argv

    def _allocate_jobs(
        self,
        task: TaskEnvelope,
        inputs: _ResolvedInputs,
        project,
        selected_gpu_keys: list[str],
    ) -> tuple[list[ScheduledExperimentJob], dict[str, str], int]:
        definitions = [
            (plan.experiment_id, seed)
            for plan in self._supplementary_plans(inputs.design)
            for seed in plan.random_seeds
        ]
        if not definitions or len(definitions) > MAX_SUPPLEMENTARY_JOBS:
            raise RemoteExecutionError("supplementary job count is empty or exceeds policy")
        by_server: dict[str, list[int]] = defaultdict(list)
        for key in selected_gpu_keys:
            server_id, raw_gpu = key.rsplit(":", 1)
            by_server[server_id].append(int(raw_gpu))
        if project.distributed_mode == "torchrun":
            slots = [
                (server_id, sorted(gpus))
                for server_id, gpus in sorted(by_server.items())
            ]
        else:
            slots = [
                (key.rsplit(":", 1)[0], [int(key.rsplit(":", 1)[1])])
                for key in selected_gpu_keys
            ]
        if not slots:
            raise RemoteExecutionError("no current accelerator slot is available for experiments")
        launch_token = file_token(task.task_id)
        run_roots = {
            server_id: str(
                PurePosixPath(
                    inputs.schedule.server_workspaces[server_id],
                    "experiment-runs",
                    launch_token,
                )
            )
            for server_id in by_server
        }
        jobs: list[ScheduledExperimentJob] = []
        for index, (experiment_id, seed) in enumerate(definitions):
            server_id, gpu_ids = slots[index % len(slots)]
            main_workspace = inputs.schedule.server_workspaces[server_id]
            run_root = run_roots[server_id]
            job_id = f"job-exp-{file_token(f'{task.task_id}:{experiment_id}:{seed}')[:16]}"
            workdir = str(PurePosixPath(run_root, "jobs", job_id))
            metrics_path = str(PurePosixPath(workdir, "metrics.jsonl"))
            jobs.append(
                ScheduledExperimentJob(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    seed=seed,
                    server_id=server_id,
                    gpu_ids=gpu_ids,
                    command_argv=self._expand_argv(
                        project,
                        experiment_id=experiment_id,
                        seed=seed,
                        workspace=main_workspace,
                        workdir=workdir,
                        metrics_path=metrics_path,
                        gpu_count=len(gpu_ids),
                    ),
                    remote_workdir=workdir,
                    log_path=str(PurePosixPath(run_root, "logs", f"{job_id}.log")),
                    metrics_path=metrics_path,
                    best_checkpoint_dir=str(PurePosixPath(run_root, "best", job_id)),
                    gpu_lock_path=str(
                        PurePosixPath(
                            run_root,
                            "runtime",
                            "gpu-locks",
                            "gpu-" + "_".join(str(value) for value in gpu_ids) + ".lock",
                        )
                    ),
                    expected_duration_seconds=project.expected_job_seconds,
                    # Main scheduling and supplementary aggregation have
                    # different evidence contracts.  Main jobs retain the
                    # project-configured validation metric; every supplementary
                    # CLI job emits this truthful, experiment-specific RMSE
                    # summary in addition to its full metric set.
                    metric_name=SUPPLEMENTARY_PRIMARY_METRIC,
                    metric_direction=project.metric_direction,
                    status="planned",
                )
            )
        makespan = math.ceil(len(jobs) / len(slots)) * project.expected_job_seconds
        return jobs, run_roots, makespan

    @staticmethod
    def _minimum_demo_argv(code: CodeImplementationManifest) -> list[str]:
        demo = next(
            (
                item
                for item in code.local_validations
                if item.validation_id == "minimal_demo" and item.status == "passed"
            ),
            None,
        )
        if demo is None or Path(demo.argv[0]).name not in {"python", "python3"}:
            raise RemoteExecutionError("passed code manifest has no valid Python minimum demo")
        argv = list(demo.argv)
        argv[0] = "python"
        return argv

    def _local_artifact_path(self, artifact: ArtifactRef) -> Path:
        self.artifacts.verify_artifact_file(artifact, max_bytes=MAX_JOB_SPEC_BYTES)
        target = (
            self.artifacts.root / artifact.uri.removeprefix("artifact://")
        ).resolve()
        target.relative_to(self.artifacts.root)
        return target

    @staticmethod
    def _launch_markdown(manifest: ExperimentCodeRunManifest) -> str:
        lines = [
            "# Experiment Code Run Launch",
            "",
            f"- Status: `{manifest.status}`",
            f"- Experiments: `{len(manifest.experiment_ids)}`",
            f"- Jobs: `{len(manifest.jobs)}`",
            f"- Selected accelerator slots: `{', '.join(manifest.selected_gpu_keys) or 'none'}`",
            f"- Estimated makespan: `{manifest.estimated_makespan_seconds}` seconds",
            f"- Next report: `{manifest.next_report_at.isoformat() if manifest.next_report_at else 'none'}`",
            "",
            "| Job | Experiment | Seed | Server | Accelerator slots | Status |",
            "|---|---|---:|---|---|---|",
        ]
        for job in manifest.jobs:
            lines.append(
                f"| `{job.job_id}` | `{job.experiment_id}` | {job.seed} | "
                f"`{job.server_id}` | `{','.join(str(value) for value in job.gpu_ids)}` | "
                f"`{job.status}` |"
            )
        if manifest.issues:
            lines.extend(["", "## Issues", ""])
            lines.extend(f"- {value}" for value in manifest.issues)
        lines.extend(
            [
                "",
                "All jobs use the reviewed immutable code and the existing managed nohup runner. ",
                "No numeric result has been predicted or generated in this launch manifest.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _write_launch(
        self, task: TaskEnvelope, manifest: ExperimentCodeRunManifest
    ) -> tuple[ArtifactRef, ArtifactRef]:
        base = (
            f"project/var/runs/{task.run_id}/experiment_code_run/"
            f"{file_token(task.task_id)}/launch"
        )
        json_ref = self.artifacts.write_agent_json(
            self.name, f"{base}.json", manifest.model_dump(mode="json")
        ).model_copy(update={"kind": "experiment_code_run_manifest"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._launch_markdown(manifest).encode("utf-8"),
            max_bytes=MAX_REPORT_BYTES,
            kind="experiment_code_run_launch_markdown",
        )
        return json_ref, markdown_ref

    def _launch(
        self,
        task: TaskEnvelope,
        inputs: _ResolvedInputs,
        coverage_refs: tuple[ArtifactRef, ArtifactRef],
    ) -> AgentResult:
        try:
            config = self._config()
            project = project_execution(config, task_project_slug(task))
        except (RemoteExecutionError, ValidationError) as exc:
            reason = f"Supplementary server configuration is unavailable: {type(exc).__name__}."
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_SERVER_CONFIG_INVALID"
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="needs_human",
                summary=reason,
                artifacts=[*coverage_refs, gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "EXPERIMENT_SERVER_CONFIG_INVALID",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_SERVER_CONFIG_INVALID:config",
                    }
                ],
                needs_human=True,
            )
        profiles = {
            item.server_id: item
            for item in config.servers
            if item.enabled
            and item.execution_enabled
            and item.server_id in inputs.schedule.server_workspaces
        }
        probes = {
            server_id: self.backend.probe(profile)
            for server_id, profile in profiles.items()
        }
        current_gpu_keys: list[str] = []
        main_gpu_keys = set(inputs.schedule.selected_gpu_keys)
        for server_id, profile in sorted(
            profiles.items(), key=lambda item: item[1].priority, reverse=True
        ):
            current_gpu_keys.extend(
                key
                for key in available_gpu_keys(
                    profile, probes[server_id], transfer_bytes=0
                )
                if key in main_gpu_keys
            )
        if not current_gpu_keys:
            reason = (
                "No accelerator slot from the passed main schedule is currently eligible for supplementary experiments."
            )
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_NO_ELIGIBLE_GPU"
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[*coverage_refs, gap],
                issues=[
                    {
                        "severity": "error",
                        "code": "EXPERIMENT_NO_ELIGIBLE_GPU",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_NO_ELIGIBLE_GPU:resources",
                    }
                ],
                route_hint=AgentName.SERVER_SCHEDULE,
            )
        try:
            jobs, run_roots, makespan = self._allocate_jobs(
                task, inputs, project, current_gpu_keys
            )
        except RemoteExecutionError as exc:
            reason = f"Supplementary allocation failed closed: {type(exc).__name__}."
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_ALLOCATION_INVALID"
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[*coverage_refs, gap],
                issues=[
                    {
                        "severity": "error",
                        "code": "EXPERIMENT_ALLOCATION_INVALID",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_ALLOCATION_INVALID:design",
                    }
                ],
                route_hint=AgentName.IDEA_DESIGN,
            )
        spec_refs: dict[str, ArtifactRef] = {}
        for job in jobs:
            spec_ref = self.artifacts.write_agent_json(
                self.name,
                (
                    f"project/var/runs/{task.run_id}/experiment_code_run/"
                    f"{file_token(task.task_id)}/specs/{job.job_id}.json"
                ),
                {
                    "job_id": job.job_id,
                    "experiment_id": job.experiment_id,
                    "seed": job.seed,
                    "gpu_ids": job.gpu_ids,
                    "command_argv": job.command_argv,
                    "remote_workdir": job.remote_workdir,
                    "log_path": job.log_path,
                    "metrics_path": job.metrics_path,
                    "best_checkpoint_dir": job.best_checkpoint_dir,
                    "gpu_lock_path": job.gpu_lock_path,
                    "metric_name": job.metric_name,
                    "metric_direction": job.metric_direction,
                },
            ).model_copy(update={"kind": "experiment_job_spec"})
            spec_refs[job.job_id] = spec_ref
        server_demos: list[ServerDemoRecord] = []
        launched_jobs: list[ScheduledExperimentJob] = []
        issues: list[str] = []
        selected_servers = sorted({item.server_id for item in jobs})
        try:
            demo_argv = self._minimum_demo_argv(inputs.code)
            for server_id in selected_servers:
                profile = profiles[server_id]
                run_root = run_roots[server_id]
                for job in (item for item in jobs if item.server_id == server_id):
                    remote_spec = str(
                        PurePosixPath(
                            run_root, "runtime", "specs", f"{job.job_id}.json"
                        )
                    )
                    self.backend.transfer_file(
                        profile,
                        local_path=self._local_artifact_path(spec_refs[job.job_id]),
                        remote_path=remote_spec,
                    )
                gpu_ids = sorted(
                    {
                        gpu_id
                        for job in jobs
                        if job.server_id == server_id
                        for gpu_id in job.gpu_ids
                    }
                )
                main_workspace = inputs.schedule.server_workspaces[server_id]
                server_demos.append(
                    self.backend.run_minimum_demo(
                        profile,
                        environment_name=inputs.schedule.environment_names[server_id],
                        code_root=str(PurePosixPath(main_workspace, "code")),
                        argv=demo_argv,
                        gpu_ids=gpu_ids,
                        log_path=str(
                            PurePosixPath(run_root, "logs", "foreground-demo.log")
                        ),
                        timeout_seconds=project.server_demo_timeout_seconds,
                    )
                )
            if any(item.status != "passed" for item in server_demos):
                issues.append(
                    "At least one supplementary server failed its foreground minimum demo; no experiment job was launched."
                )
            else:
                for job in jobs:
                    run_root = run_roots[job.server_id]
                    remote_job_id = self.backend.launch_job(
                        profiles[job.server_id],
                        environment_name=inputs.schedule.environment_names[job.server_id],
                        runner_path=str(
                            PurePosixPath(
                                inputs.schedule.server_workspaces[job.server_id],
                                "runtime",
                                "managed_runner.py",
                            )
                        ),
                        spec_path=str(
                            PurePosixPath(
                                run_root,
                                "runtime",
                                "specs",
                                f"{job.job_id}.json",
                            )
                        ),
                        bootstrap_log_path=str(
                            PurePosixPath(
                                run_root, "logs", f"{job.job_id}.bootstrap.log"
                            )
                        ),
                    )
                    started_at = utc_now()
                    launched_jobs.append(
                        job.model_copy(
                            update={
                                "status": "launched",
                                "remote_job_id": remote_job_id,
                                "started_at": started_at,
                                "next_report_at": started_at
                                + timedelta(
                                    seconds=min(
                                        REPORT_INTERVAL_SECONDS,
                                        job.expected_duration_seconds,
                                    )
                                ),
                            }
                        )
                    )
        except (RemoteExecutionError, OSError, StoragePolicyError) as exc:
            issues.append(
                f"Supplementary transfer, demo, or nohup launch failed closed: {type(exc).__name__}."
            )
        if len(launched_jobs) == len(jobs) and not issues:
            status = "launched"
        elif launched_jobs:
            status = "partial"
        else:
            status = "blocked"
        materialized_jobs = launched_jobs + jobs[len(launched_jobs) :]
        next_report_at = (
            min(item.next_report_at for item in launched_jobs if item.next_report_at)
            if launched_jobs
            else None
        )
        seed_matrix = {
            item.experiment_id: list(item.random_seeds)
            for item in self._supplementary_plans(inputs.design)
        }
        manifest = ExperimentCodeRunManifest(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            status=status,
            research_design_uri=inputs.design_ref.uri,
            code_manifest_uri=inputs.code_ref.uri,
            code_review_uri=inputs.review_ref.uri,
            main_schedule_manifest_uri=inputs.schedule_ref.uri,
            main_result_report_uri=inputs.main_result_ref.uri,
            coverage_audit_uri=coverage_refs[0].uri,
            experiment_ids=[item.experiment_id for item in self._supplementary_plans(inputs.design)],
            planned_seed_matrix=seed_matrix,
            probes=list(probes.values()),
            selected_gpu_keys=current_gpu_keys,
            server_workspaces=run_roots,
            environment_names={
                server_id: inputs.schedule.environment_names[server_id]
                for server_id in selected_servers
            },
            server_demos=server_demos,
            job_spec_artifacts=list(spec_refs.values()),
            jobs=materialized_jobs,
            estimated_makespan_seconds=makespan,
            next_report_at=next_report_at,
            issues=issues,
        )
        launch_refs = self._write_launch(task, manifest)
        artifacts = [*coverage_refs, *launch_refs, *spec_refs.values()]
        if status != "launched":
            issue_code = (
                "EXPERIMENT_FOREGROUND_DEMO_FAILED"
                if server_demos and any(item.status != "passed" for item in server_demos)
                else "EXPERIMENT_PARTIAL_LAUNCH"
            )
            reason = issues[0] if issues else "Supplementary experiment launch did not complete."
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=artifacts,
                issues=[
                    {
                        "severity": "error",
                        "code": issue_code,
                        "message": reason,
                        "root_cause_key": f"G8:{issue_code}:remote",
                    }
                ],
                route_hint=AgentName.SERVER_SCHEDULE,
            )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="in_progress",
            summary=(
                f"Main results passed; launched {len(jobs)} approved supplementary jobs for "
                f"{len(seed_matrix)} experiments across {len(current_gpu_keys)} accelerator slots after "
                f"{len(server_demos)} foreground demos. Next evidence report is due by "
                f"{next_report_at.isoformat()}."
            ),
            artifacts=artifacts,
            evidence=[
                {
                    "check": "main-result-entry-gate",
                    "verdict": inputs.main_result.verdict,
                    "metric_gate_count": len(inputs.main_result.metric_checks),
                },
                {
                    "check": "supplementary-experiment-launch",
                    "experiment_count": len(seed_matrix),
                    "job_count": len(jobs),
                    "gpu_count": len(current_gpu_keys),
                    "nohup_managed": True,
                },
            ],
        )

    def _write_log_snapshot(
        self,
        task: TaskEnvelope,
        sequence: int,
        observation: JobObservation,
    ) -> ArtifactRef:
        return self.artifacts.write_agent_bytes(
            self.name,
            (
                f"project/var/logs/jobs/{task.run_id}/experiment_code_run/"
                f"report-{sequence:04d}/{observation.job_id}.json"
            ),
            (
                json.dumps(
                    observation.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
            max_bytes=MAX_LOG_SNAPSHOT_BYTES,
            kind="experiment_job_log_snapshot",
        )

    @staticmethod
    def _failed_route(observations: list[JobObservation]) -> AgentName:
        text = " ".join(item.log_tail.lower() for item in observations)
        if any(value in text for value in ("dataset", "data file", "no such file")):
            return AgentName.DOWNLOAD_DATA
        if any(
            value in text
            for value in (
                "cuda out of memory",
                "cublas",
                "nccl",
                "ssh",
                "no space left",
                "gpu unavailable",
            )
        ):
            return AgentName.SERVER_SCHEDULE
        return AgentName.CODE_MAKE

    def _checks_and_aggregates(
        self,
        inputs: _ResolvedInputs,
        observations: list[JobObservation],
    ) -> tuple[list[ExperimentCompletenessCheck], list[ExperimentMetricAggregate]]:
        assert inputs.launch is not None
        by_job = {item.job_id: item for item in observations}
        expected_results: dict[str, list[str]] = defaultdict(list)
        for item in inputs.design.expected_results:
            expected_results[item.experiment_id].append(item.result_id)
        plans = {
            item.experiment_id: item for item in self._supplementary_plans(inputs.design)
        }
        checks: list[ExperimentCompletenessCheck] = []
        aggregates: list[ExperimentMetricAggregate] = []
        for experiment_id in inputs.launch.experiment_ids:
            plan = plans[experiment_id]
            jobs = [
                item for item in inputs.launch.jobs if item.experiment_id == experiment_id
            ]
            completed_seeds = sorted(
                item.seed
                for item in jobs
                if by_job[item.job_id].status == "completed"
            )
            seed_metrics: list[ExperimentSeedMetric] = []
            for job in jobs:
                observation = by_job[job.job_id]
                best = observation.best_metric
                if (
                    observation.status == "completed"
                    and best is not None
                    and best.metric_name == job.metric_name
                    and math.isfinite(best.value)
                ):
                    seed_metrics.append(
                        ExperimentSeedMetric(
                            job_id=job.job_id,
                            seed=job.seed,
                            value=best.value,
                            checkpoint_path=(
                                observation.best_checkpoint_path
                                or best.checkpoint_path
                            ),
                        )
                    )
            metric_seeds = sorted(item.seed for item in seed_metrics)
            expected = list(inputs.launch.planned_seed_matrix[experiment_id])
            active = any(
                by_job[item.job_id].status in {"queued", "running", "unknown"}
                for item in jobs
            )
            failed = any(by_job[item.job_id].status == "failed" for item in jobs)
            if active:
                status = "pending"
            elif failed or set(completed_seeds) != set(expected) or set(metric_seeds) != set(expected):
                status = "failed"
            else:
                status = "passed"
            checks.append(
                ExperimentCompletenessCheck(
                    experiment_id=experiment_id,
                    experiment_type=plan.experiment_type,
                    expected_seeds=expected,
                    completed_seeds=completed_seeds,
                    metric_seeds=metric_seeds,
                    expected_result_ids=expected_results.get(experiment_id, []),
                    status=status,
                    summary=(
                        f"Observed {len(completed_seeds)}/{len(expected)} completed seeds and "
                        f"{len(metric_seeds)}/{len(expected)} finite primary-metric seeds."
                    ),
                )
            )
            if status == "passed":
                values = [item.value for item in seed_metrics]
                mean = sum(values) / len(values)
                sample_std = (
                    math.sqrt(
                        sum((value - mean) ** 2 for value in values)
                        / (len(values) - 1)
                    )
                    if len(values) > 1
                    else None
                )
                aggregates.append(
                    ExperimentMetricAggregate(
                        experiment_id=experiment_id,
                        metric_name=jobs[0].metric_name,
                        direction=jobs[0].metric_direction,
                        seed_metrics=seed_metrics,
                        mean=mean,
                        sample_standard_deviation=sample_std,
                        minimum=min(values),
                        maximum=max(values),
                    )
                )
        return checks, aggregates

    @staticmethod
    def _report_markdown(
        report: ExperimentCodeRunReport,
        design: ResearchDesign,
    ) -> str:
        plans = {item.experiment_id: item for item in design.experiment_plans}
        lines = [
            "# Supplementary Experiment Result Report",
            "",
            f"- Verdict: `{report.verdict}`",
            f"- Report sequence: `{report.report_sequence}`",
            f"- Jobs observed: `{len(report.observations)}`",
            f"- Next report: `{report.next_report_at.isoformat() if report.next_report_at else 'terminal'}`",
            "",
            report.summary,
            "",
            "## Exact primary-metric aggregates",
            "",
            "| Experiment | Type | Seeds | Metric | Mean | Sample SD | Min | Max |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for aggregate in report.primary_metric_aggregates:
            plan = plans[aggregate.experiment_id]
            lines.append(
                f"| `{aggregate.experiment_id}` | `{plan.experiment_type}` | "
                f"{len(aggregate.seed_metrics)} | `{aggregate.metric_name}` | "
                f"{aggregate.mean:.12g} | "
                f"{aggregate.sample_standard_deviation if aggregate.sample_standard_deviation is not None else 'n/a'} | "
                f"{aggregate.minimum:.12g} | {aggregate.maximum:.12g} |"
            )
        lines.extend(
            [
                "",
                "## Completeness",
                "",
                "| Experiment | Status | Completed seeds | Metric seeds | Expected-result IDs |",
                "|---|---|---|---|---|",
            ]
        )
        for check in report.completeness_checks:
            lines.append(
                f"| `{check.experiment_id}` | `{check.status}` | "
                f"`{','.join(str(value) for value in check.completed_seeds) or 'none'}` | "
                f"`{','.join(str(value) for value in check.metric_seeds) or 'none'}` | "
                f"`{','.join(check.expected_result_ids) or 'none'}` |"
            )
        lines.extend(
            [
                "",
                "Every number above is recomputed from persisted per-job JSONL observations. "
                "Missing values are reported as missing and are never imputed or model-generated.",
            ]
        )
        if report.issues:
            lines.extend(["", "## Issues", ""])
            lines.extend(f"- {value}" for value in report.issues)
        return "\n".join(lines) + "\n"

    def _write_report(
        self,
        task: TaskEnvelope,
        report: ExperimentCodeRunReport,
        design: ResearchDesign,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        base = (
            f"design/{task_project_slug(task)}/experiments/results/"
            f"{file_token(task.task_id)}-report-{report.report_sequence:04d}"
        )
        json_ref = self.artifacts.write_agent_json(
            self.name, f"{base}.json", report.model_dump(mode="json")
        ).model_copy(update={"kind": "experiment_code_run_report"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._report_markdown(report, design).encode("utf-8"),
            max_bytes=MAX_REPORT_BYTES,
            kind="experiment_code_run_report_markdown",
        )
        return json_ref, markdown_ref

    def _monitor(self, task: TaskEnvelope, inputs: _ResolvedInputs) -> AgentResult:
        assert inputs.launch is not None and inputs.launch_ref is not None
        now = utc_now()
        latest_due = (
            inputs.reports[-1][0].next_report_at
            if inputs.reports and inputs.reports[-1][0].verdict == "in_progress"
            else inputs.launch.next_report_at
        )
        if latest_due is not None and now < latest_due:
            current_ref = (
                inputs.reports[-1][1] if inputs.reports else inputs.launch_ref
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="in_progress",
                summary=(
                    "Supplementary jobs are not yet due for polling; the next evidence report "
                    f"is scheduled for {latest_due.isoformat()}."
                ),
                artifacts=[current_ref],
                evidence=[
                    {
                        "check": "supplementary-poll-rate-limit",
                        "next_report_at": latest_due.isoformat(),
                        "report_interval_seconds": REPORT_INTERVAL_SECONDS,
                    }
                ],
            )
        profiles = {item.server_id: item for item in self._config().servers}
        observations: list[JobObservation] = []
        for job in inputs.launch.jobs:
            profile = profiles.get(job.server_id)
            if profile is None:
                observations.append(JobObservation(job_id=job.job_id, status="unknown"))
            else:
                observations.append(self.backend.observe_job(profile, job))
        sequence = inputs.reports[-1][0].report_sequence + 1 if inputs.reports else 1
        log_refs = [
            self._write_log_snapshot(task, sequence, item) for item in observations
        ]
        checks, aggregates = self._checks_and_aggregates(inputs, observations)
        active = [
            item for item in observations if item.status in {"queued", "running", "unknown"}
        ]
        failed = [item for item in observations if item.status == "failed"]
        issues: list[str] = []
        route: AgentName | None = None
        next_report_at = None
        if failed:
            verdict = "failed"
            route = self._failed_route(failed)
            issues.append(
                f"{len(failed)} supplementary jobs failed with persisted status and log evidence."
            )
            summary = (
                f"Observed {len(failed)} failed and {len(active)} active supplementary jobs; "
                f"repair is routed to {route.value}."
            )
        elif active:
            verdict = "in_progress"
            next_report_at = now + timedelta(seconds=REPORT_INTERVAL_SECONDS)
            summary = (
                f"{sum(item.status == 'completed' for item in observations)}/{len(observations)} "
                "supplementary jobs completed; remaining jobs will be reported within 12 hours."
            )
        elif any(item.status != "passed" for item in checks):
            verdict = "failed"
            route = AgentName.CODE_MAKE
            issues.append(
                "Terminal supplementary jobs do not provide complete seed and finite primary-metric coverage."
            )
            summary = (
                "Supplementary execution terminated, but at least one designed experiment is "
                "missing a completed seed or finite primary metric."
            )
        else:
            verdict = "passed"
            summary = (
                f"All {len(observations)} supplementary jobs completed; exact primary metrics "
                f"were aggregated for {len(aggregates)} designed experiments and every declared seed."
            )
        report = ExperimentCodeRunReport(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            run_manifest_uri=inputs.launch_ref.uri,
            report_sequence=sequence,
            verdict=verdict,
            summary=summary,
            observations=observations,
            completeness_checks=checks,
            primary_metric_aggregates=aggregates,
            log_artifacts=log_refs,
            recommended_route=route,
            next_report_at=next_report_at,
            issues=issues,
        )
        report_refs = self._write_report(task, report, inputs.design)
        artifacts = [*report_refs, *log_refs]
        evidence: list[dict[str, Any]] = [
            {
                "check": "supplementary-job-status",
                "total": len(observations),
                "completed": sum(item.status == "completed" for item in observations),
                "active": len(active),
                "failed": len(failed),
            },
            {
                "check": "exact-seed-aggregation",
                "experiment_count": len(aggregates),
                "model_generated_numeric_results": 0,
            },
            {
                "check": "monitoring-policy",
                "unbounded_monitoring_cycles": True,
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
        assert route is not None
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="failed",
            summary=summary,
            artifacts=artifacts,
            evidence=evidence,
            issues=[
                {
                    "severity": "error",
                    "code": "EXPERIMENT_RESULTS_INCOMPLETE",
                    "message": issues[0],
                    "root_cause_key": f"G8:EXPERIMENT_RESULTS_INCOMPLETE:{route.value}",
                }
            ],
            route_hint=route,
        )

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        entry_started = perf_counter()
        self._log_phase(
            task,
            event="main_result_entry_gate_started",
            status="started",
        )
        inputs = self._read_inputs(task)
        if inputs is None:
            reason = (
                "Experiment execution requires a project-matched ResearchDesign, passed Code Review, "
                "launched main schedule, and latest passed main ResultCheckReport."
            )
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_MAIN_GATE_MISSING"
            )
            self._log_phase(
                task,
                event="main_result_entry_gate_completed",
                status="blocked",
                duration_ms=round((perf_counter() - entry_started) * 1000, 3),
                details={"issue_code": "EXPERIMENT_MAIN_GATE_MISSING"},
            )
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
                        "code": "EXPERIMENT_MAIN_GATE_MISSING",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_MAIN_GATE_MISSING:inputs",
                    }
                ],
                needs_human=True,
            )
        try:
            self._validate_main_coverage(inputs)
        except RemoteExecutionError as exc:
            reason = f"Main-result entry evidence is inconsistent: {type(exc).__name__}."
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_MAIN_GATE_INVALID"
            )
            self._log_phase(
                task,
                event="main_result_entry_gate_completed",
                status="failed",
                duration_ms=round((perf_counter() - entry_started) * 1000, 3),
                details={
                    "issue_code": "EXPERIMENT_MAIN_GATE_INVALID",
                    "error_type": type(exc).__name__,
                },
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "error",
                        "code": "EXPERIMENT_MAIN_GATE_INVALID",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_MAIN_GATE_INVALID:result",
                    }
                ],
                route_hint=AgentName.CHECK_RESULT,
            )
        self._log_phase(
            task,
            event="main_result_entry_gate_completed",
            status="succeeded",
            duration_ms=round((perf_counter() - entry_started) * 1000, 3),
            details={
                "main_job_count": len(inputs.schedule.jobs),
                "main_metric_gate_count": len(inputs.main_result.metric_checks),
            },
        )
        if inputs.launch is not None:
            if inputs.launch.status != "launched":
                reason = "The latest supplementary launch is partial or blocked and must be rescheduled."
                gap = self._write_gap(
                    task, reason=reason, issue_code="EXPERIMENT_LAUNCH_NOT_MONITORABLE"
                )
                return AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=self.name,
                    status="failed",
                    summary=reason,
                    artifacts=[gap],
                    issues=[
                        {
                            "severity": "error",
                            "code": "EXPERIMENT_LAUNCH_NOT_MONITORABLE",
                            "message": reason,
                            "root_cause_key": "G8:EXPERIMENT_LAUNCH_NOT_MONITORABLE:remote",
                        }
                    ],
                    route_hint=AgentName.SERVER_SCHEDULE,
                )
            monitor_started = perf_counter()
            self._log_phase(
                task,
                event="supplementary_monitoring_started",
                status="started",
                details={"job_count": len(inputs.launch.jobs)},
            )
            try:
                result = self._monitor(task, inputs)
                self._log_phase(
                    task,
                    event="supplementary_monitoring_completed",
                    status=(
                        "succeeded"
                        if result.status == "succeeded"
                        else "info"
                        if result.status == "in_progress"
                        else "failed"
                    ),
                    duration_ms=round((perf_counter() - monitor_started) * 1000, 3),
                    details={
                        "result_status": result.status,
                        "artifact_kinds": [item.kind for item in result.artifacts],
                    },
                )
                return result
            except (
                RemoteExecutionError,
                ValidationError,
                StoragePolicyError,
                OSError,
                KeyError,
            ) as exc:
                reason = f"Supplementary monitoring failed closed: {type(exc).__name__}."
                gap = self._write_gap(
                    task, reason=reason, issue_code="EXPERIMENT_MONITOR_INVALID"
                )
                self._log_phase(
                    task,
                    event="supplementary_monitoring_completed",
                    status="failed",
                    duration_ms=round((perf_counter() - monitor_started) * 1000, 3),
                    details={
                        "issue_code": "EXPERIMENT_MONITOR_INVALID",
                        "error_type": type(exc).__name__,
                    },
                )
                return AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=self.name,
                    status="failed",
                    summary=reason,
                    artifacts=[gap],
                    issues=[
                        {
                            "severity": "error",
                            "code": "EXPERIMENT_MONITOR_INVALID",
                            "message": reason,
                            "root_cause_key": "G8:EXPERIMENT_MONITOR_INVALID:remote",
                        }
                    ],
                    route_hint=AgentName.EXPERIMENT_CODE_RUN,
                )
        coverage_started = perf_counter()
        self._log_phase(
            task,
            event="supplementary_coverage_audit_started",
            status="started",
            details={
                "designed_non_main_count": len(
                    self._supplementary_plans(inputs.design)
                )
            },
        )
        try:
            audit = self._coverage_audit(task, inputs)
        except Exception as exc:
            reason = f"Experiment coverage audit failed closed: {type(exc).__name__}."
            gap = self._write_gap(
                task, reason=reason, issue_code="EXPERIMENT_COVERAGE_INVALID"
            )
            self._log_phase(
                task,
                event="supplementary_coverage_audit_completed",
                status="blocked",
                duration_ms=round((perf_counter() - coverage_started) * 1000, 3),
                details={
                    "issue_code": "EXPERIMENT_COVERAGE_INVALID",
                    "error_type": type(exc).__name__,
                },
            )
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
                        "code": "EXPERIMENT_COVERAGE_INVALID",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_COVERAGE_INVALID:model",
                    }
                ],
                needs_human=True,
            )
        self._log_phase(
            task,
            event="supplementary_coverage_audit_completed",
            status="succeeded",
            duration_ms=round((perf_counter() - coverage_started) * 1000, 3),
            details={
                "decision": audit.decision,
                "missing_design_count": len(audit.missing_requirements),
                "missing_code_count": len(audit.missing_code_experiment_ids),
            },
        )
        coverage_refs = self._write_coverage(task, audit)
        if audit.decision == "requires_design_revision":
            reason = (
                f"Coverage audit found {len(audit.missing_requirements)} necessary computational "
                "experiments that must enter the approved design before execution."
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=list(coverage_refs),
                issues=[
                    {
                        "severity": "error",
                        "code": "EXPERIMENT_DESIGN_GAPS",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_DESIGN_GAPS:design",
                    }
                ],
                route_hint=AgentName.IDEA_DESIGN,
            )
        if audit.decision == "requires_code_revision":
            reason = (
                f"Reviewed code lacks traceable implementation for {len(audit.missing_code_experiment_ids)} "
                "designed supplementary experiments."
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=list(coverage_refs),
                issues=[
                    {
                        "severity": "error",
                        "code": "EXPERIMENT_CODE_GAPS",
                        "message": reason,
                        "root_cause_key": "G8:EXPERIMENT_CODE_GAPS:code",
                    }
                ],
                route_hint=AgentName.CODE_MAKE,
            )
        launch_started = perf_counter()
        self._log_phase(
            task,
            event="supplementary_launch_started",
            status="started",
            details={"experiment_count": len(audit.designed_experiment_ids)},
        )
        result = self._launch(task, inputs, coverage_refs)
        self._log_phase(
            task,
            event="supplementary_launch_completed",
            status=(
                "info"
                if result.status == "in_progress"
                else "succeeded"
                if result.status == "succeeded"
                else "blocked"
                if result.status == "needs_human"
                else "failed"
            ),
            duration_ms=round((perf_counter() - launch_started) * 1000, 3),
            details={
                "result_status": result.status,
                "artifact_kinds": [item.kind for item in result.artifacts],
            },
        )
        return result
