from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.project_layout import artifact_prefix, task_project_slug
from research_agents.providers import JsonModel
from research_agents.remote_execution import (
    MANAGED_RUNNER_SOURCE,
    RemoteExecutionBackend,
    RemoteExecutionError,
    SafeSSHBackend,
    available_gpu_keys,
    load_server_execution_config,
    project_execution,
    script_has_forbidden_directory_deletion,
)
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    CodeImplementationManifest,
    CodeReviewReport,
    DatasetManifest,
    ProjectExecutionConfig,
    ResearchDesign,
    ScheduledExperimentJob,
    ServerExecutionConfig,
    ServerDemoRecord,
    ServerProfileConfig,
    ServerResourceSnapshot,
    ServerScheduleManifest,
    ServerTransferItem,
    TaskEnvelope,
    utc_now,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_INPUT_BYTES,
    MAX_RUNTIME_HELPER_BYTES,
    MAX_SCHEDULE_REPORT_BYTES,
    MAX_TRANSFER_FILE_BYTES,
    REPORT_INTERVAL_SECONDS,
    environment_name,
    file_token,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.SERVER_SCHEDULE:
    raise RuntimeError("Server Schedule Agent package name mismatch")


class ServerScheduleAgent:
    """Probe, allocate, stage, and launch one immutable reviewed experiment revision."""

    name = AgentName.SERVER_SCHEDULE
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
        DatasetManifest | None,
        ArtifactRef | None,
        CodeImplementationManifest | None,
        ArtifactRef | None,
        CodeReviewReport | None,
        ArtifactRef | None,
    ]:
        slug = task_project_slug(task)
        design = data = code = review = None
        design_ref = data_ref = code_ref = review_ref = None
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
                elif artifact.kind == "data_manifest" and artifact.uri.startswith(
                    artifact_prefix("data", slug, "manifests")
                ):
                    candidate = DatasetManifest.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id and candidate.project_slug == slug:
                        data, data_ref = candidate, artifact
                elif artifact.kind == "code_implementation_manifest" and artifact.uri.startswith(
                    artifact_prefix("code", slug, "manifests")
                ):
                    candidate = CodeImplementationManifest.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id and candidate.project_slug == slug:
                        code, code_ref = candidate, artifact
                elif artifact.kind == "code_review_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "code")
                ):
                    candidate = CodeReviewReport.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id == task.run_id and candidate.project_slug == slug:
                        review, review_ref = candidate, artifact
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        if code_ref and review and review.code_manifest_uri != code_ref.uri:
            review = None
            review_ref = None
        if review and review.verdict != "passed":
            review = None
            review_ref = None
        return design, design_ref, data, data_ref, code, code_ref, review, review_ref

    def _write_gap(self, task: TaskEnvelope, reason: str, code: str) -> ArtifactRef:
        path = (
            f"project/var/runs/{task.run_id}/server_schedule/"
            f"{file_token(task.task_id)}.gap.json"
        )
        return self.artifacts.write_agent_json(
            self.name,
            path,
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": task_project_slug(task),
                "status": "blocked",
                "reason": reason,
                "issue_code": code,
            },
        ).model_copy(update={"kind": "server_schedule_gap"})

    def _config(self) -> ServerExecutionConfig:
        return self.execution_config or load_server_execution_config(self.artifacts.root)

    def _verified_local_path(self, artifact: ArtifactRef, expected_size: int) -> Path:
        measured = self.artifacts.verify_artifact_file(
            artifact, max_bytes=MAX_TRANSFER_FILE_BYTES
        )
        if measured != expected_size:
            raise StoragePolicyError("transfer file size differs from its manifest")
        relative = artifact.uri.removeprefix("artifact://")
        target = (self.artifacts.root / relative).resolve()
        target.relative_to(self.artifacts.root)
        return target

    def _base_sources(
        self,
        data: DatasetManifest,
        code: CodeImplementationManifest,
    ) -> list[tuple[str, ArtifactRef, str, int]]:
        sources: list[tuple[str, ArtifactRef, str, int]] = []
        for record in data.records:
            if record.status != "downloaded":
                continue
            for item in record.materialized_files:
                self._verified_local_path(item.artifact, item.size_bytes)
                remote = str(
                    PurePosixPath("data", record.dataset_id, item.relative_path)
                )
                sources.append(("data", item.artifact, remote, item.size_bytes))
        for item in code.files:
            self._verified_local_path(item.artifact, item.size_bytes)
            sources.append(
                ("code", item.artifact, str(PurePosixPath("code", item.path)), item.size_bytes)
            )
            normalized = f"/{item.path.strip('/')}"
            if "/models/" in normalized:
                model_suffix = normalized.split("/models/", 1)[1]
                sources.append(
                    (
                        "model",
                        item.artifact,
                        str(PurePosixPath("models", "source", model_suffix)),
                        item.size_bytes,
                    )
                )
        if not any(kind == "data" for kind, *_ in sources):
            raise StoragePolicyError("no materialized dataset body is available for transfer")
        if not any(kind == "model" for kind, *_ in sources):
            raise StoragePolicyError("reviewed implementation has no local model source")
        return sources

    @staticmethod
    def _remote_workspace(profile: ServerProfileConfig, task: TaskEnvelope) -> str:
        return str(
            PurePosixPath(
                profile.workspace_root,
                "projects",
                task_project_slug(task),
                "runs",
                task.run_id,
                file_token(task.task_id),
            )
        )

    @staticmethod
    def _expand_argv(
        project: ProjectExecutionConfig,
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

    @staticmethod
    def _minimum_demo_argv(code: CodeImplementationManifest) -> list[str]:
        demo = next(
            (
                item
                for item in code.local_validations
                if item.validation_id == "minimal_demo"
            ),
            None,
        )
        if demo is None or demo.status != "passed":
            raise RemoteExecutionError(
                "reviewed code manifest has no passed local minimum demo"
            )
        argv = list(demo.argv)
        if not argv or Path(argv[0]).name not in {"python", "python3"}:
            raise RemoteExecutionError("local minimum demo does not invoke Python")
        argv[0] = "python"
        return argv

    def _allocate_jobs(
        self,
        task: TaskEnvelope,
        design: ResearchDesign,
        project: ProjectExecutionConfig,
        profiles: dict[str, ServerProfileConfig],
        probes: dict[str, ServerResourceSnapshot],
        admitted_gpu_keys: list[str],
    ) -> tuple[list[ScheduledExperimentJob], list[str], int]:
        definitions = [
            (experiment.experiment_id, seed)
            for experiment in design.experiment_plans
            if experiment.experiment_type == "main_benchmark"
            for seed in experiment.random_seeds
        ]
        if not definitions:
            raise RemoteExecutionError(
                "research design contains no main-benchmark experiment/seed jobs"
            )
        gpu_by_server: dict[str, list[int]] = defaultdict(list)
        for key in admitted_gpu_keys:
            server_id, raw_index = key.rsplit(":", 1)
            gpu_by_server[server_id].append(int(raw_index))
        active_servers = list(gpu_by_server)
        jobs: list[ScheduledExperimentJob] = []
        selected_gpu_keys: list[str] = []
        if project.distributed_mode == "torchrun":
            active_servers = active_servers[: len(definitions)]
            for server_id in active_servers:
                selected_gpu_keys.extend(
                    f"{server_id}:{index}" for index in gpu_by_server[server_id]
                )
            slots: list[tuple[str, list[int]]] = [
                (server_id, gpu_by_server[server_id]) for server_id in active_servers
            ]
        else:
            flattened = [
                (key.rsplit(":", 1)[0], [int(key.rsplit(":", 1)[1])])
                for key in admitted_gpu_keys[: len(definitions)]
            ]
            selected_gpu_keys = [
                f"{server_id}:{gpu_ids[0]}" for server_id, gpu_ids in flattened
            ]
            slots = flattened
        if not slots:
            raise RemoteExecutionError("no GPU execution slot is available")
        for index, (experiment_id, seed) in enumerate(definitions):
            server_id, gpu_ids = slots[index % len(slots)]
            profile = profiles[server_id]
            workspace = self._remote_workspace(profile, task)
            job_id = f"job-{file_token(f'{experiment_id}:{seed}:{task.task_id}')[:16]}"
            workdir = str(PurePosixPath(workspace, "jobs", job_id))
            metrics_path = str(PurePosixPath(workdir, "metrics.jsonl"))
            log_path = str(PurePosixPath(workspace, "logs", f"{job_id}.log"))
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
                        workspace=workspace,
                        workdir=workdir,
                        metrics_path=metrics_path,
                        gpu_count=len(gpu_ids),
                    ),
                    remote_workdir=workdir,
                    log_path=log_path,
                    metrics_path=metrics_path,
                    best_checkpoint_dir=str(PurePosixPath(workspace, "best", job_id)),
                    gpu_lock_path=str(
                        PurePosixPath(
                            workspace,
                            "runtime",
                            "gpu-locks",
                            "gpu-" + "_".join(str(value) for value in gpu_ids) + ".lock",
                        )
                    ),
                    expected_duration_seconds=project.expected_job_seconds,
                    metric_name=project.metric_name,
                    metric_direction=project.metric_direction,
                    status="planned",
                )
            )
        makespan = math.ceil(len(jobs) / len(slots)) * project.expected_job_seconds
        return jobs, selected_gpu_keys, makespan

    @staticmethod
    def _markdown(manifest: ServerScheduleManifest) -> str:
        lines = [
            "# Server Schedule Report",
            "",
            f"- Status: `{manifest.status}`",
            f"- Project: `{manifest.project_slug}`",
            f"- Estimated makespan: `{manifest.estimated_makespan_seconds}` seconds",
            f"- User report interval: `{manifest.report_interval_seconds}` seconds",
            f"- Eligible/selected GPUs: `{len(manifest.available_gpu_keys)}/{len(manifest.selected_gpu_keys)}`",
            "",
            "## Resource probes",
            "",
            "| Server | Status | CPU/load | Available memory | Available disk | Conda | GPUs |",
            "|---|---|---|---:|---:|---|---|",
        ]
        for probe in manifest.probes:
            gpu_text = ", ".join(
                f"{gpu.index}:{gpu.name} free={gpu.memory_free_mib}MiB util={gpu.utilization_percent:.1f}%"
                for gpu in probe.gpus
            ) or "none"
            lines.append(
                f"| `{probe.server_id}` | `{probe.status}` | {probe.cpu_count}/{probe.load_1m:.2f} | "
                f"{probe.memory_available_bytes / 1024**3:.2f} GiB | "
                f"{probe.disk_available_bytes / 1024**3:.2f} GiB | "
                f"`{probe.conda_available}` | {gpu_text.replace('|', '/')} |"
            )
        lines.extend(
            [
                "",
                "## Allocation",
                "",
                f"- Available GPU keys: `{', '.join(manifest.available_gpu_keys) or 'none'}`",
                f"- Selected GPU keys: `{', '.join(manifest.selected_gpu_keys) or 'none'}`",
                f"- Verified transfers: `{len(manifest.transfers)}`",
                f"- Fresh Conda environments: `{json.dumps(manifest.environment_names, ensure_ascii=False)}`",
                f"- Passed foreground server demos: `{sum(item.status == 'passed' for item in manifest.server_demos)}/{len(manifest.server_demos)}`",
                "",
                "## Foreground minimum demos",
                "",
                "| Server | GPUs | Status | Return code | Duration ms | Timeout | Remote log |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for demo in manifest.server_demos:
            lines.append(
                f"| `{demo.server_id}` | `{','.join(str(value) for value in demo.gpu_ids)}` | "
                f"`{demo.status}` | `{demo.return_code if demo.return_code is not None else 'n/a'}` | "
                f"{demo.duration_ms:.3f} | {demo.timeout_seconds}s | `{demo.log_path}` |"
            )
        lines.extend(
            [
                "",
                "## Persistent jobs",
                "",
                "| Job | Experiment | Seed | Server | GPUs | State | Expected seconds | Next report |",
                "|---|---|---:|---|---|---|---:|---|",
            ]
        )
        for job in manifest.jobs:
            lines.append(
                f"| `{job.job_id}` | `{job.experiment_id}` | {job.seed} | `{job.server_id}` | "
                f"`{','.join(str(value) for value in job.gpu_ids)}` | `{job.status}` | "
                f"{job.expected_duration_seconds} | "
                f"`{job.next_report_at.isoformat() if job.next_report_at else 'not-launched'}` |"
            )
        lines.extend(
            [
                "",
                "## Enforced remote policy",
                "",
                "- Every path is below the configured server workspace and this run's attempt directory.",
                "- Data, reviewed code, and model source are transferred before installation/training.",
                "- Every attempt uses a fresh Conda environment and managed argv-only runner.",
                "- Every selected server must pass a bounded foreground minimum demo before any long job starts.",
                "- Long jobs launch only with nohup and persist PID/status, bootstrap log, job log, and JSONL metrics.",
                "- Recursive or directory deletion is forbidden; run directories are retained.",
                "- Each job has a full log, JSONL metrics, status file, and best-checkpoint directory.",
            ]
        )
        if manifest.issues:
            lines.extend(["", "## Issues", ""])
            lines.extend(f"- {value}" for value in manifest.issues)
        return "\n".join(lines) + "\n"

    def _write_manifest(
        self, task: TaskEnvelope, manifest: ServerScheduleManifest
    ) -> tuple[ArtifactRef, ArtifactRef]:
        base = (
            f"project/var/runs/{task.run_id}/server_schedule/"
            f"{file_token(task.task_id)}"
        )
        json_ref = self.artifacts.write_agent_json(
            self.name,
            f"{base}.json",
            manifest.model_dump(mode="json"),
        ).model_copy(update={"kind": "server_schedule_manifest"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._markdown(manifest).encode("utf-8"),
            max_bytes=MAX_SCHEDULE_REPORT_BYTES,
            kind="server_schedule_markdown",
        )
        return json_ref, markdown_ref

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        (
            design,
            design_ref,
            data,
            data_ref,
            code,
            code_ref,
            review,
            review_ref,
        ) = self._read_inputs(task)
        if None in (design, design_ref, data, data_ref, code, code_ref, review, review_ref):
            reason = (
                "Server scheduling requires a project-matched ResearchDesign, materialized "
                "DatasetManifest, validated CodeImplementationManifest, and passed CodeReviewReport."
            )
            gap = self._write_gap(task, reason, "SERVER_INPUT_MISSING")
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
                        "code": "SERVER_INPUT_MISSING",
                        "message": reason,
                        "root_cause_key": "G6:SERVER_INPUT_MISSING:inputs",
                    }
                ],
                needs_human=True,
            )
        assert design and design_ref and data and data_ref and code and code_ref and review and review_ref
        try:
            config = self._config()
            project = project_execution(config, task_project_slug(task))
            sources = self._base_sources(data, code)
        except (RemoteExecutionError, StoragePolicyError, ValidationError) as exc:
            reason = f"Server execution configuration or local transfer evidence is incomplete: {type(exc).__name__}."
            gap = self._write_gap(task, reason, "SERVER_CONFIG_OR_TRANSFER_INVALID")
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
                        "code": "SERVER_CONFIG_OR_TRANSFER_INVALID",
                        "message": reason,
                        "root_cause_key": "G6:SERVER_CONFIG_OR_TRANSFER_INVALID:config",
                    }
                ],
                needs_human=True,
            )

        profiles = {
            item.server_id: item
            for item in config.servers
            if item.enabled and item.execution_enabled
        }
        probes = {server_id: self.backend.probe(profile) for server_id, profile in profiles.items()}
        transfer_bytes = sum(size for _, _, _, size in sources)
        admitted: list[str] = []
        for server_id, profile in sorted(
            profiles.items(), key=lambda item: item[1].priority, reverse=True
        ):
            admitted.extend(
                available_gpu_keys(
                    profile,
                    probes[server_id],
                    transfer_bytes=transfer_bytes,
                )
            )
        if not admitted:
            manifest = ServerScheduleManifest(
                run_id=task.run_id,
                task_id=task.task_id,
                project_slug=task_project_slug(task),
                status="blocked",
                code_manifest_uri=code_ref.uri,
                data_manifest_uri=data_ref.uri,
                research_design_uri=design_ref.uri,
                code_review_uri=review_ref.uri,
                probes=list(probes.values()) or [
                    ServerResourceSnapshot(
                        server_id="unconfigured", status="unreachable", error_code="NO_SERVER"
                    )
                ],
                issues=[
                    "No explicitly execution-enabled server passed Conda, GPU, memory, disk, and utilization gates."
                ],
            )
            artifact, markdown = self._write_manifest(task, manifest)
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="needs_human",
                summary=manifest.issues[0],
                artifacts=[artifact, markdown],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "SERVER_NO_ELIGIBLE_GPU",
                        "message": manifest.issues[0],
                        "root_cause_key": "G6:SERVER_NO_ELIGIBLE_GPU:resources",
                    }
                ],
                needs_human=True,
            )

        jobs, selected_gpu_keys, makespan = self._allocate_jobs(
            task, design, project, profiles, probes, admitted
        )
        selected_servers = sorted({item.server_id for item in jobs})
        workspaces = {
            server_id: self._remote_workspace(profiles[server_id], task)
            for server_id in selected_servers
        }
        environments = {
            server_id: environment_name(task_project_slug(task), f"{task.task_id}:{server_id}")
            for server_id in selected_servers
        }
        runner_ref = self.artifacts.write_agent_bytes(
            self.name,
            (
                f"project/var/runs/{task.run_id}/server_schedule/"
                f"{file_token(task.task_id)}/managed_runner.py"
            ),
            MANAGED_RUNNER_SOURCE.encode("utf-8"),
            max_bytes=MAX_RUNTIME_HELPER_BYTES,
            kind="server_runtime_helper",
        )
        if script_has_forbidden_directory_deletion(MANAGED_RUNNER_SOURCE):
            raise RuntimeError("managed runner failed the no-directory-deletion policy")

        transfers: list[ServerTransferItem] = []
        local_paths: dict[str, Path] = {}
        for server_id in selected_servers:
            workspace = workspaces[server_id]
            for kind, artifact, remote_relative, size in sources:
                item = ServerTransferItem(
                    transfer_id=(
                        f"transfer-{file_token(f'{server_id}:{kind}:{artifact.uri}:{remote_relative}')[:20]}"
                    ),
                    server_id=server_id,
                    kind=kind,
                    local_artifact=artifact,
                    local_relative_path=artifact.uri.removeprefix("artifact://"),
                    remote_relative_path=remote_relative,
                    size_bytes=size,
                )
                transfers.append(item)
                local_paths[item.transfer_id] = self._verified_local_path(artifact, size)
            runner_item = ServerTransferItem(
                transfer_id=f"transfer-{file_token(f'{server_id}:runner:{runner_ref.uri}')[:20]}",
                server_id=server_id,
                kind="runtime",
                local_artifact=runner_ref,
                local_relative_path=runner_ref.uri.removeprefix("artifact://"),
                remote_relative_path="runtime/managed_runner.py",
                size_bytes=len(MANAGED_RUNNER_SOURCE.encode("utf-8")),
            )
            transfers.append(runner_item)
            local_paths[runner_item.transfer_id] = self._verified_local_path(
                runner_ref, runner_item.size_bytes
            )

        spec_refs: dict[str, ArtifactRef] = {}
        for job in jobs:
            spec_ref = self.artifacts.write_agent_json(
                self.name,
                (
                    f"project/var/runs/{task.run_id}/server_schedule/"
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
            ).model_copy(update={"kind": "server_job_spec"})
            spec_refs[job.job_id] = spec_ref
            item = ServerTransferItem(
                transfer_id=f"transfer-{file_token(f'{job.server_id}:spec:{job.job_id}')[:20]}",
                server_id=job.server_id,
                kind="runtime",
                local_artifact=spec_ref,
                local_relative_path=spec_ref.uri.removeprefix("artifact://"),
                remote_relative_path=f"runtime/specs/{job.job_id}.json",
                size_bytes=self.artifacts.verify_artifact_file(
                    spec_ref, max_bytes=MAX_RUNTIME_HELPER_BYTES
                ),
            )
            transfers.append(item)
            local_paths[item.transfer_id] = self._verified_local_path(
                spec_ref, item.size_bytes
            )

        launched_jobs: list[ScheduledExperimentJob] = []
        completed_transfers: list[ServerTransferItem] = []
        server_demos: list[ServerDemoRecord] = []
        issues: list[str] = []
        try:
            demo_argv = self._minimum_demo_argv(code)
            for server_id in selected_servers:
                profile = profiles[server_id]
                workspace = workspaces[server_id]
                self.backend.prepare_environment(
                    profile,
                    workspace=workspace,
                    environment_name=environments[server_id],
                    conda_dependencies=project.conda_dependencies,
                )
                for item in (value for value in transfers if value.server_id == server_id):
                    self.backend.transfer_file(
                        profile,
                        local_path=local_paths[item.transfer_id],
                        remote_path=str(PurePosixPath(workspace, item.remote_relative_path)),
                    )
                    completed_transfers.append(item)
                self.backend.install_project(
                    profile,
                    environment_name=environments[server_id],
                    code_root=str(PurePosixPath(workspace, "code")),
                    pip_install_argv=project.pip_install_argv,
                )
                server_gpu_ids = sorted(
                    {
                        gpu_id
                        for job in jobs
                        if job.server_id == server_id
                        for gpu_id in job.gpu_ids
                    }
                )
                self.backend.validate_gpu_environment(
                    profile,
                    environment_name=environments[server_id],
                    gpu_ids=server_gpu_ids,
                )
                server_demos.append(
                    self.backend.run_minimum_demo(
                        profile,
                        environment_name=environments[server_id],
                        code_root=str(PurePosixPath(workspace, "code")),
                        argv=demo_argv,
                        gpu_ids=server_gpu_ids,
                        log_path=str(
                            PurePosixPath(
                                workspace, "logs", "server-minimum-demo.log"
                            )
                        ),
                        timeout_seconds=project.server_demo_timeout_seconds,
                    )
                )
            if any(item.status != "passed" for item in server_demos):
                issues.append(
                    "At least one selected server failed the bounded foreground minimum demo; no long job was launched."
                )
            else:
                for job in jobs:
                    workspace = workspaces[job.server_id]
                    remote_job_id = self.backend.launch_job(
                        profiles[job.server_id],
                        environment_name=environments[job.server_id],
                        runner_path=str(
                            PurePosixPath(workspace, "runtime", "managed_runner.py")
                        ),
                        spec_path=str(
                            PurePosixPath(
                                workspace, "runtime", "specs", f"{job.job_id}.json"
                            )
                        ),
                        bootstrap_log_path=str(
                            PurePosixPath(
                                workspace, "logs", f"{job.job_id}.bootstrap.log"
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
                f"Remote preparation, transfer, installation, or launch failed closed: {type(exc).__name__}."
            )

        launched = len(launched_jobs) == len(jobs) and not issues
        manifest = ServerScheduleManifest(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            status="launched" if launched else "blocked",
            code_manifest_uri=code_ref.uri,
            data_manifest_uri=data_ref.uri,
            research_design_uri=design_ref.uri,
            code_review_uri=review_ref.uri,
            probes=list(probes.values()),
            available_gpu_keys=selected_gpu_keys if launched else admitted,
            selected_gpu_keys=selected_gpu_keys if launched else [],
            transfers=completed_transfers,
            environment_names=environments,
            server_workspaces=workspaces,
            server_demos=server_demos,
            jobs=launched_jobs + jobs[len(launched_jobs) :],
            estimated_makespan_seconds=makespan,
            issues=issues,
        )
        artifact, markdown = self._write_manifest(task, manifest)
        if not launched:
            demo_failed = any(item.status != "passed" for item in server_demos)
            issue_code = (
                "SERVER_MINIMUM_DEMO_FAILED"
                if demo_failed
                else "SERVER_PARTIAL_LAUNCH"
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="needs_human",
                summary=(issues[0] if issues else "Server launch did not complete."),
                artifacts=[artifact, markdown],
                issues=[
                    {
                        "severity": "blocking",
                        "code": issue_code,
                        "message": (
                            issues[0]
                            if issues
                            else "Server launch did not complete; inspect the immutable schedule manifest."
                        ),
                        "root_cause_key": f"G6:{issue_code}:remote",
                    }
                ],
                needs_human=True,
            )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded",
            summary=(
                f"Probed {len(probes)} servers, launched {len(launched_jobs)} jobs across "
                f"{len(selected_gpu_keys)} usable accelerator slots, staged {len(completed_transfers)} verified "
                f"file transfers, passed {len(server_demos)} foreground server demos, and scheduled "
                f"the next user report in 12 hours; estimated "
                f"makespan is {makespan} seconds."
            ),
            artifacts=[artifact, markdown],
            evidence=[
                {
                    "check": "server-resource-probe",
                    "servers": len(probes),
                    "selected_accelerator_count": len(selected_gpu_keys),
                },
                {
                    "check": "remote-safety-policy",
                    "fresh_conda_environments": len(environments),
                    "accelerator_preflight_servers": len(environments),
                    "foreground_demo_servers": len(server_demos),
                    "directory_deletion_commands": 0,
                    "report_interval_seconds": REPORT_INTERVAL_SECONDS,
                },
                {
                    "check": "managed-launch",
                    "job_count": len(launched_jobs),
                    "estimated_makespan_seconds": makespan,
                },
            ],
        )
