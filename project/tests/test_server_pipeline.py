from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.check_result_agent import CheckResultAgent
from agents.code_make_agent import CodeMakeAgent
from agents.code_make_agent.assaycal_scaffold import build_assaycal_scaffold
from agents.code_review_agent import CodeReviewAgent
from agents.server_schedule_agent import ServerScheduleAgent
from research_agents.remote_execution import (
    MANAGED_RUNNER_SOURCE,
    SafeLocalBackend,
    SafeSSHBackend,
    script_has_forbidden_directory_deletion,
)
from research_agents.schemas import (
    AgentName,
    CodeImplementationManifest,
    CodeReviewReport,
    GPUResourceSnapshot,
    JobObservation,
    MetricObservation,
    ProjectExecutionConfig,
    ServerExecutionConfig,
    ServerDemoRecord,
    ServerProfileConfig,
    ServerResourceSnapshot,
    ServerScheduleManifest,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore

from test_code_pipeline import FakeSourceVendor, StaticModel, _write_inputs


class FakeServerBackend:
    def __init__(
        self,
        *,
        result_status: str = "completed",
        metric_value: float = 0.2,
        demo_status: str = "passed",
    ):
        self.result_status = result_status
        self.metric_value = metric_value
        self.demo_status = demo_status
        self.probed = []
        self.prepared = []
        self.transferred = []
        self.installed = []
        self.gpu_preflights = []
        self.demos = []
        self.launched = []

    def probe(self, profile):
        self.probed.append(profile.server_id)
        return ServerResourceSnapshot(
            server_id=profile.server_id,
            status="available",
            cpu_count=32,
            load_1m=0.5,
            memory_total_bytes=128 * 1024**3,
            memory_available_bytes=96 * 1024**3,
            disk_total_bytes=2 * 1024**4,
            disk_available_bytes=1024**4,
            conda_available=True,
            gpus=[
                GPUResourceSnapshot(
                    index=0,
                    name="Test GPU",
                    memory_total_mib=24576,
                    memory_used_mib=1024,
                    utilization_percent=5,
                ),
                GPUResourceSnapshot(
                    index=1,
                    name="Test GPU",
                    memory_total_mib=24576,
                    memory_used_mib=2048,
                    utilization_percent=10,
                ),
            ],
        )

    def prepare_environment(self, profile, **kwargs):
        self.prepared.append((profile.server_id, kwargs))

    def transfer_file(self, profile, **kwargs):
        self.transferred.append((profile.server_id, kwargs))

    def install_project(self, profile, **kwargs):
        self.installed.append((profile.server_id, kwargs))

    def validate_gpu_environment(self, profile, **kwargs):
        self.gpu_preflights.append((profile.server_id, kwargs))

    def run_minimum_demo(self, profile, **kwargs):
        self.demos.append((profile.server_id, kwargs))
        return ServerDemoRecord(
            server_id=profile.server_id,
            argv=kwargs["argv"],
            gpu_ids=kwargs["gpu_ids"],
            status=self.demo_status,
            return_code=(
                None
                if self.demo_status == "timed_out"
                else (0 if self.demo_status == "passed" else 1)
            ),
            duration_ms=10.0,
            timeout_seconds=kwargs["timeout_seconds"],
            log_path=kwargs["log_path"],
            output_tail="demo output",
        )

    def launch_job(self, profile, **kwargs):
        remote_id = str(1000 + len(self.launched))
        self.launched.append((profile.server_id, kwargs, remote_id))
        return remote_id

    def observe_job(self, profile, job):
        if self.result_status == "running":
            return JobObservation(
                job_id=job.job_id,
                status="running",
                log_tail="epoch is still running",
            )
        metric = MetricObservation(
            metric_name=job.metric_name,
            value=self.metric_value,
            step=10,
            checkpoint_path=f"{job.remote_workdir}/checkpoint.pt",
        )
        return JobObservation(
            job_id=job.job_id,
            status="completed",
            exit_code=0,
            metrics=[metric],
            best_metric=metric,
            best_checkpoint_path=f"{job.best_checkpoint_dir}/checkpoint.pt",
            log_tail=f"best {job.metric_name}={self.metric_value}",
        )


def execution_config() -> ServerExecutionConfig:
    return ServerExecutionConfig(
        servers=[
            ServerProfileConfig(
                server_id="gpu-test",
                credential_prefix="GPU_TEST",
                workspace_root="/srv/research-agent-tests",
                execution_enabled=True,
                min_available_memory_bytes=1024,
                min_available_disk_bytes=1024,
            )
        ],
        projects=[
            ProjectExecutionConfig(
                project_slug="assaycal-test",
                train_argv=[
                    "python",
                    "-m",
                    "affinity.training.cli",
                    "--experiment",
                    "{experiment_id}",
                    "--seed",
                    "{seed}",
                    "--data-dir",
                    "{data_dir}",
                    "--output-dir",
                    "{output_dir}",
                    "--metrics-jsonl",
                    "{metrics_path}",
                ],
                expected_job_seconds=3600,
                metric_name="validation_mse",
                metric_direction="lower",
                target_value=0.25,
            )
        ],
    )


def reviewed_inputs(artifacts: ArtifactStore, run_id: str):
    design, design_ref, data_ref = _write_inputs(artifacts, run_id)
    draft = build_assaycal_scaffold(design)
    make_result = CodeMakeAgent(
        StaticModel(draft.model_dump(mode="json")),
        artifacts,
        source_vendor=FakeSourceVendor(artifacts),
    ).run(
        TaskEnvelope(
            run_id=run_id,
            task_id="task_code_make_server",
            agent=AgentName.CODE_MAKE,
            objective="test server scheduling",
            inputs=[design_ref, data_ref],
            context={"project_slug": "assaycal-test"},
        )
    )
    code_ref = make_result.artifacts[0]
    manifest = CodeImplementationManifest.model_validate(
        artifacts.read_artifact_json(code_ref)
    )
    review_result = CodeReviewAgent(
        StaticModel({"summary": "All deterministic checks passed.", "recommendations": []}),
        artifacts,
    ).run(
        TaskEnvelope(
            run_id=run_id,
            task_id="task_code_review_server",
            agent=AgentName.CODE_REVIEW,
            objective="test server scheduling",
            inputs=[design_ref, code_ref],
            context={"project_slug": "assaycal-test"},
        )
    )
    # Server tests exercise scheduling after an upstream acceptance gate.  The
    # intentionally tiny/hash-based Code Make fixture is correctly rejected by
    # the scientific-protocol review tests, so promote a controlled copy here
    # instead of weakening the production reviewer or coupling every server
    # test to a full scientific DTA implementation.
    review = CodeReviewReport.model_validate(
        artifacts.read_artifact_json(review_result.artifacts[0])
    )
    accepted_checks = [
        check.model_copy(
            update={
                "status": "passed",
                "summary": "Controlled upstream acceptance fixture for server tests.",
                "evidence": [
                    "Server scheduling is tested with an explicit passed-review precondition."
                ],
                "file_paths": [],
            }
        )
        if check.check_id == "review-scientific-protocol"
        else check
        for check in review.checks
    ]
    accepted_review = review.model_copy(
        update={
            "verdict": "passed",
            "checks": accepted_checks,
            "blockers": [],
            "recommendations": [],
            "summary": "Controlled passed-review fixture for downstream server tests.",
        }
    )
    accepted_review_ref = artifacts.write_agent_json(
        AgentName.CODE_REVIEW,
        f"design/assaycal-test/reviews/code/server-fixture-{run_id}.json",
        accepted_review.model_dump(mode="json"),
    ).model_copy(update={"kind": "code_review_report"})
    return design_ref, data_ref, code_ref, accepted_review_ref, manifest


class ServerPipelineTests(unittest.TestCase):
    def test_local_cpu_profile_exposes_one_bounded_accelerator_slot(self):
        from research_agents.remote_execution import available_gpu_keys

        profile = ServerProfileConfig(
            server_id="local-cpu",
            credential_prefix="LOCAL_CPU",
            accelerator="cpu",
            workspace_root="/tmp/agent-local-cpu-test",
            execution_enabled=True,
            min_available_memory_bytes=0,
            min_available_disk_bytes=0,
        )
        probe = ServerResourceSnapshot(
            server_id="local-cpu",
            status="available",
            cpu_count=4,
            memory_total_bytes=1024,
            memory_available_bytes=1024,
            disk_total_bytes=1024,
            disk_available_bytes=1024,
            conda_available=True,
            gpus=[],
        )
        self.assertEqual(
            available_gpu_keys(profile, probe, transfer_bytes=0),
            ["local-cpu:0"],
        )

    def test_local_backend_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = SafeLocalBackend(directory)
            profile = ServerProfileConfig(
                server_id="local-cpu",
                credential_prefix="LOCAL_CPU",
                accelerator="cpu",
                workspace_root="/tmp/outside-local-workspace",
                execution_enabled=True,
            )
            with self.assertRaisesRegex(Exception, "escaped"):
                backend.probe(profile)

    def test_local_launch_pins_runner_and_child_path_to_fresh_conda(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = SafeLocalBackend(directory)
            environment_name = "run_local_cpu_test"
            workspace = (
                Path(directory)
                / "project/var/local-compute/projects"
                / environment_name
            )
            environment = workspace / "conda-env"
            runner = workspace / "runtime/managed_runner.py"
            spec = workspace / "runtime/specs/job.json"
            bootstrap = workspace / "logs/bootstrap.log"
            (environment / "bin").mkdir(parents=True)
            runner.parent.mkdir(parents=True)
            spec.parent.mkdir(parents=True)
            (environment / "bin/python").touch()
            runner.touch()
            spec.write_text("{}", encoding="utf-8")
            profile = ServerProfileConfig(
                server_id="local-cpu",
                credential_prefix="LOCAL_CPU",
                accelerator="cpu",
                workspace_root=str(workspace),
                execution_enabled=True,
            )
            with patch("research_agents.remote_execution.subprocess.Popen") as popen:
                popen.return_value.pid = 1234
                backend.launch_job(
                    profile,
                    environment_name=environment_name,
                    runner_path=str(runner),
                    spec_path=str(spec),
                    bootstrap_log_path=str(bootstrap),
                )
            argv = popen.call_args.args[0]
            child_env = popen.call_args.kwargs["env"]
            resolved_environment = environment.resolve()
            self.assertEqual(argv[0], str(resolved_environment / "bin/python"))
            self.assertEqual(child_env["CONDA_PREFIX"], str(resolved_environment))
            self.assertEqual(child_env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(
                child_env["PATH"].split(":", 1)[0],
                str(resolved_environment / "bin"),
            )

    def test_managed_runner_forbids_directory_deletion(self):
        self.assertFalse(script_has_forbidden_directory_deletion(MANAGED_RUNNER_SOURCE))
        self.assertNotIn("shell=True", MANAGED_RUNNER_SOURCE)
        fixed_scripts = [
            value
            for name, value in vars(SafeSSHBackend).items()
            if name.endswith("_SCRIPT") and isinstance(value, str)
        ]
        self.assertGreaterEqual(len(fixed_scripts), 6)
        self.assertTrue(
            all(not script_has_forbidden_directory_deletion(value) for value in fixed_scripts)
        )

    def test_schedule_uses_all_eligible_gpus_and_stages_data_models_and_conda(self):
        run_id = "run_server_pipeline"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            backend = FakeServerBackend()
            result = ServerScheduleAgent(
                StaticModel({}),
                artifacts,
                backend=backend,
                execution_config=execution_config(),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_schedule",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="test server scheduling",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "succeeded", result.summary)
            schedule = ServerScheduleManifest.model_validate(
                artifacts.read_artifact_json(result.artifacts[0])
            )
            self.assertEqual(schedule.status, "launched")
            self.assertEqual(set(schedule.selected_gpu_keys), {"gpu-test:0", "gpu-test:1"})
            self.assertEqual(schedule.report_interval_seconds, 43200)
            self.assertIn("data", {item.kind for item in schedule.transfers})
            self.assertIn("model", {item.kind for item in schedule.transfers})
            self.assertEqual(len(backend.prepared), 1)
            self.assertEqual(len(backend.installed), 1)
            self.assertEqual(len(backend.gpu_preflights), 1)
            self.assertEqual(backend.gpu_preflights[0][1]["gpu_ids"], [0, 1])
            self.assertEqual(len(backend.demos), 1)
            self.assertEqual(backend.demos[0][1]["argv"][0], "python")
            self.assertEqual(len(schedule.server_demos), 1)
            self.assertEqual(schedule.server_demos[0].status, "passed")
            self.assertEqual(len(backend.launched), len(schedule.jobs))
            self.assertTrue(all(item.status == "launched" for item in schedule.jobs))
            self.assertEqual(
                len({item.gpu_lock_path for item in schedule.jobs}), 2
            )

    def test_failed_server_demo_blocks_every_nohup_launch(self):
        run_id = "run_server_demo_failed"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            backend = FakeServerBackend(demo_status="failed")
            result = ServerScheduleAgent(
                StaticModel({}),
                artifacts,
                backend=backend,
                execution_config=execution_config(),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_demo_failed",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="block persistent jobs when remote demo fails",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "needs_human")
            self.assertEqual(result.issues[0].code, "SERVER_MINIMUM_DEMO_FAILED")
            self.assertEqual(backend.launched, [])
            schedule = ServerScheduleManifest.model_validate(
                artifacts.read_artifact_json(result.artifacts[0])
            )
            self.assertEqual(schedule.status, "blocked")
            self.assertEqual(schedule.server_demos[0].status, "failed")
            self.assertTrue(all(item.status == "planned" for item in schedule.jobs))

    def test_execution_disabled_server_is_not_contacted(self):
        run_id = "run_server_execution_disabled"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            config = execution_config()
            disabled_profile = config.servers[0].model_copy(
                update={"execution_enabled": False}
            )
            disabled_config = config.model_copy(update={"servers": [disabled_profile]})
            backend = FakeServerBackend()
            result = ServerScheduleAgent(
                StaticModel({}),
                artifacts,
                backend=backend,
                execution_config=disabled_config,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_disabled",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="do not contact disabled servers",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "needs_human")
            self.assertEqual(backend.probed, [])
            self.assertEqual(backend.launched, [])

    def test_check_result_passes_complete_metrics_and_persists_best_logs(self):
        run_id = "run_result_pipeline"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            backend = FakeServerBackend(metric_value=0.2)
            config = execution_config()
            schedule_result = ServerScheduleAgent(
                StaticModel({}),
                artifacts,
                backend=backend,
                execution_config=config,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_schedule_result",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="test result checking",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            result = CheckResultAgent(
                StaticModel({}),
                artifacts,
                backend=backend,
                execution_config=config,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_check_result",
                    agent=AgentName.CHECK_RESULT,
                    objective="test result checking",
                    inputs=[design_ref, schedule_result.artifacts[0]],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "succeeded", result.summary)
            report = artifacts.read_artifact_json(result.artifacts[0])
            self.assertEqual(report["verdict"], "passed")
            self.assertEqual(report["metric_checks"][0]["status"], "passed")
            self.assertEqual(
                len(report["best_log_artifacts"]), len(report["observations"])
            )

    def test_check_result_running_is_non_failure_with_twelve_hour_report(self):
        run_id = "run_result_running"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            backend = FakeServerBackend(result_status="running")
            config = execution_config()
            schedule_result = ServerScheduleAgent(
                StaticModel({}), artifacts, backend=backend, execution_config=config
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_schedule_running",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="test running result",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            result = CheckResultAgent(
                StaticModel({}), artifacts, backend=backend, execution_config=config
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_check_result_running",
                    agent=AgentName.CHECK_RESULT,
                    objective="test running result",
                    inputs=[design_ref, schedule_result.artifacts[0]],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "in_progress")
            report = artifacts.read_artifact_json(result.artifacts[0])
            self.assertEqual(report["verdict"], "in_progress")
            self.assertEqual(report["report_interval_seconds"], 43200)
            self.assertIsNotNone(report["next_report_at"])

    def test_unsatisfactory_metric_routes_back_to_code_make(self):
        run_id = "run_result_unsatisfactory"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
                artifacts, run_id
            )
            backend = FakeServerBackend(metric_value=0.5)
            config = execution_config()
            schedule_result = ServerScheduleAgent(
                StaticModel({}), artifacts, backend=backend, execution_config=config
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_server_schedule_unsatisfactory",
                    agent=AgentName.SERVER_SCHEDULE,
                    objective="test unsatisfactory result",
                    inputs=[design_ref, data_ref, code_ref, review_ref],
                    context={"project_slug": "assaycal-test"},
                )
            )
            result = CheckResultAgent(
                StaticModel({}), artifacts, backend=backend, execution_config=config
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_check_result_unsatisfactory",
                    agent=AgentName.CHECK_RESULT,
                    objective="test unsatisfactory result",
                    inputs=[design_ref, schedule_result.artifacts[0]],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.CODE_MAKE)
            report = artifacts.read_artifact_json(result.artifacts[0])
            self.assertEqual(report["verdict"], "needs_improvement")
            self.assertEqual(report["metric_checks"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
