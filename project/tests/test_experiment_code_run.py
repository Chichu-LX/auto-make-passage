from __future__ import annotations

from datetime import timedelta
import math
import tempfile
import unittest

from agents.check_result_agent import CheckResultAgent
from agents.experiment_code_run_agent import ExperimentCodeRunAgent
from agents.idea_design_agent import IdeaDesignAgent
from agents.server_schedule_agent import ServerScheduleAgent
from research_agents.schemas import (
    AgentName,
    CodeReviewReport,
    ExperimentCodeRunManifest,
    ExperimentCodeRunReport,
    JobObservation,
    MetricObservation,
    ResultCheckReport,
    ServerScheduleManifest,
    TaskEnvelope,
    utc_now,
)
from research_agents.storage import ArtifactStore

from test_code_pipeline import StaticModel
from test_server_pipeline import (
    FakeServerBackend,
    execution_config,
    reviewed_inputs,
)


READY_COVERAGE = {
    "summary": "The approved supplementary experiment set covers the explicit claims.",
    "missing_requirements": [],
}


class SeedMetricBackend(FakeServerBackend):
    def __init__(self, values: dict[int, float] | None = None) -> None:
        super().__init__(metric_value=0.2)
        self.values = values or {7: 0.1, 17: 0.2, 27: 0.4}
        self.omit_metric = False
        self.fail_with_cuda = False

    def observe_job(self, profile, job):
        if self.result_status == "running":
            return super().observe_job(profile, job)
        if self.fail_with_cuda:
            return JobObservation(
                job_id=job.job_id,
                status="failed",
                exit_code=1,
                log_tail="CUDA out of memory while allocating the experiment batch",
            )
        if self.omit_metric:
            return JobObservation(
                job_id=job.job_id,
                status="completed",
                exit_code=0,
                log_tail="job exited without a primary metric record",
            )
        value = self.values[job.seed]
        metric = MetricObservation(
            metric_name=job.metric_name,
            value=value,
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
            log_tail=f"observed {job.metric_name}={value}",
        )


def prepare_passed_main(
    artifacts: ArtifactStore,
    run_id: str,
    backend: FakeServerBackend,
):
    design_ref, data_ref, code_ref, review_ref, _ = reviewed_inputs(
        artifacts, run_id
    )
    config = execution_config()
    schedule_result = ServerScheduleAgent(
        StaticModel({}), artifacts, backend=backend, execution_config=config
    ).run(
        TaskEnvelope(
            run_id=run_id,
            task_id=f"task_main_schedule_{run_id}",
            agent=AgentName.SERVER_SCHEDULE,
            objective="produce a satisfactory main benchmark before supplementary work",
            inputs=[design_ref, data_ref, code_ref, review_ref],
            context={"project_slug": "assaycal-test"},
        )
    )
    if schedule_result.status != "succeeded":
        raise AssertionError(schedule_result.summary)
    main_result = CheckResultAgent(
        StaticModel({}), artifacts, backend=backend, execution_config=config
    ).run(
        TaskEnvelope(
            run_id=run_id,
            task_id=f"task_main_result_{run_id}",
            agent=AgentName.CHECK_RESULT,
            objective="verify the main benchmark gate",
            inputs=[design_ref, schedule_result.artifacts[0]],
            context={"project_slug": "assaycal-test"},
        )
    )
    if main_result.status != "succeeded":
        raise AssertionError(main_result.summary)
    return {
        "design_ref": design_ref,
        "data_ref": data_ref,
        "code_ref": code_ref,
        "review_ref": review_ref,
        "schedule_ref": schedule_result.artifacts[0],
        "main_result_ref": main_result.artifacts[0],
        "config": config,
        "inputs": [
            design_ref,
            code_ref,
            review_ref,
            schedule_result.artifacts[0],
            main_result.artifacts[0],
        ],
    }


def experiment_task(run_id: str, inputs, suffix: str) -> TaskEnvelope:
    return TaskEnvelope(
        run_id=run_id,
        task_id=f"task_experiment_{suffix}",
        agent=AgentName.EXPERIMENT_CODE_RUN,
        objective="run all approved computational experiments and preserve exact evidence",
        inputs=list(inputs),
        context={"project_slug": "assaycal-test", "state_version": 14},
    )


def force_manifest_due(
    artifacts: ArtifactStore,
    run_id: str,
    manifest_ref,
):
    manifest = ExperimentCodeRunManifest.model_validate(
        artifacts.read_artifact_json(manifest_ref)
    )
    past = utc_now() - timedelta(seconds=1)
    due = manifest.model_copy(
        update={
            "next_report_at": past,
            "jobs": [
                item.model_copy(update={"next_report_at": past})
                for item in manifest.jobs
            ],
        }
    )
    return artifacts.write_agent_json(
        AgentName.EXPERIMENT_CODE_RUN,
        f"project/var/runs/{run_id}/experiment_code_run/forced-due/launch.json",
        due.model_dump(mode="json"),
    ).model_copy(update={"kind": "experiment_code_run_manifest"})


class ExperimentCodeRunTests(unittest.TestCase):
    def test_idea_design_enforces_a_plan_and_expected_result_for_each_gap(self):
        requirement = {
            "requirement_id": "experiment-gap-efficiency",
            "experiment_type": "efficiency",
            "evidence_refs": ["paper-one"],
        }
        with self.assertRaisesRegex(ValueError, "was not added"):
            IdeaDesignAgent._validate_required_experiment_revisions(
                {"experiment_plans": [], "expected_results": []}, [requirement]
            )
        IdeaDesignAgent._validate_required_experiment_revisions(
            {
                "experiment_plans": [
                    {
                        "experiment_id": "experiment-gap-efficiency",
                        "experiment_type": "efficiency",
                        "evidence_refs": ["paper-one", "paper-two"],
                    }
                ],
                "expected_results": [
                    {"experiment_id": "experiment-gap-efficiency"}
                ],
            },
            [requirement],
        )

    def test_blocks_without_a_passed_main_result(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=execution_config(),
            ).run(experiment_task("run_missing_main", [], "missing_main"))
            self.assertEqual(result.status, "blocked")
            self.assertTrue(result.needs_human)
            self.assertEqual(result.issues[0].code, "EXPERIMENT_MAIN_GATE_MISSING")
            self.assertEqual(backend.launched, [])

    def test_design_gap_routes_to_idea_design_and_launches_nothing(self):
        run_id = "run_experiment_design_gap"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared = prepare_passed_main(artifacts, run_id, backend)
            main_launches = len(backend.launched)
            response = {
                "summary": "An efficiency experiment is required to support the runtime claim.",
                "missing_requirements": [
                    {
                        "requirement_id": "experiment-gap-efficiency",
                        "experiment_type": "efficiency",
                        "objective": "Measure training time, peak memory, and inference throughput.",
                        "rationale": "The approved design discusses feasibility but has no resource comparison.",
                        "required_metrics": ["training_seconds", "peak_memory_mib"],
                        "evidence_refs": ["paper-one"],
                        "required_code_changes": ["add deterministic resource instrumentation"],
                        "computational_only": True,
                    }
                ],
            }
            result = ExperimentCodeRunAgent(
                StaticModel(response),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(experiment_task(run_id, prepared["inputs"], "design_gap"))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.IDEA_DESIGN)
            self.assertEqual(len(backend.launched), main_launches)
            audit = artifacts.read_artifact_json(result.artifacts[0])
            self.assertEqual(audit["decision"], "requires_design_revision")
            self.assertEqual(
                audit["missing_requirements"][0]["requirement_id"],
                "experiment-gap-efficiency",
            )

    def test_reviewed_code_gap_routes_to_code_make(self):
        run_id = "run_experiment_code_gap"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared = prepare_passed_main(artifacts, run_id, backend)
            review = CodeReviewReport.model_validate(
                artifacts.read_artifact_json(prepared["review_ref"])
            ).model_copy(update={"covered_experiment_ids": ["experiment-main"]})
            review_ref = artifacts.write_agent_json(
                AgentName.CODE_REVIEW,
                "design/assaycal-test/reviews/code/experiment-gap-review.json",
                review.model_dump(mode="json"),
            ).model_copy(update={"kind": "code_review_report"})
            schedule = ServerScheduleManifest.model_validate(
                artifacts.read_artifact_json(prepared["schedule_ref"])
            ).model_copy(update={"code_review_uri": review_ref.uri})
            schedule_ref = artifacts.write_agent_json(
                AgentName.SERVER_SCHEDULE,
                f"project/var/runs/{run_id}/server_schedule/experiment-gap.json",
                schedule.model_dump(mode="json"),
            ).model_copy(update={"kind": "server_schedule_manifest"})
            main_result = ResultCheckReport.model_validate(
                artifacts.read_artifact_json(prepared["main_result_ref"])
            ).model_copy(update={"schedule_manifest_uri": schedule_ref.uri})
            main_result_ref = artifacts.write_agent_json(
                AgentName.CHECK_RESULT,
                "design/assaycal-test/reviews/results/experiment-gap.json",
                main_result.model_dump(mode="json"),
            ).model_copy(update={"kind": "result_check_report"})
            main_launches = len(backend.launched)
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(
                experiment_task(
                    run_id,
                    [
                        prepared["design_ref"],
                        prepared["code_ref"],
                        review_ref,
                        schedule_ref,
                        main_result_ref,
                    ],
                    "code_gap",
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.CODE_MAKE)
            self.assertEqual(len(backend.launched), main_launches)

    def test_launches_every_non_main_experiment_and_declared_seed(self):
        run_id = "run_experiment_launch"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared = prepare_passed_main(artifacts, run_id, backend)
            main_launches = len(backend.launched)
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(experiment_task(run_id, prepared["inputs"], "launch"))
            self.assertEqual(result.status, "in_progress", result.summary)
            manifest_ref = next(
                item
                for item in result.artifacts
                if item.kind == "experiment_code_run_manifest"
            )
            manifest = ExperimentCodeRunManifest.model_validate(
                artifacts.read_artifact_json(manifest_ref)
            )
            self.assertEqual(
                set(manifest.experiment_ids),
                {"experiment-ablation", "experiment-case"},
            )
            self.assertEqual(len(manifest.jobs), 6)
            self.assertEqual(
                {(item.experiment_id, item.seed) for item in manifest.jobs},
                {
                    (experiment_id, seed)
                    for experiment_id in manifest.experiment_ids
                    for seed in (7, 17, 27)
                },
            )
            self.assertEqual(len(backend.launched) - main_launches, 6)
            self.assertTrue(all(item.status == "launched" for item in manifest.jobs))
            self.assertEqual(
                {item.metric_name for item in manifest.jobs},
                {"supplementary_primary_rmse"},
            )
            self.assertEqual(len(backend.demos), 2)

    def test_failed_foreground_demo_prevents_supplementary_nohup_launch(self):
        run_id = "run_experiment_demo_failure"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared = prepare_passed_main(artifacts, run_id, backend)
            backend.demo_status = "failed"
            main_launches = len(backend.launched)
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(experiment_task(run_id, prepared["inputs"], "demo_failure"))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.issues[0].code, "EXPERIMENT_FOREGROUND_DEMO_FAILED")
            self.assertEqual(len(backend.launched), main_launches)

    def _launch_and_force_due(self, artifacts, run_id, backend):
        prepared = prepare_passed_main(artifacts, run_id, backend)
        launch = ExperimentCodeRunAgent(
            StaticModel(READY_COVERAGE),
            artifacts,
            backend=backend,
            execution_config=prepared["config"],
        ).run(experiment_task(run_id, prepared["inputs"], "initial_launch"))
        self.assertEqual(launch.status, "in_progress")
        launch_ref = next(
            item
            for item in launch.artifacts
            if item.kind == "experiment_code_run_manifest"
        )
        forced_ref = force_manifest_due(artifacts, run_id, launch_ref)
        return prepared, forced_ref

    def test_active_jobs_return_in_progress_with_twelve_hour_report(self):
        run_id = "run_experiment_active"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared, forced_ref = self._launch_and_force_due(
                artifacts, run_id, backend
            )
            backend.result_status = "running"
            result = ExperimentCodeRunAgent(
                StaticModel({"unexpected": "model must not be called while monitoring"}),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(
                experiment_task(
                    run_id, [*prepared["inputs"], forced_ref], "active_monitor"
                )
            )
            self.assertEqual(result.status, "in_progress")
            report_ref = next(
                item for item in result.artifacts if item.kind == "experiment_code_run_report"
            )
            report = ExperimentCodeRunReport.model_validate(
                artifacts.read_artifact_json(report_ref)
            )
            self.assertEqual(report.verdict, "in_progress")
            self.assertEqual(report.report_interval_seconds, 43200)
            self.assertIsNotNone(report.next_report_at)

    def test_completed_jobs_compute_exact_seed_statistics_from_observations(self):
        run_id = "run_experiment_complete"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend({7: 0.1, 17: 0.2, 27: 0.4})
            prepared, forced_ref = self._launch_and_force_due(
                artifacts, run_id, backend
            )
            result = ExperimentCodeRunAgent(
                StaticModel({"numeric_result": 999999}),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(
                experiment_task(
                    run_id, [*prepared["inputs"], forced_ref], "complete_monitor"
                )
            )
            self.assertEqual(result.status, "succeeded", result.summary)
            report_ref = next(
                item for item in result.artifacts if item.kind == "experiment_code_run_report"
            )
            report = ExperimentCodeRunReport.model_validate(
                artifacts.read_artifact_json(report_ref)
            )
            expected_mean = (0.1 + 0.2 + 0.4) / 3
            expected_std = math.sqrt(
                sum((value - expected_mean) ** 2 for value in (0.1, 0.2, 0.4)) / 2
            )
            self.assertEqual(len(report.primary_metric_aggregates), 2)
            for aggregate in report.primary_metric_aggregates:
                self.assertAlmostEqual(aggregate.mean, expected_mean)
                self.assertAlmostEqual(
                    aggregate.sample_standard_deviation, expected_std
                )
                self.assertEqual(aggregate.minimum, 0.1)
                self.assertEqual(aggregate.maximum, 0.4)
                self.assertEqual(
                    [item.value for item in aggregate.seed_metrics], [0.1, 0.2, 0.4]
                )
            self.assertEqual(
                result.evidence[1]["model_generated_numeric_results"], 0
            )

    def test_missing_primary_metric_routes_to_code_make(self):
        run_id = "run_experiment_metric_missing"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared, forced_ref = self._launch_and_force_due(
                artifacts, run_id, backend
            )
            backend.omit_metric = True
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(
                experiment_task(
                    run_id, [*prepared["inputs"], forced_ref], "metric_missing"
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.CODE_MAKE)

    def test_cuda_job_failure_routes_to_server_schedule(self):
        run_id = "run_experiment_cuda_failure"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            backend = SeedMetricBackend()
            prepared, forced_ref = self._launch_and_force_due(
                artifacts, run_id, backend
            )
            backend.fail_with_cuda = True
            result = ExperimentCodeRunAgent(
                StaticModel(READY_COVERAGE),
                artifacts,
                backend=backend,
                execution_config=prepared["config"],
            ).run(
                experiment_task(
                    run_id, [*prepared["inputs"], forced_ref], "cuda_failure"
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.SERVER_SCHEDULE)


if __name__ == "__main__":
    unittest.main()
