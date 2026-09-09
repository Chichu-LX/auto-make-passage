from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.process_logging import redact_text
from research_agents.project_layout import artifact_prefix, scoped_path, task_project_slug
from research_agents.providers import JsonModel
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    CodeImplementationManifest,
    CodeReviewCheck,
    CodeReviewExecutionRecord,
    CodeReviewReport,
    ResearchDesign,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import MAX_INPUT_BYTES, MAX_REVIEW_ITERATIONS, file_token
from .review_engine import (
    architecture_check,
    execute_review_commands,
    execution_check,
    innovation_alignment_check,
    recommendations_for,
    reliability_check,
    scientific_protocol_check,
    static_analysis_check,
    traceability_check,
    training_update_check,
    vendored_source_check,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.CODE_REVIEW:
    raise RuntimeError("Code Review Agent package name mismatch")


class CodeReviewAgent:
    """Independently review one immutable Code Make revision without editing it."""

    name = AgentName.CODE_REVIEW
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(self, model: JsonModel, artifacts: ArtifactStore) -> None:
        self.model = model
        self.artifacts = artifacts

    def _read_inputs(
        self, task: TaskEnvelope
    ) -> tuple[ResearchDesign | None, CodeImplementationManifest | None, ArtifactRef | None]:
        slug = task_project_slug(task)
        design: ResearchDesign | None = None
        manifest: CodeImplementationManifest | None = None
        manifest_ref: ArtifactRef | None = None
        for artifact in task.inputs:
            try:
                if artifact.kind == "research_design":
                    if not artifact.uri.startswith(artifact_prefix("design", slug, "research-designs")):
                        raise StoragePolicyError("research design escaped its project namespace")
                    candidate = ResearchDesign.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id != task.run_id:
                        raise StoragePolicyError("research design belongs to another run")
                    design = candidate
                elif artifact.kind == "code_implementation_manifest":
                    if not artifact.uri.startswith(artifact_prefix("code", slug, "manifests")):
                        raise StoragePolicyError("code manifest escaped its project namespace")
                    candidate = CodeImplementationManifest.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id != task.run_id or candidate.project_slug != slug:
                        raise StoragePolicyError("code manifest belongs to another run or project")
                    manifest, manifest_ref = candidate, artifact
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        return design, manifest, manifest_ref

    def _write_gap(self, task: TaskEnvelope, reason: str) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "design",
                task_project_slug(task),
                "reviews",
                "code",
                f"{file_token(task.task_id)}.gap.json",
            ),
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": task_project_slug(task),
                "status": "blocked",
                "reason": reason,
            },
        ).model_copy(update={"kind": "code_review_gap"})

    @staticmethod
    def _skipped_executions(reason: str) -> list[CodeReviewExecutionRecord]:
        values = (
            ("minimal_demo_first", ["python", "-m", "<package>.demo", "--json"]),
            ("minimal_demo_repeat", ["python", "-m", "<package>.demo", "--json"]),
            ("unit_tests", ["python", "-m", "unittest", "discover"]),
        )
        return [
            CodeReviewExecutionRecord(
                execution_id=execution_id,
                argv=argv,
                status="failed",
                return_code=None,
                duration_ms=0,
                timeout_seconds=30,
                stderr_tail=reason,
            )
            for execution_id, argv in values
        ]

    def _model_advisory(
        self,
        task: TaskEnvelope,
        manifest: CodeImplementationManifest,
        checks: list[CodeReviewCheck],
    ) -> tuple[str, list[str]]:
        context = {
            "objective": task.objective,
            "project_slug": manifest.project_slug,
            "implementation_root": manifest.implementation_root,
            "architecture_summary": manifest.architecture_summary,
            "known_gaps": manifest.known_gaps,
            "deterministic_checks": [item.model_dump(mode="json") for item in checks],
            "policy": {
                "verdict_is_deterministic": True,
                "do_not_edit_code": True,
                "return": {"summary": "string", "recommendations": ["string"]},
            },
        }
        try:
            response = self.model.complete_json(
                PACKAGE.system_prompt,
                json.dumps(context, ensure_ascii=False),
            )
        except Exception:
            return "", []
        summary = redact_text(str(response.get("summary", "")), max_chars=2000).strip()
        raw = response.get("recommendations", [])
        recommendations: list[str] = []
        if isinstance(raw, list):
            for value in raw[:20]:
                text = redact_text(str(value), max_chars=1000).strip()
                if len(text) >= 5:
                    recommendations.append(text)
        return summary, recommendations

    @staticmethod
    def _markdown(report: CodeReviewReport) -> str:
        lines = [
            "# Code Review Report",
            "",
            f"- Verdict: `{report.verdict}`",
            f"- Review iteration: `{report.review_iteration}/{MAX_REVIEW_ITERATIONS}`",
            f"- Implementation: `{report.code_manifest_uri}`",
            "",
            report.summary,
            "",
            "## Checks",
            "",
            "| Check | Category | Status | Summary |",
            "|---|---|---|---|",
        ]
        for item in report.checks:
            summary = item.summary.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{item.check_id}` | {item.category} | `{item.status}` | {summary} |"
            )
        lines.extend(["", "## Fixed executions", ""])
        for item in report.executions:
            lines.append(
                f"- `{item.execution_id}`: `{item.status}`, return={item.return_code}, "
                f"duration={item.duration_ms:.3f} ms, timeout={item.timeout_seconds} s"
            )
        if report.blockers:
            lines.extend(["", "## Blocking findings", ""])
            lines.extend(f"- {value}" for value in report.blockers)
        if report.recommendations:
            lines.extend(["", "## Code Make repair instructions", ""])
            lines.extend(f"- {value}" for value in report.recommendations)
        lines.extend(
            [
                "",
                "## Coverage",
                "",
                f"- Components: `{', '.join(report.covered_component_ids)}`",
                f"- Datasets: `{', '.join(report.covered_dataset_keys)}`",
                f"- Experiments: `{', '.join(report.covered_experiment_ids)}`",
                f"- Local source modules: `{', '.join(report.covered_source_module_ids)}`",
                "",
            ]
        )
        return "\n".join(lines)

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        design, manifest, manifest_ref = self._read_inputs(task)
        if design is None or manifest is None or manifest_ref is None:
            reason = "Approved ResearchDesign and a validated CodeImplementationManifest are required."
            gap = self._write_gap(task, reason)
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
                        "code": "CODE_REVIEW_INPUT_MISSING",
                        "message": reason,
                        "root_cause_key": "G5:CODE_REVIEW_INPUT_MISSING:inputs",
                    }
                ],
                needs_human=True,
            )

        architecture = architecture_check(manifest)
        traceability, covered = traceability_check(manifest, design)
        alignment = innovation_alignment_check(manifest, design)
        protocol = scientific_protocol_check(manifest, design, self.artifacts)
        sources = vendored_source_check(manifest, design, self.artifacts)
        static = static_analysis_check(manifest, self.artifacts)
        pre_execution = [architecture, traceability, alignment, protocol, sources, static]
        if static.status == "failed":
            executions = self._skipped_executions(
                "Execution skipped because static integrity/safety review failed."
            )
        else:
            try:
                executions = execute_review_commands(manifest, self.artifacts)
            except (OSError, ValueError, StoragePolicyError) as exc:
                executions = self._skipped_executions(
                    f"Execution setup failed: {type(exc).__name__}."
                )
        checks = [
            *pre_execution,
            execution_check(executions),
            training_update_check(executions),
            reliability_check(executions),
        ]
        failed_checks = [item for item in checks if item.status == "failed"]
        verdict = "needs_revision" if failed_checks else "passed"
        deterministic_recommendations = recommendations_for(checks)
        advisory_summary, advisory_recommendations = self._model_advisory(
            task, manifest, checks
        )
        blockers = [item.summary for item in failed_checks]
        recommendations = list(dict.fromkeys([*deterministic_recommendations, *advisory_recommendations]))
        if verdict == "needs_revision" and not recommendations:
            recommendations = [
                "Address every failed check, regenerate the implementation manifest, and rerun the fixed local validations."
            ]
        base_summary = (
            f"Code Review iteration {min(task.attempt, MAX_REVIEW_ITERATIONS)}/"
            f"{MAX_REVIEW_ITERATIONS}: {len(checks) - len(failed_checks)} checks passed or warned, "
            f"{len(failed_checks)} failed."
        )
        summary = f"{base_summary} {advisory_summary}".strip()
        report = CodeReviewReport(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=manifest.project_slug,
            code_manifest_uri=manifest_ref.uri,
            review_iteration=min(task.attempt, MAX_REVIEW_ITERATIONS),
            verdict=verdict,
            summary=summary,
            checks=checks,
            executions=executions,
            blockers=blockers,
            recommendations=recommendations,
            covered_component_ids=covered["components"],
            covered_dataset_keys=covered["datasets"],
            covered_experiment_ids=covered["experiments"],
            covered_source_module_ids=covered["sources"],
        )
        token = file_token(task.task_id)
        json_ref = self.artifacts.write_agent_json(
            self.name,
            scoped_path("design", manifest.project_slug, "reviews", "code", f"{token}.json"),
            report.model_dump(mode="json"),
        ).model_copy(update={"kind": "code_review_report"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            scoped_path("design", manifest.project_slug, "reviews", "code", f"{token}.md"),
            self._markdown(report).encode("utf-8"),
            max_bytes=512 * 1024,
            kind="code_review_markdown",
        )
        issues = [
            {
                "severity": "blocking",
                "code": f"CODE_REVIEW_{item.check_id.removeprefix('review-').replace('-', '_').upper()}",
                "message": item.summary,
                "root_cause_key": f"G5:{item.check_id}:implementation",
            }
            for item in failed_checks
        ]
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded" if verdict == "passed" else "failed",
            summary=summary,
            artifacts=[json_ref, markdown_ref],
            evidence=[
                {
                    "check_count": len(checks),
                    "failed_check_count": len(failed_checks),
                    "review_iteration": report.review_iteration,
                    "fixed_execution_count": len(executions),
                    "model_update_verified": checks[-2].status == "passed",
                    "determinism_verified": checks[-1].status == "passed",
                }
            ],
            issues=issues,
            route_hint=AgentName.CODE_MAKE if verdict == "needs_revision" else None,
            needs_human=False,
        )
