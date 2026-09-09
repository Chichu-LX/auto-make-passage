from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.paper_evidence import (
    PaperEvidenceError,
    build_citation_evidence,
    build_observed_experiment_results,
    extract_code_symbol_catalog,
    resolve_formula_evidence,
    source_excerpt,
    validate_writing_draft,
)
from research_agents.process_logging import ProcessLogStore, redact_text
from research_agents.project_layout import artifact_prefix, task_project_slug
from research_agents.providers import JsonModel
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    CodeImplementationManifest,
    CodeReviewReport,
    DatasetManifest,
    ExperimentCodeRunManifest,
    ExperimentCodeRunReport,
    PaperManifest,
    PaperManuscriptManifest,
    PaperWritingDraft,
    ProcessLogRecord,
    ResearchDesign,
    ResultCheckReport,
    ServerScheduleManifest,
    TaskEnvelope,
    WorkflowState,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_CODE_EXCERPT_CHARS,
    MAX_CODE_SYMBOLS_IN_PROMPT,
    MAX_INPUT_BYTES,
    MAX_MANUSCRIPT_BYTES,
    MAX_MODEL_CONTEXT_CHARS,
    MAX_PAPER_ABSTRACT_CHARS,
    file_token,
)
from .rendering import render_bibtex, render_latex, render_markdown


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.WRITE_PAPER:
    raise RuntimeError("Write Paper Agent package name mismatch")


@dataclass(frozen=True)
class _ResolvedInputs:
    papers: PaperManifest
    papers_ref: ArtifactRef
    design: ResearchDesign
    design_ref: ArtifactRef
    data: DatasetManifest
    data_ref: ArtifactRef
    code: CodeImplementationManifest
    code_ref: ArtifactRef
    review: CodeReviewReport
    review_ref: ArtifactRef
    schedule: ServerScheduleManifest
    schedule_ref: ArtifactRef
    main_result: ResultCheckReport
    main_result_ref: ArtifactRef
    experiment_run: ExperimentCodeRunManifest
    experiment_run_ref: ArtifactRef
    experiment_result: ExperimentCodeRunReport
    experiment_result_ref: ArtifactRef


@dataclass(frozen=True)
class _InputResolution:
    inputs: _ResolvedInputs | None
    issue_code: str
    reason: str
    route_hint: AgentName
    missing_paper_ids: tuple[str, ...] = ()


class WritePaperAgent:
    """Render a paper only from locally verified literature, code, and results."""

    name = AgentName.WRITE_PAPER
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(self, model: JsonModel, artifacts: ArtifactStore) -> None:
        self.model = model
        self.artifacts = artifacts
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
                workflow_state=WorkflowState.WRITING,
                state_version=max(1, int(task.context.get("state_version", 1))),
                task_id=task.task_id,
                attempt=task.attempt,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def _read_candidate(self, artifact: ArtifactRef, model_type):
        return model_type.model_validate(
            self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
        )

    @staticmethod
    def _design_paper_ids(design: ResearchDesign) -> set[str]:
        """Collect every paper ID that the approved design treats as evidence."""

        refs = {
            ref
            for component in design.method_components
            for ref in component.evidence_refs
        }
        refs.update(
            ref
            for requirement in design.dataset_requirements
            for ref in requirement.paper_refs
        )
        refs.update(
            ref
            for experiment in design.experiment_plans
            for ref in experiment.evidence_refs
        )
        refs.update(
            ref
            for result in design.expected_results
            for ref in result.evidence_refs
        )
        refs.update(item.paper_id for item in design.paper_comparisons)
        refs.update(item.source_paper_id for item in design.reusable_code_modules)
        refs.update(
            ref for decision in design.design_decisions for ref in decision.evidence_refs
        )
        refs.update(
            ref for report in design.advisor_consensus for ref in report.evidence_refs
        )
        return refs

    @classmethod
    def _papers_support_design(
        cls,
        papers: PaperManifest,
        papers_ref: ArtifactRef,
        design: ResearchDesign,
    ) -> tuple[bool, list[str]]:
        """Require a source-linked, complete local-PDF set for all design citations."""

        downloaded_records = {
            record.paper_id: record
            for record in papers.records
            if record.status == "downloaded" and record.local_artifact is not None
        }
        required_ids = cls._design_paper_ids(design)
        missing_ids = sorted(required_ids - set(downloaded_records))
        local_pdf_uris = {
            record.local_artifact.uri for record in downloaded_records.values()
        }
        declared_sources = set(design.paper_evidence_uris)
        # Current Idea Design records source manifest/corpus URIs. The PDF form is
        # retained as a bounded compatibility path for older typed designs.
        source_linked = papers_ref.uri in declared_sources or declared_sources.issubset(
            local_pdf_uris
        )
        supported = (
            design.evidence_quality == "downloaded"
            and bool(required_ids)
            and not missing_ids
            and source_linked
        )
        return supported, missing_ids

    def _read_inputs(self, task: TaskEnvelope) -> _InputResolution:
        slug = task_project_slug(task)
        collections: dict[str, list[tuple[Any, ArtifactRef]]] = {
            name: []
            for name in (
                "papers",
                "design",
                "data",
                "code",
                "review",
                "schedule",
                "main_result",
                "experiment_run",
                "experiment_result",
            )
        }
        for artifact in task.inputs:
            try:
                if artifact.kind == "paper_manifest" and artifact.uri.startswith(
                    artifact_prefix("reference", slug, "manifests")
                ):
                    value = self._read_candidate(artifact, PaperManifest)
                    if value.run_id == task.run_id:
                        collections["papers"].append((value, artifact))
                elif artifact.kind == "research_design" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "research-designs")
                ):
                    value = self._read_candidate(artifact, ResearchDesign)
                    if value.run_id == task.run_id:
                        collections["design"].append((value, artifact))
                elif artifact.kind == "data_manifest" and artifact.uri.startswith(
                    artifact_prefix("data", slug, "manifests")
                ):
                    value = self._read_candidate(artifact, DatasetManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["data"].append((value, artifact))
                elif artifact.kind == "code_implementation_manifest" and artifact.uri.startswith(
                    artifact_prefix("code", slug, "manifests")
                ):
                    value = self._read_candidate(artifact, CodeImplementationManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["code"].append((value, artifact))
                elif artifact.kind == "code_review_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "code")
                ):
                    value = self._read_candidate(artifact, CodeReviewReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["review"].append((value, artifact))
                elif artifact.kind == "server_schedule_manifest" and artifact.uri.startswith(
                    f"artifact://project/var/runs/{task.run_id}/server_schedule/"
                ):
                    value = self._read_candidate(artifact, ServerScheduleManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["schedule"].append((value, artifact))
                elif artifact.kind == "result_check_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "reviews", "results")
                ):
                    value = self._read_candidate(artifact, ResultCheckReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["main_result"].append((value, artifact))
                elif artifact.kind == "experiment_code_run_manifest" and artifact.uri.startswith(
                    f"artifact://project/var/runs/{task.run_id}/experiment_code_run/"
                ):
                    value = self._read_candidate(artifact, ExperimentCodeRunManifest)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["experiment_run"].append((value, artifact))
                elif artifact.kind == "experiment_code_run_report" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "experiments", "results")
                ):
                    value = self._read_candidate(artifact, ExperimentCodeRunReport)
                    if value.run_id == task.run_id and value.project_slug == slug:
                        collections["experiment_result"].append((value, artifact))
            except (StoragePolicyError, ValidationError, ValueError):
                continue

        missing_collections = sorted(
            name for name, values in collections.items() if not values
        )
        if missing_collections:
            literature_missing = bool(
                {"papers", "design"}.intersection(missing_collections)
            )
            return _InputResolution(
                inputs=None,
                issue_code=(
                    "PAPER_WRITE_LITERATURE_EVIDENCE_MISSING"
                    if literature_missing
                    else "PAPER_WRITE_INPUT_MISSING"
                ),
                reason=(
                    "The writing evidence bundle is missing these typed inputs: "
                    + ", ".join(missing_collections)
                ),
                route_hint=(
                    AgentName.DOWNLOAD_PAPER
                    if literature_missing
                    else AgentName.EXPERIMENT_CODE_RUN
                ),
            )
        literature_gap_details: set[str] = set()
        for experiment_result, experiment_result_ref in reversed(
            collections["experiment_result"]
        ):
            if experiment_result.verdict != "passed":
                continue
            run_matches = [
                item
                for item in collections["experiment_run"]
                if item[1].uri == experiment_result.run_manifest_uri
                and item[0].status == "launched"
            ]
            for experiment_run, experiment_run_ref in reversed(run_matches):
                main_matches = [
                    item
                    for item in collections["main_result"]
                    if item[1].uri == experiment_run.main_result_report_uri
                    and item[0].verdict == "passed"
                ]
                schedule_matches = [
                    item
                    for item in collections["schedule"]
                    if item[1].uri == experiment_run.main_schedule_manifest_uri
                    and item[0].status == "launched"
                ]
                review_matches = [
                    item
                    for item in collections["review"]
                    if item[1].uri == experiment_run.code_review_uri
                    and item[0].verdict == "passed"
                ]
                code_matches = [
                    item
                    for item in collections["code"]
                    if item[1].uri == experiment_run.code_manifest_uri
                ]
                design_matches = [
                    item
                    for item in collections["design"]
                    if item[1].uri == experiment_run.research_design_uri
                ]
                if not all(
                    (
                        main_matches,
                        schedule_matches,
                        review_matches,
                        code_matches,
                        design_matches,
                    )
                ):
                    continue
                main_result, main_result_ref = main_matches[-1]
                schedule, schedule_ref = schedule_matches[-1]
                review, review_ref = review_matches[-1]
                code, code_ref = code_matches[-1]
                design, design_ref = design_matches[-1]
                data_matches = [
                    item
                    for item in collections["data"]
                    if item[1].uri == code.data_manifest_uri
                    and schedule.data_manifest_uri == item[1].uri
                ]
                paper_matches = []
                for papers, papers_ref in collections["papers"]:
                    supported, missing_ids = self._papers_support_design(
                        papers, papers_ref, design
                    )
                    if supported:
                        paper_matches.append((papers, papers_ref))
                    else:
                        literature_gap_details.update(missing_ids)
                if not data_matches or not paper_matches:
                    if not paper_matches:
                        if design.evidence_quality != "downloaded":
                            literature_gap_details.add(
                                "design evidence_quality is metadata_only"
                            )
                        if not literature_gap_details:
                            literature_gap_details.add(
                                "paper manifest is not linked to the approved design evidence sources"
                            )
                    continue
                data, data_ref = data_matches[-1]
                papers, papers_ref = paper_matches[-1]
                if not (
                    code.research_design_uri == design_ref.uri
                    and review.code_manifest_uri == code_ref.uri
                    and schedule.research_design_uri == design_ref.uri
                    and schedule.code_manifest_uri == code_ref.uri
                    and schedule.code_review_uri == review_ref.uri
                    and main_result.schedule_manifest_uri == schedule_ref.uri
                ):
                    continue
                return _InputResolution(
                    inputs=_ResolvedInputs(
                        papers=papers,
                        papers_ref=papers_ref,
                        design=design,
                        design_ref=design_ref,
                        data=data,
                        data_ref=data_ref,
                        code=code,
                        code_ref=code_ref,
                        review=review,
                        review_ref=review_ref,
                        schedule=schedule,
                        schedule_ref=schedule_ref,
                        main_result=main_result,
                        main_result_ref=main_result_ref,
                        experiment_run=experiment_run,
                        experiment_run_ref=experiment_run_ref,
                        experiment_result=experiment_result,
                        experiment_result_ref=experiment_result_ref,
                    ),
                    issue_code="",
                    reason="",
                    route_hint=AgentName.WRITE_PAPER,
                )
        if literature_gap_details:
            details = ", ".join(sorted(literature_gap_details))
            return _InputResolution(
                inputs=None,
                issue_code="PAPER_WRITE_LITERATURE_EVIDENCE_MISSING",
                reason=(
                    "Every paper ID cited by the approved Research Design must be backed "
                    f"by a source-linked, verified local PDF. Missing or invalid evidence: {details}."
                ),
                route_hint=AgentName.DOWNLOAD_PAPER,
                missing_paper_ids=tuple(
                    sorted(
                        item
                        for item in literature_gap_details
                        if item.startswith("paper-")
                    )
                ),
            )
        return _InputResolution(
            inputs=None,
            issue_code="PAPER_WRITE_INPUT_MISSING",
            reason=(
                "No single project-matched chain connects the passed code review, main "
                "result, and supplementary result artifacts required for writing."
            ),
            route_hint=AgentName.EXPERIMENT_CODE_RUN,
        )

    def _write_gap(
        self,
        task: TaskEnvelope,
        *,
        reason: str,
        issue_code: str,
        missing_paper_ids: tuple[str, ...] = (),
    ) -> ArtifactRef:
        slug = task_project_slug(task)
        return self.artifacts.write_agent_json(
            self.name,
            f"project/paper/{slug}/gaps/{file_token(task.task_id)}.json",
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": slug,
                "status": "blocked",
                "issue_code": issue_code,
                "reason": reason,
                "missing_paper_ids": list(missing_paper_ids),
            },
        ).model_copy(update={"kind": "paper_writing_gap"})

    @staticmethod
    def _dataset_context(data: DatasetManifest) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for record in data.records:
            values.append(
                {
                    "dataset_id": record.dataset_id,
                    "title": record.name,
                    "status": record.status,
                    "materialized_size_bytes": record.materialized_size_bytes,
                    "profile": (
                        record.profile.model_dump(mode="json")
                        if record.profile is not None
                        else None
                    ),
                }
            )
        return values

    def _writing_prompt(
        self,
        task: TaskEnvelope,
        inputs: _ResolvedInputs,
        citations,
        results,
        catalog,
        sources: dict[str, str],
    ) -> str:
        records = {item.paper_id: item for item in inputs.papers.records}
        selected_symbols = []
        selected_keys: set[tuple[str, str]] = set()
        for component in inputs.design.method_components:
            match = next(
                (
                    item
                    for item in catalog
                    if component.component_id in item.design_component_ids
                ),
                None,
            )
            if match is not None:
                key = (match.file_path, match.symbol)
                selected_keys.add(key)
                selected_symbols.append(match)
        for item in catalog:
            key = (item.file_path, item.symbol)
            if key not in selected_keys:
                selected_keys.add(key)
                selected_symbols.append(item)
            if len(selected_symbols) >= MAX_CODE_SYMBOLS_IN_PROMPT:
                break
        symbol_context = []
        for item in selected_symbols:
            symbol_context.append(
                {
                    **item.model_dump(mode="json"),
                    "source_excerpt": source_excerpt(
                        sources[item.file_path], item.line_start, radius=5
                    )[:MAX_CODE_EXCERPT_CHARS],
                }
            )
        context = {
            "objective": task.objective,
            "user_revision": task.context.get("user_revision"),
            "allowed_citations": [
                {
                    **item.model_dump(mode="json"),
                    "abstract": records[item.paper_id].abstract[:MAX_PAPER_ABSTRACT_CHARS],
                }
                for item in citations
            ],
            "research_design": inputs.design.model_dump(mode="json"),
            "measured_datasets": self._dataset_context(inputs.data),
            "reviewed_code": {
                "implementation_root": inputs.code.implementation_root,
                "architecture_summary": inputs.code.architecture_summary,
                "files": [item.model_dump(mode="json") for item in inputs.code.files],
                "ast_symbol_catalog": symbol_context,
            },
            "observed_result_keys": [
                {
                    "result_key": item.result_key,
                    "scope": item.scope,
                    "experiment_id": item.experiment_id,
                    "metric_name": item.metric_name,
                    "direction": item.direction,
                    "gate_status": item.gate_status,
                    "seed_count": len(item.seed_results),
                    "numeric_values_withheld_from_model": True,
                }
                for item in results
            ],
            "scientific_claim_boundary": {
                "manuscript_status": "preliminary_proxy_validation",
                "reviewed_code_known_gaps": inputs.code.known_gaps,
                "observed_metric_names": sorted(
                    {item.metric_name for item in results}
                ),
                "design_metric_names_not_observed": sorted(
                    {
                        metric
                        for experiment in inputs.design.experiment_plans
                        for metric in experiment.metrics
                        if metric not in {item.metric_name for item in results}
                    }
                ),
                "required_disclosures": [
                    "The executed model is a lightweight hashed-feature CPU proxy, not a faithful DeepDTA or GraphDTA neural reproduction.",
                    "Completed job/seed coverage does not imply that every metric or expected scientific result in the design was implemented.",
                    "Do not claim model superiority, novelty validation, publication readiness, or completion of unobserved CI, Spearman, ECE/MCE, per-target, scaffold, or neural-baseline analyses.",
                ],
            },
            "requirements": {
                "citation_ids": [item.paper_id for item in citations],
                "formula_symbols_must_come_from_catalog": True,
                "model_authored_numeric_result_literals_forbidden": True,
                "results_section_must_reference_every_result_key": True,
                "experiment_claims_must_cover_every_result_key": True,
                "all_design_components_must_have_method_or_formula_evidence": True,
                "only_computational_experiments": True,
                "title_abstract_discussion_and_conclusion_must_label_the_study_preliminary": True,
                "limitations_must_state_every_scientific_claim_boundary": True,
            },
            "response_schema": PaperWritingDraft.model_json_schema(),
        }
        value = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(value) > MAX_MODEL_CONTEXT_CHARS:
            raise PaperEvidenceError("bounded paper-writing context exceeds the model policy")
        return value

    def _failure(
        self,
        task: TaskEnvelope,
        *,
        issue_code: str,
        reason: str,
        route_hint: AgentName,
        missing_paper_ids: tuple[str, ...] = (),
    ) -> AgentResult:
        safe_reason = redact_text(reason, max_chars=2000)
        gap = self._write_gap(
            task,
            reason=safe_reason,
            issue_code=issue_code,
            missing_paper_ids=missing_paper_ids,
        )
        self._log_phase(
            task,
            event="paper_writing_failed",
            status="failed",
            details={"issue_code": issue_code, "route_hint": route_hint.value},
        )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="failed",
            summary=safe_reason,
            artifacts=[gap],
            issues=[
                {
                    "severity": "blocking",
                    "code": issue_code,
                    "message": safe_reason,
                    "root_cause_key": f"G9:{issue_code}:paper",
                }
            ],
            route_hint=route_hint,
            needs_human=False,
        )

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        started = perf_counter()
        self._log_phase(task, event="paper_evidence_resolution_started", status="started")
        resolution = self._read_inputs(task)
        inputs = resolution.inputs
        if inputs is None:
            return self._failure(
                task,
                issue_code=resolution.issue_code,
                reason=resolution.reason,
                route_hint=resolution.route_hint,
                missing_paper_ids=resolution.missing_paper_ids,
            )
        try:
            citations = build_citation_evidence(self.artifacts, inputs.papers)
            slug_pdf_prefix = artifact_prefix(
                "reference", task_project_slug(task), "papers"
            )
            if any(not item.local_pdf.uri.startswith(slug_pdf_prefix) for item in citations):
                raise PaperEvidenceError("a citation PDF escaped the current project namespace")
            results = build_observed_experiment_results(
                inputs.schedule,
                inputs.main_result,
                inputs.experiment_result,
                main_report_uri=inputs.main_result_ref.uri,
                experiment_report_uri=inputs.experiment_result_ref.uri,
            )
            catalog, sources = extract_code_symbol_catalog(
                self.artifacts, inputs.code, inputs.design
            )
            self._log_phase(
                task,
                event="paper_evidence_resolution_completed",
                status="succeeded",
                duration_ms=round((perf_counter() - started) * 1000, 3),
                details={
                    "citation_count": len(citations),
                    "result_row_count": len(results),
                    "code_symbol_count": len(catalog),
                },
            )

            model_started = perf_counter()
            self._log_phase(task, event="paper_synthesis_started", status="started")
            response = self.model.complete_json(
                PACKAGE.system_prompt,
                self._writing_prompt(
                    task, inputs, citations, results, catalog, sources
                ),
            )
            draft = PaperWritingDraft.model_validate(response)
            formulas = resolve_formula_evidence(
                draft.formulas,
                catalog,
                inputs.design,
                {item.paper_id for item in citations},
            )
            validate_writing_draft(
                draft,
                design=inputs.design,
                code=inputs.code,
                citations=citations,
                results=results,
                formulas=formulas,
            )
            self._log_phase(
                task,
                event="paper_synthesis_completed",
                status="succeeded",
                duration_ms=round((perf_counter() - model_started) * 1000, 3),
                details={
                    "section_count": len(draft.sections),
                    "formula_count": len(formulas),
                    "claim_count": len(draft.claims),
                },
            )

            render_started = perf_counter()
            self._log_phase(task, event="paper_rendering_started", status="started")
            rendering_args = {
                "title": draft.title,
                "abstract": draft.abstract,
                "keywords": draft.keywords,
                "sections": draft.sections,
                "formulas": formulas,
                "claims": draft.claims,
                "citations": citations,
                "results": results,
                "acknowledgements": draft.acknowledgements,
            }
            markdown = render_markdown(**rendering_args).encode("utf-8")
            latex = render_latex(**rendering_args).encode("utf-8")
            bibliography = render_bibtex(citations).encode("utf-8")
            token = file_token(task.task_id)
            base = f"project/paper/{task_project_slug(task)}/{token}"
            markdown_ref = self.artifacts.write_agent_bytes(
                self.name,
                f"{base}/manuscript.md",
                markdown,
                max_bytes=MAX_MANUSCRIPT_BYTES,
                kind="paper_markdown",
            )
            latex_ref = self.artifacts.write_agent_bytes(
                self.name,
                f"{base}/manuscript.tex",
                latex,
                max_bytes=MAX_MANUSCRIPT_BYTES,
                kind="paper_latex",
            )
            bibliography_ref = self.artifacts.write_agent_bytes(
                self.name,
                f"{base}/references.bib",
                bibliography,
                max_bytes=MAX_MANUSCRIPT_BYTES,
                kind="paper_bibliography",
            )
            manifest = PaperManuscriptManifest(
                run_id=task.run_id,
                task_id=task.task_id,
                project_slug=task_project_slug(task),
                title=draft.title,
                abstract=draft.abstract,
                keywords=draft.keywords,
                paper_manifest_uri=inputs.papers_ref.uri,
                research_design_uri=inputs.design_ref.uri,
                dataset_manifest_uri=inputs.data_ref.uri,
                code_manifest_uri=inputs.code_ref.uri,
                code_review_uri=inputs.review_ref.uri,
                main_schedule_uri=inputs.schedule_ref.uri,
                main_result_report_uri=inputs.main_result_ref.uri,
                experiment_run_manifest_uri=inputs.experiment_run_ref.uri,
                experiment_result_report_uri=inputs.experiment_result_ref.uri,
                citations=citations,
                closest_prior_comparisons=inputs.design.paper_comparisons,
                sections=draft.sections,
                formulas=formulas,
                claims=draft.claims,
                acknowledgements=draft.acknowledgements,
                experiment_results=results,
                markdown_artifact=markdown_ref,
                latex_artifact=latex_ref,
                bibliography_artifact=bibliography_ref,
            )
            manifest_ref = self.artifacts.write_agent_json(
                self.name,
                f"{base}/manifest.json",
                manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manuscript_manifest"})
            self._log_phase(
                task,
                event="paper_rendering_completed",
                status="succeeded",
                duration_ms=round((perf_counter() - render_started) * 1000, 3),
                details={
                    "manifest_uri": manifest_ref.uri,
                    "markdown_sha256": markdown_ref.sha256,
                    "latex_sha256": latex_ref.sha256,
                    "bibliography_sha256": bibliography_ref.sha256,
                },
            )
        except (ValidationError, PaperEvidenceError, StoragePolicyError, ValueError) as exc:
            return self._failure(
                task,
                issue_code="PAPER_WRITE_EVIDENCE_INVALID",
                reason=f"Paper writing failed closed: {type(exc).__name__}: {exc}",
                route_hint=AgentName.WRITE_PAPER,
            )
        except Exception as exc:
            return self._failure(
                task,
                issue_code="PAPER_WRITE_MODEL_FAILED",
                reason=f"Paper synthesis failed closed after {type(exc).__name__}.",
                route_hint=AgentName.WRITE_PAPER,
            )

        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded",
            summary=(
                f"Grounded manuscript written with {len(citations)} verified local citations, "
                f"{len(formulas)} code-bound formulas, and {len(results)} deterministic result rows."
            ),
            artifacts=[manifest_ref, markdown_ref, latex_ref, bibliography_ref],
            evidence=[
                {
                    "verified_local_pdf_count": len(citations),
                    "ast_bound_formula_count": len(formulas),
                    "observed_result_row_count": len(results),
                    "numeric_result_source": "persisted server observations only",
                    "model_authored_numeric_results": False,
                }
            ],
        )
