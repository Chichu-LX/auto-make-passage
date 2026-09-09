from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from time import perf_counter
from typing import Any

from pydantic import ValidationError
from pypdf import PdfReader

from agents.write_paper_agent.rendering import render_bibtex, render_latex, render_markdown
from research_agents.agent_package import load_agent_package
from research_agents.paper_evidence import (
    PaperEvidenceError,
    _inferred_component_ids_for_path,
    build_citation_evidence,
    build_observed_experiment_results,
    extract_code_symbol_catalog,
    resolve_formula_evidence,
    result_evidence_equal,
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
    FormulaCodeReferenceDraft,
    PaperClaimAudit,
    PaperCodeConsistencyReport,
    PaperConsistencyAdvisoryDraft,
    PaperConsistencyAdvisoryFinding,
    PaperConsistencyCheck,
    PaperFormulaDraft,
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
    MAX_ADVISORY_CONTEXT_CHARS,
    MAX_INPUT_BYTES,
    MAX_MANUSCRIPT_BYTES,
    MAX_PDF_EXCERPT_CHARS,
    MAX_REPORT_BYTES,
    file_token,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.CHECK_PAPER_AND_CODE:
    raise RuntimeError("Check Paper and Code Agent package name mismatch")


@dataclass(frozen=True)
class _ResolvedInputs:
    manuscript: PaperManuscriptManifest
    manuscript_ref: ArtifactRef
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


class CheckPaperAndCodeAgent:
    """Independently rebuild and compare the manuscript evidence graph."""

    name = AgentName.CHECK_PAPER_AND_CODE
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
                workflow_state=WorkflowState.CONSISTENCY_REVIEW,
                state_version=max(1, int(task.context.get("state_version", 1))),
                task_id=task.task_id,
                attempt=task.attempt,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def _read(self, artifact: ArtifactRef, model_type):
        return model_type.model_validate(
            self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
        )

    @staticmethod
    def _unique_ref(
        by_uri: dict[str, list[ArtifactRef]], uri: str, kind: str
    ) -> ArtifactRef | None:
        values = [item for item in by_uri.get(uri, []) if item.kind == kind]
        if not values or len({item.sha256 for item in values}) != 1:
            return None
        return values[-1]

    def _resolve_inputs(self, task: TaskEnvelope) -> _ResolvedInputs | None:
        slug = task_project_slug(task)
        by_uri: dict[str, list[ArtifactRef]] = {}
        for artifact in task.inputs:
            by_uri.setdefault(artifact.uri, []).append(artifact)
        manuscript_candidates: list[tuple[PaperManuscriptManifest, ArtifactRef]] = []
        for artifact in task.inputs:
            if artifact.kind != "paper_manuscript_manifest" or not artifact.uri.startswith(
                f"artifact://project/paper/{slug}/"
            ):
                continue
            try:
                value = self._read(artifact, PaperManuscriptManifest)
                if value.run_id == task.run_id and value.project_slug == slug:
                    manuscript_candidates.append((value, artifact))
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        for manuscript, manuscript_ref in reversed(manuscript_candidates):
            specs = (
                ("papers", manuscript.paper_manifest_uri, "paper_manifest", PaperManifest),
                ("design", manuscript.research_design_uri, "research_design", ResearchDesign),
                ("data", manuscript.dataset_manifest_uri, "data_manifest", DatasetManifest),
                ("code", manuscript.code_manifest_uri, "code_implementation_manifest", CodeImplementationManifest),
                ("review", manuscript.code_review_uri, "code_review_report", CodeReviewReport),
                ("schedule", manuscript.main_schedule_uri, "server_schedule_manifest", ServerScheduleManifest),
                ("main_result", manuscript.main_result_report_uri, "result_check_report", ResultCheckReport),
                ("experiment_run", manuscript.experiment_run_manifest_uri, "experiment_code_run_manifest", ExperimentCodeRunManifest),
                ("experiment_result", manuscript.experiment_result_report_uri, "experiment_code_run_report", ExperimentCodeRunReport),
            )
            loaded: dict[str, tuple[Any, ArtifactRef]] = {}
            try:
                for name, uri, kind, model_type in specs:
                    artifact = self._unique_ref(by_uri, uri, kind)
                    if artifact is None:
                        raise PaperEvidenceError("manuscript source reference is absent or ambiguous")
                    loaded[name] = (self._read(artifact, model_type), artifact)
            except (StoragePolicyError, ValidationError, ValueError, PaperEvidenceError):
                continue
            papers, papers_ref = loaded["papers"]
            design, design_ref = loaded["design"]
            data, data_ref = loaded["data"]
            code, code_ref = loaded["code"]
            review, review_ref = loaded["review"]
            schedule, schedule_ref = loaded["schedule"]
            main_result, main_result_ref = loaded["main_result"]
            experiment_run, experiment_run_ref = loaded["experiment_run"]
            experiment_result, experiment_result_ref = loaded["experiment_result"]
            if not (
                papers.run_id == task.run_id
                and design.run_id == task.run_id
                and data.run_id == task.run_id
                and data.project_slug == slug
                and code.run_id == task.run_id
                and code.project_slug == slug
                and review.run_id == task.run_id
                and review.project_slug == slug
                and review.verdict == "passed"
                and schedule.run_id == task.run_id
                and schedule.project_slug == slug
                and schedule.status == "launched"
                and main_result.run_id == task.run_id
                and main_result.project_slug == slug
                and main_result.verdict == "passed"
                and experiment_run.run_id == task.run_id
                and experiment_run.project_slug == slug
                and experiment_run.status == "launched"
                and experiment_result.run_id == task.run_id
                and experiment_result.project_slug == slug
                and experiment_result.verdict == "passed"
            ):
                continue
            if not (
                code.research_design_uri == design_ref.uri
                and code.data_manifest_uri == data_ref.uri
                and review.code_manifest_uri == code_ref.uri
                and schedule.research_design_uri == design_ref.uri
                and schedule.data_manifest_uri == data_ref.uri
                and schedule.code_manifest_uri == code_ref.uri
                and schedule.code_review_uri == review_ref.uri
                and main_result.schedule_manifest_uri == schedule_ref.uri
                and experiment_run.research_design_uri == design_ref.uri
                and experiment_run.code_manifest_uri == code_ref.uri
                and experiment_run.code_review_uri == review_ref.uri
                and experiment_run.main_schedule_manifest_uri == schedule_ref.uri
                and experiment_run.main_result_report_uri == main_result_ref.uri
                and experiment_result.run_manifest_uri == experiment_run_ref.uri
            ):
                continue
            expected_prefixes = (
                (papers_ref.uri, artifact_prefix("reference", slug, "manifests")),
                (design_ref.uri, artifact_prefix("design", slug, "research-designs")),
                (data_ref.uri, artifact_prefix("data", slug, "manifests")),
                (code_ref.uri, artifact_prefix("code", slug, "manifests")),
                (review_ref.uri, artifact_prefix("design", slug, "reviews", "code")),
                (schedule_ref.uri, f"artifact://project/var/runs/{task.run_id}/server_schedule/"),
                (main_result_ref.uri, artifact_prefix("design", slug, "reviews", "results")),
                (experiment_run_ref.uri, f"artifact://project/var/runs/{task.run_id}/experiment_code_run/"),
                (experiment_result_ref.uri, artifact_prefix("design", slug, "experiments", "results")),
            )
            if any(not uri.startswith(prefix) for uri, prefix in expected_prefixes):
                continue
            return _ResolvedInputs(
                manuscript=manuscript,
                manuscript_ref=manuscript_ref,
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
            )
        return None

    def _write_gap(self, task: TaskEnvelope, reason: str, issue_code: str) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            (
                f"project/reviews/consistency/{task_project_slug(task)}/gaps/"
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
        ).model_copy(update={"kind": "paper_code_consistency_gap"})

    @staticmethod
    def _check(
        check_id: str,
        category: str,
        passed: bool,
        summary: str,
        evidence: list[str] | None = None,
        *,
        warning: bool = False,
    ) -> PaperConsistencyCheck:
        return PaperConsistencyCheck(
            check_id=check_id,
            category=category,
            status="warning" if warning else ("passed" if passed else "failed"),
            summary=summary,
            evidence=evidence or [],
        )

    @staticmethod
    def _manifest_draft(manifest: PaperManuscriptManifest) -> PaperWritingDraft:
        formulas = [
            PaperFormulaDraft(
                formula_id=item.formula_id,
                label=item.label,
                latex=item.latex,
                explanation=item.explanation,
                design_component_ids=item.design_component_ids,
                code_references=[
                    FormulaCodeReferenceDraft(
                        file_path=binding.file_path,
                        symbol=binding.symbol,
                    )
                    for binding in item.code_bindings
                ],
                citation_ids=item.citation_ids,
            )
            for item in manifest.formulas
        ]
        return PaperWritingDraft(
            title=manifest.title,
            abstract=manifest.abstract,
            keywords=manifest.keywords,
            sections=manifest.sections,
            formulas=formulas,
            claims=manifest.claims,
            acknowledgements=manifest.acknowledgements,
        )

    @staticmethod
    def _claim_audits(
        manuscript: PaperManuscriptManifest,
        design: ResearchDesign,
        code: CodeImplementationManifest,
    ) -> list[PaperClaimAudit]:
        citations = {item.paper_id for item in manuscript.citations}
        components = {item.component_id for item in design.method_components}
        results = {item.result_key for item in manuscript.experiment_results}
        files = {item.path: item for item in code.files}
        audits: list[PaperClaimAudit] = []
        for claim in manuscript.claims:
            failures: list[str] = []
            if not set(claim.citation_ids).issubset(citations):
                failures.append("citation evidence escaped the verified PDF set")
            if not set(claim.design_component_ids).issubset(components):
                failures.append("design component evidence is unknown")
            if not set(claim.result_keys).issubset(results):
                failures.append("experiment evidence is unknown")
            for path in claim.code_file_paths:
                record = files.get(path)
                if record is None:
                    failures.append(f"reviewed code file is missing: {path}")
                else:
                    file_components = set(record.design_component_ids)
                    if not file_components:
                        file_components.update(
                            _inferred_component_ids_for_path(path, design)
                        )
                    if not set(claim.design_component_ids).issubset(file_components):
                        failures.append(
                            f"code file lacks component traceability: {path}"
                        )
            if claim.claim_type == "literature" and not claim.citation_ids:
                failures.append("literature claim has no verified citation")
            if claim.claim_type == "method" and (
                not claim.design_component_ids or not claim.code_file_paths
            ):
                failures.append("method claim lacks design or code evidence")
            if claim.claim_type == "experiment" and not claim.result_keys:
                failures.append("experiment claim lacks observed result evidence")
            if claim.claim_type == "limitation" and claim.strength != "descriptive":
                failures.append("limitation claim is not descriptive")
            audits.append(
                PaperClaimAudit(
                    claim_id=claim.claim_id,
                    status="failed" if failures else "passed",
                    citation_ids=claim.citation_ids,
                    code_file_paths=claim.code_file_paths,
                    result_keys=claim.result_keys,
                    summary=(
                        "; ".join(failures)
                        if failures
                        else "Claim references resolve to the independently verified evidence graph."
                    ),
                )
            )
        return audits

    def _pdf_context(
        self, inputs: _ResolvedInputs
    ) -> tuple[list[dict[str, str]], list[str]]:
        records = {item.paper_id: item for item in inputs.papers.records}
        values: list[dict[str, str]] = []
        warnings: list[str] = []
        for citation in inputs.manuscript.citations:
            record = records[citation.paper_id]
            text = ""
            source = "local_pdf_text"
            try:
                payload = self.artifacts.read_artifact_bytes(
                    citation.local_pdf, max_bytes=MAX_INPUT_BYTES
                )
                reader = PdfReader(BytesIO(payload), strict=False)
                chunks: list[str] = []
                for page in reader.pages[:5]:
                    chunks.append(page.extract_text() or "")
                    if sum(len(item) for item in chunks) >= MAX_PDF_EXCERPT_CHARS:
                        break
                text = "\n".join(chunks).strip()[:MAX_PDF_EXCERPT_CHARS]
            except Exception:
                text = ""
            if not text:
                text = record.abstract.strip()[:MAX_PDF_EXCERPT_CHARS]
                source = "manifest_abstract_fallback"
                warnings.append(
                    f"{citation.paper_id} had no extractable PDF text; advisory used its manifest abstract."
                )
            values.append(
                {
                    "paper_id": citation.paper_id,
                    "title": citation.title,
                    "source": source,
                    "excerpt": text or "No extractable text or abstract was available.",
                }
            )
        return values, warnings

    def _advisory(
        self,
        task: TaskEnvelope,
        inputs: _ResolvedInputs,
        catalog,
        sources: dict[str, str],
        pdf_context: list[dict[str, str]],
    ) -> tuple[PaperConsistencyAdvisoryDraft, bool]:
        symbol_map = {(item.file_path, item.symbol): item for item in catalog}
        formula_context = []
        for formula in inputs.manuscript.formulas:
            bindings = []
            for binding in formula.code_bindings:
                current = symbol_map.get((binding.file_path, binding.symbol))
                if current is None:
                    continue
                bindings.append(
                    {
                        **current.model_dump(mode="json"),
                        "source_excerpt": source_excerpt(
                            sources[current.file_path], current.line_start, radius=8
                        ),
                    }
                )
            formula_context.append(
                {
                    "formula_id": formula.formula_id,
                    "latex": formula.latex,
                    "explanation": formula.explanation,
                    "bindings": bindings,
                }
            )
        context = {
            "objective": task.objective,
            "original_paper_evidence": pdf_context,
            "approved_claim_boundaries": inputs.design.claim_boundaries,
            "approved_limitations": inputs.design.limitations,
            "claim_boundary_precedence": {
                "rule": (
                    "A design-stage statement that no experiment had yet run is a "
                    "temporal planning constraint. It is superseded only for exact rows "
                    "independently reconstructed from the passed post-design main and "
                    "supplementary result reports. All other design limitations remain active."
                ),
                "post_design_main_result_verdict": inputs.main_result.verdict,
                "post_design_supplementary_result_verdict": inputs.experiment_result.verdict,
                "admissible_observed_result_keys": [
                    item.result_key for item in inputs.manuscript.experiment_results
                ],
                "forbidden_inferences": [
                    "full neural baseline reproduction",
                    "unobserved design metrics",
                    "state-of-the-art performance",
                    "publication readiness",
                    "SSH or GPU execution",
                ],
            },
            "paper_comparisons": [
                item.model_dump(mode="json") for item in inputs.design.paper_comparisons
            ],
            "manuscript": {
                "title": inputs.manuscript.title,
                "abstract": inputs.manuscript.abstract,
                "sections": [item.model_dump(mode="json") for item in inputs.manuscript.sections],
                "claims": [item.model_dump(mode="json") for item in inputs.manuscript.claims],
                "formulas": formula_context,
                "observed_results": [
                    item.model_dump(mode="json")
                    for item in inputs.manuscript.experiment_results
                ],
            },
            "allowed_ids": {
                "claim_ids": [item.claim_id for item in inputs.manuscript.claims],
                "formula_ids": [item.formula_id for item in inputs.manuscript.formulas],
                "citation_ids": [item.paper_id for item in inputs.manuscript.citations],
            },
            "response_schema": PaperConsistencyAdvisoryDraft.model_json_schema(),
        }
        prompt = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(prompt) > MAX_ADVISORY_CONTEXT_CHARS:
            raise PaperEvidenceError("bounded consistency advisory context exceeds policy")
        try:
            response = self.model.complete_json(PACKAGE.system_prompt, prompt)
            advisory = PaperConsistencyAdvisoryDraft.model_validate(response)
            claim_ids = {item.claim_id for item in inputs.manuscript.claims}
            formula_ids = {item.formula_id for item in inputs.manuscript.formulas}
            citation_ids = {item.paper_id for item in inputs.manuscript.citations}
            for finding in advisory.findings:
                if not set(finding.claim_ids).issubset(claim_ids):
                    raise PaperEvidenceError("advisory references an unknown claim")
                if not set(finding.formula_ids).issubset(formula_ids):
                    raise PaperEvidenceError("advisory references an unknown formula")
                if not set(finding.citation_ids).issubset(citation_ids):
                    raise PaperEvidenceError("advisory references an unknown citation")
            return advisory, False
        except Exception as exc:
            return (
                PaperConsistencyAdvisoryDraft(
                    summary="The qualitative consistency advisory failed closed and must be rerun.",
                    findings=[
                        PaperConsistencyAdvisoryFinding(
                            finding_id="advisory-contract-unavailable",
                            category="claim_scope",
                            severity="error",
                            summary=(
                                "The qualitative advisory could not produce a validated bounded report."
                            ),
                            recommendation=(
                                f"Rerun this read-only consistency Agent; the bounded failure type was {type(exc).__name__}."
                            ),
                        )
                    ],
                ),
                True,
            )

    @staticmethod
    def _markdown(report: PaperCodeConsistencyReport) -> str:
        lines = [
            "# Paper and Code Consistency Report",
            "",
            f"- Verdict: `{report.verdict}`",
            f"- Manuscript: `{report.manuscript_manifest_uri}`",
            "",
            report.summary,
            "",
            "## Deterministic checks",
            "",
            "| Check | Category | Status | Summary |",
            "|---|---|---|---|",
        ]
        for item in report.checks:
            lines.append(
                f"| `{item.check_id}` | {item.category} | `{item.status}` | "
                f"{item.summary.replace('|', '\\|').replace(chr(10), ' ')} |"
            )
        lines.extend(["", "## Claim audits", ""])
        for item in report.claim_audits:
            lines.append(f"- `{item.claim_id}` — `{item.status}`: {item.summary}")
        lines.extend(["", "## Qualitative advisory", "", report.advisory.summary, ""])
        for item in report.advisory.findings:
            lines.append(
                f"- `{item.severity}` `{item.category}` `{item.finding_id}`: "
                f"{item.summary} Recommendation: {item.recommendation}"
            )
        if report.blockers:
            lines.extend(["", "## Blockers", ""])
            lines.extend(f"- {item}" for item in report.blockers)
        if report.recommendations:
            lines.extend(["", "## Repair instructions", ""])
            lines.extend(f"- {item}" for item in report.recommendations)
        return "\n".join(lines) + "\n"

    def _blocked(self, task: TaskEnvelope, reason: str) -> AgentResult:
        safe_reason = redact_text(reason, max_chars=2000)
        gap = self._write_gap(task, safe_reason, "PAPER_CONSISTENCY_INPUT_MISSING")
        self._log_phase(
            task,
            event="paper_consistency_failed",
            status="failed",
            details={"issue_code": "PAPER_CONSISTENCY_INPUT_MISSING"},
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
                    "code": "PAPER_CONSISTENCY_INPUT_MISSING",
                    "message": safe_reason,
                    "root_cause_key": "G10:PAPER_CONSISTENCY_INPUT_MISSING:paper",
                }
            ],
            route_hint=AgentName.WRITE_PAPER,
        )

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        resolve_started = perf_counter()
        self._log_phase(task, event="paper_consistency_resolution_started", status="started")
        inputs = self._resolve_inputs(task)
        if inputs is None:
            return self._blocked(
                task,
                "A content-addressed manuscript and its complete, matching upstream evidence chain are required.",
            )
        self._log_phase(
            task,
            event="paper_consistency_resolution_completed",
            status="succeeded",
            duration_ms=round((perf_counter() - resolve_started) * 1000, 3),
            details={"manuscript_uri": inputs.manuscript_ref.uri},
        )

        audit_started = perf_counter()
        self._log_phase(task, event="paper_consistency_deterministic_audit_started", status="started")
        checks: list[PaperConsistencyCheck] = []
        pdf_warnings: list[str] = []
        try:
            manuscript = inputs.manuscript
            markdown = self.artifacts.read_artifact_bytes(
                manuscript.markdown_artifact, max_bytes=MAX_MANUSCRIPT_BYTES
            )
            latex = self.artifacts.read_artifact_bytes(
                manuscript.latex_artifact, max_bytes=MAX_MANUSCRIPT_BYTES
            )
            bibliography = self.artifacts.read_artifact_bytes(
                manuscript.bibliography_artifact, max_bytes=MAX_MANUSCRIPT_BYTES
            )
            render_args = {
                "title": manuscript.title,
                "abstract": manuscript.abstract,
                "keywords": manuscript.keywords,
                "sections": manuscript.sections,
                "formulas": manuscript.formulas,
                "claims": manuscript.claims,
                "citations": manuscript.citations,
                "results": manuscript.experiment_results,
                "acknowledgements": manuscript.acknowledgements,
            }
            expected_markdown = render_markdown(**render_args).encode("utf-8")
            expected_latex = render_latex(**render_args).encode("utf-8")
            expected_bibliography = render_bibtex(manuscript.citations).encode("utf-8")
            integrity_passed = (
                markdown == expected_markdown
                and latex == expected_latex
                and bibliography == expected_bibliography
            )
            checks.append(
                self._check(
                    "paper-check-manuscript-integrity",
                    "manuscript_integrity",
                    integrity_passed,
                    (
                        "All manuscript files match their hashes and deterministic render."
                        if integrity_passed
                        else "A manuscript file differs from its content-addressed deterministic render."
                    ),
                    [
                        manuscript.markdown_artifact.sha256,
                        manuscript.latex_artifact.sha256,
                        manuscript.bibliography_artifact.sha256,
                    ],
                )
            )

            rebuilt_citations = build_citation_evidence(self.artifacts, inputs.papers)
            citation_passed = [item.model_dump(mode="json") for item in rebuilt_citations] == [
                item.model_dump(mode="json") for item in manuscript.citations
            ] and all(
                item.local_pdf.uri.startswith(
                    artifact_prefix("reference", task_project_slug(task), "papers")
                )
                for item in rebuilt_citations
            )
            checks.append(
                self._check(
                    "paper-check-citation-pdf",
                    "citation_pdf",
                    citation_passed,
                    (
                        "Every citation resolves to the independently verified local PDF set."
                        if citation_passed
                        else "The manuscript citation set differs from verified local PDF evidence."
                    ),
                    [item.local_pdf.uri for item in rebuilt_citations],
                )
            )

            related = next(
                item for item in manuscript.sections if item.section_id == "related_work"
            )
            comparison_ids = {item.paper_id for item in inputs.design.paper_comparisons}
            original_comparison_passed = comparison_ids.issubset(
                {item.paper_id for item in manuscript.citations}
            ) and comparison_ids.issubset(set(related.citation_ids))
            checks.append(
                self._check(
                    "paper-check-original-comparisons",
                    "original_paper_comparison",
                    original_comparison_passed,
                    (
                        "Related Work covers every design-declared original-paper comparison."
                        if original_comparison_passed
                        else "Related Work omits at least one required original-paper comparison."
                    ),
                    sorted(comparison_ids),
                )
            )

            catalog, sources = extract_code_symbol_catalog(
                self.artifacts, inputs.code, inputs.design
            )
            symbol_map = {(item.file_path, item.symbol): item for item in catalog}
            formula_failures: list[str] = []
            for formula in manuscript.formulas:
                for binding in formula.code_bindings:
                    current = symbol_map.get((binding.file_path, binding.symbol))
                    if current is None or current.model_dump(mode="json") != binding.model_dump(
                        mode="json"
                    ):
                        formula_failures.append(
                            f"{formula.formula_id}:{binding.file_path}:{binding.symbol}"
                        )
                bound_components = {
                    value
                    for binding in formula.code_bindings
                    for value in binding.design_component_ids
                }
                if not set(formula.design_component_ids).issubset(bound_components):
                    formula_failures.append(f"{formula.formula_id}:component-traceability")
            checks.append(
                self._check(
                    "paper-check-formula-code",
                    "formula_code",
                    not formula_failures,
                    (
                        "Every formula binding matches a current reviewed AST symbol and file hash."
                        if not formula_failures
                        else "At least one formula binding no longer matches reviewed source code."
                    ),
                    formula_failures or [f"{item.file_path}:{item.symbol}" for item in catalog[:50]],
                )
            )

            files = {item.path: item for item in inputs.code.files}
            method_failures: list[str] = []
            for claim in manuscript.claims:
                if claim.claim_type != "method":
                    continue
                for path in claim.code_file_paths:
                    record = files.get(path)
                    if record is None:
                        method_failures.append(f"{claim.claim_id}:{path}")
                        continue
                    file_components = set(record.design_component_ids)
                    if not file_components:
                        file_components.update(
                            _inferred_component_ids_for_path(path, inputs.design)
                        )
                    if not set(claim.design_component_ids).issubset(file_components):
                        method_failures.append(f"{claim.claim_id}:{path}")
            checks.append(
                self._check(
                    "paper-check-method-code",
                    "method_code",
                    not method_failures,
                    (
                        "Every method claim is traceable to reviewed files and design components."
                        if not method_failures
                        else "At least one method claim lacks reviewed file-level traceability."
                    ),
                    method_failures,
                )
            )

            rebuilt_results = build_observed_experiment_results(
                inputs.schedule,
                inputs.main_result,
                inputs.experiment_result,
                main_report_uri=inputs.main_result_ref.uri,
                experiment_report_uri=inputs.experiment_result_ref.uri,
            )
            result_passed = result_evidence_equal(
                rebuilt_results, manuscript.experiment_results
            )
            checks.append(
                self._check(
                    "paper-check-experiment-results",
                    "experiment_result",
                    result_passed,
                    (
                        "Every manuscript result row exactly matches persisted server evidence."
                        if result_passed
                        else "The manuscript result evidence differs from independently rebuilt server observations."
                    ),
                    [item.result_key for item in rebuilt_results],
                )
            )

            claim_audits = self._claim_audits(manuscript, inputs.design, inputs.code)
            claim_passed = all(item.status != "failed" for item in claim_audits)
            checks.append(
                self._check(
                    "paper-check-claim-evidence",
                    "claim_evidence",
                    claim_passed,
                    (
                        "Every claim has type-appropriate citation, code, design, or result evidence."
                        if claim_passed
                        else "At least one manuscript claim has missing or mismatched evidence."
                    ),
                    [item.claim_id for item in claim_audits if item.status == "failed"],
                )
            )

            draft = self._manifest_draft(manuscript)
            independently_resolved = resolve_formula_evidence(
                draft.formulas,
                catalog,
                inputs.design,
                {item.paper_id for item in rebuilt_citations},
            )
            validate_writing_draft(
                draft,
                design=inputs.design,
                code=inputs.code,
                citations=rebuilt_citations,
                results=rebuilt_results,
                formulas=independently_resolved,
            )
            scope_passed = [item.model_dump(mode="json") for item in independently_resolved] == [
                item.model_dump(mode="json") for item in manuscript.formulas
            ]
            checks.append(
                self._check(
                    "paper-check-claim-scope",
                    "claim_scope",
                    scope_passed,
                    (
                        "Paper structure, result language, formula graph, and claim scope satisfy policy."
                        if scope_passed
                        else "The independently resolved manuscript graph differs from stored formula evidence."
                    ),
                )
            )
            pdf_context, pdf_warnings = self._pdf_context(inputs)
            if pdf_warnings:
                checks.append(
                    self._check(
                        "paper-check-pdf-text-extraction",
                        "original_paper_comparison",
                        True,
                        "Some local PDFs had no extractable text; qualitative review used manifest abstracts.",
                        pdf_warnings,
                        warning=True,
                    )
                )
        except Exception as exc:
            safe = redact_text(f"{type(exc).__name__}: {exc}", max_chars=1000)
            claim_audits = [
                PaperClaimAudit(
                    claim_id=item.claim_id,
                    status="failed",
                    citation_ids=item.citation_ids,
                    code_file_paths=item.code_file_paths,
                    result_keys=item.result_keys,
                    summary="Independent evidence reconstruction failed before this claim could pass.",
                )
                for item in inputs.manuscript.claims
            ]
            checks = [
                self._check(
                    "paper-check-manuscript-integrity",
                    "manuscript_integrity",
                    False,
                    "Independent deterministic reconstruction failed closed.",
                    [safe],
                ),
                self._check("paper-check-citation-pdf", "citation_pdf", False, "Citation PDF verification did not complete."),
                self._check("paper-check-original-comparisons", "original_paper_comparison", False, "Original-paper comparison verification did not complete."),
                self._check("paper-check-formula-code", "formula_code", False, "Formula and code verification did not complete."),
                self._check("paper-check-method-code", "method_code", False, "Method and code verification did not complete."),
                self._check("paper-check-experiment-results", "experiment_result", False, "Experiment result verification did not complete."),
                self._check("paper-check-claim-evidence", "claim_evidence", False, "Claim evidence verification did not complete."),
            ]
            catalog, sources, pdf_context = [], {}, []
        self._log_phase(
            task,
            event="paper_consistency_deterministic_audit_completed",
            status="succeeded" if all(item.status != "failed" for item in checks) else "failed",
            duration_ms=round((perf_counter() - audit_started) * 1000, 3),
            details={
                "check_count": len(checks),
                "failed_check_count": sum(item.status == "failed" for item in checks),
                "claim_audit_count": len(claim_audits),
            },
        )

        advisory_started = perf_counter()
        self._log_phase(task, event="paper_consistency_advisory_started", status="started")
        advisory, advisory_unavailable = self._advisory(
            task, inputs, catalog, sources, pdf_context
        )
        self._log_phase(
            task,
            event="paper_consistency_advisory_completed",
            status="failed" if advisory_unavailable else "succeeded",
            duration_ms=round((perf_counter() - advisory_started) * 1000, 3),
            details={
                "finding_count": len(advisory.findings),
                "error_count": sum(item.severity == "error" for item in advisory.findings),
            },
        )

        failed_checks = [item for item in checks if item.status == "failed"]
        failed_claims = [item for item in claim_audits if item.status == "failed"]
        advisory_errors = [item for item in advisory.findings if item.severity == "error"]
        failed = bool(failed_checks or failed_claims or advisory_errors)
        blockers = [item.summary for item in failed_checks]
        blockers.extend(item.summary for item in failed_claims)
        blockers.extend(item.summary for item in advisory_errors)
        blockers = list(dict.fromkeys(blockers))
        recommendations = [
            "Regenerate the manuscript from the unchanged verified sources and resolve every failed deterministic check."
        ] if failed_checks or failed_claims else []
        recommendations.extend(item.recommendation for item in advisory_errors)
        recommendations = list(dict.fromkeys(recommendations))
        route = (
            AgentName.CHECK_PAPER_AND_CODE
            if failed and advisory_unavailable and not failed_checks and not failed_claims
            else (AgentName.WRITE_PAPER if failed else None)
        )
        report = PaperCodeConsistencyReport(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            manuscript_manifest_uri=inputs.manuscript_ref.uri,
            verdict="needs_revision" if failed else "passed",
            summary=(
                f"Paper/code consistency audit completed with {len(failed_checks)} failed checks, "
                f"{len(failed_claims)} failed claims, and {len(advisory_errors)} advisory errors."
            ),
            checks=checks,
            claim_audits=claim_audits,
            advisory=advisory,
            blockers=blockers,
            recommendations=recommendations,
            recommended_route=route,
        )
        token = file_token(task.task_id)
        base = f"project/reviews/consistency/{task_project_slug(task)}/{token}"
        json_ref = self.artifacts.write_agent_json(
            self.name, f"{base}.json", report.model_dump(mode="json")
        ).model_copy(update={"kind": "paper_code_consistency_report"})
        markdown_ref = self.artifacts.write_agent_bytes(
            self.name,
            f"{base}.md",
            self._markdown(report).encode("utf-8"),
            max_bytes=MAX_REPORT_BYTES,
            kind="paper_code_consistency_markdown",
        )
        self._log_phase(
            task,
            event="paper_consistency_report_written",
            status="failed" if failed else "succeeded",
            details={
                "verdict": report.verdict,
                "report_uri": json_ref.uri,
                "recommended_route": route.value if route else None,
            },
        )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="failed" if failed else "succeeded",
            summary=report.summary,
            artifacts=[json_ref, markdown_ref],
            evidence=[
                {
                    "deterministic_check_count": len(checks),
                    "claim_audit_count": len(claim_audits),
                    "advisory_finding_count": len(advisory.findings),
                    "verified_pdf_count": len(inputs.manuscript.citations),
                    "verified_formula_count": len(inputs.manuscript.formulas),
                    "verified_result_row_count": len(inputs.manuscript.experiment_results),
                }
            ],
            issues=[
                {
                    "severity": "blocking",
                    "code": "PAPER_CODE_CONSISTENCY_FAILED",
                    "message": value,
                    "root_cause_key": f"G10:PAPER_CODE_CONSISTENCY_FAILED:{index}",
                }
                for index, value in enumerate(blockers, start=1)
            ],
            route_hint=route,
            needs_human=False,
        )
