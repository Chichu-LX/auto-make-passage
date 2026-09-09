from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import math
import re
from typing import Iterable

from .schemas import (
    ArtifactRef,
    CodeImplementationManifest,
    CodeSymbolEvidence,
    ExperimentCodeRunReport,
    ObservedExperimentResult,
    ObservedSeedResult,
    PaperCitationEvidence,
    PaperFormulaDraft,
    PaperFormulaEvidence,
    PaperManifest,
    PaperWritingDraft,
    ResearchDesign,
    ResultCheckReport,
    ServerScheduleManifest,
)
from .storage import ArtifactStore, StoragePolicyError


MAX_CODE_FILE_BYTES = 512 * 1024
MAX_PDF_BYTES = 128 * 1024 * 1024
_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)


class PaperEvidenceError(ValueError):
    pass


def _result_key(scope: str, experiment_id: str, metric_name: str) -> str:
    digest = hashlib.sha256(
        f"{scope}:{experiment_id}:{metric_name}".encode("utf-8")
    ).hexdigest()[:12]
    return f"result-{'main' if scope == 'main' else 'supp'}-{digest}"


def _statistics(values: list[float]) -> tuple[float, float | None, float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise PaperEvidenceError("experiment result values must be non-empty and finite")
    mean = sum(values) / len(values)
    sample_std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else None
    )
    return mean, sample_std, min(values), max(values)


def build_observed_experiment_results(
    schedule: ServerScheduleManifest,
    main_report: ResultCheckReport,
    experiment_report: ExperimentCodeRunReport,
    *,
    main_report_uri: str,
    experiment_report_uri: str,
) -> list[ObservedExperimentResult]:
    """Build exact paper-facing result rows from server evidence, never model output."""

    if schedule.status != "launched" or main_report.verdict != "passed":
        raise PaperEvidenceError("paper writing requires a passed main result")
    if experiment_report.verdict != "passed":
        raise PaperEvidenceError("paper writing requires passed supplementary experiments")
    observations = {item.job_id: item for item in main_report.observations}
    if len(observations) != len(main_report.observations):
        raise PaperEvidenceError("main result observations contain duplicate jobs")
    if set(observations) != {item.job_id for item in schedule.jobs}:
        raise PaperEvidenceError("main result observations do not match the schedule")
    grouped: dict[tuple[str, str, str], list[ObservedSeedResult]] = defaultdict(list)
    for job in schedule.jobs:
        observation = observations[job.job_id]
        best = observation.best_metric
        if (
            observation.status != "completed"
            or best is None
            or best.metric_name != job.metric_name
            or not math.isfinite(best.value)
        ):
            raise PaperEvidenceError("main result lacks an exact finite best metric")
        grouped[(job.experiment_id, job.metric_name, job.metric_direction)].append(
            ObservedSeedResult(
                job_id=job.job_id,
                seed=job.seed,
                value=best.value,
                checkpoint_path=observation.best_checkpoint_path or best.checkpoint_path,
            )
        )
    targets = {
        item.metric_name: item.target_value
        for item in main_report.metric_checks
        if item.status == "passed"
    }
    results: list[ObservedExperimentResult] = []
    for (experiment_id, metric_name, direction), seed_results in sorted(grouped.items()):
        seed_results.sort(key=lambda item: item.seed)
        mean, sample_std, minimum, maximum = _statistics(
            [item.value for item in seed_results]
        )
        results.append(
            ObservedExperimentResult(
                result_key=_result_key("main", experiment_id, metric_name),
                scope="main",
                experiment_id=experiment_id,
                metric_name=metric_name,
                direction=direction,
                seed_results=seed_results,
                mean=mean,
                sample_standard_deviation=sample_std,
                minimum=minimum,
                maximum=maximum,
                target_value=targets.get(metric_name),
                source_report_uri=main_report_uri,
            )
        )
    for aggregate in sorted(
        experiment_report.primary_metric_aggregates,
        key=lambda item: (item.experiment_id, item.metric_name),
    ):
        results.append(
            ObservedExperimentResult(
                result_key=_result_key(
                    "supplementary", aggregate.experiment_id, aggregate.metric_name
                ),
                scope="supplementary",
                experiment_id=aggregate.experiment_id,
                metric_name=aggregate.metric_name,
                direction=aggregate.direction,
                seed_results=[
                    ObservedSeedResult(
                        job_id=item.job_id,
                        seed=item.seed,
                        value=item.value,
                        checkpoint_path=item.checkpoint_path,
                    )
                    for item in aggregate.seed_metrics
                ],
                mean=aggregate.mean,
                sample_standard_deviation=aggregate.sample_standard_deviation,
                minimum=aggregate.minimum,
                maximum=aggregate.maximum,
                source_report_uri=experiment_report_uri,
            )
        )
    if not results or len({item.result_key for item in results}) != len(results):
        raise PaperEvidenceError("paper result evidence is empty or has key collisions")
    return results


def _top_level_symbols(tree: ast.Module) -> Iterable[tuple[str, str, int]]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, "function", node.lineno
        elif isinstance(node, ast.AsyncFunctionDef):
            yield node.name, "async_function", node.lineno
        elif isinstance(node, ast.ClassDef):
            yield node.name, "class", node.lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    yield target.id, "constant", node.lineno


def _inferred_component_ids_for_path(
    path: str,
    design: ResearchDesign,
) -> list[str]:
    """Conservatively recover traceability for legacy full-tree revisions."""

    available = {item.component_id for item in design.method_components}
    lowered = path.casefold()
    candidates: set[str] = set()
    if "/data/adapters" in lowered or "/data/provenance" in lowered:
        candidates.add("component-data-ingestion")
    if "/data/split" in lowered:
        candidates.add("component-strict-split-manager")
    if "/models/" in lowered:
        candidates.add("component-affinity-regressor")
    if "dispersion" in lowered:
        candidates.add("component-dispersion-estimator")
    if "/calibration/conformal" in lowered:
        candidates.add("component-conformal-calibrator")
    if "/evaluation/" in lowered or "/testing/" in lowered:
        candidates.add("component-reliability-evaluator")
    if "case-study" in lowered or "case_study" in lowered:
        candidates.add("component-case-study-intervals")
    return sorted(candidates & available)


def extract_code_symbol_catalog(
    artifacts: ArtifactStore,
    code: CodeImplementationManifest,
    design: ResearchDesign | None = None,
) -> tuple[list[CodeSymbolEvidence], dict[str, str]]:
    """Parse reviewed generated Python without importing or executing it."""

    available_components = bool(design and design.method_components)

    catalog: list[CodeSymbolEvidence] = []
    sources: dict[str, str] = {}
    for record in code.files:
        if record.language != "python" or not record.path.endswith(".py"):
            continue
        payload = artifacts.read_artifact_bytes(
            record.artifact, max_bytes=MAX_CODE_FILE_BYTES
        )
        if len(payload) != record.size_bytes or record.artifact.sha256 != record.sha256:
            raise PaperEvidenceError("code file differs from its implementation manifest")
        try:
            source = payload.decode("utf-8")
            tree = ast.parse(source, filename=record.path)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise PaperEvidenceError("reviewed Python cannot be parsed for paper guidance") from exc
        sources[record.path] = source
        component_ids = list(record.design_component_ids)
        if not component_ids and available_components and design is not None:
            component_ids = _inferred_component_ids_for_path(record.path, design)
        for symbol, kind, line_start in _top_level_symbols(tree):
            catalog.append(
                CodeSymbolEvidence(
                    file_path=record.path,
                    file_sha256=record.sha256,
                    symbol=symbol,
                    symbol_kind=kind,
                    line_start=line_start,
                    design_component_ids=component_ids,
                    experiment_ids=record.experiment_ids,
                )
            )
    keys = [(item.file_path, item.symbol) for item in catalog]
    if not catalog or len(keys) != len(set(keys)):
        raise PaperEvidenceError("reviewed code symbol catalog is empty or ambiguous")
    return sorted(catalog, key=lambda item: (item.file_path, item.line_start)), sources


def build_citation_evidence(
    artifacts: ArtifactStore,
    manifest: PaperManifest,
) -> list[PaperCitationEvidence]:
    citations: list[PaperCitationEvidence] = []
    for record in manifest.records:
        if record.status != "downloaded" or record.local_artifact is None:
            continue
        payload = artifacts.read_artifact_bytes(
            record.local_artifact, max_bytes=MAX_PDF_BYTES
        )
        if not payload.startswith(b"%PDF-") or record.sha256 != record.local_artifact.sha256:
            raise StoragePolicyError("paper citation PDF failed signature or digest verification")
        citations.append(
            PaperCitationEvidence(
                paper_id=record.paper_id,
                title=record.title,
                authors=record.authors,
                year=record.year,
                doi=record.doi,
                venue=record.canonical_venue or record.venue or record.source,
                local_pdf=record.local_artifact,
            )
        )
    if not citations:
        raise PaperEvidenceError("paper writing requires at least one verified local PDF")
    return citations


def resolve_formula_evidence(
    formulas: list[PaperFormulaDraft],
    catalog: list[CodeSymbolEvidence],
    design: ResearchDesign,
    citation_ids: set[str],
) -> list[PaperFormulaEvidence]:
    symbol_map = {(item.file_path, item.symbol): item for item in catalog}
    component_ids = {item.component_id for item in design.method_components}
    resolved: list[PaperFormulaEvidence] = []
    for formula in formulas:
        if not set(formula.design_component_ids).issubset(component_ids):
            raise PaperEvidenceError("formula references an unknown design component")
        if not set(formula.citation_ids).issubset(citation_ids):
            raise PaperEvidenceError("formula cites a paper outside verified local evidence")
        bindings: list[CodeSymbolEvidence] = []
        for reference in formula.code_references:
            binding = symbol_map.get((reference.file_path, reference.symbol))
            if binding is None:
                raise PaperEvidenceError("formula references a missing reviewed code symbol")
            bindings.append(binding)
        bound_components = {
            component_id for item in bindings for component_id in item.design_component_ids
        }
        if not set(formula.design_component_ids).issubset(bound_components):
            raise PaperEvidenceError(
                "formula component IDs are not traceable through its bound code files"
            )
        resolved.append(
            PaperFormulaEvidence(
                formula_id=formula.formula_id,
                label=formula.label,
                latex=formula.latex,
                explanation=formula.explanation,
                design_component_ids=formula.design_component_ids,
                code_bindings=bindings,
                citation_ids=formula.citation_ids,
            )
        )
    return resolved


def validate_writing_draft(
    draft: PaperWritingDraft,
    *,
    design: ResearchDesign,
    code: CodeImplementationManifest,
    citations: list[PaperCitationEvidence],
    results: list[ObservedExperimentResult],
    formulas: list[PaperFormulaEvidence],
) -> None:
    citation_ids = {item.paper_id for item in citations}
    result_keys = {item.result_key for item in results}
    formula_ids = {item.formula_id for item in formulas}
    component_ids = {item.component_id for item in design.method_components}
    code_records = {item.path: item for item in code.files}
    all_section_citations = {
        citation for section in draft.sections for citation in section.citation_ids
    }
    for section in draft.sections:
        if not set(section.citation_ids).issubset(citation_ids):
            raise PaperEvidenceError("paper section cites a non-downloaded paper")
        if not set(section.formula_ids).issubset(formula_ids):
            raise PaperEvidenceError("paper section references an unresolved formula")
        if not set(section.result_keys).issubset(result_keys):
            raise PaperEvidenceError("paper section references an unobserved result")
    for claim in draft.claims:
        if not set(claim.citation_ids).issubset(citation_ids):
            raise PaperEvidenceError("paper claim cites a non-downloaded paper")
        if not set(claim.design_component_ids).issubset(component_ids):
            raise PaperEvidenceError("paper claim references an unknown method component")
        if not set(claim.result_keys).issubset(result_keys):
            raise PaperEvidenceError("paper claim references an unobserved result")
        for path in claim.code_file_paths:
            record = code_records.get(path)
            if record is None:
                raise PaperEvidenceError("paper method claim references an unknown code file")
            file_components = set(record.design_component_ids)
            if not file_components:
                file_components.update(_inferred_component_ids_for_path(path, design))
            if not set(claim.design_component_ids).issubset(file_components):
                raise PaperEvidenceError("paper method claim is not file-level traceable")
    related_work = next(item for item in draft.sections if item.section_id == "related_work")
    comparison_ids = {item.paper_id for item in design.paper_comparisons}
    if not comparison_ids.issubset(set(related_work.citation_ids)):
        raise PaperEvidenceError("related work omits a design-declared comparison paper")
    if not comparison_ids.issubset(citation_ids):
        raise PaperEvidenceError("a design comparison paper has no verified local PDF")
    if not all_section_citations:
        raise PaperEvidenceError("professional manuscript cannot omit citations")
    covered_components = {
        component
        for formula in formulas
        for component in formula.design_component_ids
    }
    covered_components.update(
        component
        for claim in draft.claims
        if claim.claim_type == "method"
        for component in claim.design_component_ids
    )
    if covered_components != component_ids:
        raise PaperEvidenceError("paper method/formulas do not cover every design component")
    results_section = next(item for item in draft.sections if item.section_id == "results")
    if set(results_section.result_keys) != result_keys:
        raise PaperEvidenceError("paper results section must include every exact result row")
    claim_result_keys = {
        key
        for claim in draft.claims
        if claim.claim_type == "experiment"
        for key in claim.result_keys
    }
    if claim_result_keys != result_keys:
        raise PaperEvidenceError("paper experiment claims must cover all and only observed results")
    protected_text = [draft.abstract]
    protected_text.extend(
        paragraph
        for section in draft.sections
        if section.section_id in {"results", "discussion", "conclusion"}
        for paragraph in section.paragraphs
    )
    protected_text.extend(
        claim.text for claim in draft.claims if claim.claim_type == "experiment"
    )
    if any(_NUMERIC_LITERAL.search(value) for value in protected_text):
        raise PaperEvidenceError(
            "model-authored result prose contains a numeric literal; exact numbers belong only in runtime-rendered tables"
        )


def source_excerpt(source: str, line_start: int, *, radius: int = 8) -> str:
    lines = source.splitlines()
    start = max(0, line_start - 1 - radius)
    end = min(len(lines), line_start + radius)
    return "\n".join(f"{index + 1}: {lines[index]}" for index in range(start, end))


def result_evidence_equal(
    left: list[ObservedExperimentResult], right: list[ObservedExperimentResult]
) -> bool:
    normalize = lambda items: sorted(
        (item.model_dump(mode="json") for item in items),
        key=lambda item: item["result_key"],
    )
    return normalize(left) == normalize(right)
