from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter

from agents.check_paper_and_code_agent import CheckPaperAndCodeAgent
from agents.experiment_code_run_agent import ExperimentCodeRunAgent
from agents.paper_review_agent import PaperReviewAgent
from agents.write_paper_agent import WritePaperAgent
from research_agents.schemas import (
    AgentName,
    DownloadPolicyRecord,
    ExperimentCodeRunReport,
    PaperCodeConsistencyReport,
    PaperFormulaDraft,
    PaperManifest,
    PaperManuscriptManifest,
    PaperRecord,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore

from test_code_pipeline import StaticModel
from test_experiment_code_run import (
    READY_COVERAGE,
    SeedMetricBackend,
    experiment_task,
    force_manifest_due,
    prepare_passed_main,
)


class PaperDraftModel:
    def __init__(self, *, bad_symbol: bool = False, numeric_results: bool = False):
        self.bad_symbol = bad_symbol
        self.numeric_results = numeric_results
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        import json

        self.calls.append((system_prompt, user_prompt))
        context = json.loads(user_prompt)
        citations = [item["paper_id"] for item in context["allowed_citations"]]
        result_keys = [item["result_key"] for item in context["observed_result_keys"]]
        components = [
            item["component_id"]
            for item in context["research_design"]["method_components"]
        ]
        catalog = context["reviewed_code"]["ast_symbol_catalog"]
        formulas = []
        method_claims = []
        formula_ids = []
        for index, component_id in enumerate(components, start=1):
            binding = next(
                item for item in catalog if component_id in item["design_component_ids"]
            )
            symbol = "missing_reviewed_symbol" if self.bad_symbol and index == 1 else binding["symbol"]
            formula_id = f"formula-component-{index}"
            formula_ids.append(formula_id)
            formulas.append(
                {
                    "formula_id": formula_id,
                    "label": f"Implemented component {index}",
                    "latex": r"\hat{y} = f_{\theta}(x)",
                    "explanation": "This equation describes the bounded transformation implemented by the referenced reviewed symbol.",
                    "design_component_ids": [component_id],
                    "code_references": [
                        {"file_path": binding["file_path"], "symbol": symbol}
                    ],
                    "citation_ids": citations,
                }
            )
            method_claims.append(
                {
                    "claim_id": f"claim-method-{index}",
                    "claim_type": "method",
                    "text": "The proposed component follows the approved design and the reviewed local implementation.",
                    "citation_ids": citations,
                    "design_component_ids": [component_id],
                    "code_file_paths": [binding["file_path"]],
                    "result_keys": [],
                    "strength": "descriptive",
                }
            )
        abstract = (
            "We present an evidence-grounded computational affinity prediction method whose literature, "
            "implementation, and evaluated claims remain connected through content-addressed artifacts and "
            "independently reviewable experiment records."
        )
        results_paragraph = (
            "The deterministic tables below report every approved main and supplementary result directly from persisted server evidence."
        )
        if self.numeric_results:
            results_paragraph += " The reported score is 1.0."
        sections = [
            {
                "section_id": "introduction",
                "title": "Introduction",
                "paragraphs": ["Reliable affinity prediction requires connected literature, implementation, and evaluation evidence."],
                "citation_ids": citations,
                "formula_ids": [],
                "result_keys": [],
            },
            {
                "section_id": "related_work",
                "title": "Related Work",
                "paragraphs": ["The downloaded comparison papers motivate the selected representation and evaluation boundaries."],
                "citation_ids": citations,
                "formula_ids": [],
                "result_keys": [],
            },
            {
                "section_id": "method",
                "title": "Method",
                "paragraphs": ["The method follows the approved component graph and exposes each implemented transformation for review."],
                "citation_ids": citations,
                "formula_ids": formula_ids,
                "result_keys": [],
            },
            {
                "section_id": "experimental_setup",
                "title": "Experimental Setup",
                "paragraphs": ["All experiments are computational, use measured local datasets, fixed controls, and declared random seeds."],
                "citation_ids": citations,
                "formula_ids": [],
                "result_keys": [],
            },
            {
                "section_id": "results",
                "title": "Results",
                "paragraphs": [results_paragraph],
                "citation_ids": [],
                "formula_ids": [],
                "result_keys": result_keys,
            },
            {
                "section_id": "discussion",
                "title": "Discussion",
                "paragraphs": ["Observed evidence supports only the bounded computational interpretation stated by the approved design."],
                "citation_ids": citations,
                "formula_ids": [],
                "result_keys": [],
            },
            {
                "section_id": "limitations",
                "title": "Limitations",
                "paragraphs": ["The conclusions remain limited to the downloaded datasets, reviewed implementation, and evaluated computational settings."],
                "citation_ids": [],
                "formula_ids": [],
                "result_keys": [],
            },
            {
                "section_id": "conclusion",
                "title": "Conclusion",
                "paragraphs": ["The evidence graph supports a reproducible manuscript without extending beyond observed computational results."],
                "citation_ids": [],
                "formula_ids": [],
                "result_keys": [],
            },
        ]
        claims = [
            {
                "claim_id": "claim-literature",
                "claim_type": "literature",
                "text": "The verified comparison papers motivate the bounded method and evaluation choices.",
                "citation_ids": citations,
                "design_component_ids": [],
                "code_file_paths": [],
                "result_keys": [],
                "strength": "descriptive",
            },
            *method_claims,
            {
                "claim_id": "claim-experiment",
                "claim_type": "experiment",
                "text": "The persisted computational evidence supports the reported bounded evaluation outcome.",
                "citation_ids": [],
                "design_component_ids": [],
                "code_file_paths": [],
                "result_keys": result_keys,
                "strength": "associational",
            },
            {
                "claim_id": "claim-limitation",
                "claim_type": "limitation",
                "text": "The study does not establish effects outside the evaluated computational datasets and protocols.",
                "citation_ids": [],
                "design_component_ids": [],
                "code_file_paths": [],
                "result_keys": [],
                "strength": "descriptive",
            },
        ]
        return {
            "title": "Evidence-Grounded Assay-Aware Drug Target Affinity Prediction",
            "abstract": abstract,
            "keywords": ["drug target affinity", "uncertainty", "reproducibility"],
            "sections": sections,
            "formulas": formulas,
            "claims": claims,
            "acknowledgements": None,
        }


def _paper_manifest(artifacts: ArtifactStore, run_id: str):
    records = []
    titles = {
        "paper-one": "Dynamic applicability domain for compound-target binding affinity estimates with local conformal prediction",
        "paper-two": "Conformal Prediction for Uncertainty Estimation in Drug-Target Interaction Prediction",
    }
    for index, paper_id in enumerate(("paper-one", "paper-two"), start=1):
        buffer = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72 + index, height=72)
        writer.write(buffer)
        payload = buffer.getvalue()
        filename = "paper.pdf" if paper_id == "paper-one" else "paper-two.pdf"
        pdf = artifacts.write_agent_bytes(
            AgentName.DOWNLOAD_PAPER,
            f"reference/assaycal-test/papers/{filename}",
            payload,
            max_bytes=4096,
            kind="paper_pdf",
        )
        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=titles[paper_id],
                authors=["Fixture Author"],
                year=2025,
                abstract="A local abstract describing a computational affinity prediction comparison and its bounded evidence.",
                source="Bioinformatics",
                venue="Bioinformatics",
                publication_type="journal",
                final_url=f"https://example.com/{filename}",
                content_type="application/pdf",
                size_bytes=len(payload),
                sha256=pdf.sha256,
                local_artifact=pdf,
                status="downloaded",
                reason="verified local fixture",
            )
        )
    manifest = PaperManifest(
        run_id=run_id,
        task_id="task_papers",
        objective="test paper writing",
        query="affinity paper query",
        search_result_count=2,
        records=records,
        downloaded_count=2,
        discovered_count=0,
        requires_review_count=0,
        policy=DownloadPolicyRecord(
            max_bytes=1024 * 1024,
            max_redirects=3,
            allowed_content_types=["application/pdf"],
        ),
    )
    return artifacts.write_agent_json(
        AgentName.DOWNLOAD_PAPER,
        "reference/assaycal-test/manifests/papers.json",
        manifest.model_dump(mode="json"),
    ).model_copy(update={"kind": "paper_manifest"})


def _full_inputs(artifacts: ArtifactStore, run_id: str):
    backend = SeedMetricBackend({7: 0.1, 17: 0.2, 27: 0.4})
    prepared = prepare_passed_main(artifacts, run_id, backend)
    launch = ExperimentCodeRunAgent(
        StaticModel(READY_COVERAGE),
        artifacts,
        backend=backend,
        execution_config=prepared["config"],
    ).run(experiment_task(run_id, prepared["inputs"], "paper_launch"))
    launch_ref = next(
        item for item in launch.artifacts if item.kind == "experiment_code_run_manifest"
    )
    forced_ref = force_manifest_due(artifacts, run_id, launch_ref)
    completed = ExperimentCodeRunAgent(
        StaticModel({"ignored": "monitoring never asks the model for numbers"}),
        artifacts,
        backend=backend,
        execution_config=prepared["config"],
    ).run(
        experiment_task(
            run_id,
            [*prepared["inputs"], forced_ref],
            "paper_complete",
        )
    )
    if completed.status != "succeeded":
        raise AssertionError(completed.summary)
    experiment_report_ref = next(
        item for item in completed.artifacts if item.kind == "experiment_code_run_report"
    )
    report = ExperimentCodeRunReport.model_validate(
        artifacts.read_artifact_json(experiment_report_ref)
    )
    if report.verdict != "passed":
        raise AssertionError(report.summary)
    paper_ref = _paper_manifest(artifacts, run_id)
    return [
        paper_ref,
        prepared["data_ref"],
        *prepared["inputs"],
        forced_ref,
        experiment_report_ref,
    ]


def _write_task(run_id: str, inputs):
    return TaskEnvelope(
        run_id=run_id,
        task_id=f"task_write_{run_id}",
        agent=AgentName.WRITE_PAPER,
        objective="write a reliable computational affinity prediction paper",
        inputs=list(inputs),
        context={"project_slug": "assaycal-test", "state_version": 20},
    )


class PaperPipelineTests(unittest.TestCase):
    def test_formula_contract_rejects_executable_or_unknown_tex_commands(self):
        with self.assertRaises(ValueError):
            PaperFormulaDraft(
                formula_id="formula-unsafe",
                label="Unsafe formula",
                latex=r"\input{secret}",
                explanation="This formula must never read an external local file.",
                design_component_ids=["component-encoder"],
                code_references=[
                    {"file_path": "src/model.py", "symbol": "forward"}
                ],
            )

    def test_grounded_write_and_independent_checker_pass(self):
        run_id = "run_paper_pipeline"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            inputs = _full_inputs(artifacts, run_id)
            write_result = WritePaperAgent(PaperDraftModel(), artifacts).run(
                _write_task(run_id, inputs)
            )
            self.assertEqual(write_result.status, "succeeded", write_result.summary)
            manifest_ref = write_result.artifacts[0]
            manifest = PaperManuscriptManifest.model_validate(
                artifacts.read_artifact_json(manifest_ref)
            )
            self.assertGreaterEqual(len(manifest.experiment_results), 3)
            self.assertTrue(all(item.code_bindings for item in manifest.formulas))
            checker = CheckPaperAndCodeAgent(
                StaticModel(
                    {
                        "summary": "No qualitative evidence inconsistency was found.",
                        "findings": [],
                    }
                ),
                artifacts,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_check_paper",
                    agent=AgentName.CHECK_PAPER_AND_CODE,
                    objective="check the manuscript against original papers, code, and results",
                    inputs=[*inputs, *write_result.artifacts],
                    context={"project_slug": "assaycal-test", "state_version": 22},
                )
            )
            self.assertEqual(checker.status, "succeeded", checker.summary)
            report = PaperCodeConsistencyReport.model_validate(
                artifacts.read_artifact_json(checker.artifacts[0])
            )
            self.assertEqual(report.verdict, "passed")
            self.assertTrue(all(item.status != "failed" for item in report.checks))

            review_inputs = [*inputs, *write_result.artifacts, *checker.artifacts]
            accepted_review = PaperReviewAgent(
                StaticModel(
                    {
                        "reviewer_role": "Senior NeurIPS and Bioinformatics reviewer",
                        "target_venue_profile": "authoritative_bioinformatics_journal",
                        "recommendation": "minor_revision",
                        "summary": "The manuscript is methodologically sound, evidence-grounded, reproducible, and ready apart from minor editorial clarification.",
                        "strengths": ["Claims are connected to immutable computational evidence."],
                        "findings": [
                            {
                                "finding_id": "review-minor-writing",
                                "category": "writing",
                                "severity": "minor",
                                "summary": "One transition paragraph could more directly connect the method and result sections.",
                                "evidence_refs": [manifest_ref.uri],
                                "required_revision": "Add a short cross-reference without changing any claim or result.",
                            }
                        ],
                        "required_revisions": ["Add the minor method-to-results cross-reference."],
                    }
                ),
                artifacts,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_final_paper_review_pass",
                    agent=AgentName.PAPER_REVIEW,
                    objective="perform an authoritative final peer review",
                    inputs=review_inputs,
                    context={"project_slug": "assaycal-test", "state_version": 24},
                )
            )
            self.assertEqual(accepted_review.status, "succeeded", accepted_review.summary)

            novelty_gap_manifest = manifest.model_copy(
                update={
                    "closest_prior_comparisons": [],
                }
            )
            novelty_gap_ref = artifacts.write_agent_json(
                AgentName.WRITE_PAPER,
                "project/paper/assaycal-test/novelty-gap/manifest.json",
                novelty_gap_manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manuscript_manifest"})
            novelty_gap_consistency = report.model_copy(
                update={"manuscript_manifest_uri": novelty_gap_ref.uri}
            )
            novelty_gap_consistency_ref = artifacts.write_agent_json(
                AgentName.CHECK_PAPER_AND_CODE,
                "project/reviews/consistency/assaycal-test/novelty-gap.json",
                novelty_gap_consistency.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_code_consistency_report"})
            deterministic_rejection = PaperReviewAgent(
                StaticModel(
                    {
                        "reviewer_role": "Senior NeurIPS and Bioinformatics reviewer",
                        "target_venue_profile": "authoritative_bioinformatics_journal",
                        "recommendation": "accept",
                        "summary": "The model reviewer did not identify a remaining issue in the supplied manuscript.",
                        "strengths": [],
                        "findings": [],
                        "required_revisions": [],
                    }
                ),
                artifacts,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_final_paper_review_deterministic_reject",
                    agent=AgentName.PAPER_REVIEW,
                    objective="perform an authoritative final peer review",
                    inputs=[novelty_gap_ref, novelty_gap_consistency_ref],
                    context={"project_slug": "assaycal-test", "state_version": 24},
                )
            )
            self.assertEqual(deterministic_rejection.status, "failed")
            self.assertEqual(
                deterministic_rejection.route_hint,
                AgentName.WRITE_PAPER,
            )

            revision_review = PaperReviewAgent(
                StaticModel(
                    {
                        "reviewer_role": "Senior NeurIPS and Bioinformatics reviewer",
                        "target_venue_profile": "authoritative_bioinformatics_journal",
                        "recommendation": "major_revision",
                        "summary": "The paper requires an additional computational robustness analysis before an authoritative reviewer can accept it.",
                        "strengths": ["The existing evidence graph is reproducible."],
                        "findings": [
                            {
                                "finding_id": "review-major-experiment",
                                "category": "experiments",
                                "severity": "major",
                                "summary": "The robustness conclusion lacks a predeclared sensitivity experiment across the approved seeds.",
                                "evidence_refs": [manifest_ref.uri],
                                "required_revision": "Run and report the missing computational sensitivity experiment with the same reviewed code path.",
                            }
                        ],
                        "required_revisions": ["Add the missing computational sensitivity experiment."],
                    }
                ),
                artifacts,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_final_paper_review_revision",
                    agent=AgentName.PAPER_REVIEW,
                    objective="perform an authoritative final peer review",
                    inputs=review_inputs,
                    context={"project_slug": "assaycal-test", "state_version": 25},
                )
            )
            self.assertEqual(revision_review.status, "failed")
            self.assertEqual(revision_review.route_hint, AgentName.EXPERIMENT_CODE_RUN)

    def test_missing_design_pdf_routes_to_download_paper(self):
        run_id = "run_paper_literature_gap"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            inputs = _full_inputs(artifacts, run_id)
            source_manifest = PaperManifest.model_validate(
                artifacts.read_artifact_json(inputs[0])
            )
            retained = [
                item for item in source_manifest.records if item.paper_id == "paper-one"
            ]
            incomplete_manifest = source_manifest.model_copy(
                update={
                    "task_id": "task_incomplete_papers",
                    "records": retained,
                    "downloaded_count": 1,
                    "discovered_count": 0,
                    "requires_review_count": 0,
                    "rejected_count": 0,
                    "authoritative_downloaded_count": 0,
                    "recent_downloaded_count": 0,
                    "github_marked_count": 0,
                }
            )
            incomplete_ref = artifacts.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                "reference/assaycal-test/manifests/incomplete.json",
                incomplete_manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manifest"})
            result = WritePaperAgent(PaperDraftModel(), artifacts).run(
                _write_task(run_id, [incomplete_ref, *inputs[1:]])
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.DOWNLOAD_PAPER)
            self.assertEqual(
                {item.code for item in result.issues},
                {"PAPER_WRITE_LITERATURE_EVIDENCE_MISSING"},
            )
            self.assertIn("paper-two", result.summary)

    def test_numeric_model_authored_result_prose_is_rejected(self):
        run_id = "run_paper_numeric_reject"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            inputs = _full_inputs(artifacts, run_id)
            result = WritePaperAgent(
                PaperDraftModel(numeric_results=True), artifacts
            ).run(_write_task(run_id, inputs))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.WRITE_PAPER)

    def test_missing_ast_symbol_is_rejected(self):
        run_id = "run_paper_symbol_reject"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            inputs = _full_inputs(artifacts, run_id)
            result = WritePaperAgent(PaperDraftModel(bad_symbol=True), artifacts).run(
                _write_task(run_id, inputs)
            )
            self.assertEqual(result.status, "failed")
            self.assertIn("PAPER_WRITE_EVIDENCE_INVALID", {item.code for item in result.issues})

    def test_checker_detects_tampered_manuscript(self):
        run_id = "run_paper_tamper"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            inputs = _full_inputs(artifacts, run_id)
            write_result = WritePaperAgent(PaperDraftModel(), artifacts).run(
                _write_task(run_id, inputs)
            )
            manifest = PaperManuscriptManifest.model_validate(
                artifacts.read_artifact_json(write_result.artifacts[0])
            )
            target = Path(directory) / manifest.markdown_artifact.uri.removeprefix(
                "artifact://"
            )
            target.write_text(target.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")
            result = CheckPaperAndCodeAgent(
                StaticModel(
                    {
                        "summary": "No qualitative evidence inconsistency was found.",
                        "findings": [],
                    }
                ),
                artifacts,
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_check_tamper",
                    agent=AgentName.CHECK_PAPER_AND_CODE,
                    objective="detect manuscript tampering",
                    inputs=[*inputs, *write_result.artifacts],
                    context={"project_slug": "assaycal-test"},
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.route_hint, AgentName.WRITE_PAPER)


if __name__ == "__main__":
    unittest.main()
