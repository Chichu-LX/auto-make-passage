import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from agents.code_make_agent import CodeMakeAgent
from agents.download_data_agent import DownloadDataAgent
from agents.download_data_agent.policy import DATA_MAX_BYTES
from agents.download_paper_agent import DownloadPaperAgent
from agents.idea_design_agent import IdeaDesignAgent
from agents.feedback_agent import FeedbackAgent
from agents.lead_agent import DeepSeekLeadAgent, LeadDecisionError
from agents.make_idea_agent import MakeIdeaAgent
from research_agents.data_profiling import canonicalize_headerless_dta_text
from research_agents.downloads import DownloadPayload
from research_agents.schemas import (
    AgentName,
    DownloadPolicyRecord,
    DatasetRequirement,
    LeadTransitionOption,
    PaperCandidate,
    PaperManifest,
    PaperRecord,
    RunSnapshot,
    StatisticalAnalysisPlan,
    TaskEnvelope,
    WorkflowState,
)
from research_agents.storage import ArtifactStore
from research_fixtures import advisor_response, valid_design_response, valid_idea_response


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return dict(self.response)


class IdeaDesignFakeModel:
    def __init__(self):
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if "read-only auxiliary Agent" in system_prompt:
            role = system_prompt.split(",", 1)[0].removeprefix("You are ")
            return advisor_response(role)
        response = valid_design_response()
        request = json.loads(user_prompt)
        required_fields = request.get("required_fields")
        if isinstance(required_fields, list):
            return {field: response[field] for field in required_fields}
        return response


class ProviderFailingModel:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        from research_agents.providers import ProviderRequestError

        self.calls += 1
        raise ProviderRequestError("bounded provider budget exhausted")


class FakeSearch:
    def search(self, query, max_results=5):
        return [
            {
                "title": "Example result",
                "url": "https://example.com/result",
                "content": "abstract or dataset description",
                "score": 0.9,
            }
        ]


class FakeDownloader:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def fetch(self, url, policy):
        self.calls.append((url, policy))
        return self.payload


class AgentContractTests(unittest.TestCase):
    @staticmethod
    def _valid_coverage_statistics() -> dict:
        return {
            "primary_metric": "strict cold-start empirical coverage at nominal 0.90",
            "secondary_metrics": ["mean interval width", "RMSE"],
            "analysis_unit": "drug-target affinity pair",
            "effect_size": "coverage must stay above the lower 0.85 and below the upper 0.95 bound",
            "confidence_interval": "paired cluster bootstrap over drug and target entities",
            "alpha": 0.05,
            "multiple_testing_correction": "Holm correction across primary model comparisons",
            "seed_aggregation": "report every seed with descriptive median and IQR; seeds are not the inference unit",
            "random_seeds": [7, 17, 27],
            "minimum_random_seeds": 3,
            "minimum_successful_seeds": 2,
            "precision_or_power_analysis": "Use simulation-based precision analysis to require enough held-out entity clusters for the maximum allowed coverage confidence-interval width.",
            "assumption_checks": ["Check within-entity dependence and residual heterogeneity."],
            "robustness_checks": ["Repeat inference under pair and entity-cluster bootstrap schemes."],
            "interval_width_normalization": "dimensionless interval width divided by the frozen per-dataset training-label IQR",
            "negative_control_policy": "Conformal coverage alone cannot judge shuffled labels; require worse normalized interval efficiency, higher point error, and uninformative width-error Spearman tracking.",
            "practical_significance_threshold": "coverage stays inside both bounds and mean interval width remains within ten percent of the control",
            "failure_decision_rule": "reject if the entity-cluster bootstrap confidence interval for coverage lies below 0.85 or above 0.95, or interval width exceeds the efficiency gate",
        }

    def test_coverage_statistics_reject_seed_only_bootstrap(self):
        payload = self._valid_coverage_statistics()
        payload["confidence_interval"] = (
            "10,000 bootstrap resamples over the three random seed folds"
        )
        with self.assertRaisesRegex(ValueError, "analysis units or entity clusters"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_coverage_statistics_require_width_or_efficiency_gate(self):
        payload = self._valid_coverage_statistics()
        payload["practical_significance_threshold"] = (
            "coverage stays inside the predeclared lower and upper bounds"
        )
        payload["failure_decision_rule"] = (
            "reject if the coverage confidence interval is below 0.85 or above 0.95"
        )
        with self.assertRaisesRegex(ValueError, "interval-width or efficiency"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_statistics_reject_cross_dataset_absolute_width_scale(self):
        payload = self._valid_coverage_statistics()
        payload["interval_width_normalization"] = (
            "use the same absolute affinity-unit threshold for every dataset"
        )
        with self.assertRaisesRegex(ValueError, "dataset-specific dimensionless"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_statistics_reject_negative_control_judged_by_coverage(self):
        payload = self._valid_coverage_statistics()
        payload["negative_control_policy"] = (
            "Reject shuffled labels whenever empirical coverage reaches the nominal band."
        )
        with self.assertRaisesRegex(ValueError, "coverage alone"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_statistics_reject_all_seed_coverage_vote(self):
        payload = self._valid_coverage_statistics()
        payload["seed_aggregation"] = (
            "report descriptive values; seeds are not the inference unit, but all seeds must pass coverage"
        )
        with self.assertRaisesRegex(ValueError, "all-seed point-estimate quorum"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_statistics_reject_coverage_point_gate_without_finite_sample_interval(self):
        payload = self._valid_coverage_statistics()
        payload["failure_decision_rule"] = (
            "reject if empirical coverage is below 0.85 or above 0.95, or interval width is excessive"
        )
        with self.assertRaisesRegex(ValueError, "finite-sample confidence interval"):
            StatisticalAnalysisPlan.model_validate(payload)

    def test_research_design_rejects_identifier_only_shift_audit(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["data_audit_plans"][0]["leakage_checks"] = [
            "drug identifier overlap by split",
            "target identifier overlap by split",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "representation- or group-level"):
            ResearchDesignDraft.model_validate(payload)

    def test_research_design_applies_data_semantics_contract_across_domains(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["dataset_requirements"][0].update(
            {
                "dataset_key": "dataset-medical-imaging",
                "candidate_names": ["Multi-site clinical imaging cohort"],
                "acquisition_query": "official multi-site medical imaging outcome dataset",
                "contents": ["patient images", "hospital site", "clinical outcome"],
                "required_fields": [
                    "patient_id",
                    "site_id",
                    "image_embedding",
                    "outcome",
                ],
                "split_strategy": "Freeze an unseen-hospital site split for external validation.",
            }
        )
        for experiment in payload["experiment_plans"]:
            experiment["dataset_keys"] = ["dataset-medical-imaging"]
        payload["data_audit_plans"] = [
            {
                "dataset_key": "dataset-medical-imaging",
                "schema_checks": [
                    "required fields and dtypes",
                    "clinical endpoint definition and outcome coding normalization by site",
                ],
                "leakage_checks": [
                    "exact image-feature and patient-group overlap across splits",
                    "hospital-site disjointness and near-duplicate image leakage audit",
                ],
                "duplicate_conflict_policy": "Quarantine every duplicate or identical model input with conflicting clinical outcomes and retain provenance.",
                "missingness_policy": "Report missingness by site and never remove rows using the outcome.",
                "distribution_shift_checks": [
                    "scanner and site drift",
                    "outcome prevalence and demographic drift",
                ],
                "acceptance_gates": [
                    "zero patient or image-feature leakage",
                    "endpoint coding is harmonized before analysis",
                ],
            }
        ]

        from research_agents.schemas import ResearchDesignDraft

        ResearchDesignDraft.model_validate(payload)
        payload["data_audit_plans"][0]["duplicate_conflict_policy"] = (
            "Drop repeated patient identifiers before training."
        )
        with self.assertRaisesRegex(ValueError, "conflicting outcomes"):
            ResearchDesignDraft.model_validate(payload)

    def test_dataset_requirement_rejects_derived_negative_control(self):
        payload = dict(valid_design_response()["dataset_requirements"][0])
        payload.update(
            {
                "dataset_key": "dataset-permuted-negative-control",
                "source_kind": "external_dataset",
                "candidate_names": ["permuted negative control"],
            }
        )
        with self.assertRaisesRegex(ValueError, "derived experimental transformations"):
            DatasetRequirement.model_validate(payload)

    def test_research_design_rejects_impossible_cold_start_tier_partition(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["objective"] = (
            "Evaluate a primary test split disjoint from training on both drug and target identities."
        )
        payload["experiment_plans"][2]["variables"] = [
            "new-drug-only",
            "new-target-only",
            "both-new",
        ]
        payload["experiment_plans"][2]["protocol"] = [
            "Partition the frozen primary test pairs into new-drug-only, new-target-only, and both-new tiers.",
            "Report every diagnostic without changing the training set.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "contains only both-new pairs"):
            ResearchDesignDraft.model_validate(payload)

    def test_research_design_allows_separate_single_sided_cold_start_splits(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["objective"] = (
            "Evaluate a primary test split disjoint from training on both drug and target identities."
        )
        payload["experiment_plans"][2]["variables"] = [
            "new-drug-only",
            "new-target-only",
            "both-new",
        ]
        payload["experiment_plans"][2]["protocol"] = [
            "Construct a separate diagnostic split for new-drug-only pairs with target overlap allowed.",
            "Construct a separate diagnostic split for new-target-only pairs with drug overlap allowed, while both-new remains the primary test split.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        ResearchDesignDraft.model_validate(payload)

    def test_research_design_rejects_drug_by_drug_test_product(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["protocol"] = [
            "Assign train pairs as D_train x T_train and primary test pairs as D_test x D_test.",
            "Report strict cold-start metrics on the frozen test pairs.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "D_test x T_test"):
            ResearchDesignDraft.model_validate(payload)

    def test_research_design_allows_invalid_products_named_as_anti_patterns(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["protocol"] = [
            "Assign train pairs as D_train x T_train and primary test pairs as D_test x T_test.",
            "Primary test pairs are exactly D_test x T_test; never D_test x D_test or T_test x T_test.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        ResearchDesignDraft.model_validate(payload)

    def test_research_design_rejects_target_by_target_test_product(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["protocol"] = [
            "Assign train pairs as D_train x T_train and primary test pairs as T_test x T_test.",
            "Report strict cold-start metrics on the frozen test pairs.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "D_test x T_test"):
            ResearchDesignDraft.model_validate(payload)

    def test_research_design_rejects_confidence_as_conformal_alpha(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][0]["protocol"] = [
            "Fit split conformal intervals at nominal confidence 0.90 with alpha=0.90.",
            "Evaluate the frozen test pairs without tuning.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "alpha=0.10"):
            ResearchDesignDraft.model_validate(payload)

    def test_research_design_rejects_confidence_levels_named_alpha(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        payload["experiment_plans"][1]["variables"] = [
            "nominal alpha levels: 0.80, 0.90, 0.95"
        ]
        payload["experiment_plans"][1]["protocol"] = [
            "Repeat alpha levels 0.80, 0.90, 0.95 on the frozen test set.",
            "Report empirical coverage for every setting.",
        ]
        from research_agents.schemas import ResearchDesignDraft

        with self.assertRaisesRegex(ValueError, "confidence/coverage levels"):
            ResearchDesignDraft.model_validate(payload)

    def test_idea_design_cache_ignores_only_volatile_review_id(self):
        first = IdeaDesignAgent._semantic_cache_request(
            {
                "objective": "bounded correction",
                "user_revision": {
                    "review_id": "review-first",
                    "comments": "repair the seed quorum",
                    "rating": 2,
                    "target_agent": "idea-design-agent",
                },
            }
        )
        second = IdeaDesignAgent._semantic_cache_request(
            {
                "objective": "bounded correction",
                "user_revision": {
                    "review_id": "review-second",
                    "comments": "repair the seed quorum",
                    "rating": 2,
                    "target_agent": "idea-design-agent",
                },
            }
        )
        self.assertEqual(first, second)
        self.assertNotIn("review_id", first["user_revision"])
        self.assertEqual(
            first["user_revision"]["comments"], "repair the seed quorum"
        )

    def test_idea_design_bounded_revision_rejects_experiment_identity_drift(self):
        payload = valid_design_response()
        payload["advisor_consensus"] = [
            advisor_response(role)
            for role in (
                "method-architecture-advisor",
                "code-reuse-architecture-advisor",
                "data-decision-advisor",
                "experiment-decision-advisor",
                "critical-risk-advisor",
            )
        ]
        from research_agents.schemas import ResearchDesignDraft

        draft = ResearchDesignDraft.model_validate(payload)
        baseline_plans = [
            {
                "experiment_id": item.experiment_id,
                "experiment_type": item.experiment_type,
            }
            for item in draft.experiment_plans
        ]
        context = {
            "revision_preservation_policy": {"experiment_identity": True},
            "revision_baseline": {"experiment_plans": baseline_plans},
        }
        IdeaDesignAgent._validate_revision_preservation(draft, context)
        drifted = draft.model_copy(
            update={"experiment_plans": draft.experiment_plans[:-1]}
        )
        with self.assertRaisesRegex(ValueError, "experiment count or identity"):
            IdeaDesignAgent._validate_revision_preservation(drifted, context)

    def test_idea_design_revision_baseline_is_newest_non_rejected_design(self):
        candidates = [
            {
                "task_id": "task_old",
                "created_at": "2026-08-13T17:00:00Z",
                "method_name": "old",
            },
            {
                "task_id": "task_baseline",
                "created_at": "2026-08-13T18:40:00Z",
                "method_name": "baseline",
            },
            {
                "task_id": "task_rejected",
                "created_at": "2026-08-22T07:27:00Z",
                "method_name": "rejected",
            },
        ]
        selected = IdeaDesignAgent._select_revision_baseline(
            candidates,
            {"review_id": "review_task_rejected"},
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["task_id"], "task_baseline")

    def test_idea_design_revision_honors_explicit_baseline_task(self):
        candidates = [
            {
                "task_id": "task_requested",
                "created_at": "2026-08-13T18:40:00Z",
            },
            {
                "task_id": "task_newer_but_rejected",
                "created_at": "2026-08-22T07:27:00Z",
            },
        ]
        selected = IdeaDesignAgent._select_revision_baseline(
            candidates,
            {
                "review_id": "review_task_failed_without_design",
                "comments": "Use task_requested / research-design baseline.",
            },
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["task_id"], "task_requested")

    def test_idea_design_extracts_only_explicit_scoped_design_token(self):
        self.assertEqual(
            IdeaDesignAgent._explicit_revision_design_token(
                {"comments": "Use research-design 3f7a26283fe7.json as baseline."}
            ),
            "3f7a26283fe7",
        )
        self.assertIsNone(
            IdeaDesignAgent._explicit_revision_design_token(
                {"comments": "Use ../../3f7a26283fe7.json as baseline."}
            )
        )

    def test_idea_design_rejects_method_name_drift_before_section_cache(self):
        context = {
            "revision_preservation_policy": {"method_name": True},
            "revision_baseline": {"method_name": "Frozen Method"},
        }
        IdeaDesignAgent._validate_revision_section_preservation(
            "method-and-code",
            {"method_name": "Frozen Method"},
            context,
        )
        with self.assertRaisesRegex(ValueError, "preserved method name"):
            IdeaDesignAgent._validate_revision_section_preservation(
                "method-and-code",
                {"method_name": "Renamed Method"},
                context,
            )

    def test_idea_design_does_not_multiply_exhausted_provider_retry_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ProviderFailingModel()
            agent = IdeaDesignAgent(model, ArtifactStore(directory))
            task = TaskEnvelope(
                run_id="run_provider_bound",
                task_id="task_provider_bound",
                agent=AgentName.IDEA_DESIGN,
                objective="verify the bounded provider failure path",
                context={"project_slug": "provider-bound"},
            )
            with self.assertRaisesRegex(ValueError, "bounded synthesis attempts"):
                agent._generate_section(
                    task,
                    name="method-and-code",
                    fields=("selected_idea_id",),
                    request_base={"selected_idea_id": "idea-one"},
                )
            self.assertEqual(model.calls, 1)

    def test_idea_design_normalizes_bounded_advisor_field_drift(self):
        normalized = IdeaDesignAgent._normalize_advisor_response(
            {
                "advisor_report": {
                    "recommendation": ["one", "two"],
                    "decision": "retain the strict split",
                    "concerns": "license must be verified",
                    "paper_refs": [{"paper_id": "paper-one"}, "UNKNOWN"],
                    "assumption": "the dataset remains public",
                    "issues": ["LICENSE_UNKNOWN"],
                }
            },
            role="data-decision-advisor",
            paper_ids={"paper-one"},
        )
        from research_agents.schemas import AdvisorReport

        report = AdvisorReport.model_validate(normalized)
        self.assertEqual(report.advisor, "data-decision-advisor")
        self.assertEqual(report.evidence_refs, ["paper-one"])
        self.assertEqual(report.decisions, ["retain the strict split"])

    def test_make_idea_rejects_unbounded_global_novelty_claim(self):
        draft = valid_idea_response()
        draft["candidates"][0]["idea"]["novelty_claim"] = (
            "This is the first calibrated affinity method in the entire literature."
        )
        from research_agents.schemas import IdeaDraft

        with self.assertRaisesRegex(ValueError, "bounded supplied-paper"):
            MakeIdeaAgent._validate_claim_boundaries(IdeaDraft.model_validate(draft))

    def test_make_idea_uses_latest_complete_consolidation_as_citation_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            policy = DownloadPolicyRecord(
                max_bytes=1024,
                max_redirects=1,
                allowed_content_types=["application/pdf"],
            )

            def manifest_ref(task_id, paper_id, purpose="portfolio_selection"):
                payload = b"%PDF-1.7\n" + paper_id.encode("utf-8")
                pdf_ref = store.write_agent_bytes(
                    AgentName.DOWNLOAD_PAPER,
                    f"reference/citation-boundary/papers/{paper_id}.pdf",
                    payload,
                    max_bytes=1024,
                    kind="paper_pdf",
                )
                record = PaperRecord(
                    paper_id=paper_id,
                    title=f"Title {paper_id}",
                    source="Bioinformatics",
                    venue="Bioinformatics",
                    publication_type="journal",
                    final_url=f"https://example.com/{paper_id}.pdf",
                    content_type="application/pdf",
                    size_bytes=len(payload),
                    sha256=pdf_ref.sha256,
                    local_artifact=pdf_ref,
                    status="downloaded",
                    reason="verified",
                    quality_tier="authoritative",
                    canonical_venue="Bioinformatics",
                    portfolio_bucket="authoritative",
                )
                manifest = PaperManifest(
                    run_id="run_boundary",
                    task_id=task_id,
                    objective="citation boundary",
                    query="",
                    search_result_count=0,
                    records=[record],
                    downloaded_count=1,
                    discovered_count=0,
                    requires_review_count=0,
                    authoritative_downloaded_count=1,
                    policy=policy,
                    manifest_purpose=purpose,
                    required_paper_ids=(
                        [paper_id] if purpose == "evidence_consolidation" else []
                    ),
                    required_evidence_complete=(purpose == "evidence_consolidation"),
                )
                return store.write_agent_json(
                    AgentName.DOWNLOAD_PAPER,
                    f"reference/citation-boundary/manifests/{task_id}.json",
                    manifest.model_dump(mode="json"),
                ).model_copy(update={"kind": "paper_manifest"})

            old_ref = manifest_ref("old", "paper-old")
            consolidated_ref = manifest_ref(
                "complete", "paper-required", "evidence_consolidation"
            )
            agent = MakeIdeaAgent(FakeModel({}), store)
            papers, sources, errors = agent._read_paper_evidence(
                TaskEnvelope(
                    run_id="run_boundary",
                    task_id="task_idea",
                    agent=AgentName.MAKE_IDEA,
                    objective="citation boundary",
                    inputs=[old_ref, consolidated_ref],
                    context={"project_slug": "citation-boundary"},
                )
            )
            self.assertEqual(errors, [])
            self.assertEqual([item["paper_id"] for item in papers], ["paper-required"])
            self.assertEqual(sources, [consolidated_ref])

    def test_paper_repair_enriches_missing_venue_from_registered_exact_id(self):
        candidate = PaperCandidate(
            paper_id="paper-required",
            title="Required affinity study",
            authors=[],
            abstract="",
            source="PubMed Central",
            publication_type="other",
        )
        historical = PaperRecord(
            paper_id="paper-required",
            title="Required affinity study",
            authors=["Fixture Author"],
            year=2025,
            abstract="Registered metadata from an earlier exact-ID manifest.",
            source="PMC",
            venue="ACS Omega",
            publication_type="journal",
            status="rejected",
            reason="PDF was not yet available",
            quality_tier="established_peer_reviewed",
            canonical_venue="ACS Omega",
            portfolio_bucket="authoritative",
        )
        enriched = DownloadPaperAgent._enrich_repair_metadata(
            candidate, historical, {"paper-required"}
        )
        self.assertEqual(enriched.year, 2025)
        self.assertEqual(enriched.venue, "ACS Omega")
        self.assertEqual(enriched.publication_type, "journal")
        self.assertEqual(enriched.authors, ["Fixture Author"])

    def test_paper_selection_normalizes_openai_compatible_field_drift(self):
        normalized = DownloadPaperAgent._normalize_selection(
            {
                "candidates": [
                    {
                        "paper_id": "DTA Live 001",
                        "title": "A valid affinity paper",
                        "authors": None,
                        "year": "2026",
                        "DOI": "10.1000/test",
                        "abstract": None,
                        "source": "Bioinformatics",
                        "venue": "Bioinformatics",
                        "publication_type": "journal_article",
                        "landing_page": "https://example.com/paper",
                        "pdf_url": "https://example.com/paper.pdf",
                        "relevance_score": 2,
                        "unexpected": "discarded",
                    }
                ]
            }
        )
        candidate = normalized["candidates"][0]
        self.assertEqual(candidate["paper_id"], "paper-dta-live-001")
        self.assertEqual(candidate["publication_type"], "journal")
        self.assertEqual(candidate["doi"], "10.1000/test")
        self.assertEqual(candidate["relevance_score"], 1.0)
        self.assertNotIn("unexpected", candidate)

    def test_deepseek_lead_validates_structured_decision(self):
        model = FakeModel(
            {
                "schema_version": "1.0",
                "run_id": "run_test",
                "observed_state_version": 1,
                "decision": "dispatch",
                "target_agent": "download-paper-agent",
                "task_template": "discover_research_papers",
                "reason": "papers are the next stage",
                "required_inputs": [],
                "needs_human": False,
            }
        )
        snapshot = RunSnapshot(
            run_id="run_test",
            objective="test topic",
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )
        decision = DeepSeekLeadAgent(model).decide(snapshot, [AgentName.DOWNLOAD_PAPER])
        self.assertEqual(decision.target_agent, AgentName.DOWNLOAD_PAPER)
        self.assertEqual(DeepSeekLeadAgent.allowed_tools, ())

    def test_deepseek_lead_selects_exact_policy_transition_option(self):
        model = FakeModel(
            {
                "schema_version": "1.0",
                "run_id": "run_test",
                "observed_state_version": 1,
                "phase": "specialist_result",
                "selected_option_id": "request_human_needs_human",
                "action": "request_human",
                "target_state": "NEEDS_HUMAN",
                "target_agent": None,
                "reason": "The available evidence is insufficient for a safe advance.",
                "considered_issue_codes": ["EVIDENCE_GAP"],
            }
        )
        snapshot = RunSnapshot(
            run_id="run_test",
            objective="test topic",
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )
        options = [
            LeadTransitionOption(
                option_id="advance_make_idea_agent",
                action="advance",
                target_state=WorkflowState.IDEATING,
                target_agent=AgentName.MAKE_IDEA,
                reason_hint="Typed paper result passed the policy checks.",
                recommended=True,
            ),
            LeadTransitionOption(
                option_id="request_human_needs_human",
                action="request_human",
                target_state=WorkflowState.NEEDS_HUMAN,
                reason_hint="Evidence is insufficient and requires human review.",
            ),
        ]
        decision = DeepSeekLeadAgent(model).decide_transition(
            snapshot,
            phase="specialist_result",
            options=options,
            context={"issue_codes": ["EVIDENCE_GAP"]},
        )
        self.assertEqual(decision.action, "request_human")
        self.assertEqual(decision.target_state, WorkflowState.NEEDS_HUMAN)

    def test_deepseek_lead_reconstructs_transition_fields_from_selected_option(self):
        model = FakeModel(
            {
                "selected_option_id": "advance_make_idea_agent",
                "action": "retry",
                "target_state": "WRITING",
                "target_agent": "write-paper-agent",
                "reason": "The paper acquisition evidence passed its typed gates.",
            }
        )
        snapshot = RunSnapshot(
            run_id="run_test",
            objective="test topic",
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )
        options = [
            LeadTransitionOption(
                option_id="advance_make_idea_agent",
                action="advance",
                target_state=WorkflowState.IDEATING,
                target_agent=AgentName.MAKE_IDEA,
                reason_hint="Typed paper result passed the policy checks.",
                recommended=True,
            ),
            LeadTransitionOption(
                option_id="request_human_needs_human",
                action="request_human",
                target_state=WorkflowState.NEEDS_HUMAN,
                reason_hint="Evidence is insufficient and requires human review.",
            ),
        ]
        decision = DeepSeekLeadAgent(model).decide_transition(
            snapshot,
            phase="specialist_result",
            options=options,
            context={"issue_codes": []},
        )
        self.assertEqual(decision.action, "advance")
        self.assertEqual(decision.target_state, WorkflowState.IDEATING)
        self.assertEqual(decision.target_agent, AgentName.MAKE_IDEA)
        self.assertEqual(decision.run_id, snapshot.run_id)
        self.assertEqual(decision.observed_state_version, snapshot.version)

    def test_deepseek_lead_rejects_unknown_transition_option(self):
        model = FakeModel(
            {
                "selected_option_id": "advance_write_paper_agent",
                "reason": "Attempt to escape the Runtime transition allowlist.",
            }
        )
        snapshot = RunSnapshot(
            run_id="run_test",
            objective="test topic",
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )
        options = [
            LeadTransitionOption(
                option_id="advance_make_idea_agent",
                action="advance",
                target_state=WorkflowState.IDEATING,
                target_agent=AgentName.MAKE_IDEA,
                reason_hint="Typed paper result passed the policy checks.",
                recommended=True,
            )
        ]
        with self.assertRaises(LeadDecisionError):
            DeepSeekLeadAgent(model).decide_transition(
                snapshot,
                phase="specialist_result",
                options=options,
                context={"issue_codes": []},
            )

    def test_deepseek_lead_fails_closed_on_wrong_agent(self):
        model = FakeModel(
            {
                "schema_version": "1.0",
                "run_id": "run_test",
                "observed_state_version": 1,
                "decision": "dispatch",
                "target_agent": "code-make-agent",
                "task_template": "implement_research_code",
                "reason": "attempted escape",
                "required_inputs": [],
                "needs_human": False,
            }
        )
        snapshot = RunSnapshot(
            run_id="run_test",
            objective="test topic",
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )
        with self.assertRaises(LeadDecisionError):
            DeepSeekLeadAgent(model).decide(snapshot, [AgentName.DOWNLOAD_PAPER])

    def test_code_make_blocks_without_design_and_data_contracts(self):
        model = FakeModel(
            {
                "summary": "implementation plan ready",
                "evidence": [{"check": "contract", "status": "passed"}],
                "issues": [],
                "needs_human": False,
            }
        )
        task = TaskEnvelope(
            run_id="run_test",
            task_id="task_test",
            agent=AgentName.CODE_MAKE,
            objective="test topic",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = CodeMakeAgent(model, ArtifactStore(directory)).run(task)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.agent, AgentName.CODE_MAKE)
        self.assertEqual(result.route_hint, AgentName.DOWNLOAD_DATA)

    def test_feedback_agent_returns_typed_control(self):
        model = FakeModel(
            {
                "summary": "review mismatch should return to code implementation",
                "evidence": [{"check": "process-log", "status": "mismatch"}],
                "issues": [],
                "control": {
                    "schema_version": "1.0",
                    "action": "retry",
                    "severity": "error",
                    "reason": "Code review mismatch is traceable in the process log.",
                    "target_agent": "code-make-agent",
                    "issue_codes": ["CODE_REVIEW_MISMATCH"],
                },
            }
        )
        result = FeedbackAgent(model).run(
            TaskEnvelope(
                run_id="run_test",
                task_id="feedback_task",
                agent=AgentName.FEEDBACK,
                objective="monitor workflow",
                context={
                    "allowed_control_targets": ["code-make-agent"],
                    "monitoring_summary": {"record_count": 12},
                },
            )
        )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.feedback_control.action, "retry")
        self.assertEqual(result.feedback_control.target_agent, AgentName.CODE_MAKE)
        self.assertEqual(result.feedback_control.observed_log_count, 12)

    def test_feedback_normalizes_openai_compatible_control_drift(self):
        payload = FeedbackAgent._normalize_control(
            {
                "summary": "No blocking anomaly was detected.",
                "feedback_control": {
                    "action": "continue_workflow",
                    "severity": "low",
                    "target": "code-make-agent",
                },
            },
            log_count=8,
        )
        self.assertEqual(payload["action"], "continue")
        self.assertEqual(payload["severity"], "info")
        self.assertIsNone(payload["target_agent"])
        self.assertEqual(payload["observed_log_count"], 8)

    def test_feedback_agent_fails_closed_on_invalid_target(self):
        model = FakeModel(
            {
                "summary": "invalid route",
                "evidence": [],
                "issues": [],
                "control": {
                    "schema_version": "1.0",
                    "action": "retry",
                    "severity": "error",
                    "reason": "Attempted a route not supplied by Runtime.",
                    "target_agent": "download-paper-agent",
                    "issue_codes": [],
                },
            }
        )
        result = FeedbackAgent(model).run(
            TaskEnvelope(
                run_id="run_test",
                task_id="feedback_task",
                agent=AgentName.FEEDBACK,
                objective="monitor workflow",
                context={
                    "allowed_control_targets": ["code-make-agent"],
                    "monitoring_summary": {"record_count": 3},
                },
            )
        )
        self.assertEqual(result.feedback_control.action, "request_human")
        self.assertIn(
            "FEEDBACK_MODEL_CONTRACT_RECOVERED", result.feedback_control.issue_codes
        )

    def test_feedback_recovers_malformed_control_from_source_evidence(self):
        result = FeedbackAgent(FakeModel({"summary": "missing control"})).run(
            TaskEnvelope(
                run_id="run_test",
                task_id="feedback_task",
                agent=AgentName.FEEDBACK,
                objective="monitor workflow",
                context={
                    "source_status": "needs_human",
                    "source_issue_codes": ["NO_VERIFIED_DATASET"],
                    "monitoring_summary": {"record_count": 21},
                },
            )
        )
        self.assertEqual(result.feedback_control.action, "request_human")
        self.assertEqual(
            result.feedback_control.issue_codes, ["NO_VERIFIED_DATASET"]
        )
        self.assertIn("status=needs_human", result.feedback_control.reason)
        self.assertEqual(result.feedback_control.observed_log_count, 21)

    def test_headerless_dta_text_is_canonicalized_to_csv(self):
        payload = (
            b"drug-1 target-1 CCO MKT 7.5\n"
            b"drug-2 target-2 C#N MPEPTIDE 6.25\n"
        )
        canonical = canonicalize_headerless_dta_text(payload)
        self.assertIsNotNone(canonical)
        text = canonical.decode("utf-8")
        self.assertTrue(
            text.startswith(
                "drug_id,target_id,compound_iso_smiles,target_sequence,affinity"
            )
        )
        self.assertIn("drug-2,target-2,C#N,MPEPTIDE,6.25", text)

    def test_data_agent_resolves_search_matched_zenodo_record(self):
        metadata = {
            "metadata": {
                "doi": "10.5281/zenodo.123",
                "license": {"id": "cc-by-4.0"},
            },
            "files": [
                {
                    "key": "Davis.txt",
                    "size": 1234,
                    "links": {
                        "self": "https://zenodo.org/api/records/123/files/Davis.txt/content"
                    },
                }
            ],
        }
        encoded = json.dumps(metadata).encode("utf-8")
        downloader = FakeDownloader(
            DownloadPayload(
                content=encoded,
                final_url="https://zenodo.org/api/records/123",
                content_type="application/json",
                sha256=hashlib.sha256(encoded).hexdigest(),
                redirect_count=0,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            agent = DownloadDataAgent(
                FakeSearch(), FakeModel({"candidates": []}), ArtifactStore(directory), downloader
            )
            candidates = agent._zenodo_dataset_candidates(
                [{"dataset_key": "dataset-davis"}],
                [{"url": "https://zenodo.org/records/123"}],
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].requirement_key, "dataset-davis")
        self.assertEqual(candidates[0].license_id, "cc-by-4.0")
        self.assertEqual(candidates[0].format.value, "txt")

    def test_paper_agent_downloads_only_verified_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "paper_id": "paper-one",
                            "title": "Example paper",
                            "authors": ["A. Researcher"],
                            "year": 2025,
                            "doi": "10.1000/example",
                            "abstract": "A grounded abstract.",
                            "source": "Example Journal",
                            "landing_page_url": "https://example.com/paper",
                            "direct_pdf_url": "https://example.com/paper.pdf",
                            "relevance_score": 0.9,
                        }
                    ]
                }
            )
            payload = b"%PDF-1.7\nverified test payload"
            downloader = FakeDownloader(
                DownloadPayload(
                    content=payload,
                    final_url="https://example.com/paper.pdf",
                    content_type="application/pdf",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    redirect_count=0,
                )
            )
            agent = DownloadPaperAgent(
                FakeSearch(), model, ArtifactStore(directory), downloader
            )
            result = agent.run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="task_test",
                    agent=AgentName.DOWNLOAD_PAPER,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.artifacts[0].kind, "paper_manifest")
            manifest_path = Path(directory) / result.artifacts[0].uri.removeprefix("artifact://")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["records"][0]["status"], "downloaded")
            self.assertEqual(manifest["records"][0]["portfolio_bucket"], "recent")
            self.assertEqual(manifest["recent_downloaded_count"], 1)
            self.assertTrue(manifest["quality_policy"]["require_verified_pdf"])
            self.assertTrue(any((Path(directory) / "reference/legacy/papers").glob("*.pdf")))

    def test_paper_agent_does_not_claim_landing_page_was_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "paper_id": "paper-one",
                            "title": "Metadata only",
                            "authors": [],
                            "year": 2024,
                            "doi": None,
                            "abstract": "Only metadata was available.",
                            "source": "Repository",
                            "landing_page_url": "https://example.com/paper",
                            "direct_pdf_url": None,
                            "relevance_score": 0.8,
                        }
                    ]
                }
            )
            downloader = FakeDownloader(None)
            result = DownloadPaperAgent(
                FakeSearch(), model, ArtifactStore(directory), downloader
            ).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="task_test",
                    agent=AgentName.DOWNLOAD_PAPER,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "needs_human")
            self.assertEqual(downloader.calls, [])
            manifest_path = Path(directory) / result.artifacts[0].uri.removeprefix("artifact://")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["records"][0]["status"], "rejected")
            self.assertIn("hard PDF gate", manifest["records"][0]["reason"])

    def test_paper_repair_reads_typed_gap_and_preserves_required_ids(self):
        run_id = "run_paper_repair"
        slug = "assaycal-test"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            prior_record = PaperRecord(
                paper_id="paper-required",
                title="Required affinity study",
                authors=[],
                year=2025,
                abstract="A prior manifest title used for an exact PDF repair.",
                source="Bioinformatics",
                venue="Bioinformatics",
                publication_type="journal",
                status="rejected",
                reason="direct PDF was unavailable",
                quality_tier="authoritative",
                canonical_venue="Bioinformatics",
                portfolio_bucket="authoritative",
            )
            manifest = PaperManifest(
                run_id=run_id,
                task_id="task_prior_papers",
                objective="repair missing paper",
                query="prior query",
                search_result_count=1,
                records=[prior_record],
                downloaded_count=0,
                discovered_count=0,
                requires_review_count=0,
                rejected_count=1,
                policy=DownloadPolicyRecord(
                    max_bytes=1024 * 1024,
                    max_redirects=3,
                    allowed_content_types=["application/pdf"],
                ),
            )
            manifest_ref = artifacts.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                f"reference/{slug}/manifests/prior.json",
                manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manifest"})
            gap_ref = artifacts.write_agent_json(
                AgentName.WRITE_PAPER,
                f"project/paper/{slug}/gaps/missing.json",
                {
                    "run_id": run_id,
                    "issue_code": "PAPER_WRITE_LITERATURE_EVIDENCE_MISSING",
                    "missing_paper_ids": ["paper-required"],
                },
            ).model_copy(update={"kind": "paper_writing_gap"})
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "paper_id": "paper-required",
                            "title": "Required affinity study",
                            "authors": ["Fixture Author"],
                            "year": 2025,
                            "abstract": "Exact required paper from supplied search evidence.",
                            "source": "Bioinformatics",
                            "venue": "Bioinformatics",
                            "publication_type": "journal",
                            "direct_pdf_url": "https://example.com/required.pdf",
                            "relevance_score": 1.0,
                        }
                    ]
                }
            )
            payload = b"%PDF-1.7\nverified targeted repair"
            result = DownloadPaperAgent(
                FakeSearch(),
                model,
                artifacts,
                FakeDownloader(
                    DownloadPayload(
                        content=payload,
                        final_url="https://example.com/required.pdf",
                        content_type="application/pdf",
                        sha256=hashlib.sha256(payload).hexdigest(),
                        redirect_count=0,
                    )
                ),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_targeted_repair",
                    agent=AgentName.DOWNLOAD_PAPER,
                    objective="repair the writing literature evidence",
                    inputs=[manifest_ref, gap_ref],
                    context={"project_slug": slug},
                )
            )
            self.assertEqual(result.status, "succeeded", result.summary)
            request = json.loads(model.calls[0][1])
            self.assertEqual(
                request["required_paper_repairs"],
                [
                    {
                        "paper_id": "paper-required",
                        "title": "Required affinity study",
                    }
                ],
            )
            repaired = PaperManifest.model_validate(
                artifacts.read_artifact_json(result.artifacts[0])
            )
            self.assertEqual(repaired.records[0].paper_id, "paper-required")
            self.assertEqual(repaired.records[0].status, "downloaded")

    def test_paper_repair_reuses_verified_same_run_pdf_without_provider_call(self):
        run_id = "run_paper_history"
        slug = "paper-history"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            payload = b"%PDF-1.7\nverified historical paper"
            paper_ref = artifacts.write_agent_bytes(
                AgentName.DOWNLOAD_PAPER,
                f"reference/{slug}/papers/required.pdf",
                payload,
                max_bytes=1024 * 1024,
                kind="paper_pdf",
            )
            records = []
            for paper_id, title in (
                ("paper-required", "Required affinity study"),
                ("paper-companion", "Companion affinity study"),
            ):
                local_ref = paper_ref
                if paper_id == "paper-required":
                    local_ref = paper_ref
                else:
                    companion_payload = b"%PDF-1.7\nverified companion paper"
                    local_ref = artifacts.write_agent_bytes(
                        AgentName.DOWNLOAD_PAPER,
                        f"reference/{slug}/papers/companion.pdf",
                        companion_payload,
                        max_bytes=1024 * 1024,
                        kind="paper_pdf",
                    )
                local_payload = artifacts.read_artifact_bytes(
                    local_ref, max_bytes=1024 * 1024
                )
                records.append(
                    PaperRecord(
                        paper_id=paper_id,
                        title=title,
                        year=2025,
                        abstract="Previously downloaded exact evidence.",
                        source="Bioinformatics",
                        venue="Bioinformatics",
                        publication_type="journal",
                        final_url=f"https://example.com/{paper_id}.pdf",
                        content_type="application/pdf",
                        size_bytes=len(local_payload),
                        sha256=local_ref.sha256,
                        local_artifact=local_ref,
                        status="downloaded",
                        reason="verified",
                        quality_tier="authoritative",
                        canonical_venue="Bioinformatics",
                        portfolio_bucket="authoritative",
                    )
                )
            manifest = PaperManifest(
                run_id=run_id,
                task_id="task_prior",
                objective="repair evidence",
                query="prior",
                search_result_count=1,
                records=records,
                downloaded_count=2,
                discovered_count=0,
                requires_review_count=0,
                authoritative_downloaded_count=2,
                policy=DownloadPolicyRecord(
                    max_bytes=1024 * 1024,
                    max_redirects=3,
                    allowed_content_types=["application/pdf"],
                ),
            )
            manifest_ref = artifacts.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                f"reference/{slug}/manifests/prior.json",
                manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manifest"})
            gap_ref = artifacts.write_agent_json(
                AgentName.WRITE_PAPER,
                f"project/paper/{slug}/gaps/missing.json",
                {
                    "run_id": run_id,
                    "issue_code": "PAPER_WRITE_LITERATURE_EVIDENCE_MISSING",
                    "missing_paper_ids": ["paper-required"],
                },
            ).model_copy(update={"kind": "paper_writing_gap"})
            model = FakeModel({"candidates": []})
            downloader = FakeDownloader(None)

            result = DownloadPaperAgent(
                FakeSearch(), model, artifacts, downloader
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_merge_history",
                    agent=AgentName.DOWNLOAD_PAPER,
                    objective="merge existing exact evidence",
                    inputs=[manifest_ref, gap_ref],
                    context={"project_slug": slug},
                )
            )

            self.assertEqual(result.status, "succeeded", result.summary)
            self.assertEqual(model.calls, [])
            self.assertEqual(downloader.calls, [])
            repaired = PaperManifest.model_validate(
                artifacts.read_artifact_json(result.artifacts[0])
            )
            self.assertEqual(repaired.downloaded_count, 1)
            self.assertEqual(repaired.manifest_purpose, "evidence_consolidation")
            self.assertTrue(repaired.required_evidence_complete)
            repaired_by_id = {item.paper_id: item for item in repaired.records}
            self.assertEqual(repaired_by_id["paper-required"].local_artifact, paper_ref)
            self.assertEqual(set(repaired_by_id), {"paper-required"})

    def test_paper_repair_contract_keeps_all_design_ids_for_consolidation(self):
        run_id = "run_paper_design_contract"
        slug = "paper-design-contract"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            # Reuse the typed design fixture and make every evidence reference point
            # at one exact paper ID; the old-style writing gap has no explicit ID list.
            design_payload = valid_design_response()
            for component in design_payload["method_components"]:
                component["evidence_refs"] = ["paper-required"]
            for requirement in design_payload["dataset_requirements"]:
                requirement["paper_refs"] = ["paper-required"]
            for experiment in design_payload["experiment_plans"]:
                experiment["evidence_refs"] = ["paper-required"]
            for result in design_payload["expected_results"]:
                result["evidence_refs"] = ["paper-required"]
            for comparison in design_payload["paper_comparisons"]:
                comparison["paper_id"] = "paper-required"
            for module in design_payload["reusable_code_modules"]:
                module["source_paper_id"] = "paper-required"
            for decision in design_payload["design_decisions"]:
                decision["evidence_refs"] = ["paper-required"]
            design_payload["advisor_consensus"] = [
                advisor_response(role)
                for role in (
                    "method-architecture-advisor",
                    "code-reuse-architecture-advisor",
                    "data-decision-advisor",
                    "experiment-decision-advisor",
                    "critical-risk-advisor",
                )
            ]
            for report in design_payload["advisor_consensus"]:
                report["evidence_refs"] = ["paper-required"]
            design_payload.update(
                {
                    "run_id": run_id,
                    "task_id": "task_design",
                    "objective": "consolidate design evidence",
                    "idea_proposal_uri": (
                        f"artifact://design/{slug}/ideas/proposal.json"
                    ),
                    "paper_evidence_uris": [
                        f"artifact://reference/{slug}/manifests/prior.json"
                    ],
                    "evidence_quality": "metadata_only",
                }
            )
            design_ref = artifacts.write_agent_json(
                AgentName.IDEA_DESIGN,
                f"design/{slug}/research-designs/design.json",
                design_payload,
            ).model_copy(update={"kind": "research_design"})
            gap_ref = artifacts.write_agent_json(
                AgentName.WRITE_PAPER,
                f"project/paper/{slug}/gaps/missing.json",
                {
                    "run_id": run_id,
                    "issue_code": "PAPER_WRITE_LITERATURE_EVIDENCE_MISSING",
                    "reason": "old gap without missing_paper_ids",
                },
            ).model_copy(update={"kind": "paper_writing_gap"})
            task = TaskEnvelope(
                run_id=run_id,
                task_id="task_consolidate",
                agent=AgentName.DOWNLOAD_PAPER,
                objective="consolidate",
                inputs=[design_ref, gap_ref],
                context={"project_slug": slug},
            )
            agent = DownloadPaperAgent(
                FakeSearch(), FakeModel({"candidates": []}), artifacts
            )
            self.assertEqual(
                agent._required_paper_repairs(task),
                [{"paper_id": "paper-required", "title": "paper-required"}],
            )

    def test_data_agent_requires_license_then_downloads_safe_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "dataset_id": "dataset-one",
                            "name": "Example dataset",
                            "description": "A public benchmark dataset.",
                            "source": "Example Repository",
                            "landing_page_url": "https://example.com/data",
                            "direct_download_url": "https://example.com/data.csv",
                            "license_id": "cc-by-4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "license_evidence": "Licensed under CC BY 4.0.",
                            "format": "csv",
                            "estimated_size_bytes": 32,
                            "relevance_score": 0.9,
                        }
                    ]
                }
            )
            payload = b"label,value\na,1\nb,2\n"
            downloader = FakeDownloader(
                DownloadPayload(
                    content=payload,
                    final_url="https://example.com/data.csv",
                    content_type="text/csv",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    redirect_count=0,
                )
            )
            result = DownloadDataAgent(
                FakeSearch(), model, ArtifactStore(directory), downloader
            ).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="task_data",
                    agent=AgentName.DOWNLOAD_DATA,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "succeeded")
            self.assertTrue(any((Path(directory) / "data/legacy/raw").glob("*.csv")))
            materialized = list(
                (Path(directory) / "data/legacy/datasets/dataset-one").glob(
                    "*/dataset.csv"
                )
            )
            self.assertEqual(len(materialized), 1)
            self.assertEqual(materialized[0].read_bytes(), payload)
            manifest_path = Path(directory) / result.artifacts[0].uri.removeprefix("artifact://")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(manifest["materialized_count"], 1)
            self.assertEqual(manifest["total_materialized_bytes"], len(payload))
            self.assertEqual(len(manifest["records"][0]["materialized_files"]), 1)
            self.assertEqual(manifest["total_profiled_rows"], 2)
            self.assertEqual(manifest["records"][0]["profile"]["column_count"], 2)

    def test_data_agent_does_not_download_unverified_license(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "dataset_id": "dataset-one",
                            "name": "License unclear dataset",
                            "description": "The result does not prove a reusable license.",
                            "source": "Example Repository",
                            "landing_page_url": "https://example.com/data",
                            "direct_download_url": "https://example.com/data.csv",
                            "license_id": "custom-research-only",
                            "license_url": "https://example.com/terms",
                            "license_evidence": "Research use only.",
                            "format": "csv",
                            "estimated_size_bytes": 32,
                            "relevance_score": 0.9,
                        }
                    ]
                }
            )
            downloader = FakeDownloader(None)
            result = DownloadDataAgent(
                FakeSearch(), model, ArtifactStore(directory), downloader
            ).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="task_data",
                    agent=AgentName.DOWNLOAD_DATA,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "needs_human")
            self.assertEqual(downloader.calls, [])

    def test_data_agent_revalidates_same_run_cached_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "dataset_id": "dataset-davis-source",
                            "requirement_key": "dataset-davis",
                            "name": "Davis test dataset",
                            "description": "A bounded licensed DTA table.",
                            "source": "Test repository",
                            "landing_page_url": "https://example.com/data",
                            "direct_download_url": "https://example.com/data.csv",
                            "license_id": "cc-by-4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "license_evidence": "Licensed under CC BY 4.0.",
                            "format": "csv",
                            "estimated_size_bytes": 128,
                            "relevance_score": 1.0,
                        }
                    ]
                }
            )
            payload = (
                b"compound_iso_smiles,target_sequence,target_key,affinity\n"
                b"CCO,MPEPTIDE,target-1,7.1\n"
            )
            downloader = FakeDownloader(
                DownloadPayload(
                    content=payload,
                    final_url="https://example.com/data.csv",
                    content_type="text/csv",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    redirect_count=0,
                )
            )
            agent = DownloadDataAgent(FakeSearch(), model, store, downloader)
            first = agent.run(
                TaskEnvelope(
                    run_id="run_cache",
                    task_id="task_data_first",
                    agent=AgentName.DOWNLOAD_DATA,
                    objective="cache test",
                )
            )
            cached = agent._verified_cached_records(
                TaskEnvelope(
                    run_id="run_cache",
                    task_id="task_data_second",
                    agent=AgentName.DOWNLOAD_DATA,
                    objective="cache test",
                    inputs=[first.artifacts[0]],
                ),
                [{"dataset_key": "dataset-davis"}],
            )
        self.assertEqual(set(cached), {"dataset-davis"})
        self.assertIn("revalidating", cached["dataset-davis"].reason)

    def test_data_agent_generates_server_script_above_ten_gib(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(
                {
                    "candidates": [
                        {
                            "dataset_id": "dataset-large",
                            "name": "Large public dataset",
                            "description": "A dataset that must be fetched on the server.",
                            "source": "Example Repository",
                            "landing_page_url": "https://example.com/data",
                            "direct_download_url": "https://example.com/data.zip",
                            "license_id": "cc-by-4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "license_evidence": "Licensed under CC BY 4.0.",
                            "format": "zip",
                            "estimated_size_bytes": DATA_MAX_BYTES + 1,
                            "relevance_score": 0.9,
                        }
                    ]
                }
            )
            downloader = FakeDownloader(None)
            result = DownloadDataAgent(
                FakeSearch(), model, ArtifactStore(directory), downloader
            ).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="task_large_data",
                    agent=AgentName.DOWNLOAD_DATA,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "needs_human")
            self.assertEqual(downloader.calls, [])
            manifest_path = Path(directory) / result.artifacts[0].uri.removeprefix(
                "artifact://"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = manifest["records"][0]
            self.assertEqual(record["status"], "server_script_required")
            self.assertEqual(manifest["server_script_count"], 1)
            script_path = Path(directory) / record["server_download_script"][
                "uri"
            ].removeprefix("artifact://")
            self.assertTrue(script_path.is_file())
            self.assertIn("curl --fail", script_path.read_text(encoding="utf-8"))

    def test_make_idea_then_idea_design_write_typed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            policy = DownloadPolicyRecord(
                max_bytes=1024,
                max_redirects=1,
                allowed_content_types=["application/pdf"],
            )
            records = []
            for index, paper_id in enumerate(("paper-one", "paper-two"), start=1):
                paper_buffer = io.BytesIO()
                paper_writer = PdfWriter()
                paper_writer.add_blank_page(width=72 + index, height=72)
                paper_writer.write(paper_buffer)
                paper_payload = paper_buffer.getvalue()
                paper_artifact = store.write_agent_bytes(
                    AgentName.DOWNLOAD_PAPER,
                    f"reference/papers/{paper_id}.pdf",
                    paper_payload,
                    max_bytes=1024,
                    kind="paper_pdf",
                )
                records.append(
                    PaperRecord(
                        paper_id=paper_id,
                        title=f"Grounded affinity paper {index}",
                        abstract="Evidence for a testable affinity prediction gap.",
                        source="Bioinformatics",
                        final_url=f"https://example.com/{paper_id}.pdf",
                        content_type="application/pdf",
                        size_bytes=len(paper_payload),
                        sha256=paper_artifact.sha256,
                        local_artifact=paper_artifact,
                        status="downloaded",
                        reason="verified",
                        github_repository_url=(
                            "https://github.com/example/affinity"
                            if paper_id == "paper-one"
                            else None
                        ),
                        github_status=(
                            "verified_open_source"
                            if paper_id == "paper-one"
                            else "not_found"
                        ),
                        github_license_id="MIT" if paper_id == "paper-one" else None,
                    )
                )
            paper_manifest = PaperManifest(
                run_id="run_test",
                task_id="paper_task",
                objective="test topic",
                query="paper query",
                search_result_count=2,
                records=records,
                downloaded_count=2,
                discovered_count=0,
                requires_review_count=0,
                github_marked_count=1,
                policy=policy,
            )
            paper_ref = store.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                "reference/manifests/paper.json",
                paper_manifest.model_dump(mode="json"),
            ).model_copy(update={"kind": "paper_manifest"})
            model = FakeModel(valid_idea_response())
            result = MakeIdeaAgent(model, store).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="idea_task",
                    agent=AgentName.MAKE_IDEA,
                    objective="test topic",
                    inputs=[paper_ref],
                )
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.artifacts[0].kind, "idea_proposal")
            proposal_path = Path(directory) / result.artifacts[0].uri.removeprefix("artifact://")
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertEqual(
                proposal["candidates"][0]["idea"]["evidence_refs"],
                ["paper-one", "paper-two"],
            )
            self.assertFalse(proposal["degraded"])
            idea_request = json.loads(model.calls[0][1])
            self.assertIn("response_contract", idea_request)
            self.assertEqual(
                idea_request["allowed_paper_ids"], ["paper-one", "paper-two"]
            )

            design_model = IdeaDesignFakeModel()
            design_result = IdeaDesignAgent(design_model, store).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="design_task",
                    agent=AgentName.IDEA_DESIGN,
                    objective="test topic",
                    inputs=[paper_ref, result.artifacts[0]],
                )
            )
            self.assertEqual(design_result.status, "succeeded")
            self.assertEqual(design_result.artifacts[0].kind, "research_design")
            design_path = Path(directory) / design_result.artifacts[0].uri.removeprefix("artifact://")
            design = json.loads(design_path.read_text(encoding="utf-8"))
            self.assertEqual(design["selected_idea_id"], "idea-one")
            self.assertTrue(all(item["computational_only"] for item in design["experiment_plans"]))
            self.assertEqual(design["expected_results"][0]["claim_status"], "predicted_not_observed")
            self.assertEqual(len(design["advisor_consensus"]), 5)
            self.assertEqual(len(design_model.calls), 12)
            synthesis_sections = [
                json.loads(user_prompt).get("section_name")
                for _, user_prompt in design_model.calls
                if json.loads(user_prompt).get("internal_stage")
                == "bounded_research_design_section"
            ]
            self.assertCountEqual(
                synthesis_sections,
                [
                    "method-and-code",
                    "data-and-comparison",
                    "experiments-and-results",
                    "design-decisions",
                    "statistical-analysis",
                    "experiment-stages",
                    "roadmap-and-risks",
                ],
            )
            experiment_request = next(
                json.loads(user_prompt)
                for _, user_prompt in design_model.calls
                if json.loads(user_prompt).get("section_name")
                == "experiments-and-results"
            )
            constraints = " ".join(experiment_request["section_constraints"])
            self.assertIn("random_seeds", constraints)
            self.assertIn("even to negate them", constraints)
            self.assertIn("predicted_not_observed", constraints)
            self.assertIn('"dataset-davis"', constraints)
            self.assertIn("Do not invent synthetic", constraints)
            self.assertIn("separate diagnostic split", constraints)
            self.assertEqual(design["statistical_analysis"]["minimum_random_seeds"], 3)
            self.assertEqual(
                design["statistical_analysis"]["random_seeds"], [7, 17, 27]
            )

            cached_result = IdeaDesignAgent(design_model, store).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="design_task_retry",
                    agent=AgentName.IDEA_DESIGN,
                    objective="test topic",
                    inputs=[paper_ref, result.artifacts[0]],
                )
            )
            self.assertEqual(cached_result.status, "succeeded")
            self.assertEqual(
                len(design_model.calls),
                12,
                "advisor and synthesis sections should load from validated local cache",
            )

    def test_make_idea_blocks_without_paper_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(valid_idea_response())
            result = MakeIdeaAgent(model, ArtifactStore(directory)).run(
                TaskEnvelope(
                    run_id="run_test",
                    task_id="idea_task",
                    agent=AgentName.MAKE_IDEA,
                    objective="test topic",
                )
            )
            self.assertEqual(result.status, "blocked")
            self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
