import unittest
from types import SimpleNamespace

from agents.download_paper_agent.policy import (
    RECENT_YEAR_FLOOR,
    canonical_authoritative_venue,
    classify_paper_quality,
    github_repository_from_results,
    portfolio_targets,
)
from agents.paper_review_agent.policy import (
    deterministic_manuscript_findings,
    repair_route,
)
from research_agents.schemas import AgentName


class PaperQualityPolicyTests(unittest.TestCase):
    @staticmethod
    def _manuscript(*, citation_titles, text, comparison_count=2, claim_strength="descriptive"):
        artifact = SimpleNamespace(uri="artifact://project/paper/manuscript.md")
        return SimpleNamespace(
            title="Conformal Drug-Target Affinity Study",
            abstract=text,
            sections=[],
            claims=[
                SimpleNamespace(
                    claim_id="claim-experiment",
                    claim_type="experiment",
                    text=text,
                    strength=claim_strength,
                    result_keys=["result-one"] if claim_strength == "comparative" else [],
                )
            ],
            citations=[
                SimpleNamespace(paper_id=f"paper-{index}", title=title)
                for index, title in enumerate(citation_titles, start=1)
            ],
            markdown_artifact=artifact,
            latex_artifact=SimpleNamespace(uri="artifact://project/paper/manuscript.tex"),
            bibliography_artifact=SimpleNamespace(uri="artifact://project/paper/references.bib"),
            reproducibility_bundle_artifact=None,
            closest_prior_comparisons=[
                SimpleNamespace(paper_id=f"paper-{index}")
                for index in range(1, comparison_count + 1)
            ],
        )

    def test_deterministic_review_blocks_unsupported_truth_claims(self):
        manuscript = self._manuscript(
            citation_titles=["A generic affinity model"],
            comparison_count=0,
            text=(
                "Our preregistered conformal drug-target affinity study proves the result, "
                "and an artifact bundle accompanies the manuscript."
            ),
        )
        findings = deterministic_manuscript_findings(manuscript)
        identifiers = {item.finding_id for item in findings}
        self.assertEqual(
            identifiers,
            {
                "review-deterministic-closest-prior-comparisons",
                "review-deterministic-preregistration-evidence",
                "review-deterministic-reproducibility-bundle",
                "review-deterministic-proof-language",
            },
        )
        self.assertEqual(repair_route(findings), AgentName.WRITE_PAPER)

    def test_deterministic_review_accepts_direct_prior_art_when_claims_are_bounded(self):
        manuscript = self._manuscript(
            citation_titles=[
                "Dynamic applicability domain for compound-target binding affinity estimates with local conformal prediction",
                "Conformal Prediction for Uncertainty Estimation in Drug-Target Interaction Prediction",
            ],
            text=(
                "This conformal drug-target affinity comparison is pre-specified in a versioned "
                "configuration and remains limited to the evaluated protocol."
            ),
        )
        self.assertEqual(deterministic_manuscript_findings(manuscript), [])

    def test_future_preregistration_recommendation_is_not_a_current_claim(self):
        manuscript = self._manuscript(
            citation_titles=[
                "Dynamic applicability domain for compound-target binding affinity estimates with local conformal prediction",
                "Conformal Prediction for Uncertainty Estimation in Drug-Target Interaction Prediction",
            ],
            text=(
                "This conformal drug-target affinity pilot remains bounded. The next study should be "
                "preregistered before a full benchmark is executed."
            ),
        )
        self.assertEqual(deterministic_manuscript_findings(manuscript), [])

    def test_superiority_language_requires_result_bound_comparative_claim(self):
        unsupported = self._manuscript(
            citation_titles=["Prior method one", "Prior method two"],
            text="Our method outperforms all baselines under the evaluated protocol.",
        )
        identifiers = {
            item.finding_id for item in deterministic_manuscript_findings(unsupported)
        }
        self.assertIn("review-deterministic-unsupported-superiority", identifiers)

        supported = self._manuscript(
            citation_titles=["Prior method one", "Prior method two"],
            text="Our method outperforms the matched baseline under the evaluated protocol.",
            claim_strength="comparative",
        )
        self.assertEqual(deterministic_manuscript_findings(supported), [])

    def test_canonical_venues_are_boundary_aware(self):
        self.assertEqual(canonical_authoritative_venue("NeurIPS 2025"), "NeurIPS")
        self.assertEqual(canonical_authoritative_venue("Bioinformatics"), "Bioinformatics")
        self.assertIsNone(canonical_authoritative_venue("Molecular Systems"))

    def test_quality_tiers_distinguish_core_established_and_recent(self):
        self.assertEqual(
            classify_paper_quality("Bioinformatics", None, 2019),
            ("authoritative", "Bioinformatics"),
        )
        self.assertEqual(
            classify_paper_quality("BMC Bioinformatics", None, 2020),
            ("established_peer_reviewed", "BMC Bioinformatics"),
        )
        self.assertEqual(
            classify_paper_quality("arXiv", None, RECENT_YEAR_FLOOR),
            ("recent_arxiv", None),
        )

    def test_portfolio_target_is_seven_to_three(self):
        self.assertEqual(portfolio_targets(10), (7, 3))

    def test_github_match_requires_title_or_method_evidence(self):
        results = [
            {
                "title": "DeepDTA source",
                "url": "https://github.com/example/DeepDTA",
                "content": "Source code for DeepDTA: deep drug-target binding affinity prediction",
            },
            {
                "title": "Unrelated DTA project",
                "url": "https://github.com/example/unrelated",
                "content": "Generic graph model",
            },
        ]
        repository, evidence = github_repository_from_results(
            "DeepDTA: deep drug-target binding affinity prediction", results
        )
        self.assertEqual(repository, "https://github.com/example/DeepDTA")
        self.assertEqual(evidence, results[0]["url"])


if __name__ == "__main__":
    unittest.main()
