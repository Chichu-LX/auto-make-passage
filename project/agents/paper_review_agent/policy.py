from __future__ import annotations

import hashlib
import re

from research_agents.schemas import (
    AgentName,
    PaperManuscriptManifest,
    PaperReviewFinding,
)


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REVIEW_CONTEXT_BYTES = 2 * 1024 * 1024
MAX_REVIEW_REPORT_BYTES = 2 * 1024 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def repair_route(findings: list[PaperReviewFinding]) -> AgentName:
    if any(
        item.finding_id == "review-deterministic-closest-prior-comparisons"
        for item in findings
    ):
        return AgentName.WRITE_PAPER
    categories = {item.category for item in findings}
    if "novelty" in categories:
        return AgentName.MAKE_IDEA
    if "methodology" in categories:
        return AgentName.IDEA_DESIGN
    if categories.intersection({"experiments", "reproducibility", "results"}):
        return AgentName.EXPERIMENT_CODE_RUN
    if "citations" in categories:
        return AgentName.DOWNLOAD_PAPER
    return AgentName.WRITE_PAPER


def deterministic_manuscript_findings(
    manuscript: PaperManuscriptManifest,
) -> list[PaperReviewFinding]:
    """Find auditable claim defects that must not depend on reviewer-model recall."""

    section_text = " ".join(
        paragraph for section in manuscript.sections for paragraph in section.paragraphs
    )
    claim_text = " ".join(item.text for item in manuscript.claims)
    full_text = " ".join(
        (manuscript.title, manuscript.abstract, section_text, claim_text)
    ).casefold()
    findings: list[PaperReviewFinding] = []

    closest_comparisons = list(
        getattr(manuscript, "closest_prior_comparisons", []) or []
    )
    if len(closest_comparisons) < 2:
        findings.append(
            PaperReviewFinding(
                finding_id="review-deterministic-closest-prior-comparisons",
                category="novelty",
                severity="blocking",
                summary=(
                    "The manuscript lacks at least two structured closest-prior comparisons "
                    "that connect the shared problem or mechanism to the claimed delta and "
                    "the conditions required for a fair evaluation."
                ),
                evidence_refs=[item.paper_id for item in manuscript.citations[:100]],
                required_revision=(
                    "Add at least two cited closest-prior entries with shared elements, precise "
                    "proposed differences, reusable elements, and fair-comparison conditions; "
                    "then align the related-work and novelty claims to those entries."
                ),
            )
        )

    preregistration_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?。！？]+", full_text)
        if any(
            marker in sentence
            for marker in ("preregistered", "pre-registered", "pre registered", "预注册")
        )
    ]
    future_only_markers = (
        "future",
        "next study",
        "next step",
        "continuation",
        "should be",
        "should preregister",
        "remain necessary",
        "remains necessary",
        "before a full",
        "will be preregistered",
        "not evidence that",
        "not preregistered",
        "was not preregistered",
    )
    preregistration_claimed = any(
        not any(marker in sentence for marker in future_only_markers)
        for sentence in preregistration_sentences
    )
    registration_evidence = any(
        marker in full_text
        for marker in (
            "osf.io/",
            "aspredicted.org/",
            "as.predicted/",
            "registered report",
            "time-stamped registration",
            "timestamped registration",
            "registration identifier",
        )
    )
    if preregistration_claimed and not registration_evidence:
        findings.append(
            PaperReviewFinding(
                finding_id="review-deterministic-preregistration-evidence",
                category="reproducibility",
                severity="major",
                summary=(
                    "The manuscript claims preregistration without an external time-stamped registry "
                    "identifier or registered-report record in the manuscript evidence."
                ),
                evidence_refs=[manuscript.markdown_artifact.uri],
                required_revision=(
                    "Replace 'preregistered' with 'pre-specified in the versioned configuration' "
                    "unless an independently time-stamped registration record is supplied and cited."
                ),
            )
        )

    bundle_claimed = any(
        marker in full_text
        for marker in (
            "artifact bundle",
            "reproducibility bundle",
            "replication package",
            "reproduction package",
        )
    )
    archive_evidence = manuscript.reproducibility_bundle_artifact is not None
    if bundle_claimed and not archive_evidence:
        findings.append(
            PaperReviewFinding(
                finding_id="review-deterministic-reproducibility-bundle",
                category="reproducibility",
                severity="major",
                summary=(
                    "The manuscript claims a reproducibility/replication bundle, but the manifest "
                    "contains no content-addressed archive artifact supporting that claim."
                ),
                evidence_refs=[
                    manuscript.markdown_artifact.uri,
                    manuscript.latex_artifact.uri,
                    manuscript.bibliography_artifact.uri,
                ],
                required_revision=(
                    "Create and verify a content-addressed archive containing code, frozen configs, "
                    "split manifests, environment lock, raw-to-result provenance, and result tables, "
                    "or remove the bundle-availability claim."
                ),
            )
        )

    strong_claims = [
        item.claim_id
        for item in manuscript.claims
        if item.claim_type in {"method", "experiment"}
        and re.search(
            r"\b(?:prove[sd]?|proven|definitively establish(?:es|ed)?|confirm(?:s|ed)? causality)\b|证明|证实因果",
            item.text,
            flags=re.IGNORECASE,
        )
    ]
    if strong_claims:
        findings.append(
            PaperReviewFinding(
                finding_id="review-deterministic-proof-language",
                category="results",
                severity="major",
                summary=(
                    "Experimental or method claims use proof/causal-confirmation language that exceeds "
                    "the allowed descriptive, associational, or comparative evidence strengths."
                ),
                evidence_refs=strong_claims[:100],
                required_revision=(
                    "Replace proof language with a bounded statement of the observed comparison and "
                    "its uncertainty, explicitly limited to the evaluated data and protocol."
                ),
            )
        )
    superiority_language = bool(
        re.search(
            r"\b(?:state[- ]of[- ]the[- ]art|outperform(?:s|ed|ing)?|superior to|"
            r"improves? over|best[- ]performing)\b|优于|领先于|最佳性能",
            full_text,
            flags=re.IGNORECASE,
        )
    )
    supported_comparative_claim = any(
        item.claim_type == "experiment"
        and getattr(item, "strength", None) == "comparative"
        and bool(getattr(item, "result_keys", []))
        for item in manuscript.claims
    )
    if superiority_language and not supported_comparative_claim:
        findings.append(
            PaperReviewFinding(
                finding_id="review-deterministic-unsupported-superiority",
                category="results",
                severity="major",
                summary=(
                    "The manuscript uses superiority language without a comparative experiment "
                    "claim bound to observed result keys."
                ),
                evidence_refs=[manuscript.markdown_artifact.uri],
                required_revision=(
                    "Bind the comparison to observed results under matched evaluation conditions "
                    "and quantify uncertainty, or replace the superiority statement with a bounded description."
                ),
            )
        )
    return findings
