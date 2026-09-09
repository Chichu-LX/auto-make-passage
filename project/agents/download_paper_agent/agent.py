from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.downloads import DownloadPolicyError, SafeHttpDownloader
from research_agents.providers import JsonModel, SearchProvider
from research_agents.project_layout import artifact_prefix, scoped_path, task_project_slug
from research_agents.schemas import (
    AgentName,
    AgentResult,
    DownloadPolicyRecord,
    PaperManifest,
    PaperCandidate,
    PaperQualityPolicyRecord,
    PaperRecord,
    PaperSelection,
    ResearchDesign,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    AUTHORITATIVE_RATIO,
    AUTHORITATIVE_VENUES,
    PAPER_POLICY,
    PORTFOLIO_SIZE,
    RECENT_RATIO,
    RECENT_YEAR_FLOOR,
    bounded_search_results,
    classify_paper_quality,
    file_token,
    github_repository_from_results,
    portfolio_targets,
    priority_search_queries,
)


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.DOWNLOAD_PAPER:
    raise RuntimeError("Download Paper Agent package name mismatch")


class DownloadPaperAgent:
    """Discover papers, validate model metadata, and download direct PDFs safely."""

    name = AgentName.DOWNLOAD_PAPER
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(
        self,
        search: SearchProvider,
        model: JsonModel,
        artifacts: ArtifactStore,
        downloader: SafeHttpDownloader | None = None,
    ) -> None:
        self.search = search
        self.model = model
        self.artifacts = artifacts
        self.downloader = downloader or SafeHttpDownloader()

    def _select_candidates(
        self,
        task: TaskEnvelope,
        search_results: list[dict[str, Any]],
        required_papers: list[dict[str, str]] | None = None,
    ) -> PaperSelection:
        response = self.model.complete_json(
            PACKAGE.system_prompt,
            json.dumps(
                {
                    "objective": task.objective,
                    "user_revision": task.context.get("user_revision"),
                    "required_paper_repairs": required_papers or [],
                    "untrusted_search_results": search_results,
                    "response_contract": PaperSelection.model_json_schema(),
                },
                ensure_ascii=False,
            ),
        )
        normalized = self._normalize_selection(response)
        candidates: list[PaperCandidate] = []
        for item in normalized["candidates"]:
            try:
                candidates.append(PaperCandidate.model_validate(item))
            except ValidationError:
                continue
        return PaperSelection(candidates=candidates)

    @staticmethod
    def _normalize_selection(response: dict[str, Any]) -> dict[str, Any]:
        """Normalize common OpenAI-compatible JSON naming drift before strict validation."""

        raw_candidates = response.get("candidates")
        if not isinstance(raw_candidates, list):
            return {"candidates": []}
        candidates: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for index, raw in enumerate(raw_candidates[:10], start=1):
            if not isinstance(raw, dict):
                continue
            title = raw.get("title")
            source = raw.get("source") or raw.get("venue")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(source, str) or not source.strip():
                continue
            raw_id = str(raw.get("paper_id") or f"candidate-{index}").lower()
            token = re.sub(r"[^a-z0-9_-]+", "-", raw_id).strip("-_")[:56]
            token = token.removeprefix("paper-") or f"candidate-{index}"
            paper_id = f"paper-{token}"
            if paper_id in used_ids:
                paper_id = f"paper-{token}-{index}"
            used_ids.add(paper_id)
            publication_raw = str(raw.get("publication_type") or "other").lower()
            if "conference" in publication_raw or "proceeding" in publication_raw:
                publication_type = "conference"
            elif "journal" in publication_raw or "article" in publication_raw:
                publication_type = "journal"
            elif "preprint" in publication_raw or "arxiv" in publication_raw:
                publication_type = "preprint"
            else:
                publication_type = "other"
            year = raw.get("year")
            if isinstance(year, str) and year.strip().isdigit():
                year = int(year.strip())
            if not isinstance(year, int):
                year = None
            score = raw.get("relevance_score", 0.0)
            if not isinstance(score, (int, float)):
                score = 0.0
            authors = raw.get("authors")
            if not isinstance(authors, list):
                authors = []
            abstract = raw.get("abstract")
            if not isinstance(abstract, str):
                abstract = ""
            candidates.append(
                {
                    "paper_id": paper_id,
                    "title": title.strip(),
                    "authors": [str(value) for value in authors[:100]],
                    "year": year,
                    "doi": raw.get("doi") or raw.get("DOI"),
                    "abstract": abstract,
                    "source": source.strip(),
                    "venue": raw.get("venue"),
                    "publication_type": publication_type,
                    "landing_page_url": raw.get("landing_page_url")
                    or raw.get("landing_page"),
                    "direct_pdf_url": raw.get("direct_pdf_url") or raw.get("pdf_url"),
                    "relevance_score": min(1.0, max(0.0, float(score))),
                }
            )
        return {"candidates": candidates}

    @staticmethod
    def _normalized_title(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    @classmethod
    def _repair_candidate_source(
        cls,
        candidate: PaperCandidate,
        search_results: list[dict[str, Any]],
        required_ids: set[str],
    ) -> PaperCandidate:
        """Prefer exact-title authority evidence over an indirect PubMed landing page."""

        if candidate.paper_id not in required_ids:
            return candidate
        candidate_title = cls._normalized_title(candidate.title)
        matches = []
        domain_rank = {
            "pmc.ncbi.nlm.nih.gov": 0,
            "europepmc.org": 1,
            "nature.com": 2,
            "www.nature.com": 2,
            "academic.oup.com": 3,
            "biorxiv.org": 4,
            "www.biorxiv.org": 4,
            "pubs.rsc.org": 5,
        }
        for item in search_results:
            if item.get("required_paper_id") != candidate.paper_id:
                continue
            title = cls._normalized_title(str(item.get("title") or ""))
            if not title or not (
                title == candidate_title
                or title in candidate_title
                or candidate_title in title
            ):
                continue
            url = str(item.get("url") or "")
            host = (urlsplit(url).hostname or "").lower()
            matches.append((domain_rank.get(host, 99), url))
        if not matches:
            return candidate
        _, evidence_url = min(matches)
        path = urlsplit(evidence_url).path.lower()
        direct_pdf_url = candidate.direct_pdf_url
        landing_page_url = evidence_url
        if path.endswith(".pdf") or ".full.pdf" in path:
            direct_pdf_url = evidence_url
            landing_page_url = candidate.landing_page_url or evidence_url
        return candidate.model_copy(
            update={
                "landing_page_url": landing_page_url,
                "direct_pdf_url": direct_pdf_url,
            }
        )

    def _search_by_priority(
        self,
        objective: str,
        required_papers: list[dict[str, str]] | None = None,
        *,
        include_portfolio: bool = True,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        combined: list[dict[str, Any]] = []
        queries: list[str] = []
        seen_urls: set[str] = set()
        for paper in required_papers or []:
            query = (
                f'"{paper["title"]}" open access PDF '
                "PMC OR Europe PMC OR Nature OR Bioinformatics OR bioRxiv"
            )
            queries.append(query)
            for item in bounded_search_results(
                self.search.search(query, max_results=10),
                phase="required_repair",
                limit=5,
            ):
                url = item["url"]
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                combined.append(
                    {
                        **item,
                        "required_paper_id": paper["paper_id"],
                        "required_title": paper["title"],
                    }
                )
        if include_portfolio:
            for phase, query in priority_search_queries(objective):
                queries.append(query)
                for item in bounded_search_results(
                    self.search.search(query, max_results=10), phase=phase, limit=10
                ):
                    url = item["url"]
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    combined.append(item)
        return combined[:80], queries

    def _verified_historical_downloads(
        self, task: TaskEnvelope
    ) -> dict[str, PaperRecord]:
        """Load only same-run PDF records whose registered local bytes still verify."""

        slug = task_project_slug(task)
        verified: dict[str, PaperRecord] = {}
        for artifact in task.inputs:
            if artifact.kind != "paper_manifest" or not artifact.uri.startswith(
                artifact_prefix("reference", slug, "manifests")
            ):
                continue
            try:
                manifest = PaperManifest.model_validate(
                    self.artifacts.read_artifact_json(
                        artifact, max_bytes=2 * 1024 * 1024
                    )
                )
                if manifest.run_id != task.run_id:
                    continue
                for record in manifest.records:
                    local = record.local_artifact
                    if (
                        record.status != "downloaded"
                        or local is None
                        or not local.uri.startswith(
                            artifact_prefix("reference", slug, "papers")
                        )
                        or record.content_type != "application/pdf"
                        or record.sha256 != local.sha256
                    ):
                        continue
                    payload = self.artifacts.read_artifact_bytes(
                        local, max_bytes=PAPER_POLICY.max_bytes
                    )
                    if not payload.startswith(b"%PDF-"):
                        continue
                    if record.size_bytes is not None and record.size_bytes != len(payload):
                        continue
                    verified[record.paper_id] = record
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        return verified

    def _historical_paper_metadata(
        self, task: TaskEnvelope
    ) -> dict[str, PaperRecord]:
        """Select the strongest registered same-run metadata for each paper ID."""

        slug = task_project_slug(task)
        candidates: dict[str, list[PaperRecord]] = {}
        for artifact in task.inputs:
            if artifact.kind != "paper_manifest" or not artifact.uri.startswith(
                artifact_prefix("reference", slug, "manifests")
            ):
                continue
            try:
                manifest = PaperManifest.model_validate(
                    self.artifacts.read_artifact_json(
                        artifact, max_bytes=2 * 1024 * 1024
                    )
                )
                if manifest.run_id != task.run_id:
                    continue
                for record in manifest.records:
                    candidates.setdefault(record.paper_id, []).append(record)
            except (StoragePolicyError, ValidationError, ValueError):
                continue

        def score(record: PaperRecord) -> tuple[int, int, int, int, int]:
            return (
                int(record.canonical_venue is not None),
                int(record.venue is not None),
                int(record.year is not None),
                int(record.doi is not None),
                len(record.abstract),
            )

        return {
            paper_id: max(records, key=score)
            for paper_id, records in candidates.items()
        }

    @staticmethod
    def _enrich_repair_metadata(
        candidate: PaperCandidate,
        historical: PaperRecord | None,
        required_ids: set[str],
    ) -> PaperCandidate:
        """Fill missing fields only from a registered same-run exact-ID record."""

        if candidate.paper_id not in required_ids or historical is None:
            return candidate
        return candidate.model_copy(
            update={
                "authors": candidate.authors or historical.authors,
                "year": candidate.year if candidate.year is not None else historical.year,
                "doi": candidate.doi or historical.doi,
                "abstract": candidate.abstract or historical.abstract,
                "venue": candidate.venue or historical.venue or historical.canonical_venue,
                "publication_type": (
                    historical.publication_type
                    if candidate.publication_type == "other"
                    else candidate.publication_type
                ),
            }
        )

    def _required_paper_repairs(self, task: TaskEnvelope) -> list[dict[str, str]]:
        """Resolve typed Write Paper gaps to exact IDs and prior manifest titles."""

        slug = task_project_slug(task)
        required_ids: set[str] = set()
        design_ids: set[str] = set()
        downloaded_ids: set[str] = set()
        titles: dict[str, str] = {}
        literature_gap = False
        for artifact in task.inputs:
            try:
                if artifact.kind == "paper_writing_gap" and artifact.uri.startswith(
                    f"artifact://project/paper/{slug}/gaps/"
                ):
                    payload = self.artifacts.read_artifact_json(
                        artifact, max_bytes=128 * 1024
                    )
                    if payload.get("run_id") != task.run_id:
                        continue
                    literature_gap = (
                        payload.get("issue_code")
                        == "PAPER_WRITE_LITERATURE_EVIDENCE_MISSING"
                    ) or literature_gap
                    values = payload.get("missing_paper_ids", [])
                    if isinstance(values, list):
                        required_ids.update(
                            value
                            for value in values
                            if isinstance(value, str)
                            and re.fullmatch(r"paper-[a-zA-Z0-9_-]{1,64}", value)
                        )
                elif artifact.kind == "paper_manifest" and artifact.uri.startswith(
                    artifact_prefix("reference", slug, "manifests")
                ):
                    manifest = PaperManifest.model_validate(
                        self.artifacts.read_artifact_json(
                            artifact, max_bytes=2 * 1024 * 1024
                        )
                    )
                    if manifest.run_id != task.run_id:
                        continue
                    for record in manifest.records:
                        titles[record.paper_id] = record.title
                        if record.status == "downloaded":
                            downloaded_ids.add(record.paper_id)
                elif artifact.kind == "research_design" and artifact.uri.startswith(
                    artifact_prefix("design", slug, "research-designs")
                ):
                    design = ResearchDesign.model_validate(
                        self.artifacts.read_artifact_json(
                            artifact, max_bytes=2 * 1024 * 1024
                        )
                    )
                    if design.run_id != task.run_id:
                        continue
                    design_ids.update(
                        ref
                        for component in design.method_components
                        for ref in component.evidence_refs
                    )
                    design_ids.update(
                        ref
                        for requirement in design.dataset_requirements
                        for ref in requirement.paper_refs
                    )
                    design_ids.update(
                        ref
                        for experiment in design.experiment_plans
                        for ref in experiment.evidence_refs
                    )
                    design_ids.update(
                        ref
                        for result in design.expected_results
                        for ref in result.evidence_refs
                    )
                    design_ids.update(item.paper_id for item in design.paper_comparisons)
                    design_ids.update(
                        item.source_paper_id for item in design.reusable_code_modules
                    )
                    design_ids.update(
                        ref
                        for decision in design.design_decisions
                        for ref in decision.evidence_refs
                    )
                    design_ids.update(
                        ref
                        for report in design.advisor_consensus
                        for ref in report.evidence_refs
                    )
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        if literature_gap:
            # Keep the full approved design set as the repair contract. The search
            # stage separately subtracts verified historical downloads, while the
            # output stage merges them into one source-linked manifest for writing.
            required_ids.update(design_ids)
        return [
            {"paper_id": paper_id, "title": titles.get(paper_id, paper_id)}
            for paper_id in sorted(required_ids)
        ]

    def _github_evidence(self, title: str) -> dict[str, Any]:
        query = f'"{title}" GitHub source code repository'
        results = bounded_search_results(
            self.search.search(query, max_results=5), phase="github", limit=5
        )
        repository_url, evidence_url = github_repository_from_results(title, results)
        if not repository_url:
            return {
                "github_repository_url": None,
                "github_evidence_url": None,
                "github_status": "not_found",
                "github_license_id": None,
            }
        return {
            "github_repository_url": repository_url,
            "github_evidence_url": evidence_url,
            "github_status": "search_matched",
            "github_license_id": None,
        }

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        revision = task.context.get("user_revision")
        revision_comment = (
            str(revision.get("comments", ""))[:1200]
            if isinstance(revision, dict)
            else ""
        )
        search_objective = " ".join(
            item for item in (task.objective, revision_comment) if item
        )
        required_papers = self._required_paper_repairs(task)
        historical_downloads = self._verified_historical_downloads(task)
        historical_metadata = self._historical_paper_metadata(task)
        missing_repairs = [
            item
            for item in required_papers
            if item["paper_id"] not in historical_downloads
        ]
        search_results, queries = self._search_by_priority(
            search_objective,
            missing_repairs,
            include_portfolio=not bool(required_papers),
        )
        query = " || ".join(queries)
        issues: list[dict[str, str]] = []
        if required_papers and not missing_repairs:
            selection = PaperSelection()
        else:
            try:
                selection = self._select_candidates(
                    task, search_results, missing_repairs or None
                )
            except (ValidationError, ValueError, TypeError) as exc:
                selection = PaperSelection()
                issues.append(
                    {
                        "severity": "blocking",
                        "code": "PAPER_METADATA_INVALID",
                        "message": f"DeepSeek paper metadata failed schema validation: {type(exc).__name__}",
                        "root_cause_key": "G1:PAPER_METADATA_INVALID:selection",
                    }
                )

        records: list[PaperRecord] = []
        downloaded_by_digest: dict[str, str] = {}
        task_token = file_token(task.task_id)
        authoritative_target, recent_target = portfolio_targets(PORTFOLIO_SIZE)
        bucket_downloads = {"authoritative": 0, "recent": 0, "fallback": 0}
        required_ids = {item["paper_id"] for item in required_papers}

        classified: list[tuple[int, Any, str, str | None, str]] = []
        priority = {
            "authoritative": 0,
            "established_peer_reviewed": 1,
            "recent_peer_reviewed": 2,
            "recent_arxiv": 3,
            "arxiv_fallback": 4,
            "other": 5,
        }
        for original_candidate in selection.candidates:
            candidate = self._repair_candidate_source(
                original_candidate, search_results, required_ids
            )
            candidate = self._enrich_repair_metadata(
                candidate,
                historical_metadata.get(candidate.paper_id),
                required_ids,
            )
            quality_tier, canonical_venue = classify_paper_quality(
                candidate.source, candidate.venue, candidate.year
            )
            bucket = (
                "authoritative"
                if quality_tier in {"authoritative", "established_peer_reviewed"}
                else "recent"
                if quality_tier in {"recent_peer_reviewed", "recent_arxiv"}
                else "fallback"
                if quality_tier == "arxiv_fallback"
                else "ineligible"
            )
            classified.append(
                (priority[quality_tier], candidate, quality_tier, canonical_venue, bucket)
            )

        for _, candidate, quality_tier, canonical_venue, bucket in sorted(
            classified, key=lambda item: (item[0], -item[1].relevance_score)
        ):
            base = candidate.model_dump(mode="json")
            is_required_repair = candidate.paper_id in required_ids
            # GitHub availability is useful even when a candidate's PDF later fails
            # the binary gate. Repository status remains search_matched until an
            # explicit license check upgrades it to verified_open_source.
            github = self._github_evidence(candidate.title)
            quality = {
                "quality_tier": quality_tier,
                "canonical_venue": canonical_venue,
                "portfolio_bucket": bucket,
            }
            if bucket == "ineligible":
                records.append(
                    PaperRecord(
                        **base,
                        **quality,
                        **github,
                        status="rejected",
                        reason="Venue is neither authoritative nor inside the recent-paper window.",
                    )
                )
                continue
            direct_pdf_url = candidate.direct_pdf_url
            if not direct_pdf_url and candidate.landing_page_url:
                try:
                    resolver = getattr(
                        self.downloader, "resolve_citation_pdf_url", None
                    )
                    if callable(resolver):
                        direct_pdf_url = resolver(candidate.landing_page_url)
                except DownloadPolicyError:
                    direct_pdf_url = None
            if not direct_pdf_url:
                records.append(
                    PaperRecord(
                        **base,
                        **quality,
                        **github,
                        status="rejected",
                        reason="Rejected by the hard PDF gate: no direct PDF URL was supplied.",
                    )
                )
                continue
            if (
                not is_required_repair
                and bucket == "authoritative"
                and bucket_downloads[bucket] >= authoritative_target
            ):
                records.append(
                    PaperRecord(**base, **quality, **github, status="rejected", reason="Authoritative 70% quota is already full.")
                )
                continue
            if (
                not is_required_repair
                and bucket == "recent"
                and bucket_downloads[bucket] >= recent_target
            ):
                records.append(
                    PaperRecord(**base, **quality, **github, status="rejected", reason="Recent 30% quota is already full.")
                )
                continue
            try:
                payload = self.downloader.fetch(direct_pdf_url, PAPER_POLICY)
                duplicate_of = downloaded_by_digest.get(payload.sha256)
                if duplicate_of:
                    records.append(
                        PaperRecord(
                            **base,
                            **quality,
                            **github,
                            status="duplicate",
                            reason="Binary content duplicates an earlier paper in this manifest.",
                            final_url=payload.final_url,
                            content_type=payload.content_type,
                            size_bytes=payload.size_bytes,
                            sha256=payload.sha256,
                            duplicate_of=duplicate_of,
                        )
                    )
                    continue
                relative_path = scoped_path(
                    "reference",
                    task_project_slug(task),
                    "papers",
                    f"{task_token}_{candidate.paper_id}.pdf",
                )
                artifact = self.artifacts.write_agent_bytes(
                    self.name,
                    relative_path,
                    payload.content,
                    max_bytes=PAPER_POLICY.max_bytes,
                    kind="paper_pdf",
                )
                downloaded_by_digest[payload.sha256] = candidate.paper_id
                records.append(
                    PaperRecord(
                        **base,
                        **quality,
                        **github,
                        status="downloaded",
                        reason="Direct PDF passed URL, MIME, size, redirect, signature, and venue/time policy checks.",
                        final_url=payload.final_url,
                        content_type=payload.content_type,
                        size_bytes=payload.size_bytes,
                        sha256=payload.sha256,
                        local_artifact=artifact,
                    )
                )
                bucket_downloads[bucket] += 1
            except DownloadPolicyError as exc:
                records.append(
                    PaperRecord(
                        **base,
                        **quality,
                        **github,
                        status="requires_review",
                        reason=str(exc),
                    )
                )

        # A targeted repair produces one complete evidence manifest, not a sequence
        # of mutually incomplete deltas. Reuse only locally re-verified records from
        # this run, and prefer a newly downloaded record for the same paper ID.
        if required_ids:
            current_by_id = {record.paper_id: record for record in records}
            for paper_id in required_ids:
                historical = historical_downloads.get(paper_id)
                current = current_by_id.get(paper_id)
                if historical is not None and (
                    current is None or current.status != "downloaded"
                ):
                    current_by_id[paper_id] = historical
            records = list(current_by_id.values())

        downloaded_count = sum(item.status == "downloaded" for item in records)
        downloaded_ids = {
            item.paper_id for item in records if item.status == "downloaded"
        }
        unresolved_required_ids = sorted(required_ids - downloaded_ids)
        discovered_count = sum(item.status == "discovered" for item in records)
        review_count = sum(item.status == "requires_review" for item in records)
        rejected_count = sum(item.status == "rejected" for item in records)
        authoritative_count = sum(
            item.status == "downloaded" and item.portfolio_bucket == "authoritative"
            for item in records
        )
        recent_count = sum(
            item.status == "downloaded" and item.portfolio_bucket == "recent"
            for item in records
        )
        github_count = sum(item.github_status != "not_found" for item in records)
        portfolio_compliant = (
            downloaded_count == PORTFOLIO_SIZE
            and authoritative_count == authoritative_target
            and recent_count == recent_target
        )
        manifest = PaperManifest(
            run_id=task.run_id,
            task_id=task.task_id,
            objective=task.objective,
            query=query,
            search_result_count=len(search_results),
            records=records,
            downloaded_count=downloaded_count,
            discovered_count=discovered_count,
            requires_review_count=review_count,
            rejected_count=rejected_count,
            authoritative_downloaded_count=authoritative_count,
            recent_downloaded_count=recent_count,
            github_marked_count=github_count,
            portfolio_compliant=portfolio_compliant,
            manifest_purpose=(
                "evidence_consolidation" if required_ids else "portfolio_selection"
            ),
            required_paper_ids=sorted(required_ids),
            required_evidence_complete=(
                bool(required_ids) and not unresolved_required_ids
            ),
            policy=DownloadPolicyRecord(
                max_bytes=PAPER_POLICY.max_bytes,
                max_redirects=PAPER_POLICY.max_redirects,
                allowed_content_types=sorted(PAPER_POLICY.allowed_content_types),
            ),
            quality_policy=PaperQualityPolicyRecord(
                authoritative_ratio=AUTHORITATIVE_RATIO,
                recent_ratio=RECENT_RATIO,
                recent_since_year=RECENT_YEAR_FLOOR,
                portfolio_size=PORTFOLIO_SIZE,
                authoritative_venues=list(AUTHORITATIVE_VENUES),
                require_verified_pdf=True,
                arxiv_is_fallback=True,
            ),
        )
        manifest_ref = self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "reference",
                task_project_slug(task),
                "manifests",
                f"{task_token}.json",
            ),
            manifest.model_dump(mode="json"),
        ).model_copy(update={"kind": "paper_manifest"})

        if downloaded_count == 0 and not issues:
            issues.append(
                {
                    "severity": "blocking",
                    "code": "NO_VERIFIED_PDF",
                    "message": "No paper passed the automatic direct-PDF download policy.",
                    "root_cause_key": "G1:NO_VERIFIED_PDF:papers",
                }
            )
        if unresolved_required_ids:
            issues.append(
                {
                    "severity": "blocking",
                    "code": "REQUIRED_PDF_EVIDENCE_MISSING",
                    "message": (
                        "The targeted writing repair did not download every required PDF: "
                        + ", ".join(unresolved_required_ids)
                    ),
                    "root_cause_key": "G1:REQUIRED_PDF_EVIDENCE_MISSING:papers",
                }
            )
        succeeded = downloaded_count > 0 and not any(
            item["severity"] == "blocking" for item in issues
        )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded" if succeeded else "needs_human",
            summary=(
                f"论文候选 {len(records)} 篇；已验证下载 {downloaded_count} 篇，"
                f"权威 {authoritative_count} 篇、近期 {recent_count} 篇、GitHub 标注 {github_count} 篇；"
                f"拒绝 {rejected_count} 篇，需人工复核 {review_count} 篇。"
            ),
            artifacts=[manifest_ref],
            evidence=[
                {
                    "check": "paper-download-policy",
                    "downloaded": downloaded_count,
                    "discovered_only": discovered_count,
                    "requires_review": review_count,
                    "rejected": rejected_count,
                    "authoritative_downloaded": authoritative_count,
                    "recent_downloaded": recent_count,
                    "github_marked": github_count,
                    "portfolio_compliant": portfolio_compliant,
                    "manifest_purpose": (
                        "evidence_consolidation"
                        if required_ids
                        else "portfolio_selection"
                    ),
                    "required_evidence_complete": (
                        bool(required_ids) and not unresolved_required_ids
                    ),
                    "required_repair_count": len(required_ids),
                    "unresolved_required_paper_ids": unresolved_required_ids,
                }
            ],
            issues=issues,
            needs_human=not succeeded,
        )
