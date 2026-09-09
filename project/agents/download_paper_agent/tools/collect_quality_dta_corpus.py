#!/usr/bin/env python3
"""Build a PDF-complete, venue-prioritized 100-paper DTA corpus."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from agents.download_paper_agent.policy import (
    AUTHORITATIVE_VENUES,
    PAPER_POLICY,
    RECENT_YEAR_FLOOR,
    classify_paper_quality,
)
from agents.download_paper_agent.tools.collect_dta_corpus import (
    corpus_track,
    innovation_evidence,
    innovation_summary_zh,
    is_relevant,
    method_tags,
    normalized_title,
    relevance_score,
    same_work,
    utc_now,
    write_outputs,
)
from research_agents.downloads import DownloadPayload, DownloadPolicyError, SafeHttpDownloader
from research_agents.schemas import AgentName
from research_agents.storage import ArtifactStore


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = "reference/assaycal-dta/literature"
SEMANTIC_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
GITHUB_SEARCH = "https://api.github.com/search/repositories"
GITHUB_CACHE = WORKSPACE_ROOT / "reference" / ".cache" / "dta_github_catalog.json"
QUERIES = (
    "drug target affinity prediction",
    "drug target binding affinity prediction",
    "compound protein binding affinity prediction",
    "protein ligand binding affinity prediction",
    "deep learning drug target affinity",
    "machine learning protein ligand binding affinity",
    "structure based protein ligand binding affinity prediction",
    "PDBbind machine learning scoring function",
    "protein ligand binding free energy machine learning",
    "deep learning scoring function protein ligand",
    "graph neural network drug target affinity",
)
SEMANTIC_FIELDS = ",".join(
    (
        "title",
        "abstract",
        "venue",
        "year",
        "citationCount",
        "externalIds",
        "openAccessPdf",
        "url",
        "authors",
        "publicationDate",
        "publicationTypes",
    )
)
CACHE_ROOT = WORKSPACE_ROOT / "reference" / ".cache" / "semantic_scholar_dta"
PDF_CACHE_ROOT = WORKSPACE_ROOT / "reference" / ".cache" / "dta_verified_pdfs"
TRACKS = ("drug_target_dta", "structure_based_plba")
# The matrix preserves the earlier 50/50 topic balance while satisfying the
# user's global 70/30 quality-time split. DTA is the faster-moving subfield, so
# its half intentionally contains more recent papers.
GROUP_QUOTAS = {
    ("drug_target_dta", "authoritative"): 25,
    ("drug_target_dta", "recent"): 25,
    ("structure_based_plba", "authoritative"): 45,
    ("structure_based_plba", "recent"): 5,
}
GITHUB_QUERIES = (
    "drug target affinity prediction",
    "protein ligand binding affinity prediction",
    "DTA deep learning",
)
CURATED_RECENT_VENUES = frozenset(
    {
        "bmc biology",
        "npj digital medicine",
        "interdisciplinary sciences computational life sciences",
        "bmc medical genomics",
        "frontiers in genetics",
        "heliyon",
        "international conference on information and knowledge management",
        "communications chemistry",
        "pharmaceuticals",
        "bioengineering",
        "acs bio & med chem au",
        "epj quantum technology",
        "biophysical journal",
        "ieee open journal of engineering in medicine and biology",
        "database j. biol. databases curation",
        "current drug targets",
        "bmc chemistry",
        "journal of chemical theory and computation",
    }
)
OFFICIAL_PDF_HOSTS = {
    "aclanthology.org",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "ojs.aaai.org",
    "nature.com",
    "www.nature.com",
    "academic.oup.com",
    "jcheminf.biomedcentral.com",
    "bmcbioinformatics.biomedcentral.com",
    "pubs.acs.org",
}
VENUE_PRIORITY = {
    name: index
    for index, name in enumerate(
        (
            "NeurIPS",
            "ICML",
            "ICLR",
            "ACL",
            "EMNLP",
            "AAAI",
            "Nature Communications",
            "Nature Machine Intelligence",
            "Nature Methods",
            "Nature Computational Science",
            "Bioinformatics",
            "Briefings in Bioinformatics",
            "Journal of Chemical Information and Modeling",
            "Journal of Cheminformatics",
            "Nucleic Acids Research",
            "PLOS Computational Biology",
            "PNAS",
            "Science Advances",
            "Nature Biotechnology",
            "Communications Biology",
            "Patterns",
            "Chemical Science",
            "Computational and Structural Biotechnology Journal",
            "Journal of Computer-Aided Molecular Design",
            "WIREs Computational Molecular Science",
            "Information Sciences",
            "Molecular Informatics",
            "iScience",
            "Methods",
            "IEEE/ACM TCBB",
            "BMC Bioinformatics",
            "IJCAI",
            "KDD",
        )
    )
}


@dataclass(frozen=True)
class DownloadedCandidate:
    record: dict[str, Any]
    payload: DownloadPayload
    source_url: str
    attempts: list[dict[str, str]]


def _request_json(
    client: httpx.Client, url: str, *, params: dict[str, Any], attempts: int = 5
) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30.0, 2 ** attempt))
            continue
        if response.status_code != 429:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("remote API returned a non-object JSON payload")
            return payload
        delay = min(30.0, float(response.headers.get("retry-after") or (2 ** attempt)))
        time.sleep(delay)
    raise RuntimeError(f"rate limit persisted for {url}")


def semantic_search(query: str) -> list[dict[str, Any]]:
    cache_key = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    cache_path = CACHE_ROOT / f"{cache_key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            return [item for item in cached if isinstance(item, dict)]
    headers = {"User-Agent": "auto-make-passage/0.2 quality-paper-agent"}
    with httpx.Client(timeout=90.0, headers=headers, trust_env=False) as client:
        payload = _request_json(
            client,
            SEMANTIC_SEARCH,
            params={
                "query": query,
                "fields": SEMANTIC_FIELDS,
                "year": "2000-2026",
            },
        )
    data = payload.get("data") or []
    results = [item for item in data if isinstance(item, dict)]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(results, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results


def _inferred_venue(venue: str, doi: str) -> str:
    lower = doi.lower()
    mappings = (
        ("10.1093/bioinformatics/", "Bioinformatics"),
        ("10.1093/bib/", "Briefings in Bioinformatics"),
        ("10.1038/s41467", "Nature Communications"),
        ("10.1038/s42256", "Nature Machine Intelligence"),
        ("10.1186/s13321", "Journal of Cheminformatics"),
        ("10.1186/s12859", "BMC Bioinformatics"),
        ("10.1021/acs.jcim", "Journal of Chemical Information and Modeling"),
        ("10.1609/aaai", "AAAI"),
    )
    for prefix, canonical in mappings:
        if lower.startswith(prefix):
            return canonical
    return venue


def _strictly_relevant(title: str, abstract: str) -> bool:
    if not is_relevant(title, abstract):
        return False
    title_lower = title.lower().replace("–", "-").replace("—", "-")
    text = f"{title} {abstract}".lower()
    title_signal = any(
        term in title_lower
        for term in (
            "drug-target affinity",
            "drug target affinity",
            "drug-target binding affinity",
            "protein-ligand binding affinity",
            "protein ligand binding affinity",
            "binding affinity prediction",
            "binding free energy prediction",
            "scoring function",
            "pdbbind",
        )
    )
    abstract_signal = any(
        term in text
        for term in (
            "predict drug-target binding affinity",
            "drug-target affinity prediction",
            "protein-ligand binding affinity prediction",
            "predicting protein-ligand binding affinity",
            "binding affinity scoring",
        )
    )
    method_named = any(
        name in title_lower
        for name in ("deepdta", "graphdta", "widedta", "simboost", "deepaffinity", "monn")
    )
    excluded_title = any(
        term in title_lower
        for term in (
            "antibody affinity",
            "targeted molecule generation",
            "drug-target interaction prediction",
            "virtual screening",
            "molecular docking study",
        )
    ) and "affinity prediction" not in title_lower
    return (title_signal or abstract_signal or method_named) and not excluded_title


def _pdf_candidates(external: dict[str, Any], oa_url: str | None, doi: str) -> list[str]:
    urls: list[str] = []
    if doi.lower().startswith("10.18653/v1/"):
        anthology_id = doi.split("/", 1)[1]
        urls.append(f"https://aclanthology.org/{anthology_id}.pdf")
    if oa_url:
        parsed = urlsplit(oa_url)
        host = (parsed.hostname or "").lower()
        if host in OFFICIAL_PDF_HOSTS or oa_url.lower().endswith(".pdf") or "pdf=" in oa_url.lower():
            urls.append(oa_url)
    pmcid = external.get("PubMedCentral")
    if pmcid:
        value = str(pmcid).upper()
        if not value.startswith("PMC"):
            value = f"PMC{value}"
        urls.append(f"https://europepmc.org/articles/{value}?pdf=render")
    arxiv = external.get("ArXiv")
    if arxiv:
        urls.append(f"https://arxiv.org/pdf/{arxiv}")
    if oa_url and oa_url not in urls and not oa_url.lower().startswith("https://doi.org/"):
        urls.append(oa_url)
    return list(dict.fromkeys(urls))


def transform(item: dict[str, Any], query: str) -> dict[str, Any] | None:
    title = str(item.get("title") or "").strip()
    abstract = str(item.get("abstract") or "").strip()
    external = item.get("externalIds") or {}
    doi = str(external.get("DOI") or "").lower().removeprefix("https://doi.org/")
    if not title or not doi or not _strictly_relevant(title, abstract):
        return None
    venue = _inferred_venue(str(item.get("venue") or "").strip(), doi)
    year = item.get("year") if isinstance(item.get("year"), int) else None
    quality_tier, canonical_venue = classify_paper_quality(venue, venue, year)
    if quality_tier in {"authoritative", "established_peer_reviewed"}:
        portfolio_bucket = "authoritative"
    elif year is not None and year >= RECENT_YEAR_FLOOR:
        portfolio_bucket = "recent"
    else:
        return None
    track = corpus_track(title, abstract)
    if track not in TRACKS:
        return None
    oa_url = (item.get("openAccessPdf") or {}).get("url")
    pdf_candidates = _pdf_candidates(external, oa_url, doi)
    if not pdf_candidates:
        return None
    citations = int(item.get("citationCount") or 0)
    tags = method_tags(title, abstract)
    evidence, basis = innovation_evidence(title, abstract)
    publication_types = [str(value) for value in (item.get("publicationTypes") or [])]
    publication_type = (
        "conference"
        if any("Conference" in value for value in publication_types)
        else "journal"
        if any(value in {"JournalArticle", "Review"} for value in publication_types)
        else "preprint"
        if external.get("ArXiv") and not venue
        else "other"
    )
    return {
        "semantic_scholar_id": item.get("paperId"),
        "openalex_id": None,
        "doi": doi,
        "title": title,
        "publication_year": year,
        "publication_date": item.get("publicationDate"),
        "work_type": publication_type,
        "authors": [
            str(author.get("name"))
            for author in (item.get("authors") or [])
            if isinstance(author, dict) and author.get("name")
        ],
        "venue": venue or None,
        "canonical_venue": canonical_venue,
        "quality_tier": quality_tier,
        "portfolio_bucket": portfolio_bucket,
        "cited_by_count": citations,
        "abstract": abstract,
        "keywords": [],
        "primary_url": item.get("url"),
        "pdf_url": pdf_candidates[0],
        "pdf_candidate_urls": pdf_candidates,
        "oa_landing_page_url": item.get("url"),
        "oa_status": (item.get("openAccessPdf") or {}).get("status"),
        "is_oa": True,
        "oa_license": (item.get("openAccessPdf") or {}).get("license"),
        "matched_queries": [query],
        "relevance_score": relevance_score(title, abstract, citations),
        "method_tags": tags,
        "innovation_summary_zh": innovation_summary_zh(tags, title),
        "innovation_evidence": evidence,
        "innovation_basis": basis,
        "innovation_needs_fulltext_review": basis == "title_only",
        "corpus_track": track,
        "publication_type": publication_type,
        "github_repository_url": None,
        "github_status": "not_found",
        "github_license_id": None,
        "github_evidence": None,
    }


def collect_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for query in QUERIES:
        for item in semantic_search(query):
            record = transform(item, query)
            if record is None:
                continue
            existing = next((value for value in candidates if same_work(value, record)), None)
            if existing:
                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)
                existing["pdf_candidate_urls"] = list(
                    dict.fromkeys(existing["pdf_candidate_urls"] + record["pdf_candidate_urls"])
                )
            else:
                candidates.append(record)
    return candidates


def _rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    venue_rank = VENUE_PRIORITY.get(record.get("canonical_venue"), len(VENUE_PRIORITY))
    return (
        venue_rank,
        -float(record.get("relevance_score") or 0),
        -int(record.get("cited_by_count") or 0),
        -int(record.get("publication_year") or 0),
        normalized_title(record["title"]),
    )


def _recent_rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    venue = str(record.get("venue") or "").lower()
    is_preprint = any(name in venue for name in ("arxiv", "biorxiv", "medrxiv", "chemrxiv"))
    return (
        1 if is_preprint else 0,
        -int(record.get("publication_year") or 0),
        -float(record.get("relevance_score") or 0),
        -int(record.get("cited_by_count") or 0),
        normalized_title(record["title"]),
    )


def fetch_pdf(record: dict[str, Any]) -> DownloadedCandidate | None:
    cache_key = hashlib.sha256(record["doi"].encode("utf-8")).hexdigest()
    cache_pdf = PDF_CACHE_ROOT / f"{cache_key}.pdf"
    cache_meta = PDF_CACHE_ROOT / f"{cache_key}.json"
    if cache_pdf.is_file() and cache_meta.is_file():
        content = cache_pdf.read_bytes()
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        digest = hashlib.sha256(content).hexdigest()
        if content.startswith(b"%PDF-") and digest == meta.get("sha256"):
            payload = DownloadPayload(
                content=content,
                final_url=meta["final_url"],
                content_type=meta["content_type"],
                sha256=digest,
                redirect_count=int(meta.get("redirect_count") or 0),
            )
            return DownloadedCandidate(
                record, payload, meta["source_url"], list(meta.get("failed_url_attempts") or [])
            )
    attempts: list[dict[str, str]] = []
    downloader = SafeHttpDownloader(timeout_seconds=45.0)
    for url in record["pdf_candidate_urls"]:
        try:
            payload = downloader.fetch(url, PAPER_POLICY)
            PDF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            cache_pdf.write_bytes(payload.content)
            cache_meta.write_text(
                json.dumps(
                    {
                        "doi": record["doi"],
                        "source_url": url,
                        "final_url": payload.final_url,
                        "content_type": payload.content_type,
                        "sha256": payload.sha256,
                        "redirect_count": payload.redirect_count,
                        "failed_url_attempts": attempts,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return DownloadedCandidate(record, payload, url, attempts)
        except DownloadPolicyError as exc:
            attempts.append({"url": url, "error": str(exc)})
    return None


def fill_group(
    candidates: list[dict[str, Any]], quota: int, *, workers: int
) -> list[DownloadedCandidate]:
    selected: list[DownloadedCandidate] = []
    batch_size = max(workers * 2, 8)
    for offset in range(0, len(candidates), batch_size):
        if len(selected) >= quota:
            break
        batch = candidates[offset : offset + batch_size]
        completed: dict[str, DownloadedCandidate] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(fetch_pdf, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    completed[str(result.record["semantic_scholar_id"])] = result
        for item in batch:
            result = completed.get(str(item["semantic_scholar_id"]))
            if result is not None and len(selected) < quota:
                selected.append(result)
    return selected


def github_catalog() -> list[dict[str, Any]]:
    if GITHUB_CACHE.is_file():
        cached = json.loads(GITHUB_CACHE.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            return [item for item in cached if isinstance(item, dict)]
    repositories: dict[str, dict[str, Any]] = {}
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "auto-make-passage/0.2 quality-paper-agent",
    }
    with httpx.Client(timeout=45.0, headers=headers, trust_env=False) as client:
        for query in GITHUB_QUERIES:
            response = client.get(GITHUB_SEARCH, params={"q": query, "per_page": 100})
            if response.status_code in {403, 429}:
                break
            response.raise_for_status()
            for item in response.json().get("items", []):
                if isinstance(item, dict) and item.get("html_url"):
                    repositories[str(item["html_url"])] = item
    result = list(repositories.values())
    GITHUB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GITHUB_CACHE.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _method_name(title: str) -> str | None:
    match = re.match(r"^([A-Za-z][A-Za-z0-9_-]{2,30})\s*[:：]", title)
    return match.group(1).lower() if match else None


def attach_github(records: list[dict[str, Any]], repositories: list[dict[str, Any]]) -> None:
    for record in records:
        record.update(
            {
                "github_repository_url": None,
                "github_status": "not_found",
                "github_license_id": None,
                "github_evidence": None,
            }
        )
        normalized_paper_title = normalized_title(record["title"])
        title_tokens = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", record["title"].lower())
            if token not in {"with", "from", "using", "based", "deep", "learning", "prediction", "binding", "affinity"}
        }
        method = _method_name(record["title"])
        best: tuple[int, dict[str, Any]] | None = None
        for repository in repositories:
            text = " ".join(
                str(repository.get(key) or "")
                for key in ("name", "full_name", "description", "homepage", "html_url")
            ).lower()
            name = str(repository.get("name") or "").lower()
            normalized_name = re.sub(r"[^a-z0-9]", "", name)
            normalized_method = re.sub(r"[^a-z0-9]", "", method or "")
            description = normalized_title(
                " ".join(str(repository.get(key) or "") for key in ("description", "homepage"))
            )
            overlap = sum(token in text for token in title_tokens)
            coverage = overlap / max(len(title_tokens), 1)
            similarity = difflib.SequenceMatcher(None, normalized_paper_title, description).ratio()
            method_exact = bool(
                normalized_method
                and (
                    normalized_method == normalized_name
                    or re.search(
                        rf"(?<![a-z0-9]){re.escape(method or '')}(?![a-z0-9])", text
                    )
                )
            )
            title_exact = bool(
                len(normalized_paper_title) >= 25
                and normalized_paper_title in normalized_title(text)
            )
            strong_title = len(title_tokens) >= 5 and coverage >= 0.80 and similarity >= 0.72
            accepted = (
                method_exact or title_exact
                if method
                else title_exact or strong_title
            )
            if not accepted:
                continue
            score = (
                (100 if method_exact else 0)
                + (50 if title_exact else 0)
                + round(coverage * 20)
                + round(similarity * 20)
            )
            if best is None or score > best[0]:
                best = (score, repository)
        if best is None:
            continue
        repository = best[1]
        license_id = (repository.get("license") or {}).get("spdx_id")
        verified = bool(license_id and license_id not in {"NOASSERTION", "OTHER"})
        record.update(
            {
                "github_repository_url": repository["html_url"],
                "github_status": "verified_open_source" if verified else "search_matched",
                "github_license_id": license_id if verified else None,
                "github_evidence": {
                    "match_score": best[0],
                    "repository_name": repository.get("full_name"),
                    "description": repository.get("description"),
                    "homepage": repository.get("homepage"),
                    "archived": bool(repository.get("archived")),
                    "fork": bool(repository.get("fork")),
                },
            }
        )


def write_github_outputs(records: list[dict[str, Any]], root: Path) -> None:
    report = [
        "# GitHub 代码仓库标注",
        "",
        "> `verified_open_source` 表示论文标题或方法名强匹配，且 GitHub API 返回 SPDX 许可证；"
        "`search_matched` 表示强匹配但许可证未验证。未达到阈值时保持 `not_found`。",
        "",
        "| ID | 论文 | 状态 | 仓库 | 许可证 |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        if record.get("github_status") == "not_found":
            continue
        title = record["title"].replace("|", "\\|")
        report.append(
            f"| {record['paper_id']} | {title} | {record['github_status']} | "
            f"{record.get('github_repository_url') or ''} | {record.get('github_license_id') or ''} |"
        )
    report_path = root / "github_repositories.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    with (root / "papers.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "paper_id", "title", "publication_year", "doi", "venue", "canonical_venue",
            "quality_tier", "portfolio_bucket", "corpus_track", "cited_by_count",
            "pdf_status", "github_status", "github_repository_url", "github_license_id",
            "method_tags", "innovation_summary_zh",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fields},
                    "pdf_status": (record.get("pdf_download") or {}).get("status"),
                    "method_tags": "; ".join(record.get("method_tags") or []),
                }
            )


def refresh_github_only(output: str) -> None:
    root = WORKSPACE_ROOT / output
    papers_path = root / "papers.json"
    records = json.loads(papers_path.read_text(encoding="utf-8"))
    attach_github(records, github_catalog())
    papers_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for record in records:
        (root / "records" / f"{record['paper_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_github_outputs(records, root)
    manifest_path = root / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["github_counts"] = {
        status: sum(item["github_status"] == status for item in records)
        for status in ("verified_open_source", "search_matched", "not_found")
    }
    manifest.setdefault("checksums", {})["papers.json"] = hashlib.sha256(
        papers_path.read_bytes()
    ).hexdigest()
    manifest["checksums"]["github_repositories.md"] = hashlib.sha256(
        (root / "github_repositories.md").read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
def materialize(
    selected: list[DownloadedCandidate], output: str, repositories: list[dict[str, Any]]
) -> None:
    records = [dict(item.record) for item in selected]
    attach_github(records, repositories)
    store = ArtifactStore(WORKSPACE_ROOT)
    downloads: list[dict[str, Any]] = []
    for index, (record, selected_item) in enumerate(zip(records, selected, strict=True), start=1):
        paper_id = f"DTA-{index:03d}"
        record["paper_id"] = paper_id
        relative = f"{output}/papers/{paper_id}.pdf"
        artifact = store.write_agent_bytes(
            AgentName.DOWNLOAD_PAPER,
            relative,
            selected_item.payload.content,
            max_bytes=PAPER_POLICY.max_bytes,
            kind="paper_pdf",
        )
        downloads.append(
            {
                "paper_id": paper_id,
                "source_url": selected_item.source_url,
                "status": "downloaded",
                "reason": "PDF passed source priority, public URL, MIME, size and signature checks.",
                "final_url": selected_item.payload.final_url,
                "content_type": selected_item.payload.content_type,
                "size_bytes": selected_item.payload.size_bytes,
                "sha256": selected_item.payload.sha256,
                "artifact": artifact.model_dump(mode="json"),
                "failed_url_attempts": selected_item.attempts,
            }
        )
    write_outputs(records, downloads, output)
    root = WORKSPACE_ROOT / output
    # write_outputs attaches the verified PDF audit; now export the additional
    # quality and code-availability columns from those final records.
    records = json.loads((root / "papers.json").read_text(encoding="utf-8"))
    write_github_outputs(records, root)
    manifest_path = root / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": "2.0",
            "source": "Semantic Scholar Graph API with official/PMC/arXiv PDF resolution",
            "source_url": SEMANTIC_SEARCH,
            "queries": list(QUERIES),
            "pdf_required": True,
            "pdf_downloaded_count": len(records),
            "pdf_requires_review_count": 0,
            "authoritative_ratio_target": 0.70,
            "recent_ratio_target": 0.30,
            "recent_since_year": RECENT_YEAR_FLOOR,
            "quality_bucket_counts": {
                "authoritative": sum(item["portfolio_bucket"] == "authoritative" for item in records),
                "recent": sum(item["portfolio_bucket"] == "recent" for item in records),
            },
            "github_counts": {
                "verified_open_source": sum(item["github_status"] == "verified_open_source" for item in records),
                "search_matched": sum(item["github_status"] == "search_matched" for item in records),
                "not_found": sum(item["github_status"] == "not_found" for item in records),
            },
            "authoritative_venue_allowlist": list(AUTHORITATIVE_VENUES),
            "limitations": [
                "Semantic Scholar metadata may contain indexing errors; DOI and title deduplication are deterministic.",
                "Every included record has a locally verified PDF and SHA-256 digest.",
                "GitHub is marked verified_open_source only when repository matching succeeds and GitHub reports an SPDX license.",
            ],
        }
    )
    manifest.setdefault("checksums", {})["papers.csv"] = hashlib.sha256(
        (root / "papers.csv").read_bytes()
    ).hexdigest()
    manifest["checksums"]["github_repositories.md"] = hashlib.sha256(
        (root / "github_repositories.md").read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# DTA affinity quality corpus v2\n\n"
        "本目录只收录已通过 `%PDF-` 签名、MIME、大小和 SHA-256 校验的论文。组合严格为 "
        "70% 权威会议/期刊和 30% 近三年论文；正式出版版本优先，PMC/arXiv 可作为合法全文镜像。"
        "`papers.json` 与 `papers.csv` 包含 venue 等级、组合 bucket、GitHub 仓库、匹配状态和许可证。\n",
        encoding="utf-8",
    )


def select_and_download(
    candidates: list[dict[str, Any]], *, workers: int
) -> list[DownloadedCandidate]:
    successful: dict[tuple[str, str], list[DownloadedCandidate]] = {}
    desired_successes = {
        ("drug_target_dta", "authoritative"): 42,
        ("drug_target_dta", "recent"): 15,
        ("structure_based_plba", "authoritative"): 42,
        ("structure_based_plba", "recent"): 15,
    }
    for track in TRACKS:
        groups = (
            (
                "authoritative",
                GROUP_QUOTAS[(track, "authoritative")],
                sorted(
                    [
                        item
                        for item in candidates
                        if item["corpus_track"] == track and item["portfolio_bucket"] == "authoritative"
                    ],
                    key=_rank_key,
                ),
            ),
            (
                "recent",
                GROUP_QUOTAS[(track, "recent")],
                sorted(
                    [
                        item
                        for item in candidates
                        if item["corpus_track"] == track and item["portfolio_bucket"] == "recent"
                    ],
                    key=_recent_rank_key,
                ),
            ),
        )
        for bucket, quota, pool in groups:
            downloaded = fill_group(pool, desired_successes[(track, bucket)], workers=workers)
            successful[(track, bucket)] = downloaded
            print(
                json.dumps(
                    {
                        "track": track,
                        "bucket": bucket,
                        "candidates": len(pool),
                        "pdf_verified": len(downloaded),
                    }
                )
            )
    def allocate(item: DownloadedCandidate, bucket: str, reason: str) -> DownloadedCandidate:
        record = dict(item.record)
        record["portfolio_bucket"] = bucket
        record["portfolio_allocation_reason"] = reason
        return DownloadedCandidate(record, item.payload, item.source_url, item.attempts)

    selected: list[DownloadedCandidate] = []
    for track in TRACKS:
        recent_pool = successful[(track, "recent")]
        authority_pool = successful[(track, "authoritative")]
        formal_recent = [
            item
            for item in recent_pool
            if str(item.record.get("venue") or "").lower().strip() in CURATED_RECENT_VENUES
        ]
        recent_selected = [
            allocate(item, "recent", "recent publication window")
            for item in formal_recent[:15]
        ]
        recent_deficit = 15 - len(recent_selected)
        reassigned: list[DownloadedCandidate] = []
        reassigned_dois: set[str] = set()
        if recent_deficit:
            for item in authority_pool:
                if int(item.record.get("publication_year") or 0) < RECENT_YEAR_FLOOR:
                    continue
                reassigned.append(
                    allocate(item, "recent", "authoritative venue and recent; allocated to 30% recency quota")
                )
                reassigned_dois.add(item.record["doi"])
                if len(reassigned) == recent_deficit:
                    break
        if len(recent_selected) + len(reassigned) < 15:
            already = {item.record["doi"] for item in recent_selected + reassigned}
            for item in recent_pool:
                if item.record["doi"] in already:
                    continue
                reassigned.append(
                    allocate(item, "recent", "last-resort recent source after curated formal venues")
                )
                if len(recent_selected) + len(reassigned) == 15:
                    break
        if len(recent_selected) + len(reassigned) != 15:
            raise RuntimeError(f"recent PDF quota failed for {track}")
        authority_selected = [
            allocate(item, "authoritative", "curated authoritative or established peer-reviewed venue")
            for item in authority_pool
            if item.record["doi"] not in reassigned_dois
        ][:35]
        if len(authority_selected) != 35:
            raise RuntimeError(f"authoritative PDF quota failed for {track}")
        selected.extend(authority_selected + recent_selected + reassigned)
    print(
        json.dumps(
            {
                "portfolio_matrix": {
                    "drug_target_dta:authoritative": 35,
                    "drug_target_dta:recent": 15,
                    "structure_based_plba:authoritative": 35,
                    "structure_based_plba:recent": 15,
                }
            }
        )
    )
    selected.sort(
        key=lambda item: (
            0 if item.record["portfolio_bucket"] == "authoritative" else 1,
            TRACKS.index(item.record["corpus_track"]),
            _rank_key(item.record)
            if item.record["portfolio_bucket"] == "authoritative"
            else _recent_rank_key(item.record),
        )
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-github-only", action="store_true")
    args = parser.parse_args()
    if not args.output.startswith("reference/") or ".." in Path(args.output).parts:
        raise SystemExit("--output must stay under reference/")
    if args.refresh_github_only:
        refresh_github_only(args.output)
        print(json.dumps({"status": "github_refreshed", "output": args.output}))
        return 0
    candidates = collect_candidates()
    counts = {
        f"{track}:{bucket}": sum(
            item["corpus_track"] == track and item["portfolio_bucket"] == bucket
            for item in candidates
        )
        for track in TRACKS
        for bucket in ("authoritative", "recent")
    }
    print(json.dumps({"candidate_count": len(candidates), "groups": counts}, ensure_ascii=False))
    if args.dry_run:
        return 0
    selected = select_and_download(candidates, workers=max(1, args.workers))
    if len(selected) != 100:
        raise RuntimeError(f"expected 100 PDF-complete records, got {len(selected)}")
    materialize(selected, args.output, github_catalog())
    print(
        json.dumps(
            {
                "status": "valid",
                "paper_count": 100,
                "pdf_count": 100,
                "authoritative_count": 70,
                "recent_count": 30,
                "output": args.output,
                "created_at": utc_now(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
