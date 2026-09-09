#!/usr/bin/env python3
"""Validate the local 100-paper DTA corpus without network access."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = WORKSPACE_ROOT / "reference" / "assaycal-dta" / "literature"
EXPECTED_TRACKS = {"drug_target_dta": 50, "structure_based_plba": 50}
EVIDENCE_LEVELS = {"abstract_explicit", "abstract_method_sentence", "title_only"}
PDF_STATUSES = {"downloaded", "requires_review", "not_available"}


class CorpusValidationError(RuntimeError):
    """Raised when one or more corpus invariants fail."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusValidationError(f"cannot read valid JSON: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _artifact_path(corpus: Path, uri: str) -> Path:
    prefix = "artifact://reference/assaycal-dta/literature/"
    if not uri.startswith(prefix):
        raise CorpusValidationError(f"unexpected PDF artifact URI: {uri}")
    target = (corpus / uri.removeprefix(prefix)).resolve()
    try:
        target.relative_to(corpus.resolve())
    except ValueError as exc:
        raise CorpusValidationError(f"PDF artifact escapes corpus: {uri}") from exc
    return target


def validate(corpus: Path) -> dict[str, Any]:
    corpus = corpus.resolve()
    required = {
        "papers.json",
        "papers.csv",
        "references.bib",
        "innovation_points.md",
        "pdf_downloads.json",
        "corpus_manifest.json",
        "research_ideas.md",
        "github_repositories.md",
    }
    missing = sorted(name for name in required if not (corpus / name).is_file())
    if missing:
        raise CorpusValidationError(f"missing corpus files: {', '.join(missing)}")

    papers = _load_json(corpus / "papers.json")
    downloads = _load_json(corpus / "pdf_downloads.json")
    manifest = _load_json(corpus / "corpus_manifest.json")
    if not isinstance(papers, list) or len(papers) != 100:
        raise CorpusValidationError("papers.json must contain exactly 100 records")
    if not isinstance(downloads, list) or len(downloads) != 100:
        raise CorpusValidationError("pdf_downloads.json must contain exactly 100 records")

    expected_ids = [f"DTA-{index:03d}" for index in range(1, 101)]
    paper_ids = [item.get("paper_id") for item in papers]
    if paper_ids != expected_ids:
        raise CorpusValidationError("paper IDs must be ordered DTA-001 through DTA-100")

    dois = [str(item.get("doi") or "").lower().removeprefix("https://doi.org/") for item in papers]
    if any(not doi for doi in dois) or len(set(dois)) != 100:
        raise CorpusValidationError("all 100 papers must have a unique DOI")
    titles = [_normalized_title(str(item.get("title") or "")) for item in papers]
    if any(not title for title in titles) or len(set(titles)) != 100:
        raise CorpusValidationError("all 100 papers must have a unique non-empty title")
    for left in range(len(titles)):
        for right in range(left + 1, len(titles)):
            if difflib.SequenceMatcher(None, titles[left], titles[right]).ratio() >= 0.965:
                raise CorpusValidationError(
                    f"probable duplicate titles: {paper_ids[left]} and {paper_ids[right]}"
                )

    track_counts = Counter(str(item.get("corpus_track")) for item in papers)
    if dict(track_counts) != EXPECTED_TRACKS:
        raise CorpusValidationError(f"unexpected corpus tracks: {dict(track_counts)}")
    manifest_tracks = manifest.get("corpus_track_counts") or {}
    if any(manifest_tracks.get(key) != value for key, value in EXPECTED_TRACKS.items()):
        raise CorpusValidationError("manifest track counts do not match the corpus")
    if any(value for key, value in manifest_tracks.items() if key not in EXPECTED_TRACKS):
        raise CorpusValidationError("manifest contains a non-empty unexpected corpus track")

    evidence_counts = Counter(str(item.get("innovation_basis")) for item in papers)
    if set(evidence_counts) - EVIDENCE_LEVELS:
        raise CorpusValidationError("unknown innovation evidence level")
    for item in papers:
        evidence = item.get("innovation_evidence")
        summary = str(item.get("innovation_summary_zh") or "").strip()
        title_only = item.get("innovation_basis") == "title_only"
        if not isinstance(evidence, list) or not evidence or not summary:
            raise CorpusValidationError(f"missing innovation evidence: {item['paper_id']}")
        if bool(item.get("innovation_needs_fulltext_review")) != title_only:
            raise CorpusValidationError(f"incorrect full-text review flag: {item['paper_id']}")
    if manifest.get("innovation_evidence_counts") != dict(evidence_counts):
        raise CorpusValidationError("manifest innovation counts do not match the corpus")

    quality_counts = Counter(str(item.get("portfolio_bucket")) for item in papers)
    if quality_counts != Counter({"authoritative": 70, "recent": 30}):
        raise CorpusValidationError(f"portfolio must be exactly 70/30: {dict(quality_counts)}")
    if manifest.get("quality_bucket_counts") != {"authoritative": 70, "recent": 30}:
        raise CorpusValidationError("manifest quality buckets do not match the 70/30 portfolio")
    recent_since = int(manifest.get("recent_since_year") or 0)
    if recent_since < 2000:
        raise CorpusValidationError("manifest recent-paper window is missing")
    for item in papers:
        if item.get("portfolio_bucket") == "recent" and int(item.get("publication_year") or 0) < recent_since:
            raise CorpusValidationError(f"recent quota contains an old paper: {item['paper_id']}")
        if item.get("portfolio_bucket") == "authoritative" and item.get("quality_tier") not in {
            "authoritative", "established_peer_reviewed"
        }:
            raise CorpusValidationError(f"authority quota contains an unapproved venue: {item['paper_id']}")

        github_status = item.get("github_status")
        github_url = item.get("github_repository_url")
        license_id = item.get("github_license_id")
        if github_status not in {"not_found", "search_matched", "verified_open_source"}:
            raise CorpusValidationError(f"unknown GitHub status: {item['paper_id']}")
        if github_status == "not_found" and (github_url or license_id):
            raise CorpusValidationError(f"unmatched GitHub record contains a claim: {item['paper_id']}")
        if github_status != "not_found" and not str(github_url or "").startswith("https://github.com/"):
            raise CorpusValidationError(f"GitHub match has no repository URL: {item['paper_id']}")
        if github_status == "verified_open_source" and not license_id:
            raise CorpusValidationError(f"open-source claim has no SPDX license: {item['paper_id']}")

    record_dir = corpus / "records"
    record_files = sorted(record_dir.glob("DTA-*.json"))
    if len(record_files) != 100:
        raise CorpusValidationError("records/ must contain exactly 100 paper JSON files")
    for paper, path in zip(papers, record_files, strict=True):
        if _load_json(path) != paper:
            raise CorpusValidationError(f"record differs from papers.json: {path.name}")

    with (corpus / "papers.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 100 or [row.get("paper_id") for row in csv_rows] != expected_ids:
        raise CorpusValidationError("papers.csv does not match the ordered corpus IDs")
    bib_entries = re.findall(r"(?m)^@[A-Za-z]+\{", (corpus / "references.bib").read_text(encoding="utf-8"))
    if len(bib_entries) != 100:
        raise CorpusValidationError("references.bib must contain exactly 100 entries")

    download_by_id = {item.get("paper_id"): item for item in downloads}
    if set(download_by_id) != set(expected_ids):
        raise CorpusValidationError("PDF audit records must map one-to-one to paper IDs")
    pdf_status_counts: Counter[str] = Counter()
    for paper in papers:
        item = download_by_id[paper["paper_id"]]
        status = str(item.get("status"))
        if status not in PDF_STATUSES:
            raise CorpusValidationError(f"unknown PDF status: {status}")
        if paper.get("pdf_download") != item:
            raise CorpusValidationError(f"PDF audit mismatch: {paper['paper_id']}")
        pdf_status_counts[status] += 1
        if status != "downloaded":
            continue
        artifact = item.get("artifact") or {}
        path = _artifact_path(corpus, str(artifact.get("uri") or ""))
        if not path.is_file() or path.read_bytes()[:5] != b"%PDF-":
            raise CorpusValidationError(f"missing or invalid PDF: {paper['paper_id']}")
        digest = _sha256(path)
        if digest != item.get("sha256") or digest != artifact.get("sha256"):
            raise CorpusValidationError(f"PDF checksum mismatch: {paper['paper_id']}")
        if path.stat().st_size != item.get("size_bytes"):
            raise CorpusValidationError(f"PDF size mismatch: {paper['paper_id']}")

    expected_pdf_counts = {
        "downloaded": manifest.get("pdf_downloaded_count"),
        "requires_review": manifest.get("pdf_requires_review_count"),
    }
    if any(pdf_status_counts[key] != value for key, value in expected_pdf_counts.items()):
        raise CorpusValidationError("manifest PDF counts do not match the audit records")

    if manifest.get("paper_count") != 100 or manifest.get("unique_doi_count") != 100:
        raise CorpusValidationError("manifest paper/DOI counts are invalid")
    if manifest.get("schema_version") != "2.0" or manifest.get("pdf_required") is not True:
        raise CorpusValidationError("corpus must use the PDF-complete v2 policy")
    if pdf_status_counts != Counter({"downloaded": 100}):
        raise CorpusValidationError("every included paper must have a verified local PDF")
    github_counts = Counter(str(item.get("github_status")) for item in papers)
    if manifest.get("github_counts") != dict(github_counts):
        raise CorpusValidationError("manifest GitHub counts do not match paper records")
    for name, expected_digest in (manifest.get("checksums") or {}).items():
        path = corpus / name
        if not path.is_file() or _sha256(path) != expected_digest:
            raise CorpusValidationError(f"manifest checksum mismatch: {name}")

    idea_json = (
        WORKSPACE_ROOT
        / "design"
        / "assaycal-dta"
        / "ideas"
        / "dta_research_ideas_v1.json"
    )
    ideas = _load_json(idea_json)
    if ideas.get("corpus_paper_count") != 100 or len(ideas.get("ideas", [])) != 5:
        raise CorpusValidationError("research idea artifact is incomplete")
    for idea in ideas["ideas"]:
        for evidence in idea.get("evidence", []):
            if evidence.get("paper_id") not in set(expected_ids):
                raise CorpusValidationError(f"idea references an unknown paper: {idea.get('idea_id')}")

    return {
        "status": "valid",
        "paper_count": len(papers),
        "unique_doi_count": len(set(dois)),
        "track_counts": dict(track_counts),
        "innovation_evidence_counts": dict(evidence_counts),
        "pdf_status_counts": dict(pdf_status_counts),
        "quality_bucket_counts": dict(quality_counts),
        "github_counts": dict(github_counts),
        "idea_count": len(ideas["ideas"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    try:
        result = validate(args.corpus)
    except CorpusValidationError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
