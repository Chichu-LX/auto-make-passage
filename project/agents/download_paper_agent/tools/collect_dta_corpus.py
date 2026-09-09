#!/usr/bin/env python3
"""Build a reproducible 100-paper DTA/protein-ligand affinity corpus from OpenAlex."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agents.download_paper_agent.policy import PAPER_POLICY
from research_agents.downloads import DownloadPolicyError, SafeHttpDownloader
from research_agents.schemas import AgentName
from research_agents.storage import ArtifactStore


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = "reference/assaycal-dta/literature"
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
QUERIES = (
    "drug target affinity prediction",
    "protein ligand binding affinity prediction",
    "deep learning drug target affinity",
    "graph neural network drug target affinity",
    "structure based protein ligand binding affinity machine learning",
    "transformer drug target affinity prediction",
    "multimodal drug target affinity prediction",
    "PDBbind binding affinity prediction",
    "DeepDTA WideDTA GraphDTA drug target affinity",
    "SimBoost DeepAffinity MONN drug target binding affinity",
    "GraphDTA",
    "WideDTA",
    "SimBoost drug target affinity",
    "DeepAffinity compound protein affinity",
    "MONN drug target affinity",
)
SELECT_FIELDS = ",".join(
    (
        "id",
        "doi",
        "title",
        "publication_year",
        "publication_date",
        "type",
        "authorships",
        "primary_location",
        "best_oa_location",
        "open_access",
        "abstract_inverted_index",
        "cited_by_count",
        "keywords",
    )
)
METHOD_TERMS = {
    "graph neural network": ("graph neural", "graph convolution", "gnn", "molecular graph"),
    "transformer/attention": ("transformer", "attention", "self-attention"),
    "3D structure": ("3d", "three-dimensional", "structure-based", "protein structure", "complex structure"),
    "multimodal fusion": ("multimodal", "multi-modal", "fusion", "multiple representations"),
    "pretraining": ("pretrain", "pre-train", "self-supervised", "contrastive"),
    "sequence modeling": ("sequence-based", "protein sequence", "smiles", "convolutional neural"),
    "geometric/equivariant": ("equivariant", "geometric", "spatial graph", "distance-aware"),
    "uncertainty/explainability": ("uncertainty", "explainab", "interpretability", "interpretable"),
    "benchmark/data": ("benchmark", "dataset", "database", "pdbbind", "bindingdb", "davis", "kiba"),
    "transfer/generalization": ("transfer", "generalization", "generalisation", "cold-start", "unseen"),
}
INNOVATION_MARKERS = (
    "we propose",
    "we present",
    "we introduce",
    "we develop",
    "we designed",
    "we construct",
    "we devise",
    "novel ",
    "new framework",
    "new method",
    "first ",
    "our model",
    "our method",
    "this work",
)
CANONICAL_METHOD_PREFIXES = (
    "DeepDTA:",
    "GraphDTA:",
    "WideDTA:",
    "SimBoost:",
    "DeepAffinity:",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in index.items():
        for position in indexes:
            positions[position] = word
    return " ".join(positions[position] for position in sorted(positions))


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    return value.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def clean_title(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def corpus_track(title: str, abstract: str) -> str:
    title_lower = title.lower().replace("–", "-").replace("—", "-")
    if any(term in title_lower for term in ("drug-target", "drug target", "dta")):
        return "drug_target_dta"
    if any(term in title_lower for term in ("protein-ligand", "protein–ligand", "protein ligand")):
        return "structure_based_plba"
    text = f"{title} {abstract}".lower()
    if any(term in text for term in ("smiles", "protein sequence", "davis", "kiba")):
        return "drug_target_dta"
    if any(term in text for term in ("pdbbind", "complex structure", "binding pose", "scoring function")):
        return "structure_based_plba"
    return "benchmark_review_other"


def same_work(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("doi") and left.get("doi") == right.get("doi"):
        return True
    left_title = normalized_title(left["title"])
    right_title = normalized_title(right["title"])
    if left_title == right_title:
        return True
    similarity = difflib.SequenceMatcher(None, left_title, right_title).ratio()
    if similarity >= 0.965:
        return True
    left_name = re.match(r"^([A-Za-z0-9][A-Za-z0-9_-]{3,30}):", left["title"])
    right_name = re.match(r"^([A-Za-z0-9][A-Za-z0-9_-]{3,30}):", right["title"])
    return bool(
        left_name
        and right_name
        and left_name.group(1).lower() == right_name.group(1).lower()
        and similarity >= 0.9
    )


def relevance_score(title: str, abstract: str, citations: int) -> float:
    title_lower = title.lower()
    text = f"{title} {abstract}".lower()
    score = 0.0
    title_phrases = {
        "drug-target affinity": 14,
        "drug target affinity": 14,
        "protein-ligand binding affinity": 14,
        "protein–ligand binding affinity": 14,
        "binding affinity prediction": 11,
        "affinity prediction": 8,
        "dta": 4,
    }
    for phrase, points in title_phrases.items():
        if phrase in title_lower:
            score += points
    if any(
        method in title_lower
        for method in ("deepdta", "graphdta", "widedta", "simboost", "deepaffinity", "monn")
    ):
        score += 15
    for phrase, points in (
        ("drug-target affinity", 6),
        ("drug target affinity", 6),
        ("protein-ligand", 5),
        ("protein ligand", 5),
        ("binding affinity", 4),
        ("pdbbind", 2),
        ("bindingdb", 2),
        ("davis", 1),
        ("kiba", 1),
    ):
        if phrase in text:
            score += points
    if any(term in text for term in ("predict", "machine learning", "deep learning", "neural", "scoring function")):
        score += 4
    if any(term in text for term in ("drug", "ligand", "compound")) and any(
        term in text for term in ("protein", "target", "receptor")
    ):
        score += 4
    score += min(math.log10(max(citations, 0) + 1), 3.0)
    return round(score, 4)


def is_relevant(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    affinity = any(
        term in text for term in ("affinity", "binding free energy", "binding strength", " dta ")
    )
    molecular_pair = any(term in text for term in ("drug", "ligand", "compound")) and any(
        term in text for term in ("protein", "target", "receptor")
    )
    modeling = any(
        term in text
        for term in (
            "predict",
            "model",
            "machine learning",
            "deep learning",
            "neural",
            "scoring function",
            "benchmark",
            "database",
        )
    )
    return affinity and molecular_pair and modeling


def method_tags(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    return [label for label, terms in METHOD_TERMS.items() if any(term in text for term in terms)]


def innovation_evidence(title: str, abstract: str) -> tuple[list[str], str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", abstract)
        if len(item.strip()) >= 30
    ]
    candidates = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in INNOVATION_MARKERS)
    ]
    if not candidates:
        candidates = [
            sentence
            for sentence in sentences
            if any(term in sentence.lower() for term in ("framework", "architecture", "method", "model"))
        ]
    evidence = [sentence[:800] for sentence in candidates[:2]]
    if evidence:
        return evidence, "abstract_explicit" if any(
            marker in evidence[0].lower() for marker in INNOVATION_MARKERS
        ) else "abstract_method_sentence"
    return [f"Title evidence: {title}"], "title_only"


def innovation_summary_zh(tags: list[str], title: str) -> str:
    if tags:
        return f"主要创新方向：{'、'.join(tags[:3])}；具体创新以摘要证据句为准。"
    return f"围绕“{title}”提出或评估亲和力预测方法；摘要未提供足够细节，需全文复核。"


def openalex_get(query: str) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "per-page": "200",
        "filter": "from_publication_date:2000-01-01,to_publication_date:2026-12-31",
        "select": SELECT_FIELDS,
    }
    headers = {"User-Agent": "auto-make-passage/0.1 DTA literature corpus"}
    with httpx.Client(timeout=45.0, trust_env=False) as client:
        response = client.get(OPENALEX_ENDPOINT, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("OpenAlex response has no result list")
    return [item for item in results if isinstance(item, dict)]


def transform_work(work: dict[str, Any], query: str) -> dict[str, Any] | None:
    title = clean_title(str(work.get("title") or ""))
    if not title:
        return None
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not is_relevant(title, abstract):
        return None
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        name = author.get("display_name") if isinstance(author, dict) else None
        if name:
            authors.append(str(name))
    primary = work.get("primary_location") or {}
    best_oa = work.get("best_oa_location") or {}
    source = primary.get("source") or {}
    open_access = work.get("open_access") or {}
    tags = method_tags(title, abstract)
    evidence, basis = innovation_evidence(title, abstract)
    citations = int(work.get("cited_by_count") or 0)
    return {
        "openalex_id": work.get("id"),
        "doi": normalize_doi(work.get("doi")),
        "title": title,
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "work_type": work.get("type"),
        "authors": authors,
        "venue": source.get("display_name") if isinstance(source, dict) else None,
        "cited_by_count": citations,
        "abstract": abstract,
        "keywords": [
            keyword.get("display_name")
            for keyword in (work.get("keywords") or [])[:15]
            if isinstance(keyword, dict) and keyword.get("display_name")
        ],
        "primary_url": primary.get("landing_page_url"),
        "pdf_url": best_oa.get("pdf_url"),
        "oa_landing_page_url": best_oa.get("landing_page_url"),
        "oa_status": open_access.get("oa_status"),
        "is_oa": bool(open_access.get("is_oa")),
        "oa_license": best_oa.get("license"),
        "matched_queries": [query],
        "relevance_score": relevance_score(title, abstract, citations),
        "method_tags": tags,
        "innovation_summary_zh": innovation_summary_zh(tags, title),
        "innovation_evidence": evidence,
        "innovation_basis": basis,
        "innovation_needs_fulltext_review": basis == "title_only",
        "corpus_track": corpus_track(title, abstract),
    }


def collect(limit: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for query in QUERIES:
        for work in openalex_get(query):
            record = transform_work(work, query)
            if record is None:
                continue
            existing = next((item for item in candidates if same_work(item, record)), None)
            if existing:
                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)
                if record["relevance_score"] > existing["relevance_score"]:
                    record["matched_queries"] = existing["matched_queries"]
                    candidates[candidates.index(existing)] = record
            else:
                candidates.append(record)
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["relevance_score"],
            item["cited_by_count"],
            item["publication_year"] or 0,
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for prefix in CANONICAL_METHOD_PREFIXES:
        canonical = next(
            (item for item in ranked if item["title"].lower().startswith(prefix.lower())),
            None,
        )
        if canonical is not None and canonical not in selected:
            selected.append(canonical)
    quotas = {
        "drug_target_dta": min(50, limit),
        "structure_based_plba": min(45, max(0, limit - min(50, limit))),
    }
    for track, quota in quotas.items():
        existing_count = sum(item["corpus_track"] == track for item in selected)
        needed = max(0, quota - existing_count)
        selected.extend(
            [
                item
                for item in ranked
                if item["corpus_track"] == track and item not in selected
            ][:needed]
        )
    selected_ids = {id(item) for item in selected}
    selected.extend(item for item in ranked if id(item) not in selected_ids)
    selected = selected[:limit]
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} relevant unique works were found")
    for index, record in enumerate(selected, start=1):
        record["paper_id"] = f"DTA-{index:03d}"
    return selected


def download_one(
    store: ArtifactStore, output: str, record: dict[str, Any]
) -> dict[str, Any]:
    result = {
        "paper_id": record["paper_id"],
        "source_url": record.get("pdf_url"),
        "status": "not_available",
        "reason": "No OpenAlex open-access PDF URL.",
    }
    if not record.get("is_oa") or not record.get("pdf_url"):
        return result
    try:
        payload = SafeHttpDownloader().fetch(str(record["pdf_url"]), PAPER_POLICY)
        relative = f"{output}/papers/{record['paper_id']}.pdf"
        artifact = store.write_agent_bytes(
            AgentName.DOWNLOAD_PAPER,
            relative,
            payload.content,
            max_bytes=PAPER_POLICY.max_bytes,
            kind="paper_pdf",
        )
        result.update(
            {
                "status": "downloaded",
                "reason": "OpenAlex OA PDF passed the safe download policy.",
                "final_url": payload.final_url,
                "content_type": payload.content_type,
                "size_bytes": payload.size_bytes,
                "sha256": payload.sha256,
                "artifact": artifact.model_dump(mode="json"),
            }
        )
    except DownloadPolicyError as exc:
        result.update({"status": "requires_review", "reason": str(exc)})
    return result


def download_pdfs(
    records: list[dict[str, Any]], output: str, workers: int
) -> list[dict[str, Any]]:
    store = ArtifactStore(WORKSPACE_ROOT)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(download_one, store, output, record): record["paper_id"]
            for record in records
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["paper_id"])


def bibtex_key(record: dict[str, Any]) -> str:
    author = record["authors"][0].split()[-1] if record["authors"] else "Unknown"
    year = record.get("publication_year") or "nd"
    return re.sub(r"[^A-Za-z0-9]", "", f"{author}{year}{record['paper_id']}")


def write_outputs(
    records: list[dict[str, Any]], downloads: list[dict[str, Any]], output: str
) -> None:
    root = WORKSPACE_ROOT / output
    records_root = root / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    active_pdf_paths = {
        WORKSPACE_ROOT / str(item["artifact"]["uri"]).removeprefix("artifact://")
        for item in downloads
        if item.get("status") == "downloaded" and item.get("artifact")
    }
    for existing_pdf in (root / "papers").glob("DTA-*.pdf"):
        if existing_pdf not in active_pdf_paths:
            existing_pdf.unlink()
    download_by_id = {item["paper_id"]: item for item in downloads}
    for record in records:
        record["pdf_download"] = download_by_id.get(record["paper_id"])
        (records_root / f"{record['paper_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    papers_json = root / "papers.json"
    papers_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "pdf_downloads.json").write_text(
        json.dumps(downloads, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (root / "papers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "paper_id",
                "title",
                "publication_year",
                "doi",
                "venue",
                "cited_by_count",
                "relevance_score",
                "oa_status",
                "pdf_status",
                "method_tags",
                "innovation_summary_zh",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in writer.fieldnames},
                    "pdf_status": (record.get("pdf_download") or {}).get("status"),
                    "method_tags": "; ".join(record["method_tags"]),
                }
            )

    bib_entries = []
    for record in records:
        fields = {
            "title": "{" + record["title"].replace("{", "").replace("}", "") + "}",
            "author": " and ".join(record["authors"]),
            "year": str(record.get("publication_year") or ""),
            "journal": record.get("venue") or "",
            "doi": record.get("doi") or "",
            "url": record.get("primary_url") or record.get("oa_landing_page_url") or "",
        }
        lines = [f"@article{{{bibtex_key(record)},"]
        lines.extend(f"  {name} = {{{value}}}," for name, value in fields.items() if value)
        lines.append("}")
        bib_entries.append("\n".join(lines))
    (root / "references.bib").write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")

    rows = [
        "# DTA / 蛋白质–配体亲和力预测：100 篇论文创新点",
        "",
        "> 创新点来自 OpenAlex 摘要中的显式方法句；`title_only` 项必须在使用前复核全文。",
        "",
        "| ID | 年份 | 论文 | 方法标签 | 创新点 | 证据级别 |",
        "|---|---:|---|---|---|---|",
    ]
    for record in records:
        title = record["title"].replace("|", "\\|")
        summary = record["innovation_summary_zh"].replace("|", "\\|")
        rows.append(
            f"| {record['paper_id']} | {record.get('publication_year') or ''} | {title} | "
            f"{', '.join(record['method_tags']) or '待全文分类'} | {summary} | {record['innovation_basis']} |"
        )
    rows.extend(["", "## 逐篇创新证据", ""])
    for record in records:
        doi = record.get("doi")
        source = f"https://doi.org/{doi}" if doi else record.get("primary_url") or ""
        rows.extend(
            [
                f"### {record['paper_id']} · {record['title']}",
                "",
                f"- 年份：{record.get('publication_year') or '未知'}",
                f"- 方法标签：{', '.join(record['method_tags']) or '待全文分类'}",
                f"- 创新概括：{record['innovation_summary_zh']}",
                f"- 来源：{source}",
                f"- 证据等级：`{record['innovation_basis']}`",
                "- 摘要证据：",
            ]
        )
        rows.extend(f"  > {sentence}" for sentence in record["innovation_evidence"])
        rows.append("")
    (root / "innovation_points.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "topic": "drug-target affinity and protein-ligand binding affinity prediction",
        "created_at": utc_now(),
        "source": "OpenAlex REST API",
        "source_url": OPENALEX_ENDPOINT,
        "queries": list(QUERIES),
        "paper_count": len(records),
        "unique_doi_count": sum(bool(record.get("doi")) for record in records),
        "abstract_count": sum(bool(record.get("abstract")) for record in records),
        "open_access_count": sum(bool(record.get("is_oa")) for record in records),
        "pdf_downloaded_count": sum(item["status"] == "downloaded" for item in downloads),
        "pdf_requires_review_count": sum(item["status"] == "requires_review" for item in downloads),
        "innovation_evidence_counts": {
            basis: sum(record["innovation_basis"] == basis for record in records)
            for basis in ("abstract_explicit", "abstract_method_sentence", "title_only")
        },
        "corpus_track_counts": {
            track: sum(record["corpus_track"] == track for record in records)
            for track in ("drug_target_dta", "structure_based_plba", "benchmark_review_other")
        },
        "checksums": {
            "papers.json": hashlib.sha256(papers_json.read_bytes()).hexdigest(),
            "references.bib": hashlib.sha256((root / "references.bib").read_bytes()).hexdigest(),
            "innovation_points.md": hashlib.sha256((root / "innovation_points.md").read_bytes()).hexdigest(),
        },
        "limitations": [
            "Metadata and abstracts are sourced from OpenAlex and may contain indexing errors.",
            "Innovation summaries are evidence extraction, not a substitute for full-text peer review.",
            "Only OpenAlex-designated open-access direct PDF URLs are eligible for download.",
        ],
    }
    (root / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# DTA affinity 100-paper corpus\n\n"
        "本目录包含 100 篇药物–靶点/蛋白质–配体亲和力预测论文。`papers.json` 是完整记录，"
        "`records/` 是逐篇 JSON，`references.bib` 可直接用于引用，`innovation_points.md` 汇总创新点，"
        "`pdf_downloads.json` 记录开放 PDF 的下载或拒绝原因。未下载不代表论文无效，只表示没有通过"
        "自动开放访问与安全策略。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 500:
        raise SystemExit("--limit must be between 1 and 500")
    if not args.output.startswith("reference/") or ".." in Path(args.output).parts:
        raise SystemExit("--output must stay under reference/")
    records = collect(args.limit)
    downloads = (
        download_pdfs(records, args.output, args.workers)
        if args.download_pdfs
        else [
            {
                "paper_id": record["paper_id"],
                "source_url": record.get("pdf_url"),
                "status": "not_attempted",
                "reason": "Run with --download-pdfs to fetch eligible OA files.",
            }
            for record in records
        ]
    )
    write_outputs(records, downloads, args.output)
    print(
        json.dumps(
            {
                "paper_count": len(records),
                "output": args.output,
                "pdf_downloaded": sum(item["status"] == "downloaded" for item in downloads),
                "pdf_requires_review": sum(item["status"] == "requires_review" for item in downloads),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
