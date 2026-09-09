from __future__ import annotations

import hashlib
from typing import Any

from research_agents.schemas import DatasetFormat


# Datasets up to 10 GiB are downloaded and materialized locally. Larger datasets
# produce a server-side shell script and remain behind a human execution gate.
DATA_MAX_BYTES = 10 * 1024 * 1024 * 1024
ALLOWED_LICENSE_IDS = frozenset(
    {
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc-by-3.0",
        "cc-by-sa-3.0",
        "mit",
        "apache-2.0",
        "odc-by-1.0",
        "pddl-1.0",
        "public-domain",
    }
)
LICENSE_URLS: dict[str, tuple[str, ...]] = {
    "cc0-1.0": ("https://creativecommons.org/publicdomain/zero/1.0",),
    "cc-by-4.0": ("https://creativecommons.org/licenses/by/4.0",),
    "cc-by-sa-4.0": ("https://creativecommons.org/licenses/by-sa/4.0",),
    "cc-by-3.0": ("https://creativecommons.org/licenses/by/3.0/us",),
    "cc-by-sa-3.0": ("https://creativecommons.org/licenses/by-sa/3.0",),
    "mit": ("https://opensource.org/license/mit", "https://spdx.org/licenses/mit"),
    "apache-2.0": (
        "https://www.apache.org/licenses/license-2.0",
        "https://spdx.org/licenses/apache-2.0",
    ),
    "odc-by-1.0": ("https://opendatacommons.org/licenses/by/1-0",),
    "pddl-1.0": ("https://opendatacommons.org/licenses/pddl/1-0",),
    "public-domain": ("https://creativecommons.org/publicdomain/mark/1.0",),
}
CONTENT_TYPES_BY_FORMAT: dict[DatasetFormat, frozenset[str]] = {
    DatasetFormat.CSV: frozenset({"text/csv", "application/csv", "text/plain"}),
    DatasetFormat.TSV: frozenset({"text/tab-separated-values", "text/plain"}),
    DatasetFormat.JSON: frozenset({"application/json", "text/json"}),
    DatasetFormat.JSONL: frozenset(
        {"application/x-ndjson", "application/json", "text/plain"}
    ),
    DatasetFormat.PARQUET: frozenset(
        {"application/vnd.apache.parquet", "application/octet-stream"}
    ),
    DatasetFormat.TXT: frozenset({"text/plain"}),
    DatasetFormat.ZIP: frozenset(
        {"application/zip", "application/x-zip-compressed", "application/octet-stream"}
    ),
    DatasetFormat.SEVEN_Z: frozenset(
        {"application/x-7z-compressed", "application/octet-stream"}
    ),
}


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def bounded_search_results(
    results: list[dict[str, Any]], *, limit: int = 5
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for item in results[:limit]:
        score = item.get("score")
        bounded.append(
            {
                "title": str(item.get("title", ""))[:500],
                "url": str(item.get("url", ""))[:4096],
                "content": str(item.get("content", ""))[:4000],
                "score": score if isinstance(score, (int, float)) else None,
            }
        )
    return bounded


def eligibility_reason(
    record: dict[str, Any], *, enforce_local_size_limit: bool = True
) -> str | None:
    license_id = record.get("license_id")
    if license_id not in ALLOWED_LICENSE_IDS:
        return "License is missing or is not in the automatic-download allowlist."
    license_url = (
        str(record.get("license_url") or "")
        .lower()
        .split("#", 1)[0]
        .split("?", 1)[0]
        .rstrip("/")
    )
    if license_url not in LICENSE_URLS.get(str(license_id), ()):
        return "License URL is missing or does not match the declared allowlisted license."
    if len(str(record.get("license_evidence") or "").strip()) < 5:
        return "No traceable license evidence was extracted from the search results."
    if record.get("format") is None:
        return "Dataset format is unknown or unsupported."
    if not record.get("direct_download_url"):
        return "No direct dataset download URL was identified."
    size = record.get("estimated_size_bytes")
    if size is None or size <= 0:
        return "Dataset size is unknown and requires manual review."
    if enforce_local_size_limit and size > DATA_MAX_BYTES:
        return "Estimated dataset size exceeds the automatic-download limit."
    return None
