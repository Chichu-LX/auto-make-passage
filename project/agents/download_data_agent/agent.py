from __future__ import annotations

import json
import re
import shlex
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.data_materialization import materialize_dataset_payload
from research_agents.data_profiling import (
    canonicalize_headerless_dta_text,
    profile_dataset_payload,
)
from research_agents.downloads import DownloadPolicy, DownloadPolicyError, SafeHttpDownloader, validate_dataset_payload
from research_agents.providers import JsonModel, SearchProvider
from research_agents.project_layout import artifact_prefix, scoped_path, task_project_slug
from research_agents.schemas import (
    AgentName,
    AgentResult,
    DatasetCandidate,
    DatasetFormat,
    DatasetManifest,
    DatasetRecord,
    DatasetSelection,
    DownloadPolicyRecord,
    PaperManifest,
    ResearchDesign,
    TaskEnvelope,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    ALLOWED_LICENSE_IDS,
    CONTENT_TYPES_BY_FORMAT,
    DATA_MAX_BYTES,
    LICENSE_URLS,
    bounded_search_results,
    eligibility_reason,
    file_token,
)

PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.DOWNLOAD_DATA:
    raise RuntimeError("Download Data Agent package name mismatch")


class DownloadDataAgent:
    """Discover datasets and download only explicitly licensed, bounded data files."""

    name = AgentName.DOWNLOAD_DATA
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
        self.downloader = downloader or SafeHttpDownloader(timeout_seconds=90.0)

    def _fetch_with_retries(
        self, url: str, policy: DownloadPolicy, *, attempts: int = 2
    ) -> Any:
        last_error: DownloadPolicyError | None = None
        for _ in range(attempts):
            try:
                return self.downloader.fetch(url, policy)
            except DownloadPolicyError as exc:
                last_error = exc
        if last_error is None:
            raise DownloadPolicyError("bounded dataset download did not execute")
        raise last_error

    def _paper_hints(self, task: TaskEnvelope) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        for artifact in task.inputs:
            if artifact.kind != "paper_manifest":
                continue
            try:
                if not artifact.uri.startswith(
                    artifact_prefix("reference", task_project_slug(task), "manifests")
                ):
                    raise StoragePolicyError("paper manifest URI is outside reference/manifests")
                manifest = PaperManifest.model_validate(
                    self.artifacts.read_artifact_json(artifact, max_bytes=2 * 1024 * 1024)
                )
                if manifest.run_id != task.run_id:
                    raise StoragePolicyError("paper manifest belongs to another run")
            except (StoragePolicyError, ValidationError):
                continue
            for record in manifest.records[:5]:
                hints.append(
                    {
                        "paper_id": record.paper_id,
                        "title": record.title,
                        "doi": record.doi,
                    }
                )
        return hints[:5]

    def _design_requirements(self, task: TaskEnvelope) -> list[dict[str, Any]]:
        requirements: list[dict[str, Any]] = []
        for artifact in task.inputs:
            if artifact.kind != "research_design":
                continue
            try:
                if not artifact.uri.startswith(
                    artifact_prefix("design", task_project_slug(task), "research-designs")
                ):
                    raise StoragePolicyError("research design is outside its declared scope")
                design = ResearchDesign.model_validate(
                    self.artifacts.read_artifact_json(artifact, max_bytes=2 * 1024 * 1024)
                )
                if design.run_id != task.run_id:
                    raise StoragePolicyError("research design belongs to another run")
            except (StoragePolicyError, ValidationError):
                continue
            requirements.extend(
                item.model_dump(mode="json") for item in design.dataset_requirements
            )
        return requirements[:10]

    def _design_repository_urls(self, task: TaskEnvelope) -> list[str]:
        repositories: list[str] = []
        for artifact in task.inputs:
            if artifact.kind != "research_design":
                continue
            try:
                design = ResearchDesign.model_validate(
                    self.artifacts.read_artifact_json(
                        artifact, max_bytes=2 * 1024 * 1024
                    )
                )
                if design.run_id != task.run_id:
                    raise StoragePolicyError("research design belongs to another run")
            except (StoragePolicyError, ValidationError):
                continue
            for module in design.reusable_code_modules:
                if module.repository_url not in repositories:
                    repositories.append(module.repository_url)
        return repositories[:10]

    def _verified_cached_records(
        self,
        task: TaskEnvelope,
        design_requirements: list[dict[str, Any]],
    ) -> dict[str, DatasetRecord]:
        """Reuse only same-run local datasets after full artifact revalidation."""

        slug = task_project_slug(task)
        requirement_keys = {
            str(item.get("dataset_key"))
            for item in design_requirements
            if item.get("dataset_key")
        }
        verified: dict[str, DatasetRecord] = {}
        for artifact in task.inputs:
            if artifact.kind != "data_manifest" or not artifact.uri.startswith(
                artifact_prefix("data", slug, "manifests")
            ):
                continue
            try:
                manifest = DatasetManifest.model_validate(
                    self.artifacts.read_artifact_json(
                        artifact, max_bytes=2 * 1024 * 1024
                    )
                )
                if manifest.run_id != task.run_id or manifest.project_slug != slug:
                    raise StoragePolicyError(
                        "cached data manifest belongs to another run or project"
                    )
                for record in manifest.records:
                    if (
                        record.status != "downloaded"
                        or record.requirement_key not in requirement_keys
                        or record.requirement_key is None
                        or record.local_artifact is None
                        or record.profile is None
                        or not record.materialized_files
                        or eligibility_reason(record.model_dump(mode="json")) is not None
                    ):
                        continue
                    raw_size = self.artifacts.verify_artifact_file(
                        record.local_artifact, max_bytes=DATA_MAX_BYTES
                    )
                    if raw_size != record.size_bytes:
                        raise StoragePolicyError(
                            "cached raw dataset size differs from its manifest"
                        )
                    materialized_size = 0
                    for item in record.materialized_files:
                        measured = self.artifacts.verify_artifact_file(
                            item.artifact, max_bytes=DATA_MAX_BYTES
                        )
                        if measured != item.size_bytes:
                            raise StoragePolicyError(
                                "cached materialized dataset size differs from its manifest"
                            )
                        materialized_size += measured
                    if materialized_size != record.materialized_size_bytes:
                        raise StoragePolicyError(
                            "cached materialized byte total differs from its manifest"
                        )
                    verified[record.requirement_key] = record.model_copy(
                        update={
                            "reason": (
                                "Reused a same-run, same-project local dataset after "
                                "revalidating raw and materialized SHA-256, byte size, "
                                "license metadata, and deterministic Profile evidence."
                            )
                        }
                    )
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        return verified

    def _github_json(self, url: str, *, max_bytes: int = 5 * 1024 * 1024) -> dict[str, Any]:
        payload = self.downloader.fetch(
            url,
            DownloadPolicy(
                max_bytes=max_bytes,
                allowed_content_types=frozenset(
                    {"application/json", "application/vnd.github+json"}
                ),
                max_redirects=2,
            ),
        )
        value = json.loads(payload.content.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("GitHub API response must be a JSON object")
        return value

    def _zenodo_dataset_candidates(
        self,
        design_requirements: list[dict[str, Any]],
        search_results: list[dict[str, Any]],
    ) -> list[DatasetCandidate]:
        """Resolve search-matched Zenodo records through their public JSON API."""

        suffix_formats = {
            ".csv": DatasetFormat.CSV,
            ".tsv": DatasetFormat.TSV,
            ".json": DatasetFormat.JSON,
            ".jsonl": DatasetFormat.JSONL,
            ".parquet": DatasetFormat.PARQUET,
            ".txt": DatasetFormat.TXT,
            ".zip": DatasetFormat.ZIP,
            ".7z": DatasetFormat.SEVEN_Z,
        }
        record_ids: list[str] = []
        for result in search_results:
            match = re.search(
                r"https?://(?:www\.)?zenodo\.org/(?:records?|record)/(\d+)",
                str(result.get("url") or ""),
                flags=re.IGNORECASE,
            )
            if match and match.group(1) not in record_ids:
                record_ids.append(match.group(1))

        candidates: list[DatasetCandidate] = []
        for record_id in record_ids[:10]:
            try:
                metadata = self._github_json(
                    f"https://zenodo.org/api/records/{record_id}",
                    max_bytes=5 * 1024 * 1024,
                )
            except (
                DownloadPolicyError,
                ValueError,
                UnicodeError,
                json.JSONDecodeError,
            ):
                continue
            record_metadata = metadata.get("metadata")
            if not isinstance(record_metadata, dict):
                continue
            license_metadata = record_metadata.get("license")
            if not isinstance(license_metadata, dict):
                continue
            license_id = str(license_metadata.get("id") or "").lower()
            if license_id not in ALLOWED_LICENSE_IDS or license_id not in LICENSE_URLS:
                continue
            files = metadata.get("files")
            if not isinstance(files, list):
                continue
            for requirement in design_requirements:
                requirement_key = str(requirement.get("dataset_key") or "")
                if not requirement_key.startswith("dataset-"):
                    continue
                token = requirement_key.removeprefix("dataset-").casefold()
                matches: list[tuple[int, str, str, DatasetFormat]] = []
                for file_record in files:
                    if not isinstance(file_record, dict):
                        continue
                    filename = str(file_record.get("key") or "")
                    size = file_record.get("size")
                    links = file_record.get("links")
                    direct_url = links.get("self") if isinstance(links, dict) else None
                    suffix = next(
                        (
                            value
                            for value in suffix_formats
                            if filename.casefold().endswith(value)
                        ),
                        None,
                    )
                    if (
                        token not in filename.casefold()
                        or suffix is None
                        or not isinstance(size, int)
                        or size <= 0
                        or not isinstance(direct_url, str)
                    ):
                        continue
                    matches.append((size, filename, direct_url, suffix_formats[suffix]))
                if not matches:
                    continue
                size, filename, direct_url, dataset_format = min(
                    matches, key=lambda item: item[0]
                )
                doi = str(record_metadata.get("doi") or f"zenodo.{record_id}")
                candidates.append(
                    DatasetCandidate(
                        dataset_id=f"{requirement_key}-zenodo-{record_id}",
                        requirement_key=requirement_key,
                        name=f"{requirement_key.removeprefix('dataset-').upper()} Zenodo data",
                        description=(
                            f"File {filename} resolved from a search-matched Zenodo record "
                            "and verified through the public record API."
                        ),
                        source=f"Zenodo record {record_id}",
                        landing_page_url=f"https://zenodo.org/records/{record_id}",
                        direct_download_url=direct_url,
                        license_id=license_id,
                        license_url=LICENSE_URLS[license_id][0],
                        license_evidence=(
                            f"Zenodo record API metadata declares license.id={license_id}."
                        ),
                        format=dataset_format,
                        estimated_size_bytes=size,
                        expected_entity_scale=(
                            "Unknown until deterministic local canonicalization and profiling."
                        ),
                        version=doi,
                        relevance_score=1.0,
                    )
                )
        return candidates[:10]

    def _repository_dataset_candidates(
        self,
        design_requirements: list[dict[str, Any]],
        repository_urls: list[str],
    ) -> list[DatasetCandidate]:
        """Find small named datasets in paper-linked repos with API-verified licenses."""

        candidates: list[DatasetCandidate] = []
        suffix_formats = {
            ".csv": DatasetFormat.CSV,
            ".tsv": DatasetFormat.TSV,
            ".json": DatasetFormat.JSON,
            ".jsonl": DatasetFormat.JSONL,
            ".parquet": DatasetFormat.PARQUET,
            ".txt": DatasetFormat.TXT,
            ".zip": DatasetFormat.ZIP,
            ".7z": DatasetFormat.SEVEN_Z,
        }
        for repository_url in repository_urls:
            match = re.fullmatch(
                r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
                repository_url.rstrip("/"),
            )
            if not match:
                continue
            owner, repository = match.groups()
            api_root = f"https://api.github.com/repos/{owner}/{repository}"
            try:
                license_payload = self._github_json(f"{api_root}/license", max_bytes=1024 * 1024)
                license_info = license_payload.get("license")
                if not isinstance(license_info, dict):
                    continue
                license_id = str(license_info.get("spdx_id") or "").lower()
                if license_id not in ALLOWED_LICENSE_IDS or license_id not in LICENSE_URLS:
                    continue
                metadata = self._github_json(api_root, max_bytes=1024 * 1024)
                branch = str(metadata.get("default_branch") or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9._/-]{1,200}", branch):
                    continue
                tree = self._github_json(
                    f"{api_root}/git/trees/{quote(branch, safe='')}?recursive=1"
                )
            except (
                DownloadPolicyError,
                ValueError,
                UnicodeError,
                json.JSONDecodeError,
            ):
                continue
            blobs = tree.get("tree")
            if not isinstance(blobs, list):
                continue
            for requirement in design_requirements:
                requirement_key = str(requirement.get("dataset_key") or "")
                if not requirement_key.startswith("dataset-"):
                    continue
                token = requirement_key.removeprefix("dataset-").casefold()
                matches: list[tuple[int, str, DatasetFormat]] = []
                for blob in blobs:
                    if not isinstance(blob, dict) or blob.get("type") != "blob":
                        continue
                    path = str(blob.get("path") or "")
                    size = blob.get("size")
                    suffix = next(
                        (value for value in suffix_formats if path.lower().endswith(value)),
                        None,
                    )
                    if (
                        token not in path.casefold()
                        or suffix is None
                        or not isinstance(size, int)
                        or size <= 0
                        or size > DATA_MAX_BYTES
                    ):
                        continue
                    matches.append((size, path, suffix_formats[suffix]))
                if not matches:
                    continue
                size, path, dataset_format = min(matches, key=lambda item: item[0])
                safe_path = quote(path, safe="/")
                candidate_id = re.sub(r"[^a-z0-9_-]+", "-", token)[:48]
                candidate = DatasetCandidate(
                    dataset_id=f"dataset-{candidate_id}-paper-repo",
                    requirement_key=requirement_key,
                    name=f"{requirement_key.removeprefix('dataset-').upper()} paper-linked repository data",
                    description=(
                        f"Data blob {path} discovered through the GitHub tree API in a "
                        "paper-linked repository with a root license identified by GitHub."
                    ),
                    source=f"GitHub paper-linked repository {owner}/{repository}",
                    landing_page_url=f"{repository_url}/blob/{quote(branch, safe='')}/{safe_path}",
                    direct_download_url=(
                        f"https://raw.githubusercontent.com/{owner}/{repository}/"
                        f"{quote(branch, safe='')}/{safe_path}"
                    ),
                    license_id=license_id,
                    license_url=LICENSE_URLS[license_id][0],
                    license_evidence=(
                        "GitHub license API identified the repository root license as "
                        f"{license_info.get('name') or license_id} ({license_info.get('spdx_id')})."
                    ),
                    format=dataset_format,
                    estimated_size_bytes=size,
                    expected_entity_scale="Unknown until deterministic local profiling.",
                    version=f"{owner}/{repository}@{branch}",
                    relevance_score=1.0,
                )
                candidates.append(candidate)
        return candidates[:5]

    def _select_candidates(
        self,
        task: TaskEnvelope,
        search_results: list[dict[str, Any]],
        paper_hints: list[dict[str, Any]],
        design_requirements: list[dict[str, Any]],
    ) -> DatasetSelection:
        response = self.model.complete_json(
            PACKAGE.system_prompt,
            json.dumps(
                {
                    "objective": task.objective,
                    "user_revision": task.context.get("user_revision"),
                    "paper_hints": paper_hints,
                    "research_design_dataset_requirements": design_requirements,
                    "untrusted_search_results": search_results,
                    "allowed_requirement_keys": [
                        item.get("dataset_key")
                        for item in design_requirements
                        if item.get("dataset_key")
                    ],
                    "response_contract": DatasetSelection.model_json_schema(),
                },
                ensure_ascii=False,
            ),
        )
        selection = DatasetSelection.model_validate(response)
        requirement_keys = {
            str(item.get("dataset_key"))
            for item in design_requirements
            if item.get("dataset_key")
        }
        for candidate in selection.candidates:
            if requirement_keys and candidate.requirement_key not in requirement_keys:
                raise ValueError("dataset candidate does not map to a supplied design requirement")
        return selection

    @staticmethod
    def _markdown(manifest: DatasetManifest) -> str:
        lines = [
            f"# {manifest.project_slug} 数据获取与规模报告",
            "",
            f"- 候选数：{len(manifest.records)}",
            f"- 已验证下载：{manifest.downloaded_count}",
            f"- 已落地数据本体：{manifest.materialized_count}",
            f"- 需人工复核：{manifest.requires_review_count}",
            f"- 待服务器脚本：{manifest.server_script_count}",
            f"- 实际下载：{manifest.total_downloaded_bytes:,} bytes",
            f"- 解压后规模：{manifest.total_uncompressed_bytes:,} bytes",
            f"- 本地数据本体：{manifest.total_materialized_bytes:,} bytes",
            f"- 已统计记录：{manifest.total_profiled_rows:,}",
            "",
        ]
        for record in manifest.records:
            lines.extend(
                [
                    f"## {record.name} · {record.dataset_id}",
                    "",
                    f"- 设计需求：{record.requirement_key or '未映射'}",
                    f"- 状态：{record.status}",
                    f"- 版本：{record.version or '未确认'}",
                    f"- 来源：{record.source}",
                    f"- 许可：{record.license_id or '未验证'}",
                    f"- 预估下载大小：{record.estimated_size_bytes if record.estimated_size_bytes is not None else '未知'} bytes",
                    f"- 实际下载大小：{record.size_bytes if record.size_bytes is not None else '未下载'} bytes",
                    f"- 预估记录数：{record.expected_record_count if record.expected_record_count is not None else '未知'}",
                    f"- 预估实体规模：{record.expected_entity_scale or '未知'}",
                    f"- 原因/说明：{record.reason}",
                ]
            )
            if record.materialized_root:
                lines.append(f"- 数据本体目录：`{record.materialized_root}`")
            for item in record.materialized_files:
                lines.append(
                    f"- 数据文件：`{item.artifact.uri.removeprefix('artifact://')}` "
                    f"({item.size_bytes:,} bytes, SHA-256 `{item.sha256}`)"
                )
            if record.server_download_script is not None:
                lines.append(
                    "- 服务器下载脚本："
                    f"`{record.server_download_script.uri.removeprefix('artifact://')}`"
                )
            profile = record.profile
            if profile is not None:
                lines.extend(
                    [
                        f"- 解压后大小：{profile.uncompressed_size_bytes:,} bytes",
                        f"- 表：{profile.source_table}",
                        f"- 行/列：{profile.row_count:,} / {profile.column_count}",
                        f"- 化合物数：{profile.unique_compounds if profile.unique_compounds is not None else '未识别'}",
                        f"- 靶点数：{profile.unique_targets if profile.unique_targets is not None else '未识别'}",
                        f"- 唯一配对数：{profile.unique_pairs if profile.unique_pairs is not None else '未识别'}",
                        f"- 重复精确输入组：{profile.repeated_exact_input_group_count}",
                        f"- 同输入异结果组/行：{profile.conflicting_exact_input_label_group_count} / {profile.conflicting_exact_input_label_row_count}",
                        f"- 表征层审计完整：{profile.representation_audit_complete}",
                        f"- 缺失单元格：{profile.missing_cell_count:,} ({profile.missing_fraction:.6f})",
                    ]
                )
                for representation in profile.entity_representation_profiles:
                    lines.append(
                        "- 实体表征映射："
                        f"{representation.entity_role}，标识字段 `{representation.identifier_field}`，"
                        f"表征字段 `{representation.representation_field}`，"
                        f"唯一标识/表征 {representation.unique_identifier_count} / "
                        f"{representation.unique_representation_count}，"
                        f"共享表征组 {representation.shared_representation_group_count}"
                    )
                lines.append(f"- 关键字段覆盖率：{profile.key_field_coverage}")
                for label in profile.label_statistics:
                    lines.extend(
                        [
                            f"- 标签字段：{label.field}",
                            f"- 标签有效数：{label.observed_count:,}",
                            f"- 标签范围：{label.minimum} .. {label.maximum}",
                            f"- 标签均值/标准差：{label.mean} / {label.standard_deviation}",
                            f"- 标签分位数：{label.quantiles}",
                        ]
                    )
                lines.extend(f"- Profile 备注：{note}" for note in profile.notes)
            lines.append("")
        lines.extend(
            [
                "## 进入 Code Make 前的门",
                "",
                "- 仅 `downloaded` 且同时带原始包、本地数据本体、SHA-256、许可证据与 Profile 的数据可进入实现。",
                "- 预计大于 10 GiB 的数据只生成服务器 Shell 脚本；用户执行并复核前不会进入 Code Make。",
                "- 规模和分布以本报告的实际 Profile 为准；论文或页面数字只保留为预估值。",
                "- 任何许可、字段覆盖、泄漏或重复冲突问题必须在训练前关闭或显式降级。",
                "",
            ]
        )
        return "\n".join(lines)

    def _write_server_script(self, task: TaskEnvelope, candidate: Any) -> Any:
        dataset_format = candidate.format.value
        slug = task_project_slug(task)
        filename = f"{candidate.dataset_id}.{dataset_format}"
        destination = scoped_path(
            "data", slug, "server-downloads", candidate.dataset_id, filename
        )
        url = shlex.quote(candidate.direct_download_url)
        script = f"""#!/usr/bin/env bash
set -euo pipefail

workspace_root="${{1:-$(pwd)}}"
destination="$workspace_root/{destination}"
mkdir -p "$(dirname "$destination")"

curl --fail --location --proto '=https' --tlsv1.2 \\
  --retry 5 --retry-all-errors --continue-at - \\
  --output "$destination.part" {url}
mv "$destination.part" "$destination"
shasum -a 256 "$destination"

printf '%s\n' "Downloaded server dataset to: $destination"
printf '%s\n' "Do not continue until Download Data Agent validates and materializes it."
"""
        return self.artifacts.write_agent_bytes(
            self.name,
            scoped_path(
                "data",
                slug,
                "scripts",
                f"download_{candidate.dataset_id}.sh",
            ),
            script.encode("utf-8"),
            max_bytes=64 * 1024,
            kind="dataset_server_download_script",
        )

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        paper_hints = self._paper_hints(task)
        design_requirements = self._design_requirements(task)
        repository_urls = self._design_repository_urls(task)
        cached_records = self._verified_cached_records(task, design_requirements)
        design_text = " ".join(
            [
                *[
                    " ".join(item.get("candidate_names", []))
                    for item in design_requirements
                ],
                *[item.get("acquisition_query", "") for item in design_requirements],
            ]
        )[:1800]
        hint_text = " ".join(item["title"] for item in paper_hints[:3])[:800]
        revision = task.context.get("user_revision")
        revision_comment = (
            str(revision.get("comments", ""))[:1200]
            if isinstance(revision, dict)
            else ""
        )
        query = f"{task.objective} {revision_comment} {hint_text} public dataset repository license download".strip()
        if design_text:
            query = f"{task.objective} {revision_comment} {design_text} official dataset license direct download".strip()
        authority_names = " ".join(
            str(requirement.get("dataset_key") or "")
            .removeprefix("dataset-")
            .upper()
            for requirement in design_requirements
            if requirement.get("dataset_key")
        )[:500]
        search_queries = [query]
        if authority_names:
            search_queries.append(
                f"{authority_names} dataset source code Zenodo"
            )
        raw_results: list[dict[str, Any]] = []
        seen_result_urls: set[str] = set()
        for search_query in search_queries:
            for result in self.search.search(search_query, max_results=10):
                url = str(result.get("url") or "")
                if not url or url in seen_result_urls:
                    continue
                seen_result_urls.add(url)
                raw_results.append(result)
        search_results = bounded_search_results(raw_results, limit=20)
        issues: list[dict[str, str]] = []
        try:
            selection = self._select_candidates(
                task, search_results, paper_hints, design_requirements
            )
        except (ValidationError, ValueError, TypeError) as exc:
            selection = DatasetSelection()
            issues.append(
                {
                    "severity": "blocking",
                    "code": "DATASET_METADATA_INVALID",
                    "message": f"DeepSeek dataset metadata failed schema validation: {type(exc).__name__}",
                    "root_cause_key": "G1:DATASET_METADATA_INVALID:selection",
                }
            )

        authority_candidates = self._zenodo_dataset_candidates(
            design_requirements, search_results
        )
        repository_candidates = self._repository_dataset_candidates(
            design_requirements, repository_urls
        )
        selected_by_requirement = {
            candidate.requirement_key: candidate for candidate in selection.candidates
        }
        for candidate in [*authority_candidates, *repository_candidates]:
            existing = selected_by_requirement.get(candidate.requirement_key)
            existing_reason = (
                eligibility_reason(existing.model_dump(mode="json"))
                if existing is not None
                else "missing"
            )
            if existing is None or existing_reason is not None:
                selected_by_requirement[candidate.requirement_key] = candidate
        selection = DatasetSelection(
            candidates=list(selected_by_requirement.values())[:5]
        )

        records: list[DatasetRecord] = list(cached_records.values())
        downloaded_by_digest: dict[str, str] = {
            str(record.sha256): record.dataset_id
            for record in records
            if record.sha256
        }
        task_token = file_token(task.task_id)
        for candidate in selection.candidates:
            if candidate.requirement_key in cached_records:
                continue
            base = candidate.model_dump(mode="json")
            estimated_size = candidate.estimated_size_bytes or 0
            oversized = estimated_size > DATA_MAX_BYTES
            reason = eligibility_reason(
                base, enforce_local_size_limit=not oversized
            )
            if reason:
                records.append(DatasetRecord(**base, status="requires_review", reason=reason))
                continue

            if oversized:
                records.append(
                    DatasetRecord(
                        **base,
                        status="server_script_required",
                        reason=(
                            "Estimated dataset size exceeds 10 GiB; generated a server "
                            "download script and paused for user execution."
                        ),
                        server_download_script=self._write_server_script(task, candidate),
                    )
                )
                continue

            dataset_format = candidate.format
            if dataset_format is None or candidate.direct_download_url is None:
                records.append(
                    DatasetRecord(
                        **base,
                        status="requires_review",
                        reason="Dataset did not pass deterministic eligibility checks.",
                    )
                )
                continue
            policy = DownloadPolicy(
                max_bytes=DATA_MAX_BYTES,
                allowed_content_types=CONTENT_TYPES_BY_FORMAT[dataset_format],
                max_redirects=3,
                max_archive_uncompressed_bytes=DATA_MAX_BYTES,
            )
            try:
                payload = self._fetch_with_retries(
                    candidate.direct_download_url, policy
                )
                validate_dataset_payload(payload.content, dataset_format.value, policy)
                materialized_payload = payload.content
                materialized_format = dataset_format.value
                if dataset_format == DatasetFormat.TXT:
                    canonical = canonicalize_headerless_dta_text(payload.content)
                    if canonical is not None:
                        materialized_payload = canonical
                        materialized_format = DatasetFormat.CSV.value
                profile = profile_dataset_payload(
                    materialized_payload, materialized_format
                )
                duplicate_of = downloaded_by_digest.get(payload.sha256)
                if duplicate_of:
                    records.append(
                        DatasetRecord(
                            **base,
                            status="duplicate",
                            reason="Binary content duplicates an earlier dataset in this manifest.",
                            final_url=payload.final_url,
                            content_type=payload.content_type,
                            size_bytes=payload.size_bytes,
                            sha256=payload.sha256,
                            duplicate_of=duplicate_of,
                        )
                    )
                    continue
                relative_path = scoped_path(
                    "data",
                    task_project_slug(task),
                    "raw",
                    f"{task_token}_{candidate.dataset_id}.{dataset_format.value}",
                )
                artifact = self.artifacts.write_agent_bytes(
                    self.name,
                    relative_path,
                    payload.content,
                    max_bytes=DATA_MAX_BYTES,
                    kind="dataset_raw",
                )
                materialized_root = scoped_path(
                    "data",
                    task_project_slug(task),
                    "datasets",
                    candidate.dataset_id,
                    task_token,
                )
                materialized_files = materialize_dataset_payload(
                    materialized_payload,
                    materialized_format,
                    artifacts=self.artifacts,
                    agent=self.name,
                    relative_root=materialized_root,
                    max_bytes=DATA_MAX_BYTES,
                )
                downloaded_by_digest[payload.sha256] = candidate.dataset_id
                records.append(
                    DatasetRecord(
                        **base,
                        status="downloaded",
                        reason=(
                            "License, format, size, URL, MIME, payload, and local "
                            "materialization checks passed."
                        ),
                        final_url=payload.final_url,
                        content_type=payload.content_type,
                        size_bytes=payload.size_bytes,
                        sha256=payload.sha256,
                        local_artifact=artifact,
                        profile=profile,
                        materialized_root=materialized_root,
                        materialized_files=materialized_files,
                        materialized_size_bytes=sum(
                            item.size_bytes for item in materialized_files
                        ),
                    )
                )
            except (
                DownloadPolicyError,
                StoragePolicyError,
                ValueError,
                UnicodeError,
            ) as exc:
                records.append(
                    DatasetRecord(
                        **base,
                        status="requires_review",
                        reason=str(exc),
                    )
                )

        downloaded_count = sum(item.status == "downloaded" for item in records)
        review_count = sum(item.status == "requires_review" for item in records)
        materialized_count = sum(
            item.status == "downloaded" and bool(item.materialized_files)
            for item in records
        )
        server_script_count = sum(
            item.status == "server_script_required" for item in records
        )
        all_types = sorted(
            {content_type for values in CONTENT_TYPES_BY_FORMAT.values() for content_type in values}
        )
        manifest = DatasetManifest(
            run_id=task.run_id,
            project_slug=task_project_slug(task),
            task_id=task.task_id,
            objective=task.objective,
            query=query,
            search_result_count=len(search_results),
            records=records,
            downloaded_count=downloaded_count,
            requires_review_count=review_count,
            materialized_count=materialized_count,
            server_script_count=server_script_count,
            total_downloaded_bytes=sum(
                item.size_bytes or 0 for item in records if item.status == "downloaded"
            ),
            total_uncompressed_bytes=sum(
                item.profile.uncompressed_size_bytes
                for item in records
                if item.status == "downloaded" and item.profile is not None
            ),
            total_materialized_bytes=sum(
                item.materialized_size_bytes
                for item in records
                if item.status == "downloaded"
            ),
            total_profiled_rows=sum(
                item.profile.row_count
                for item in records
                if item.status == "downloaded" and item.profile is not None
            ),
            policy=DownloadPolicyRecord(
                max_bytes=DATA_MAX_BYTES,
                max_redirects=3,
                allowed_content_types=all_types,
            ),
            allowed_license_ids=sorted(ALLOWED_LICENSE_IDS),
        )
        manifest_ref = self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "data",
                task_project_slug(task),
                "manifests",
                f"{task_token}.json",
            ),
            manifest.model_dump(mode="json"),
        ).model_copy(update={"kind": "data_manifest"})
        report_ref = self.artifacts.write_agent_bytes(
            self.name,
            scoped_path(
                "data",
                task_project_slug(task),
                "reports",
                f"{task_token}.md",
            ),
            self._markdown(manifest).encode("utf-8"),
            max_bytes=512 * 1024,
            kind="data_profile_report",
        )

        if downloaded_count == 0 and server_script_count == 0 and not issues:
            issues.append(
                {
                    "severity": "blocking",
                    "code": "NO_VERIFIED_DATASET",
                    "message": "No dataset passed automatic license and download checks.",
                    "root_cause_key": "G1:NO_VERIFIED_DATASET:data",
                }
            )
        required_keys = {
            str(item.get("dataset_key"))
            for item in design_requirements
            if item.get("priority") == "required" and item.get("dataset_key")
        }
        downloaded_keys = {
            str(item.requirement_key)
            for item in records
            if item.status == "downloaded" and item.requirement_key
        }
        missing_required_keys = sorted(required_keys - downloaded_keys)
        if missing_required_keys:
            issues.append(
                {
                    "severity": "blocking",
                    "code": "REQUIRED_DATASET_UNSATISFIED",
                    "message": (
                        "Required research-design datasets were not verified locally: "
                        + ", ".join(missing_required_keys)
                    ),
                    "root_cause_key": "G1:REQUIRED_DATASET_UNSATISFIED:data",
                }
            )
        if server_script_count:
            issues.append(
                {
                    "severity": "blocking",
                    "code": "SERVER_DATA_DOWNLOAD_REQUIRED",
                    "message": (
                        "One or more datasets exceed 10 GiB; run the generated Shell "
                        "script on the server before continuing."
                    ),
                    "root_cause_key": "G1:SERVER_DATA_DOWNLOAD_REQUIRED:data",
                }
            )
        succeeded = downloaded_count > 0 and server_script_count == 0 and not any(
            item["severity"] == "blocking" for item in issues
        )
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded" if succeeded else "needs_human",
            summary=(
                f"数据候选 {len(records)} 个；已验证下载 {downloaded_count} 个，"
                f"已落地本体 {materialized_count} 个，待服务器脚本 {server_script_count} 个，"
                f"需人工复核 {review_count} 个；实际下载 {manifest.total_downloaded_bytes:,} bytes，"
                f"统计 {manifest.total_profiled_rows:,} 条记录。"
            ),
            artifacts=[manifest_ref, report_ref],
            evidence=[
                {
                    "check": "dataset-download-policy",
                    "downloaded": downloaded_count,
                    "materialized": materialized_count,
                    "server_scripts": server_script_count,
                    "requires_review": review_count,
                    "license_allowlist_size": len(ALLOWED_LICENSE_IDS),
                    "design_requirement_count": len(design_requirements),
                    "project_slug": manifest.project_slug,
                    "downloaded_bytes": manifest.total_downloaded_bytes,
                    "uncompressed_bytes": manifest.total_uncompressed_bytes,
                    "materialized_bytes": manifest.total_materialized_bytes,
                    "profiled_rows": manifest.total_profiled_rows,
                    "reused_verified_datasets": len(cached_records),
                }
            ],
            issues=issues,
            needs_human=not succeeded,
        )
