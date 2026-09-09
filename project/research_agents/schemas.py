from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import re
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowState(str, Enum):
    ACQUIRING_PAPERS = "ACQUIRING_PAPERS"
    IDEATING = "IDEATING"
    DESIGNING_IDEA = "DESIGNING_IDEA"
    ACQUIRING_DATA = "ACQUIRING_DATA"
    IMPLEMENTING = "IMPLEMENTING"
    CODE_REVIEW = "CODE_REVIEW"
    RUNNING = "RUNNING"
    RESULT_CHECK = "RESULT_CHECK"
    EXPERIMENT_CODE_RUN = "EXPERIMENT_CODE_RUN"
    WRITING = "WRITING"
    CONSISTENCY_REVIEW = "CONSISTENCY_REVIEW"
    PAPER_REVIEW = "PAPER_REVIEW"
    AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


class AgentName(str, Enum):
    LEAD = "lead-agent"
    DOWNLOAD_PAPER = "download-paper-agent"
    DOWNLOAD_DATA = "download-data-agent"
    MAKE_IDEA = "make-idea-agent"
    IDEA_DESIGN = "idea-design-agent"
    CODE_MAKE = "code-make-agent"
    CODE_REVIEW = "code-review-agent"
    SERVER_SCHEDULE = "server-schedule-agent"
    CHECK_RESULT = "check-result-agent"
    EXPERIMENT_CODE_RUN = "experiment-code-run-agent"
    CHECK_PAPER_AND_CODE = "check-paper-and-code-agent"
    WRITE_PAPER = "write-paper-agent"
    PAPER_REVIEW = "paper-review-agent"
    FEEDBACK = "feedback-agent"


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    sha256: str = ""
    kind: str = "generic"


def _validate_http_url(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("a valid HTTP(S) URL is required")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are forbidden")
    return candidate


class DownloadPolicyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(gt=0)
    max_redirects: int = Field(ge=0, le=10)
    allowed_content_types: list[str] = Field(min_length=1)


class PaperQualityPolicyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authoritative_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    recent_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    recent_since_year: int = Field(ge=2000, le=2200)
    portfolio_size: int = Field(default=10, ge=1, le=500)
    authoritative_venues: list[str] = Field(min_length=1)
    require_verified_pdf: bool = True
    arxiv_is_fallback: bool = True

    @model_validator(mode="after")
    def validate_ratio(self) -> "PaperQualityPolicyRecord":
        if abs(self.authoritative_ratio + self.recent_ratio - 1.0) > 1e-9:
            raise ValueError("paper quality ratios must sum to 1")
        return self


class PaperCandidate(BaseModel):
    """Model-extracted paper metadata. It never represents a completed download."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(pattern=r"^paper-[a-zA-Z0-9_-]{1,64}$")
    title: str = Field(min_length=1, max_length=500)
    authors: list[str] = Field(default_factory=list, max_length=100)
    year: int | None = Field(default=None, ge=1800, le=2200)
    doi: str | None = Field(default=None, max_length=300)
    abstract: str = Field(default="", max_length=6000)
    source: str = Field(min_length=1, max_length=300)
    venue: str | None = Field(default=None, max_length=300)
    publication_type: Literal["conference", "journal", "preprint", "other"] = "other"
    landing_page_url: str | None = None
    direct_pdf_url: str | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    _landing_url = field_validator("landing_page_url")(_validate_http_url)
    _pdf_url = field_validator("direct_pdf_url")(_validate_http_url)


class PaperSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[PaperCandidate] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "PaperSelection":
        values = [item.paper_id for item in self.candidates]
        if len(values) != len(set(values)):
            raise ValueError("paper candidate ids must be unique")
        return self


class PaperRecord(PaperCandidate):
    status: Literal["downloaded", "discovered", "requires_review", "duplicate", "rejected"]
    reason: str = Field(default="", max_length=1000)
    quality_tier: Literal[
        "authoritative", "established_peer_reviewed", "recent_peer_reviewed", "recent_arxiv", "arxiv_fallback", "other"
    ] = "other"
    canonical_venue: str | None = Field(default=None, max_length=300)
    portfolio_bucket: Literal["authoritative", "recent", "fallback", "ineligible"] = "ineligible"
    github_repository_url: str | None = None
    github_evidence_url: str | None = None
    github_status: Literal["not_found", "search_matched", "verified_open_source"] = "not_found"
    github_license_id: str | None = Field(default=None, max_length=100)
    final_url: str | None = None
    content_type: str | None = Field(default=None, max_length=200)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_artifact: ArtifactRef | None = None
    duplicate_of: str | None = None

    _final_url = field_validator("final_url")(_validate_http_url)
    _github_url = field_validator("github_repository_url")(_validate_http_url)
    _github_evidence_url = field_validator("github_evidence_url")(_validate_http_url)

    @model_validator(mode="after")
    def validate_download_claim(self) -> "PaperRecord":
        if self.status == "downloaded":
            if (
                self.local_artifact is None
                or not self.sha256
                or not self.final_url
                or not self.content_type
                or not self.size_bytes
            ):
                raise ValueError("downloaded paper must include complete binary evidence")
            paper_path = self.local_artifact.uri.removeprefix("artifact://")
            if not (
                self.local_artifact.uri.startswith("artifact://reference/")
                and "/papers/" in paper_path
            ):
                raise ValueError("downloaded paper artifact is outside a project papers directory")
            if self.local_artifact.sha256 != self.sha256:
                raise ValueError("paper record and artifact digests differ")
        elif self.local_artifact is not None:
            raise ValueError("only downloaded papers may have a local artifact")
        if self.status == "duplicate" and (not self.duplicate_of or not self.sha256):
            raise ValueError("duplicate paper must identify its source and digest")
        if self.github_status != "not_found" and not self.github_repository_url:
            raise ValueError("GitHub status requires a repository URL")
        if self.github_status == "verified_open_source" and not self.github_license_id:
            raise ValueError("verified open-source repository requires a license ID")
        return self


class PaperManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    objective: str
    query: str
    search_result_count: int = Field(ge=0)
    records: list[PaperRecord] = Field(default_factory=list)
    downloaded_count: int = Field(ge=0)
    discovered_count: int = Field(ge=0)
    requires_review_count: int = Field(ge=0)
    rejected_count: int = Field(default=0, ge=0)
    authoritative_downloaded_count: int = Field(default=0, ge=0)
    recent_downloaded_count: int = Field(default=0, ge=0)
    github_marked_count: int = Field(default=0, ge=0)
    portfolio_compliant: bool = False
    manifest_purpose: Literal["portfolio_selection", "evidence_consolidation"] = (
        "portfolio_selection"
    )
    required_paper_ids: list[str] = Field(default_factory=list, max_length=500)
    required_evidence_complete: bool = False
    policy: DownloadPolicyRecord
    quality_policy: PaperQualityPolicyRecord | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_counts(self) -> "PaperManifest":
        if len({item.paper_id for item in self.records}) != len(self.records):
            raise ValueError("paper_id values must be unique within a manifest")
        expected = {
            "downloaded": sum(item.status == "downloaded" for item in self.records),
            "discovered": sum(item.status == "discovered" for item in self.records),
            "review": sum(item.status == "requires_review" for item in self.records),
            "rejected": sum(item.status == "rejected" for item in self.records),
            "authoritative": sum(
                item.status == "downloaded" and item.portfolio_bucket == "authoritative"
                for item in self.records
            ),
            "recent": sum(
                item.status == "downloaded" and item.portfolio_bucket == "recent"
                for item in self.records
            ),
            "github": sum(item.github_status != "not_found" for item in self.records),
        }
        if (
            self.downloaded_count != expected["downloaded"]
            or self.discovered_count != expected["discovered"]
            or self.requires_review_count != expected["review"]
            or self.rejected_count != expected["rejected"]
            or self.authoritative_downloaded_count != expected["authoritative"]
            or self.recent_downloaded_count != expected["recent"]
            or self.github_marked_count != expected["github"]
        ):
            raise ValueError("paper manifest counts do not match record statuses")
        downloaded_ids = {
            item.paper_id for item in self.records if item.status == "downloaded"
        }
        if self.manifest_purpose == "evidence_consolidation":
            if not self.required_paper_ids:
                raise ValueError("evidence consolidation requires paper IDs")
            expected_complete = set(self.required_paper_ids).issubset(downloaded_ids)
            if self.required_evidence_complete != expected_complete:
                raise ValueError(
                    "required evidence completeness does not match downloaded records"
                )
        elif self.required_paper_ids or self.required_evidence_complete:
            raise ValueError(
                "portfolio manifests cannot claim a targeted evidence contract"
            )
        return self


class DatasetFormat(str, Enum):
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    TXT = "txt"
    ZIP = "zip"
    SEVEN_Z = "7z"


class DatasetLabelStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=300)
    observed_count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0)
    quantiles: dict[str, float] = Field(default_factory=dict, max_length=10)


class DatasetRepresentationProfile(BaseModel):
    """Measured identifier-to-model-representation cardinality for one entity role."""

    model_config = ConfigDict(extra="forbid")

    entity_role: str = Field(min_length=1, max_length=100)
    identifier_field: str = Field(min_length=1, max_length=300)
    representation_field: str = Field(min_length=1, max_length=300)
    unique_identifier_count: int = Field(ge=0)
    unique_representation_count: int = Field(ge=0)
    shared_representation_group_count: int = Field(ge=0)


class DatasetProfile(BaseModel):
    """Measured local profile; values are derived from the downloaded bytes."""

    model_config = ConfigDict(extra="forbid")

    profile_status: Literal["complete", "partial"]
    source_table: str = Field(min_length=1, max_length=1000)
    archive_member_count: int = Field(default=1, ge=1)
    compressed_size_bytes: int = Field(ge=1)
    uncompressed_size_bytes: int = Field(ge=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list, max_length=300)
    unique_compounds: int | None = Field(default=None, ge=0)
    unique_targets: int | None = Field(default=None, ge=0)
    unique_pairs: int | None = Field(default=None, ge=0)
    entity_representation_profiles: list[DatasetRepresentationProfile] = Field(
        default_factory=list, max_length=20
    )
    repeated_exact_input_group_count: int = Field(default=0, ge=0)
    conflicting_exact_input_label_group_count: int = Field(default=0, ge=0)
    conflicting_exact_input_label_row_count: int = Field(default=0, ge=0)
    representation_audit_complete: bool = False
    missing_cell_count: int = Field(default=0, ge=0)
    missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    key_field_coverage: dict[str, float] = Field(default_factory=dict, max_length=30)
    label_statistics: list[DatasetLabelStatistics] = Field(default_factory=list, max_length=20)
    notes: list[str] = Field(default_factory=list, max_length=30)


class DatasetCandidate(BaseModel):
    """Model-extracted dataset metadata; code independently decides eligibility."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^dataset-[a-zA-Z0-9_-]{1,64}$")
    requirement_key: str | None = Field(
        default=None, pattern=r"^dataset-[a-zA-Z0-9_-]{1,64}$"
    )
    name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=6000)
    source: str = Field(min_length=1, max_length=300)
    landing_page_url: str | None = None
    direct_download_url: str | None = None
    license_id: str | None = Field(default=None, max_length=100)
    license_url: str | None = None
    license_evidence: str = Field(default="", max_length=1000)
    format: DatasetFormat | None = None
    estimated_size_bytes: int | None = Field(default=None, ge=0)
    expected_record_count: int | None = Field(default=None, ge=0)
    expected_entity_scale: str = Field(default="", max_length=2000)
    version: str | None = Field(default=None, max_length=200)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    _landing_url = field_validator("landing_page_url")(_validate_http_url)
    _download_url = field_validator("direct_download_url")(_validate_http_url)
    _license_url = field_validator("license_url")(_validate_http_url)

    @field_validator("license_id")
    @classmethod
    def normalize_license_id(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None

    @field_validator("format", mode="before")
    @classmethod
    def normalize_format(cls, value: Any) -> Any:
        return value.strip().lower().lstrip(".") if isinstance(value, str) else value


class DatasetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[DatasetCandidate] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "DatasetSelection":
        values = [item.dataset_id for item in self.candidates]
        if len(values) != len(set(values)):
            raise ValueError("dataset candidate ids must be unique")
        return self


class DatasetMaterializedFile(BaseModel):
    """One persisted, directly usable file from a downloaded dataset."""

    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1, max_length=2000)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_materialized_file(self) -> "DatasetMaterializedFile":
        path = self.relative_path.strip().replace("\\", "/")
        if path.startswith(("/", "~")) or ".." in path.split("/"):
            raise ValueError("materialized dataset paths must be relative and traversal-free")
        if not self.artifact.uri.startswith("artifact://data/") or "/datasets/" not in self.artifact.uri:
            raise ValueError("materialized dataset file must be stored below data/<project>/datasets")
        if self.artifact.sha256 != self.sha256:
            raise ValueError("materialized file and artifact digests differ")
        return self


class DatasetRecord(DatasetCandidate):
    status: Literal[
        "downloaded",
        "requires_review",
        "duplicate",
        "server_script_required",
    ]
    reason: str = Field(default="", max_length=1000)
    final_url: str | None = None
    content_type: str | None = Field(default=None, max_length=200)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_artifact: ArtifactRef | None = None
    duplicate_of: str | None = None
    profile: DatasetProfile | None = None
    materialized_root: str | None = Field(default=None, max_length=2000)
    materialized_files: list[DatasetMaterializedFile] = Field(
        default_factory=list, max_length=1000
    )
    materialized_size_bytes: int = Field(default=0, ge=0)
    server_download_script: ArtifactRef | None = None

    _final_url = field_validator("final_url")(_validate_http_url)

    @model_validator(mode="after")
    def validate_download_claim(self) -> "DatasetRecord":
        if self.status == "downloaded":
            if (
                self.local_artifact is None
                or not self.sha256
                or not self.final_url
                or not self.content_type
                or not self.size_bytes
                or self.format is None
                or not self.license_id
                or not self.license_url
                or not self.license_evidence.strip()
                or self.profile is None
                or not self.materialized_root
                or not self.materialized_files
                or self.materialized_size_bytes <= 0
            ):
                raise ValueError(
                    "downloaded dataset must include raw and materialized binary evidence"
                )
            data_path = self.local_artifact.uri.removeprefix("artifact://")
            if not (
                self.local_artifact.uri.startswith("artifact://data/")
                and "/raw/" in data_path
            ):
                raise ValueError("downloaded dataset artifact is outside a project raw directory")
            if self.local_artifact.sha256 != self.sha256:
                raise ValueError("dataset record and artifact digests differ")
            if not self.materialized_root.startswith("data/") or "/datasets/" not in self.materialized_root:
                raise ValueError("materialized dataset root is outside data/<project>/datasets")
            if self.materialized_size_bytes != sum(
                item.size_bytes for item in self.materialized_files
            ):
                raise ValueError("materialized byte total does not match its file records")
            if self.server_download_script is not None:
                raise ValueError("downloaded datasets must not carry a server script")
        elif self.status == "server_script_required":
            if self.server_download_script is None:
                raise ValueError("large datasets must include a server download script")
            if (
                not self.server_download_script.uri.startswith("artifact://data/")
                or "/scripts/" not in self.server_download_script.uri
            ):
                raise ValueError("server download script is outside data/<project>/scripts")
            if self.local_artifact is not None or self.materialized_files:
                raise ValueError("server-script datasets must not claim a local download")
        elif self.local_artifact is not None or self.materialized_files:
            raise ValueError("only downloaded datasets may have local dataset artifacts")
        if self.status == "duplicate" and (not self.duplicate_of or not self.sha256):
            raise ValueError("duplicate dataset must identify its source and digest")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    run_id: str
    project_slug: str = Field(
        pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$"
    )
    task_id: str
    objective: str
    query: str
    search_result_count: int = Field(ge=0)
    records: list[DatasetRecord] = Field(default_factory=list)
    downloaded_count: int = Field(ge=0)
    requires_review_count: int = Field(ge=0)
    materialized_count: int = Field(ge=0)
    server_script_count: int = Field(ge=0)
    total_downloaded_bytes: int = Field(ge=0)
    total_uncompressed_bytes: int = Field(ge=0)
    total_materialized_bytes: int = Field(ge=0)
    total_profiled_rows: int = Field(ge=0)
    policy: DownloadPolicyRecord
    allowed_license_ids: list[str] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_counts(self) -> "DatasetManifest":
        if len({item.dataset_id for item in self.records}) != len(self.records):
            raise ValueError("dataset_id values must be unique within a manifest")
        downloaded = sum(item.status == "downloaded" for item in self.records)
        review = sum(item.status == "requires_review" for item in self.records)
        materialized = sum(
            item.status == "downloaded" and bool(item.materialized_files)
            for item in self.records
        )
        scripts = sum(item.status == "server_script_required" for item in self.records)
        if (
            self.downloaded_count != downloaded
            or self.requires_review_count != review
            or self.materialized_count != materialized
            or self.server_script_count != scripts
        ):
            raise ValueError("dataset manifest counts do not match record statuses")
        downloaded_records = [item for item in self.records if item.status == "downloaded"]
        if self.total_downloaded_bytes != sum(item.size_bytes or 0 for item in downloaded_records):
            raise ValueError("dataset manifest downloaded bytes do not match records")
        if self.total_uncompressed_bytes != sum(
            item.profile.uncompressed_size_bytes
            for item in downloaded_records
            if item.profile is not None
        ):
            raise ValueError("dataset manifest uncompressed bytes do not match profiles")
        if self.total_materialized_bytes != sum(
            item.materialized_size_bytes for item in downloaded_records
        ):
            raise ValueError("dataset manifest materialized bytes do not match records")
        if self.total_profiled_rows != sum(
            item.profile.row_count
            for item in downloaded_records
            if item.profile is not None
        ):
            raise ValueError("dataset manifest row count does not match profiles")
        return self


class ResearchIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea_id: str = Field(pattern=r"^idea-[a-zA-Z0-9_-]{1,64}$")
    title: str = Field(min_length=1, max_length=500)
    research_question: str = Field(min_length=10, max_length=3000)
    novelty_claim: str = Field(min_length=10, max_length=3000)
    rationale: str = Field(min_length=10, max_length=5000)
    evidence_refs: list[str] = Field(min_length=2, max_length=30)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(pattern=r"^hypothesis-[a-zA-Z0-9_-]{1,64}$")
    statement: str = Field(min_length=10, max_length=3000)
    expected_direction: str = Field(min_length=1, max_length=1000)
    falsification_criterion: str = Field(min_length=10, max_length=3000)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(pattern=r"^metric-[a-zA-Z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=300)
    definition: str = Field(min_length=5, max_length=3000)
    direction: Literal["higher", "lower", "two_sided"]
    primary: bool = False


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^experiment-[a-zA-Z0-9_-]{1,64}$")
    objective: str = Field(min_length=10, max_length=3000)
    dataset_ids: list[str] = Field(min_length=1, max_length=20)
    baseline_methods: list[str] = Field(min_length=1, max_length=20)
    proposed_method: str = Field(min_length=5, max_length=3000)
    metric_ids: list[str] = Field(min_length=1, max_length=20)
    ablations: list[str] = Field(min_length=1, max_length=30)
    controls: list[str] = Field(default_factory=list, max_length=30)
    random_seeds: list[int] = Field(min_length=1, max_length=50)
    success_criteria: list[str] = Field(min_length=1, max_length=20)


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_id: str = Field(pattern=r"^risk-[a-zA-Z0-9_-]{1,64}$")
    category: Literal["data", "method", "evaluation", "ethics", "reproducibility"]
    description: str = Field(min_length=5, max_length=3000)
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high"]
    mitigation: str = Field(min_length=5, max_length=3000)


class IdeaCandidate(BaseModel):
    """A paper-grounded novelty candidate, intentionally free of implementation detail."""

    model_config = ConfigDict(extra="forbid")

    idea: ResearchIdea
    hypothesis: Hypothesis
    innovation_type: Literal[
        "data_representation",
        "model_architecture",
        "learning_objective",
        "evaluation_protocol",
        "uncertainty_calibration",
        "efficiency",
        "interpretability",
        "hybrid",
    ]
    baseline_gap: str = Field(min_length=10, max_length=3000)
    expected_contribution: str = Field(min_length=10, max_length=3000)
    differentiation: list[str] = Field(min_length=1, max_length=12)
    feasibility_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    implementation_risk: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def validate_candidate_evidence(self) -> "IdeaCandidate":
        idea_refs = set(self.idea.evidence_refs)
        hypothesis_refs = set(self.hypothesis.evidence_refs)
        if not hypothesis_refs.issubset(idea_refs):
            raise ValueError("hypothesis evidence must be included in the idea evidence")
        return self


class IdeaDraft(BaseModel):
    """Make Idea output: rank novelty candidates; do not design code or datasets here."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[IdeaCandidate] = Field(min_length=3, max_length=8)
    recommended_idea_id: str
    selection_rationale: str = Field(min_length=20, max_length=5000)
    diversity_note: str = Field(min_length=10, max_length=3000)
    limitations: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_portfolio(self) -> "IdeaDraft":
        idea_ids = [item.idea.idea_id for item in self.candidates]
        if len(idea_ids) != len(set(idea_ids)):
            raise ValueError("idea candidate ids must be unique")
        if self.recommended_idea_id not in idea_ids:
            raise ValueError("recommended_idea_id must identify a supplied candidate")
        if len({item.innovation_type for item in self.candidates}) < 2:
            raise ValueError("idea portfolio must cover at least two innovation types")
        return self


class IdeaProposal(IdeaDraft):
    schema_version: Literal["2.0"] = "2.0"
    run_id: str
    task_id: str
    objective: str
    degraded: bool = False
    evidence_quality: Literal["downloaded", "metadata_only"]
    paper_evidence_uris: list[str] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class MethodComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(pattern=r"^component-[a-zA-Z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=10, max_length=3000)
    inputs: list[str] = Field(min_length=1, max_length=20)
    outputs: list[str] = Field(min_length=1, max_length=20)
    innovation: str = Field(min_length=10, max_length=3000)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    dependency_ids: list[str] = Field(default_factory=list, max_length=20)


class ReusableCodeModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str = Field(pattern=r"^reuse-[a-zA-Z0-9_-]{1,64}$")
    repository_url: str
    source_paper_id: str
    repository_status: Literal["verified_open_source", "search_matched"]
    license_id: str | None = Field(default=None, max_length=100)
    reuse_mode: Literal["integrate", "reference_only", "reimplement"]
    reusable_parts: list[str] = Field(min_length=1, max_length=20)
    integration_strategy: str = Field(min_length=10, max_length=3000)
    maturity_evidence: list[str] = Field(default_factory=list, max_length=20)
    fallback_implementation: str = Field(min_length=10, max_length=3000)

    _repository_url = field_validator("repository_url")(_validate_http_url)

    @model_validator(mode="after")
    def validate_reuse_rights(self) -> "ReusableCodeModule":
        if self.reuse_mode == "integrate":
            if self.repository_status != "verified_open_source" or not self.license_id:
                raise ValueError(
                    "code integration requires a verified open-source repository and license"
                )
        return self


class CodePackageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=500)
    responsibility: str = Field(min_length=10, max_length=2000)
    key_interfaces: list[str] = Field(min_length=1, max_length=20)
    reference_module_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("path")
    @classmethod
    def validate_relative_code_path(cls, value: str) -> str:
        candidate = value.strip()
        if candidate.startswith(("/", "~")) or ".." in candidate.split("/"):
            raise ValueError("code package paths must be relative and traversal-free")
        return candidate


class CodeBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: list[str] = Field(min_length=5, max_length=30)
    packages: list[CodePackageSpec] = Field(min_length=3, max_length=30)
    entrypoints: list[str] = Field(min_length=1, max_length=20)
    config_contracts: list[str] = Field(min_length=1, max_length=30)
    tests: list[str] = Field(min_length=3, max_length=40)
    dependencies: list[str] = Field(min_length=1, max_length=40)
    minimum_viable_implementation: list[str] = Field(min_length=3, max_length=20)


class NumericRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float
    high: float

    @model_validator(mode="after")
    def validate_order(self) -> "NumericRange":
        if self.low > self.high:
            raise ValueError("range low must not exceed high")
        return self


class ExpectedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(pattern=r"^result-[a-zA-Z0-9_-]{1,64}$")
    experiment_id: str
    metric: str = Field(min_length=1, max_length=300)
    expected_direction: Literal["higher", "lower", "stable", "two_sided"]
    target_delta: NumericRange | None = None
    expectation: str = Field(min_length=10, max_length=3000)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    failure_threshold: str = Field(min_length=10, max_length=2000)
    claim_status: Literal["predicted_not_observed"] = "predicted_not_observed"


class PaperComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    method_name: str = Field(min_length=1, max_length=500)
    shared_elements: list[str] = Field(min_length=1, max_length=20)
    proposed_differences: list[str] = Field(min_length=1, max_length=20)
    reusable_elements: list[str] = Field(default_factory=list, max_length=20)
    fair_comparison_conditions: list[str] = Field(min_length=1, max_length=20)


class DatasetRequirement(BaseModel):
    """A pre-download requirement; reported scale/distribution are hypotheses to verify."""

    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(pattern=r"^dataset-[a-zA-Z0-9_-]{1,64}$")
    source_kind: Literal["external_dataset"]
    candidate_names: list[str] = Field(min_length=1, max_length=10)
    priority: Literal["required", "preferred", "optional"]
    acquisition_query: str = Field(min_length=5, max_length=1000)
    official_source_urls: list[str] = Field(default_factory=list, max_length=10)
    contents: list[str] = Field(min_length=1, max_length=30)
    required_fields: list[str] = Field(min_length=1, max_length=40)
    reported_scale: str = Field(min_length=5, max_length=2000)
    expected_distribution: str = Field(min_length=5, max_length=3000)
    distribution_checks: list[str] = Field(min_length=2, max_length=30)
    split_strategy: str = Field(min_length=10, max_length=3000)
    reliability_basis: str = Field(min_length=10, max_length=3000)
    license_requirement: str = Field(min_length=5, max_length=1000)
    paper_refs: list[str] = Field(min_length=1, max_length=30)
    alternatives: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("official_source_urls")
    @classmethod
    def validate_source_urls(cls, values: list[str]) -> list[str]:
        return [str(_validate_http_url(value)) for value in values]

    @model_validator(mode="after")
    def reject_derived_experiment_as_download(self) -> "DatasetRequirement":
        text = " ".join(
            [
                self.dataset_key,
                *self.candidate_names,
                self.acquisition_query,
                *self.contents,
            ]
        ).casefold()
        derived_markers = (
            "permuted",
            "shuffled label",
            "shuffled-label",
            "negative control",
            "synthetic residual",
        )
        if any(marker in text for marker in derived_markers):
            raise ValueError(
                "derived experimental transformations must not be external dataset acquisition requirements"
            )
        return self


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^experiment-[a-zA-Z0-9_-]{1,64}$")
    experiment_type: Literal[
        "main_benchmark",
        "ablation",
        "case_study",
        "module_analysis",
        "feasibility",
        "sensitivity",
        "ood_generalization",
        "efficiency",
        "uncertainty_calibration",
    ]
    objective: str = Field(min_length=10, max_length=3000)
    dataset_keys: list[str] = Field(min_length=1, max_length=20)
    baselines: list[str] = Field(min_length=1, max_length=30)
    variables: list[str] = Field(min_length=1, max_length=30)
    controls: list[str] = Field(min_length=1, max_length=30)
    metrics: list[str] = Field(min_length=1, max_length=30)
    random_seeds: list[int] = Field(min_length=3, max_length=50)
    protocol: list[str] = Field(min_length=2, max_length=40)
    success_criteria: list[str] = Field(min_length=1, max_length=20)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    computational_only: Literal[True] = True

    @model_validator(mode="after")
    def reject_non_computational_experiments(self) -> "ExperimentPlan":
        text = " ".join([self.objective, *self.protocol]).lower()
        forbidden = (
            "wet lab",
            "wet-lab",
            "cell culture",
            "animal experiment",
            "clinical trial",
            "生物实验",
            "湿实验",
            "动物实验",
            "临床实验",
        )
        if any(term in text for term in forbidden):
            raise ValueError("only computational experiments are allowed")
        return self


class ImplementationMilestone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    milestone_id: str = Field(pattern=r"^milestone-[a-zA-Z0-9_-]{1,64}$")
    deliverable: str = Field(min_length=5, max_length=2000)
    prerequisites: list[str] = Field(default_factory=list, max_length=20)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=20)
    fallback: str = Field(min_length=5, max_length=2000)


class ImplementationRoadmap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    milestones: list[ImplementationMilestone] = Field(min_length=3, max_length=20)
    estimated_compute: str = Field(min_length=5, max_length=2000)
    stop_conditions: list[str] = Field(min_length=1, max_length=20)


class DesignDecision(BaseModel):
    """One ADR-style research or engineering decision with an explicit revisit rule."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(pattern=r"^decision-[a-zA-Z0-9_-]{1,64}$")
    question: str = Field(min_length=10, max_length=2000)
    options: list[str] = Field(min_length=2, max_length=12)
    selected_option: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=10, max_length=3000)
    tradeoffs: list[str] = Field(min_length=1, max_length=20)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    revisit_trigger: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def selected_option_must_be_declared(self) -> "DesignDecision":
        if self.selected_option not in self.options:
            raise ValueError("selected design option must be one of the declared options")
        return self


class DataAuditPlan(BaseModel):
    """Pre-download and post-download checks that gate every dataset used by the design."""

    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(pattern=r"^dataset-[a-zA-Z0-9_-]{1,64}$")
    schema_checks: list[str] = Field(min_length=2, max_length=30)
    leakage_checks: list[str] = Field(min_length=2, max_length=30)
    duplicate_conflict_policy: str = Field(min_length=10, max_length=3000)
    missingness_policy: str = Field(min_length=10, max_length=3000)
    distribution_shift_checks: list[str] = Field(min_length=2, max_length=30)
    acceptance_gates: list[str] = Field(min_length=2, max_length=30)


class StatisticalAnalysisPlan(BaseModel):
    """Predeclared statistical interpretation shared by all computational experiments."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = Field(min_length=1, max_length=500)
    secondary_metrics: list[str] = Field(min_length=1, max_length=30)
    analysis_unit: str = Field(min_length=5, max_length=1000)
    effect_size: str = Field(min_length=5, max_length=1000)
    confidence_interval: str = Field(min_length=5, max_length=1000)
    alpha: float = Field(gt=0.0, lt=1.0)
    multiple_testing_correction: str = Field(min_length=5, max_length=1000)
    seed_aggregation: str = Field(min_length=5, max_length=1000)
    random_seeds: list[int] = Field(min_length=3, max_length=50)
    minimum_random_seeds: int = Field(ge=3, le=50)
    minimum_successful_seeds: int = Field(ge=2, le=50)
    precision_or_power_analysis: str = Field(min_length=10, max_length=2000)
    assumption_checks: list[str] = Field(min_length=1, max_length=30)
    robustness_checks: list[str] = Field(min_length=1, max_length=30)
    interval_width_normalization: str | None = Field(
        default=None, min_length=10, max_length=1000
    )
    negative_control_policy: str | None = Field(
        default=None, min_length=20, max_length=1500
    )
    practical_significance_threshold: str = Field(min_length=5, max_length=1000)
    failure_decision_rule: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def validate_fixed_seed_contract(self) -> "StatisticalAnalysisPlan":
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("statistical random seeds must be unique")
        if self.minimum_random_seeds != len(self.random_seeds):
            raise ValueError(
                "minimum_random_seeds must equal the predeclared fixed seed count"
            )
        if self.minimum_successful_seeds > self.minimum_random_seeds:
            raise ValueError(
                "minimum_successful_seeds cannot exceed the fixed seed count"
            )
        plan_text = " ".join(
            (
                self.primary_metric,
                *self.secondary_metrics,
                self.effect_size,
                self.practical_significance_threshold,
                self.failure_decision_rule,
            )
        ).casefold()
        interval_analysis = any(
            marker in plan_text
            for marker in (
                "coverage",
                "prediction interval",
                "confidence interval width",
                "interval width",
                "conformal",
            )
        )
        normalization_text = (self.interval_width_normalization or "").casefold()
        if interval_analysis and not any(
            marker in normalization_text
            for marker in (
                "dimensionless",
                "iqr",
                "standard deviation",
                "rmse",
                "dataset-specific",
                "per-dataset",
            )
        ):
            raise ValueError(
                "interval analyses must use a predeclared dataset-specific dimensionless width normalization"
            )
        negative_text = (self.negative_control_policy or "").casefold()
        conformal_negative_control = (
            any(marker in negative_text for marker in ("shuffle", "permut", "negative control"))
            and any(marker in plan_text for marker in ("coverage", "conformal"))
        )
        if conformal_negative_control and not (
            ("coverage alone" in negative_text or "coverage by itself" in negative_text)
            and any(
                marker in negative_text
                for marker in ("width", "efficiency", "error", "informativeness")
            )
        ):
            raise ValueError(
                "conformal negative controls cannot be judged by coverage alone and require informativeness diagnostics"
            )
        precision_text = self.precision_or_power_analysis.casefold()
        if not any(
            marker in precision_text
            for marker in (
                "power",
                "precision",
                "minimum sample",
                "minimum entity",
                "detectable effect",
                "confidence interval width",
                "simulation-based",
            )
        ):
            raise ValueError(
                "statistical plans require a prospective power, precision, or minimum-sample justification"
            )
        if "seed" in self.analysis_unit.casefold():
            raise ValueError("random seeds cannot be the primary inferential analysis unit")
        confidence_text = self.confidence_interval.casefold()
        if (
            "bootstrap" in confidence_text
            and "seed" in confidence_text
            and not any(
                re.search(rf"\b{re.escape(unit)}s?\b", confidence_text)
                for unit in (
                    "pair",
                    "drug",
                    "target",
                    "entity",
                    "cluster",
                    "sample",
                    "observation",
                )
            )
        ):
            raise ValueError(
                "bootstrap confidence intervals must resample paired analysis units or entity clusters, not only random seeds"
            )
        seed_text = self.seed_aggregation.casefold()
        if not (
            "descriptive" in seed_text
            and any(
                marker in seed_text
                for marker in (
                    "not the inference unit",
                    "not an inference unit",
                    "not used as the inference unit",
                    "not inferential units",
                )
            )
        ):
            raise ValueError(
                "random seeds must be summarized descriptively and explicitly excluded as the inferential unit"
            )
        decision_text = " ".join(
            (
                self.effect_size,
                self.practical_significance_threshold,
                self.failure_decision_rule,
            )
        ).casefold()
        coverage_is_gated = "coverage" in decision_text
        all_seed_gate = any(
            marker in f"{seed_text} {decision_text}"
            for marker in (
                "all seeds must pass",
                "every seed must pass",
                "all fixed seeds pass",
                "every fixed seed pass",
                "3/3",
                "3-of-3",
                "three-of-three",
            )
        )
        if coverage_is_gated and all_seed_gate:
            raise ValueError(
                "empirical coverage cannot use an all-seed point-estimate quorum as its scientific acceptance test"
            )
        if coverage_is_gated:
            coverage_decision = self.failure_decision_rule.casefold()
            if not any(
                marker in coverage_decision
                for marker in (
                    "exact binomial",
                    "clopper-pearson",
                    "wilson",
                    "bootstrap interval",
                    "bootstrap confidence interval",
                    "confidence interval for coverage",
                    "coverage confidence interval",
                )
            ):
                raise ValueError(
                    "coverage decisions require a predeclared finite-sample confidence interval or exact binomial test"
                )
            if not any(
                marker in confidence_text
                for marker in (
                    "pair",
                    "drug",
                    "target",
                    "entity",
                    "cluster",
                    "sample",
                    "observation",
                )
            ):
                raise ValueError(
                    "coverage uncertainty must be estimated over observations or predeclared entity clusters"
                )
        primary_text = self.primary_metric.casefold()
        if "coverage" in primary_text:
            decision_text = " ".join(
                (
                    self.effect_size,
                    self.practical_significance_threshold,
                    self.failure_decision_rule,
                )
            ).casefold()
            if not (
                ("upper" in decision_text or "above" in decision_text or ">" in decision_text)
                and ("lower" in decision_text or "below" in decision_text or "<" in decision_text)
            ):
                raise ValueError(
                    "coverage-primary plans require explicit lower and upper acceptance bounds"
                )
            if not any(
                efficiency in decision_text
                for efficiency in ("interval width", "mean width", "median width", "efficiency")
            ):
                raise ValueError(
                    "coverage-primary plans require a predeclared interval-width or efficiency gate"
                )
        return self


class ExperimentStagePlan(BaseModel):
    """Budgeted stage gate for baseline-first, falsifiable experiment execution."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(pattern=r"^stage-[a-zA-Z0-9_-]{1,64}$")
    stage_type: Literal[
        "data_validation",
        "baseline_reproduction",
        "tuning",
        "proposed_method",
        "ablation",
        "stress_test",
    ]
    objective: str = Field(min_length=10, max_length=3000)
    attempt_budget: int = Field(ge=1, le=20)
    entry_gate: list[str] = Field(min_length=1, max_length=20)
    exit_gate: list[str] = Field(min_length=1, max_length=20)
    stop_conditions: list[str] = Field(min_length=1, max_length=20)
    escalation: str = Field(min_length=10, max_length=2000)


class AdvisorReport(BaseModel):
    """Read-only, strongly typed output of one bounded Idea Design auxiliary Agent."""

    model_config = ConfigDict(extra="forbid")

    advisor: Literal[
        "method-architecture-advisor",
        "code-reuse-architecture-advisor",
        "data-decision-advisor",
        "experiment-decision-advisor",
        "critical-risk-advisor",
    ]
    recommendations: list[str] = Field(min_length=2, max_length=20)
    decisions: list[str] = Field(min_length=1, max_length=20)
    blocking_concerns: list[str] = Field(default_factory=list, max_length=20)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    assumptions: list[str] = Field(min_length=1, max_length=20)
    issue_codes: list[str] = Field(default_factory=list, max_length=20)


class ResearchDesignDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_idea_id: str
    method_name: str = Field(min_length=1, max_length=500)
    executive_summary: str = Field(min_length=20, max_length=6000)
    method_components: list[MethodComponent] = Field(min_length=2, max_length=30)
    reusable_code_modules: list[ReusableCodeModule] = Field(min_length=1, max_length=20)
    code_blueprint: CodeBlueprint
    dataset_requirements: list[DatasetRequirement] = Field(min_length=1, max_length=20)
    experiment_plans: list[ExperimentPlan] = Field(min_length=3, max_length=30)
    expected_results: list[ExpectedResult] = Field(min_length=1, max_length=30)
    paper_comparisons: list[PaperComparison] = Field(min_length=2, max_length=30)
    design_decisions: list[DesignDecision] = Field(min_length=3, max_length=30)
    data_audit_plans: list[DataAuditPlan] = Field(min_length=1, max_length=20)
    statistical_analysis: StatisticalAnalysisPlan
    experiment_stages: list[ExperimentStagePlan] = Field(min_length=4, max_length=20)
    advisor_consensus: list[AdvisorReport] = Field(min_length=5, max_length=5)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=30)
    claim_boundaries: list[str] = Field(min_length=1, max_length=30)
    implementation_roadmap: ImplementationRoadmap
    risks: list[Risk] = Field(min_length=1, max_length=30)
    limitations: list[str] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "ResearchDesignDraft":
        component_ids = {item.component_id for item in self.method_components}
        for component in self.method_components:
            if not set(component.dependency_ids).issubset(component_ids):
                raise ValueError("method component references an unknown dependency")
            if component.component_id in component.dependency_ids:
                raise ValueError("method component cannot depend on itself")
        module_ids = {item.module_id for item in self.reusable_code_modules}
        for package in self.code_blueprint.packages:
            if not set(package.reference_module_ids).issubset(module_ids):
                raise ValueError("code package references an unknown reusable module")
        dataset_keys = {item.dataset_key for item in self.dataset_requirements}
        experiment_ids = {item.experiment_id for item in self.experiment_plans}
        if len(dataset_keys) != len(self.dataset_requirements):
            raise ValueError("dataset requirement keys must be unique")
        if len(experiment_ids) != len(self.experiment_plans):
            raise ValueError("experiment ids must be unique")
        for experiment in self.experiment_plans:
            if not set(experiment.dataset_keys).issubset(dataset_keys):
                raise ValueError("experiment references an unknown dataset requirement")
            if experiment.random_seeds != self.statistical_analysis.random_seeds:
                raise ValueError(
                    "every experiment must use the statistical plan's fixed random seeds"
                )
        self._validate_cold_start_tier_semantics()
        if not all(item.experiment_id in experiment_ids for item in self.expected_results):
            raise ValueError("expected result references an unknown experiment")
        experiment_types = {item.experiment_type for item in self.experiment_plans}
        if "main_benchmark" not in experiment_types or "ablation" not in experiment_types:
            raise ValueError("design must include a main benchmark and ablation")
        if not experiment_types.intersection({"case_study", "module_analysis", "feasibility"}):
            raise ValueError("design must include case study, module analysis, or feasibility")
        audit_dataset_keys = {item.dataset_key for item in self.data_audit_plans}
        if audit_dataset_keys != dataset_keys:
            raise ValueError("every dataset requirement must have exactly one data audit plan")
        self._validate_dataset_semantics()
        stage_types = {item.stage_type for item in self.experiment_stages}
        required_stages = {
            "baseline_reproduction",
            "tuning",
            "proposed_method",
            "ablation",
        }
        if not required_stages.issubset(stage_types):
            raise ValueError("experiment stages must cover baseline, tuning, method, and ablation")
        advisor_roles = [item.advisor for item in self.advisor_consensus]
        if len(set(advisor_roles)) != 5:
            raise ValueError("advisor consensus must contain all five distinct advisors")
        return self

    def _validate_dataset_semantics(self) -> None:
        """Require domain-appropriate data semantics before scientific claims."""

        audits = {item.dataset_key: item for item in self.data_audit_plans}
        outcome_markers = (
            "label", "outcome", "response", "score", "affinity", "class",
            "endpoint", "target value", "dependent variable",
        )
        shift_markers = (
            "cold", "ood", "out-of-distribution", "unseen", "held-out",
            "held out", "group split", "cluster split", "domain shift",
            "temporal split", "time split", "site split", "geographic split",
            "scaffold split",
        )
        representation_markers = (
            "representation", "feature", "exact input", "identical input",
            "similarity", "cluster", "group", "subject", "patient", "site",
            "time", "geograph", "scaffold", "sequence", "image", "document",
            "speaker",
        )
        measurement_markers = (
            "measurement", "label", "outcome", "endpoint", "response",
        )
        semantic_markers = (
            "unit", "scale", "coding", "definition", "harmonization",
            "normalization", "ontology",
        )

        for requirement in self.dataset_requirements:
            requirement_text = " ".join(
                (
                    requirement.dataset_key,
                    *requirement.candidate_names,
                    requirement.acquisition_query,
                    *requirement.contents,
                    *requirement.required_fields,
                    requirement.split_strategy,
                )
            ).casefold()
            if not any(marker in requirement_text for marker in outcome_markers):
                continue
            audit = audits[requirement.dataset_key]
            audit_text = " ".join(
                (
                    *audit.schema_checks,
                    *audit.leakage_checks,
                    audit.duplicate_conflict_policy,
                    audit.missingness_policy,
                    *audit.distribution_shift_checks,
                    *audit.acceptance_gates,
                )
            ).casefold()
            duplicate_text = audit.duplicate_conflict_policy.casefold()
            if not (
                "conflict" in duplicate_text
                and any(
                    marker in duplicate_text
                    for marker in (
                        "duplicate", "same input", "identical input",
                        "exact input", "model input", "feature vector",
                        "representation",
                    )
                )
            ):
                raise ValueError(
                    f"{requirement.dataset_key} must define a policy for repeated model inputs with conflicting outcomes"
                )
            if not (
                any(marker in audit_text for marker in measurement_markers)
                and any(marker in audit_text for marker in semantic_markers)
            ):
                raise ValueError(
                    f"{requirement.dataset_key} must audit measurement/outcome meaning, units, scale, coding, or harmonization"
                )
            if any(marker in requirement.split_strategy.casefold() for marker in shift_markers):
                leakage_text = " ".join(
                    (*audit.leakage_checks, *audit.acceptance_gates)
                ).casefold()
                if not (
                    any(marker in leakage_text for marker in ("overlap", "leak", "disjoint"))
                    and any(marker in leakage_text for marker in representation_markers)
                ):
                    raise ValueError(
                        f"{requirement.dataset_key} claims a shifted/held-out split without a representation- or group-level leakage audit"
                    )

        for experiment in self.experiment_plans:
            if len(set(experiment.dataset_keys)) < 2:
                continue
            protocol_text = " ".join((experiment.objective, *experiment.protocol)).casefold()
            if not any(
                marker in protocol_text
                for marker in (
                    "per-dataset", "separate model", "separately", "no pooling",
                    "not pool", "harmonization", "harmonized", "meta-analysis",
                    "hierarchical", "domain-specific", "dataset-specific",
                    "dataset covariate",
                )
            ):
                raise ValueError(
                    f"{experiment.experiment_id} combines datasets without an explicit separation, harmonization, or hierarchical synthesis protocol"
                )

    def _validate_cold_start_tier_semantics(self) -> None:
        """Reject impossible subgroup claims under a jointly disjoint test split.

        If the primary split makes both drug and target identities/clusters disjoint
        from training, every member of that split is necessarily ``both-new``.  A
        design may still study drug-only and target-only cold start, but they must be
        separately constructed diagnostic splits rather than partitions of the
        jointly disjoint test set.
        """

        main_plans = [
            item for item in self.experiment_plans if item.experiment_type == "main_benchmark"
        ]
        for plan in main_plans:
            text = " ".join((plan.objective, *plan.protocol)).casefold()
            invalid_products = re.finditer(
                r"\b(?:d_test\s*[x×*]\s*d_test|t_test\s*[x×*]\s*t_test)\b",
                text,
            )
            if any(
                not self._cartesian_product_is_explicitly_rejected(text, match.start())
                for match in invalid_products
            ):
                raise ValueError(
                    "drug-target test pairs must use the drug-by-target product D_test x T_test, not D_test x D_test or T_test x T_test"
                )
        for plan in self.experiment_plans:
            text = " ".join(
                (plan.objective, *plan.variables, *plan.protocol)
            ).casefold()
            nominal_ninety = any(
                marker in text
                for marker in (
                    "nominal 0.90",
                    "nominal confidence 0.90",
                    "90% confidence",
                    "90 percent confidence",
                )
            )
            if nominal_ninety and re.search(
                r"\balpha\s*(?:=|of)?\s*0\.90\b", text
            ):
                raise ValueError(
                    "90% conformal confidence requires miscoverage alpha=0.10, not alpha=0.90"
                )
            if re.search(
                r"\b(?:nominal\s+)?alpha\s+(?:levels?|values?)\s*[:=]?\s*0\.80\s*,\s*0\.90\s*,\s*0\.95\b",
                text,
            ):
                raise ValueError(
                    "0.80, 0.90 and 0.95 are nominal confidence/coverage levels, not conformal alpha values"
                )
        joint_disjoint = any(
            (
                ("pairwise disjoint" in text or "disjoint" in text)
                and "drug" in text
                and "target" in text
                and any(marker in text for marker in ("train", "training"))
                and "test" in text
            )
            for plan in main_plans
            for text in [" ".join((plan.objective, *plan.protocol)).casefold()]
        )
        if not joint_disjoint:
            return
        for plan in self.experiment_plans:
            text = " ".join(
                (plan.objective, *plan.variables, *plan.controls, *plan.protocol)
            ).casefold()
            has_all_tiers = all(
                marker in text
                for marker in ("new-drug-only", "new-target-only", "both-new")
            )
            partitions_primary_test = (
                "partition" in text
                and "test" in text
                and not any(
                    marker in text
                    for marker in (
                        "separate diagnostic split",
                        "separately constructed diagnostic split",
                        "independent diagnostic split",
                    )
                )
            )
            if has_all_tiers and partitions_primary_test:
                raise ValueError(
                    "a jointly drug-and-target-disjoint primary test set contains only both-new pairs; drug-only and target-only cold start require separately constructed diagnostic splits"
                )

    @staticmethod
    def _cartesian_product_is_explicitly_rejected(text: str, start: int) -> bool:
        """Allow an invalid product to be named only as an explicit anti-pattern.

        Prompts and generated protocols often state the valid product followed by
        wording such as ``never D_test x D_test``.  Treating that educational
        warning as if the design selected the invalid product creates a false
        positive.  Limit negation detection to the current clause so a rejection
        elsewhere cannot mask an affirmative invalid assignment.
        """

        clause_start = max(
            text.rfind(".", 0, start),
            text.rfind(";", 0, start),
            text.rfind(":", 0, start),
            text.rfind("\n", 0, start),
        )
        prefix = text[clause_start + 1 : start]
        return bool(
            re.search(
                r"\b(?:never|not|invalid|incorrect|forbidden|prohibited|avoid|"
                r"reject(?:ed|s)?|rather\s+than|instead\s+of)\b[^.;:\n]{0,80}$",
                prefix,
            )
        )


class ResearchDesign(ResearchDesignDraft):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    objective: str
    idea_proposal_uri: str
    paper_evidence_uris: list[str] = Field(min_length=1)
    evidence_quality: Literal["downloaded", "metadata_only"]
    created_at: datetime = Field(default_factory=utc_now)


class GeneratedCodeFileDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=10, max_length=2000)
    language: Literal["python", "toml", "yaml", "json", "markdown", "text"]
    content: str = Field(min_length=1, max_length=120_000)
    design_component_ids: list[str] = Field(default_factory=list, max_length=30)
    dataset_keys: list[str] = Field(default_factory=list, max_length=20)
    experiment_ids: list[str] = Field(default_factory=list, max_length=30)
    source_module_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("path")
    @classmethod
    def validate_code_file_path(cls, value: str) -> str:
        candidate = value.strip().replace("\\", "/")
        parts = candidate.split("/")
        if candidate.startswith(("/", "~")) or ".." in parts or any(not part for part in parts):
            raise ValueError("generated code path must be relative and traversal-free")
        allowed_suffixes = (".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt")
        if not candidate.endswith(allowed_suffixes):
            raise ValueError("generated code file extension is not allowlisted")
        return candidate


class CodeTraceabilityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(min_length=5, max_length=2000)
    file_paths: list[str] = Field(min_length=1, max_length=30)
    design_component_ids: list[str] = Field(default_factory=list, max_length=30)
    dataset_keys: list[str] = Field(default_factory=list, max_length=20)
    experiment_ids: list[str] = Field(default_factory=list, max_length=30)
    source_module_ids: list[str] = Field(default_factory=list, max_length=20)
    acceptance_test: str = Field(min_length=5, max_length=2000)


class CodeGenerationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    architecture_summary: str = Field(min_length=20, max_length=6000)
    files: list[GeneratedCodeFileDraft] = Field(min_length=10, max_length=80)
    entrypoints: list[str] = Field(min_length=1, max_length=20)
    dependencies: list[str] = Field(min_length=1, max_length=50)
    validation_commands: list[str] = Field(min_length=2, max_length=30)
    traceability: list[CodeTraceabilityItem] = Field(min_length=2, max_length=40)
    known_gaps: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_generated_tree(self) -> "CodeGenerationDraft":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("generated code paths must be unique")
        package_root = f"src/{self.package_name}/"
        if not any(path.startswith(package_root) and path.endswith(".py") for path in paths):
            raise ValueError("generated code must include its declared Python src package")
        if not any(path.startswith("tests/") and path.endswith(".py") for path in paths):
            raise ValueError("generated code must include tests")
        if not any(path.endswith(("pyproject.toml", "requirements.txt")) for path in paths):
            raise ValueError("generated code must declare dependencies")
        required_prefixes = (
            f"{package_root}models/",
            f"{package_root}experiments/",
            f"{package_root}training/",
            f"{package_root}testing/",
            "configs/",
        )
        for prefix in required_prefixes:
            if not any(path.startswith(prefix) for path in paths):
                raise ValueError(f"generated code is missing required layer: {prefix}")
        model_files = [
            path.removeprefix(f"{package_root}models/")
            for path in paths
            if path.startswith(f"{package_root}models/") and path.endswith(".py")
        ]
        if not any("/" in path for path in model_files):
            raise ValueError("each model must live in a named subdirectory")
        experiment_files = [
            path.removeprefix(f"{package_root}experiments/")
            for path in paths
            if path.startswith(f"{package_root}experiments/") and path.endswith(".py")
        ]
        if not any("/" in path for path in experiment_files):
            raise ValueError("each experiment must live in a named subdirectory")
        if f"{package_root}demo.py" not in paths:
            raise ValueError("generated code must include a local minimal demo entrypoint")
        path_set = set(paths)
        for item in self.traceability:
            if not set(item.file_paths).issubset(path_set):
                raise ValueError("traceability references an unknown generated file")
        return self


class GeneratedCodePatchFileDraft(BaseModel):
    """One complete replacement file for an existing failed Code Make candidate."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=10, max_length=2000)
    language: Literal["python", "toml", "yaml", "json", "markdown", "text"]
    content: str = Field(min_length=1, max_length=120_000)

    @field_validator("path")
    @classmethod
    def validate_patch_path(cls, value: str) -> str:
        candidate = value.strip().replace("\\", "/")
        parts = candidate.split("/")
        if candidate.startswith(("/", "~")) or ".." in parts or any(not part for part in parts):
            raise ValueError("generated patch path must be relative and traversal-free")
        allowed_suffixes = (".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt")
        if not candidate.endswith(allowed_suffixes):
            raise ValueError("generated patch file extension is not allowlisted")
        return candidate


class CodeRevisionPatchDraft(BaseModel):
    """Bounded replacements applied only to a verified, failed generated-code tree."""

    model_config = ConfigDict(extra="forbid")

    baseline_task_id: str = Field(pattern=r"^task_[A-Za-z0-9_-]{3,120}$")
    architecture_summary: str = Field(min_length=20, max_length=6000)
    files: list[GeneratedCodePatchFileDraft] = Field(min_length=1, max_length=8)
    known_gaps: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_unique_patch_paths(self) -> "CodeRevisionPatchDraft":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("generated patch paths must be unique")
        return self


class GeneratedCodeFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    purpose: str
    language: str
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact: ArtifactRef
    design_component_ids: list[str] = Field(default_factory=list)
    dataset_keys: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    source_module_ids: list[str] = Field(default_factory=list)


class VendoredSourceRecord(BaseModel):
    """A non-executed, content-addressed local snapshot of an integrated repository."""

    model_config = ConfigDict(extra="forbid")

    module_id: str = Field(pattern=r"^reuse-[a-zA-Z0-9_-]{1,64}$")
    repository_url: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_id: str = Field(min_length=1, max_length=100)
    reuse_mode: Literal["integrate"] = "integrate"
    local_root: str = Field(min_length=1, max_length=500)
    file_count: int = Field(ge=1, le=20_000)
    total_bytes: int = Field(ge=1)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest: ArtifactRef

    _repository_url = field_validator("repository_url")(_validate_http_url)

    @model_validator(mode="after")
    def validate_local_snapshot(self) -> "VendoredSourceRecord":
        expected = f"artifact://{self.local_root.rstrip('/')}/SOURCE_MANIFEST.json"
        if self.source_manifest.uri != expected:
            raise ValueError("vendored source manifest is outside its declared local root")
        return self


class LocalValidationRecord(BaseModel):
    """One fixed local command run by Code Make before requesting review."""

    model_config = ConfigDict(extra="forbid")

    validation_id: Literal["minimal_demo", "unit_tests"]
    argv: list[str] = Field(min_length=2, max_length=20)
    status: Literal["passed", "failed", "timed_out"]
    return_code: int | None = None
    duration_ms: float = Field(ge=0)
    timeout_seconds: int = Field(ge=1, le=120)
    stdout_tail: str = Field(default="", max_length=8000)
    stderr_tail: str = Field(default="", max_length=8000)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "LocalValidationRecord":
        if self.status == "timed_out" and self.return_code is not None:
            raise ValueError("timed-out validation cannot claim a return code")
        if self.status == "passed" and self.return_code != 0:
            raise ValueError("passed validation requires return code zero")
        return self


class CodeImplementationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    run_id: str
    task_id: str
    project_slug: str = Field(
        pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$"
    )
    package_name: str
    status: Literal["generated_validated_unreviewed"] = "generated_validated_unreviewed"
    implementation_root: str = Field(min_length=1, max_length=500)
    research_design_uri: str
    data_manifest_uri: str
    architecture_summary: str
    files: list[GeneratedCodeFileRecord] = Field(min_length=10, max_length=80)
    total_bytes: int = Field(ge=1)
    entrypoints: list[str]
    dependencies: list[str]
    validation_commands: list[str]
    traceability: list[CodeTraceabilityItem]
    vendored_sources: list[VendoredSourceRecord] = Field(default_factory=list, max_length=20)
    local_validations: list[LocalValidationRecord] = Field(min_length=2, max_length=10)
    known_gaps: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_file_totals(self) -> "CodeImplementationManifest":
        if self.total_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("code manifest total bytes do not match file records")
        expected_root = f"code/{self.project_slug}/implementations/"
        if not self.implementation_root.startswith(expected_root):
            raise ValueError("implementation root escaped its project revision namespace")
        artifact_root = f"artifact://{self.implementation_root.rstrip('/')}/"
        if any(not item.artifact.uri.startswith(artifact_root) for item in self.files):
            raise ValueError("generated file escaped its project code namespace")
        validation_ids = {item.validation_id for item in self.local_validations}
        if validation_ids != {"minimal_demo", "unit_tests"}:
            raise ValueError("code manifest must include demo and unit-test validations")
        if any(item.status != "passed" for item in self.local_validations):
            raise ValueError("unvalidated code cannot claim a review-ready manifest")
        module_ids = [item.module_id for item in self.vendored_sources]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("vendored source module IDs must be unique")
        return self


class CodeReviewCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^review-[a-z0-9-]{2,80}$")
    category: Literal[
        "architecture",
        "traceability",
        "design_alignment",
        "scientific_protocol",
        "vendored_source",
        "static_analysis",
        "execution",
        "training_update",
        "reliability",
    ]
    status: Literal["passed", "failed", "warning"]
    summary: str = Field(min_length=5, max_length=3000)
    evidence: list[str] = Field(default_factory=list, max_length=50)
    file_paths: list[str] = Field(default_factory=list, max_length=50)


class CodeReviewExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: Literal["minimal_demo_first", "minimal_demo_repeat", "unit_tests"]
    argv: list[str] = Field(min_length=2, max_length=20)
    status: Literal["passed", "failed", "timed_out"]
    return_code: int | None = None
    duration_ms: float = Field(ge=0)
    timeout_seconds: int = Field(ge=1, le=120)
    stdout_tail: str = Field(default="", max_length=8000)
    stderr_tail: str = Field(default="", max_length=8000)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)


class CodeReviewReport(BaseModel):
    """Deterministic review verdict plus actionable Code Make repair instructions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    code_manifest_uri: str
    review_iteration: int = Field(ge=1, le=5)
    verdict: Literal["passed", "needs_revision"]
    summary: str = Field(min_length=10, max_length=6000)
    checks: list[CodeReviewCheck] = Field(min_length=7, max_length=100)
    executions: list[CodeReviewExecutionRecord] = Field(min_length=3, max_length=10)
    blockers: list[str] = Field(default_factory=list, max_length=100)
    recommendations: list[str] = Field(default_factory=list, max_length=100)
    covered_component_ids: list[str] = Field(default_factory=list)
    covered_dataset_keys: list[str] = Field(default_factory=list)
    covered_experiment_ids: list[str] = Field(default_factory=list)
    covered_source_module_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_verdict(self) -> "CodeReviewReport":
        failed = [item for item in self.checks if item.status == "failed"]
        if self.verdict == "passed":
            if failed or self.blockers:
                raise ValueError("passed review cannot contain failed checks or blockers")
        elif not failed or not self.blockers or not self.recommendations:
            raise ValueError("revision verdict requires failed checks, blockers, and recommendations")
        return self


class ServerProfileConfig(BaseModel):
    """Credential-free SSH profile for one explicitly configured compute server."""

    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    credential_prefix: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    accelerator: Literal["cuda", "cpu"] = "cuda"
    workspace_root: str = Field(min_length=5, max_length=500)
    enabled: bool = True
    execution_enabled: bool = False
    priority: int = Field(default=50, ge=0, le=100)
    gpu_memory_reserve_mib: int = Field(default=2048, ge=0, le=262144)
    max_gpu_utilization_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    min_available_memory_bytes: int = Field(default=4 * 1024**3, ge=0)
    min_available_disk_bytes: int = Field(default=20 * 1024**3, ge=0)
    conda_executable: str = Field(default="conda", pattern=r"^[A-Za-z0-9_./-]{1,200}$")

    @field_validator("workspace_root")
    @classmethod
    def validate_workspace_root(cls, value: str, info) -> str:
        candidate = value.strip().rstrip("/")
        path = PurePosixPath(candidate)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("server workspace_root must be an absolute traversal-free path")
        if candidate in {"", "/", "/home", "/root", "/tmp", "/var", "/srv"}:
            raise ValueError("server workspace_root is too broad")
        if (
            info.data.get("accelerator") != "cpu"
            and any(character.isspace() for character in candidate)
        ):
            raise ValueError("server workspace_root cannot contain whitespace")
        return candidate

    @field_validator("conda_executable")
    @classmethod
    def validate_conda_executable(cls, value: str) -> str:
        candidate = value.strip()
        path = PurePosixPath(candidate)
        if ".." in path.parts or PurePosixPath(candidate).name != "conda":
            raise ValueError("conda_executable must identify a conda binary without traversal")
        return candidate


class ExecutionMetricGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1, max_length=200)
    direction: Literal["higher", "lower"]
    target_value: float

    @field_validator("target_value")
    @classmethod
    def validate_finite_target(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric gate target must be finite")
        return value


class ProjectExecutionConfig(BaseModel):
    """Explicit project command and metric contract; argv is never evaluated by a shell."""

    model_config = ConfigDict(extra="forbid")

    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    train_argv: list[str] = Field(min_length=2, max_length=64)
    distributed_mode: Literal["seed_parallel", "torchrun"] = "seed_parallel"
    expected_job_seconds: int = Field(default=3600, ge=60, le=30 * 86400)
    server_demo_timeout_seconds: int = Field(default=120, ge=10, le=1800)
    metric_name: str = Field(min_length=1, max_length=200)
    metric_direction: Literal["higher", "lower"]
    target_value: float | None = None
    additional_metric_gates: list[ExecutionMetricGateConfig] = Field(
        default_factory=list, max_length=20
    )
    conda_dependencies: list[str] = Field(
        default_factory=lambda: ["python=3.11", "pip"], min_length=2, max_length=50
    )
    pip_install_argv: list[str] = Field(
        default_factory=lambda: ["python", "-m", "pip", "install", "-e", "."],
        min_length=4,
        max_length=30,
    )

    @field_validator("train_argv", "pip_install_argv")
    @classmethod
    def validate_argv(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            token = value.strip()
            if not token or len(token) > 1000 or any(char in token for char in ("\x00", "\n", "\r")):
                raise ValueError("execution argv contains an invalid token")
            cleaned.append(token)
        executable = PurePosixPath(cleaned[0]).name
        if executable not in {"python", "python3", "torchrun"}:
            raise ValueError("execution argv must use python, python3, or torchrun")
        return cleaned

    @field_validator("target_value")
    @classmethod
    def validate_target_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("target metric must be finite")
        return value

    @model_validator(mode="after")
    def validate_metric_gates(self) -> "ProjectExecutionConfig":
        names = [self.metric_name, *(item.metric_name for item in self.additional_metric_gates)]
        if len(names) != len(set(names)):
            raise ValueError("execution metric gate names must be unique")
        return self


class ServerExecutionConfig(BaseModel):
    """Local, gitignored fleet configuration with no password or private-key fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    report_interval_seconds: Literal[43200] = 43200
    servers: list[ServerProfileConfig] = Field(min_length=1, max_length=64)
    projects: list[ProjectExecutionConfig] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> "ServerExecutionConfig":
        server_ids = [item.server_id for item in self.servers]
        project_slugs = [item.project_slug for item in self.projects]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("server IDs must be unique")
        if len(project_slugs) != len(set(project_slugs)):
            raise ValueError("project execution configs must be unique")
        return self


class GPUResourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, le=1024)
    name: str = Field(min_length=1, max_length=300)
    memory_total_mib: int = Field(gt=0)
    memory_used_mib: int = Field(ge=0)
    utilization_percent: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_memory(self) -> "GPUResourceSnapshot":
        if self.memory_used_mib > self.memory_total_mib:
            raise ValueError("GPU used memory exceeds total memory")
        return self

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


class ServerResourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    status: Literal["available", "unreachable", "ineligible"]
    cpu_count: int = Field(default=0, ge=0)
    load_1m: float = Field(default=0.0, ge=0.0)
    memory_total_bytes: int = Field(default=0, ge=0)
    memory_available_bytes: int = Field(default=0, ge=0)
    disk_total_bytes: int = Field(default=0, ge=0)
    disk_available_bytes: int = Field(default=0, ge=0)
    conda_available: bool = False
    gpus: list[GPUResourceSnapshot] = Field(default_factory=list, max_length=128)
    error_code: str | None = Field(default=None, max_length=200)
    probed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_totals(self) -> "ServerResourceSnapshot":
        if self.memory_available_bytes > self.memory_total_bytes:
            raise ValueError("available host memory exceeds total memory")
        if self.disk_available_bytes > self.disk_total_bytes:
            raise ValueError("available disk exceeds total disk")
        indices = [item.index for item in self.gpus]
        if len(indices) != len(set(indices)):
            raise ValueError("GPU indices must be unique per server")
        return self


class ServerTransferItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transfer_id: str = Field(pattern=r"^transfer-[a-zA-Z0-9_-]{1,80}$")
    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    kind: Literal["data", "model", "code", "runtime"]
    local_artifact: ArtifactRef
    local_relative_path: str = Field(min_length=1, max_length=1000)
    remote_relative_path: str = Field(min_length=1, max_length=1000)
    size_bytes: int = Field(gt=0)

    @field_validator("local_relative_path", "remote_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = value.strip().replace("\\", "/")
        path = PurePosixPath(candidate)
        if path.is_absolute() or ".." in path.parts or not candidate:
            raise ValueError("transfer paths must be relative and traversal-free")
        return candidate


class ScheduledExperimentJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^job-[a-zA-Z0-9_-]{1,100}$")
    experiment_id: str = Field(pattern=r"^experiment-[a-zA-Z0-9_-]{1,64}$")
    seed: int
    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    gpu_ids: list[int] = Field(min_length=1, max_length=128)
    command_argv: list[str] = Field(min_length=2, max_length=80)
    remote_workdir: str = Field(min_length=5, max_length=1000)
    log_path: str = Field(min_length=5, max_length=1000)
    metrics_path: str = Field(min_length=5, max_length=1000)
    best_checkpoint_dir: str = Field(min_length=5, max_length=1000)
    gpu_lock_path: str = Field(min_length=5, max_length=1000)
    expected_duration_seconds: int = Field(ge=60, le=30 * 86400)
    metric_name: str = Field(min_length=1, max_length=200)
    metric_direction: Literal["higher", "lower"]
    status: Literal["planned", "launched"]
    remote_job_id: str | None = Field(default=None, max_length=200)
    started_at: datetime | None = None
    next_report_at: datetime | None = None

    @model_validator(mode="after")
    def validate_launch(self) -> "ScheduledExperimentJob":
        if len(self.gpu_ids) != len(set(self.gpu_ids)):
            raise ValueError("job GPU IDs must be unique")
        for value in (
            self.remote_workdir,
            self.log_path,
            self.metrics_path,
            self.best_checkpoint_dir,
            self.gpu_lock_path,
        ):
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("remote job paths must be absolute and traversal-free")
        if self.status == "launched" and not (self.remote_job_id and self.started_at and self.next_report_at):
            raise ValueError("launched jobs require remote ID and report timestamps")
        return self


class ServerDemoRecord(BaseModel):
    """Foreground server-side smoke test that must pass before persistent launch."""

    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    argv: list[str] = Field(min_length=2, max_length=30)
    gpu_ids: list[int] = Field(min_length=1, max_length=128)
    status: Literal["passed", "failed", "timed_out"]
    return_code: int | None = None
    duration_ms: float = Field(ge=0.0)
    timeout_seconds: int = Field(ge=10, le=1800)
    log_path: str = Field(min_length=5, max_length=1000)
    output_tail: str = Field(default="", max_length=16000)

    @model_validator(mode="after")
    def validate_demo(self) -> "ServerDemoRecord":
        if self.status == "passed" and self.return_code != 0:
            raise ValueError("passed server demo requires return code zero")
        if self.status == "timed_out" and self.return_code is not None:
            raise ValueError("timed-out server demo cannot claim a return code")
        path = PurePosixPath(self.log_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("server demo log path must be absolute and traversal-free")
        return self


class ServerScheduleManifest(BaseModel):
    """Immutable fleet probe, transfer, environment, allocation, and launch evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    status: Literal["launched", "blocked"]
    code_manifest_uri: str
    data_manifest_uri: str
    research_design_uri: str
    code_review_uri: str
    report_interval_seconds: Literal[43200] = 43200
    probes: list[ServerResourceSnapshot] = Field(min_length=1, max_length=64)
    available_gpu_keys: list[str] = Field(default_factory=list, max_length=4096)
    selected_gpu_keys: list[str] = Field(default_factory=list, max_length=4096)
    transfers: list[ServerTransferItem] = Field(default_factory=list, max_length=5000)
    environment_names: dict[str, str] = Field(default_factory=dict, max_length=64)
    server_workspaces: dict[str, str] = Field(default_factory=dict, max_length=64)
    server_demos: list[ServerDemoRecord] = Field(default_factory=list, max_length=64)
    jobs: list[ScheduledExperimentJob] = Field(default_factory=list, max_length=2000)
    estimated_makespan_seconds: int = Field(default=0, ge=0)
    issues: list[str] = Field(default_factory=list, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_schedule(self) -> "ServerScheduleManifest":
        job_ids = [item.job_id for item in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("scheduled job IDs must be unique")
        if len(self.available_gpu_keys) != len(set(self.available_gpu_keys)):
            raise ValueError("available GPU keys must be unique")
        if len(self.selected_gpu_keys) != len(set(self.selected_gpu_keys)):
            raise ValueError("selected GPU keys must be unique")
        if self.status == "launched":
            transfer_kinds = {item.kind for item in self.transfers}
            if not {"data", "model"}.issubset(transfer_kinds):
                raise ValueError("launch requires local data and model transfers")
            if not self.jobs or any(item.status != "launched" for item in self.jobs):
                raise ValueError("launched schedule requires launched jobs")
            if set(self.selected_gpu_keys) != set(self.available_gpu_keys):
                raise ValueError("launched schedule must use every eligible GPU")
            if not self.environment_names or not self.server_workspaces:
                raise ValueError("launched schedule requires fresh conda environments and workspaces")
            demo_servers = {item.server_id for item in self.server_demos if item.status == "passed"}
            if demo_servers != set(self.environment_names):
                raise ValueError("every selected server must pass its foreground minimum demo")
        return self


class MetricObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1, max_length=200)
    value: float
    step: int | None = Field(default=None, ge=0)
    checkpoint_path: str | None = Field(default=None, max_length=1000)

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric observations must be finite")
        return value


class JobObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^job-[a-zA-Z0-9_-]{1,100}$")
    status: Literal["queued", "running", "completed", "failed", "unknown"]
    exit_code: int | None = None
    metrics: list[MetricObservation] = Field(default_factory=list, max_length=10000)
    best_metric: MetricObservation | None = None
    best_checkpoint_path: str | None = Field(default=None, max_length=1000)
    log_tail: str = Field(default="", max_length=16000)
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_completion(self) -> "JobObservation":
        if self.status == "completed" and self.exit_code != 0:
            raise ValueError("completed job requires exit code zero")
        if self.status == "failed" and self.exit_code == 0:
            raise ValueError("failed job cannot claim exit code zero")
        return self


class MetricGateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^metric-[a-z0-9-]{2,100}$")
    metric_name: str = Field(min_length=1, max_length=200)
    status: Literal["passed", "failed", "pending"]
    observed_value: float | None = None
    target_value: float | None = None
    direction: Literal["higher", "lower"]
    summary: str = Field(min_length=5, max_length=2000)


class ResultCheckReport(BaseModel):
    """One persistent result-monitoring report; running reports repeat without a retry cap."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    schedule_manifest_uri: str
    report_sequence: int = Field(ge=1)
    verdict: Literal["in_progress", "passed", "needs_improvement", "failed"]
    summary: str = Field(min_length=10, max_length=6000)
    observations: list[JobObservation] = Field(min_length=1, max_length=2000)
    metric_checks: list[MetricGateCheck] = Field(default_factory=list, max_length=200)
    best_log_artifacts: list[ArtifactRef] = Field(default_factory=list, max_length=2000)
    recommended_route: AgentName | None = None
    report_interval_seconds: Literal[43200] = 43200
    next_report_at: datetime | None = None
    issues: list[str] = Field(default_factory=list, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_verdict(self) -> "ResultCheckReport":
        statuses = {item.status for item in self.observations}
        failed_checks = [item for item in self.metric_checks if item.status == "failed"]
        if self.verdict == "in_progress":
            if self.next_report_at is None or not statuses.intersection({"queued", "running", "unknown"}):
                raise ValueError("in-progress report requires active jobs and next report time")
            if self.recommended_route is not None:
                raise ValueError("in-progress report cannot recommend a repair route")
        elif self.verdict == "passed":
            if statuses != {"completed"} or failed_checks:
                raise ValueError("passed result report requires completed jobs and passing metrics")
        elif self.recommended_route is None:
            raise ValueError("non-passing terminal result requires a repair route")
        return self


class MissingExperimentRequirement(BaseModel):
    """A computational experiment that must enter Idea Design before execution."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^experiment-gap-[a-zA-Z0-9_-]{1,64}$")
    experiment_type: Literal[
        "ablation",
        "case_study",
        "module_analysis",
        "feasibility",
        "sensitivity",
        "ood_generalization",
        "efficiency",
        "uncertainty_calibration",
    ]
    objective: str = Field(min_length=10, max_length=3000)
    rationale: str = Field(min_length=10, max_length=3000)
    required_metrics: list[str] = Field(min_length=1, max_length=20)
    evidence_refs: list[str] = Field(min_length=1, max_length=30)
    required_code_changes: list[str] = Field(default_factory=list, max_length=30)
    computational_only: Literal[True] = True

    @model_validator(mode="after")
    def reject_non_computational_gap(self) -> "MissingExperimentRequirement":
        text = " ".join(
            [self.objective, self.rationale, *self.required_code_changes]
        ).lower()
        forbidden = (
            "wet lab",
            "wet-lab",
            "cell culture",
            "animal experiment",
            "clinical trial",
            "生物实验",
            "湿实验",
            "动物实验",
            "临床实验",
        )
        if any(value in text for value in forbidden):
            raise ValueError("supplementary requirements must be computational only")
        return self


class ExperimentCoverageDraft(BaseModel):
    """DeepSeek advisory draft; deterministic policy decides whether execution is legal."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=10, max_length=6000)
    missing_requirements: list[MissingExperimentRequirement] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def validate_unique_requirements(self) -> "ExperimentCoverageDraft":
        identifiers = [item.requirement_id for item in self.missing_requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("missing experiment requirement IDs must be unique")
        return self


class ExperimentCoverageAudit(BaseModel):
    """Grounded audit of designed experiments and reviewed-code coverage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    research_design_uri: str
    code_manifest_uri: str
    main_result_report_uri: str
    decision: Literal["ready", "requires_design_revision", "requires_code_revision"]
    designed_experiment_ids: list[str] = Field(default_factory=list, max_length=30)
    code_covered_experiment_ids: list[str] = Field(default_factory=list, max_length=30)
    missing_code_experiment_ids: list[str] = Field(default_factory=list, max_length=30)
    missing_requirements: list[MissingExperimentRequirement] = Field(
        default_factory=list, max_length=20
    )
    summary: str = Field(min_length=10, max_length=6000)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_decision(self) -> "ExperimentCoverageAudit":
        designed = set(self.designed_experiment_ids)
        code_covered = set(self.code_covered_experiment_ids)
        missing_code = set(self.missing_code_experiment_ids)
        if len(designed) != len(self.designed_experiment_ids):
            raise ValueError("designed supplementary experiment IDs must be unique")
        if not code_covered.issubset(designed) or not missing_code.issubset(designed):
            raise ValueError("code coverage must refer to designed supplementary experiments")
        if code_covered.intersection(missing_code) or code_covered | missing_code != designed:
            raise ValueError("code-covered and missing-code experiment sets must partition design")
        if self.decision == "ready" and (self.missing_requirements or missing_code):
            raise ValueError("ready audit cannot contain design or code gaps")
        if self.decision == "requires_design_revision" and not self.missing_requirements:
            raise ValueError("design revision requires explicit missing experiments")
        if self.decision == "requires_code_revision" and (
            self.missing_requirements or not missing_code
        ):
            raise ValueError("code revision requires only concrete reviewed-code gaps")
        return self


class ExperimentCodeRunManifest(BaseModel):
    """Immutable supplementary experiment launch evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    status: Literal["launched", "partial", "blocked"]
    research_design_uri: str
    code_manifest_uri: str
    code_review_uri: str
    main_schedule_manifest_uri: str
    main_result_report_uri: str
    coverage_audit_uri: str
    report_interval_seconds: Literal[43200] = 43200
    experiment_ids: list[str] = Field(min_length=1, max_length=30)
    planned_seed_matrix: dict[str, list[int]] = Field(min_length=1, max_length=30)
    probes: list[ServerResourceSnapshot] = Field(min_length=1, max_length=64)
    selected_gpu_keys: list[str] = Field(default_factory=list, max_length=4096)
    server_workspaces: dict[str, str] = Field(default_factory=dict, max_length=64)
    environment_names: dict[str, str] = Field(default_factory=dict, max_length=64)
    server_demos: list[ServerDemoRecord] = Field(default_factory=list, max_length=64)
    job_spec_artifacts: list[ArtifactRef] = Field(default_factory=list, max_length=2000)
    jobs: list[ScheduledExperimentJob] = Field(min_length=1, max_length=2000)
    estimated_makespan_seconds: int = Field(default=0, ge=0)
    next_report_at: datetime | None = None
    issues: list[str] = Field(default_factory=list, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_launch(self) -> "ExperimentCodeRunManifest":
        if len(self.experiment_ids) != len(set(self.experiment_ids)):
            raise ValueError("supplementary experiment IDs must be unique")
        if set(self.experiment_ids) != set(self.planned_seed_matrix):
            raise ValueError("planned seed matrix must cover every experiment exactly")
        for seeds in self.planned_seed_matrix.values():
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError("supplementary experiment seeds must be non-empty and unique")
        job_keys = [(item.experiment_id, item.seed) for item in self.jobs]
        expected_keys = {
            (experiment_id, seed)
            for experiment_id, seeds in self.planned_seed_matrix.items()
            for seed in seeds
        }
        if set(job_keys) != expected_keys or len(job_keys) != len(set(job_keys)):
            raise ValueError("supplementary jobs must exactly cover the planned seed matrix")
        launched_count = sum(item.status == "launched" for item in self.jobs)
        if self.status == "launched":
            if launched_count != len(self.jobs) or not self.next_report_at:
                raise ValueError("launched experiment run requires every job and next report time")
            if not self.selected_gpu_keys or not self.job_spec_artifacts:
                raise ValueError("launched experiment run requires GPUs and persisted job specs")
            passed_servers = {
                item.server_id for item in self.server_demos if item.status == "passed"
            }
            if passed_servers != {item.server_id for item in self.jobs}:
                raise ValueError("every supplementary server must pass a foreground demo")
        elif self.status == "blocked" and launched_count:
            raise ValueError("blocked supplementary run cannot contain launched jobs")
        elif self.status == "partial" and not 0 < launched_count < len(self.jobs):
            raise ValueError("partial supplementary run requires a strict subset of launched jobs")
        return self


class ExperimentSeedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^job-[a-zA-Z0-9_-]{1,100}$")
    seed: int
    value: float
    checkpoint_path: str | None = Field(default=None, max_length=1000)

    @field_validator("value")
    @classmethod
    def validate_finite_seed_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("experiment seed metrics must be finite")
        return value


class ExperimentMetricAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    metric_name: str = Field(min_length=1, max_length=200)
    direction: Literal["higher", "lower"]
    seed_metrics: list[ExperimentSeedMetric] = Field(min_length=1, max_length=50)
    mean: float
    sample_standard_deviation: float | None = Field(default=None, ge=0.0)
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_statistics(self) -> "ExperimentMetricAggregate":
        values = [item.value for item in self.seed_metrics]
        seeds = [item.seed for item in self.seed_metrics]
        if len(seeds) != len(set(seeds)):
            raise ValueError("experiment aggregate cannot duplicate seeds")
        calculated_mean = sum(values) / len(values)
        if not all(
            math.isfinite(value)
            for value in (calculated_mean, self.mean, self.minimum, self.maximum)
        ):
            raise ValueError("experiment aggregate statistics must be finite")
        calculated_std = (
            math.sqrt(
                sum((value - calculated_mean) ** 2 for value in values)
                / (len(values) - 1)
            )
            if len(values) > 1
            else None
        )
        if not math.isclose(self.mean, calculated_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("experiment aggregate mean does not match seed evidence")
        if not math.isclose(self.minimum, min(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("experiment aggregate minimum does not match seed evidence")
        if not math.isclose(self.maximum, max(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("experiment aggregate maximum does not match seed evidence")
        if calculated_std is None:
            if self.sample_standard_deviation is not None:
                raise ValueError("single-seed aggregate cannot claim a sample deviation")
        elif self.sample_standard_deviation is None or not math.isclose(
            self.sample_standard_deviation,
            calculated_std,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("experiment aggregate deviation does not match seed evidence")
        return self


class ExperimentCompletenessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    experiment_type: Literal[
        "ablation",
        "case_study",
        "module_analysis",
        "feasibility",
        "sensitivity",
        "ood_generalization",
        "efficiency",
        "uncertainty_calibration",
    ]
    expected_seeds: list[int] = Field(min_length=1, max_length=50)
    completed_seeds: list[int] = Field(default_factory=list, max_length=50)
    metric_seeds: list[int] = Field(default_factory=list, max_length=50)
    expected_result_ids: list[str] = Field(default_factory=list, max_length=30)
    status: Literal["passed", "failed", "pending"]
    summary: str = Field(min_length=10, max_length=3000)

    @model_validator(mode="after")
    def validate_seed_sets(self) -> "ExperimentCompletenessCheck":
        expected = set(self.expected_seeds)
        completed = set(self.completed_seeds)
        metric = set(self.metric_seeds)
        if len(expected) != len(self.expected_seeds):
            raise ValueError("expected supplementary seeds must be unique")
        if not completed.issubset(expected) or not metric.issubset(completed):
            raise ValueError("observed supplementary seed sets escaped the plan")
        if self.status == "passed" and (completed != expected or metric != expected):
            raise ValueError("passed experiment completeness requires every seed and metric")
        return self


class ExperimentCodeRunReport(BaseModel):
    """Observed supplementary experiment results; no value may be model-generated."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    run_manifest_uri: str
    report_sequence: int = Field(ge=1)
    verdict: Literal["in_progress", "passed", "failed"]
    summary: str = Field(min_length=10, max_length=6000)
    observations: list[JobObservation] = Field(min_length=1, max_length=2000)
    completeness_checks: list[ExperimentCompletenessCheck] = Field(
        min_length=1, max_length=30
    )
    primary_metric_aggregates: list[ExperimentMetricAggregate] = Field(
        default_factory=list, max_length=30
    )
    log_artifacts: list[ArtifactRef] = Field(default_factory=list, max_length=2000)
    recommended_route: AgentName | None = None
    report_interval_seconds: Literal[43200] = 43200
    next_report_at: datetime | None = None
    issues: list[str] = Field(default_factory=list, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_verdict(self) -> "ExperimentCodeRunReport":
        observation_ids = [item.job_id for item in self.observations]
        check_ids = [item.experiment_id for item in self.completeness_checks]
        aggregate_ids_list = [
            item.experiment_id for item in self.primary_metric_aggregates
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("experiment report cannot duplicate job observations")
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("experiment report cannot duplicate completeness checks")
        if len(aggregate_ids_list) != len(set(aggregate_ids_list)):
            raise ValueError("experiment report cannot duplicate metric aggregates")
        active = any(
            item.status in {"queued", "running", "unknown"}
            for item in self.observations
        )
        failed_checks = [
            item for item in self.completeness_checks if item.status == "failed"
        ]
        if self.verdict == "in_progress":
            if not active or self.next_report_at is None or self.recommended_route is not None:
                raise ValueError("in-progress experiment report requires active jobs and next report")
        elif self.verdict == "passed":
            if (
                active
                or failed_checks
                or any(item.status != "passed" for item in self.completeness_checks)
                or self.recommended_route is not None
            ):
                raise ValueError("passed supplementary report cannot have active or failed evidence")
            if any(item.status != "completed" for item in self.observations):
                raise ValueError("passed supplementary report requires completed jobs")
            passed_ids = {item.experiment_id for item in self.completeness_checks}
            aggregate_ids = {item.experiment_id for item in self.primary_metric_aggregates}
            if passed_ids != aggregate_ids:
                raise ValueError("passed supplementary report requires an aggregate per experiment")
        elif self.recommended_route is None or not self.issues:
            raise ValueError("failed supplementary report requires a route and issue evidence")
        return self


class ObservedSeedResult(BaseModel):
    """One exact per-seed metric imported from a persisted server observation."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^job-[a-zA-Z0-9_-]{1,100}$")
    seed: int
    value: float
    checkpoint_path: str | None = Field(default=None, max_length=1000)

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("paper result evidence must be finite")
        return value


class ObservedExperimentResult(BaseModel):
    """Deterministically aggregated experiment evidence available to paper writing."""

    model_config = ConfigDict(extra="forbid")

    result_key: str = Field(pattern=r"^result-(?:main|supp)-[a-f0-9]{12}$")
    scope: Literal["main", "supplementary"]
    experiment_id: str = Field(pattern=r"^experiment-[a-zA-Z0-9_-]{1,64}$")
    metric_name: str = Field(min_length=1, max_length=200)
    direction: Literal["higher", "lower"]
    seed_results: list[ObservedSeedResult] = Field(min_length=1, max_length=50)
    mean: float
    sample_standard_deviation: float | None = Field(default=None, ge=0.0)
    minimum: float
    maximum: float
    target_value: float | None = None
    gate_status: Literal["passed"] = "passed"
    source_report_uri: str

    @model_validator(mode="after")
    def validate_exact_statistics(self) -> "ObservedExperimentResult":
        values = [item.value for item in self.seed_results]
        seeds = [item.seed for item in self.seed_results]
        jobs = [item.job_id for item in self.seed_results]
        if len(seeds) != len(set(seeds)) or len(jobs) != len(set(jobs)):
            raise ValueError("paper result evidence cannot duplicate a seed or job")
        calculated_mean = sum(values) / len(values)
        calculated_std = (
            math.sqrt(
                sum((value - calculated_mean) ** 2 for value in values)
                / (len(values) - 1)
            )
            if len(values) > 1
            else None
        )
        for value in (self.mean, self.minimum, self.maximum, calculated_mean):
            if not math.isfinite(value):
                raise ValueError("paper aggregate values must be finite")
        if self.target_value is not None and not math.isfinite(self.target_value):
            raise ValueError("paper target values must be finite")
        if not math.isclose(self.mean, calculated_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("paper result mean differs from seed evidence")
        if not math.isclose(self.minimum, min(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("paper result minimum differs from seed evidence")
        if not math.isclose(self.maximum, max(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("paper result maximum differs from seed evidence")
        if calculated_std is None:
            if self.sample_standard_deviation is not None:
                raise ValueError("single-seed paper result cannot claim sample deviation")
        elif self.sample_standard_deviation is None or not math.isclose(
            self.sample_standard_deviation,
            calculated_std,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("paper result deviation differs from seed evidence")
        return self


class CodeSymbolEvidence(BaseModel):
    """AST-derived symbol evidence from one content-addressed generated code file."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1, max_length=500)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    symbol_kind: Literal["function", "async_function", "class", "constant"]
    line_start: int = Field(ge=1)
    design_component_ids: list[str] = Field(default_factory=list, max_length=30)
    experiment_ids: list[str] = Field(default_factory=list, max_length=30)


class FormulaCodeReferenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1, max_length=500)
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class PaperFormulaDraft(BaseModel):
    """Model-authored equation proposal that must resolve to reviewed AST symbols."""

    model_config = ConfigDict(extra="forbid")

    formula_id: str = Field(pattern=r"^formula-[a-zA-Z0-9_-]{1,64}$")
    label: str = Field(min_length=3, max_length=300)
    latex: str = Field(min_length=3, max_length=4000)
    explanation: str = Field(min_length=10, max_length=3000)
    design_component_ids: list[str] = Field(min_length=1, max_length=30)
    code_references: list[FormulaCodeReferenceDraft] = Field(min_length=1, max_length=20)
    citation_ids: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("latex")
    @classmethod
    def reject_executable_latex(cls, value: str) -> str:
        lowered = value.lower()
        forbidden = (
            "\\write18",
            "\\write",
            "\\read",
            "\\input",
            "\\include",
            "\\begin",
            "\\end",
            "\\openout",
            "\\openin",
            "\\usepackage",
            "\\documentclass",
            "\\newcommand",
            "\\renewcommand",
            "\\def",
            "\\edef",
            "\\catcode",
            "\\csname",
            "\\special",
            "\\immediate",
            "\\includegraphics",
            "\\bibliography",
            "\\addbibresource",
            "\\href",
            "\\url",
            "\\label",
            "\\tag",
        )
        if any(token in lowered for token in forbidden):
            raise ValueError("formula LaTeX contains a forbidden executable directive")
        safe_math_commands = {
            "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
            "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta",
            "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma",
            "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
            "frac", "dfrac", "tfrac", "sqrt", "sum", "prod", "int", "oint", "lim", "min", "max",
            "argmin", "argmax", "log", "ln", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "operatorname", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak",
            "text", "left", "right", "big", "Big", "bigg", "Bigg", "lVert", "rVert", "langle", "rangle",
            "lceil", "rceil", "lfloor", "rfloor",
            "vert", "Vert", "mid", "cdot", "times", "div", "pm", "mp", "le", "leq", "ge", "geq",
            "neq", "approx", "sim", "simeq", "propto", "in", "notin", "subset", "subseteq", "supset",
            "supseteq", "cup", "cap", "to", "mapsto", "rightarrow", "leftarrow", "leftrightarrow",
            "Rightarrow", "Leftarrow", "Leftrightarrow", "infty", "partial", "nabla", "forall", "exists",
            "neg", "land", "lor", "oplus", "otimes", "odot", "hat", "widehat", "bar", "overline",
            "underline", "tilde", "widetilde", "vec", "dot", "ddot", "top", "perp", "dagger", "ell",
            "quad", "qquad",
        }
        commands = set(re.findall(r"\\([A-Za-z]+)", value))
        unsupported = sorted(commands - safe_math_commands)
        if unsupported:
            raise ValueError(
                "formula LaTeX contains non-allowlisted commands: "
                + ", ".join(unsupported[:10])
            )
        if any(character in value for character in ("%", "#")):
            raise ValueError("formula LaTeX contains unsafe TeX metacharacters")
        return value.strip()


class PaperFormulaEvidence(BaseModel):
    """Resolved formula-to-code mapping stored with the paper."""

    model_config = ConfigDict(extra="forbid")

    formula_id: str = Field(pattern=r"^formula-[a-zA-Z0-9_-]{1,64}$")
    label: str = Field(min_length=3, max_length=300)
    latex: str = Field(min_length=3, max_length=4000)
    explanation: str = Field(min_length=10, max_length=3000)
    design_component_ids: list[str] = Field(min_length=1, max_length=30)
    code_bindings: list[CodeSymbolEvidence] = Field(min_length=1, max_length=20)
    citation_ids: list[str] = Field(default_factory=list, max_length=30)


class PaperSectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Literal[
        "introduction",
        "related_work",
        "method",
        "experimental_setup",
        "results",
        "discussion",
        "limitations",
        "conclusion",
    ]
    title: str = Field(min_length=2, max_length=200)
    paragraphs: list[str] = Field(min_length=1, max_length=20)
    citation_ids: list[str] = Field(default_factory=list, max_length=100)
    formula_ids: list[str] = Field(default_factory=list, max_length=30)
    result_keys: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("paragraphs")
    @classmethod
    def validate_paragraphs(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            paragraph = value.strip()
            if not 20 <= len(paragraph) <= 6000:
                raise ValueError("paper paragraphs must be substantive and bounded")
            if "<script" in paragraph.lower() or "\\write18" in paragraph.lower():
                raise ValueError("paper paragraph contains executable markup")
            cleaned.append(paragraph)
        return cleaned


class PaperClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^claim-[a-zA-Z0-9_-]{1,80}$")
    claim_type: Literal["literature", "method", "experiment", "limitation"]
    text: str = Field(min_length=10, max_length=3000)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    design_component_ids: list[str] = Field(default_factory=list, max_length=30)
    code_file_paths: list[str] = Field(default_factory=list, max_length=30)
    result_keys: list[str] = Field(default_factory=list, max_length=100)
    strength: Literal["descriptive", "associational", "comparative"]

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> "PaperClaimDraft":
        if self.claim_type == "literature" and not self.citation_ids:
            raise ValueError("literature claims require citations")
        if self.claim_type == "method" and (
            not self.design_component_ids or not self.code_file_paths
        ):
            raise ValueError("method claims require design and code evidence")
        if self.claim_type == "experiment" and not self.result_keys:
            raise ValueError("experiment claims require observed result keys")
        if self.claim_type == "limitation" and self.strength != "descriptive":
            raise ValueError("limitation claims must remain descriptive")
        return self


class PaperWritingDraft(BaseModel):
    """DeepSeek writing plan. Runtime resolves every code and result reference."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=10, max_length=500)
    abstract: str = Field(min_length=100, max_length=5000)
    keywords: list[str] = Field(min_length=3, max_length=12)
    sections: list[PaperSectionDraft] = Field(min_length=8, max_length=8)
    formulas: list[PaperFormulaDraft] = Field(min_length=1, max_length=30)
    claims: list[PaperClaimDraft] = Field(min_length=4, max_length=200)
    acknowledgements: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_structure(self) -> "PaperWritingDraft":
        section_ids = [item.section_id for item in self.sections]
        required = {
            "introduction",
            "related_work",
            "method",
            "experimental_setup",
            "results",
            "discussion",
            "limitations",
            "conclusion",
        }
        if set(section_ids) != required or len(section_ids) != len(set(section_ids)):
            raise ValueError("paper draft must contain each required section exactly once")
        formula_ids = [item.formula_id for item in self.formulas]
        claim_ids = [item.claim_id for item in self.claims]
        if len(formula_ids) != len(set(formula_ids)):
            raise ValueError("paper formula IDs must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("paper claim IDs must be unique")
        known_formulas = set(formula_ids)
        for section in self.sections:
            if not set(section.formula_ids).issubset(known_formulas):
                raise ValueError("paper section references an unknown formula")
        if not any(item.claim_type == "limitation" for item in self.claims):
            raise ValueError("paper draft requires an explicit limitation claim")
        return self


class PaperCitationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(pattern=r"^paper-[a-zA-Z0-9_-]{1,64}$")
    title: str = Field(min_length=1, max_length=500)
    authors: list[str] = Field(default_factory=list, max_length=100)
    year: int | None = Field(default=None, ge=1800, le=2200)
    doi: str | None = Field(default=None, max_length=300)
    venue: str = Field(min_length=1, max_length=300)
    local_pdf: ArtifactRef


class PaperManuscriptManifest(BaseModel):
    """Content-addressed manuscript whose formulas, citations, and results are grounded."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    status: Literal["draft_grounded"] = "draft_grounded"
    title: str = Field(min_length=10, max_length=500)
    abstract: str = Field(min_length=100, max_length=5000)
    keywords: list[str] = Field(min_length=3, max_length=12)
    paper_manifest_uri: str
    research_design_uri: str
    dataset_manifest_uri: str
    code_manifest_uri: str
    code_review_uri: str
    main_schedule_uri: str
    main_result_report_uri: str
    experiment_run_manifest_uri: str
    experiment_result_report_uri: str
    citations: list[PaperCitationEvidence] = Field(min_length=1, max_length=200)
    closest_prior_comparisons: list[PaperComparison] = Field(
        default_factory=list, max_length=30
    )
    sections: list[PaperSectionDraft] = Field(min_length=8, max_length=8)
    formulas: list[PaperFormulaEvidence] = Field(min_length=1, max_length=30)
    claims: list[PaperClaimDraft] = Field(min_length=4, max_length=200)
    acknowledgements: str | None = Field(default=None, max_length=2000)
    experiment_results: list[ObservedExperimentResult] = Field(min_length=1, max_length=100)
    markdown_artifact: ArtifactRef
    latex_artifact: ArtifactRef
    bibliography_artifact: ArtifactRef
    reproducibility_bundle_artifact: ArtifactRef | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> "PaperManuscriptManifest":
        if self.reproducibility_bundle_artifact is not None:
            bundle_uri = self.reproducibility_bundle_artifact.uri.casefold()
            if not bundle_uri.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                raise ValueError("reproducibility bundle must be an archive artifact")
        citation_ids = [item.paper_id for item in self.citations]
        formula_ids = [item.formula_id for item in self.formulas]
        claim_ids = [item.claim_id for item in self.claims]
        result_keys = [item.result_key for item in self.experiment_results]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("paper citations must be unique")
        if len(formula_ids) != len(set(formula_ids)):
            raise ValueError("resolved paper formulas must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("resolved paper claims must be unique")
        if len(result_keys) != len(set(result_keys)):
            raise ValueError("paper experiment result keys must be unique")
        citation_set = set(citation_ids)
        formula_set = set(formula_ids)
        result_set = set(result_keys)
        comparison_ids = {item.paper_id for item in self.closest_prior_comparisons}
        if not comparison_ids.issubset(citation_set):
            raise ValueError("closest-prior comparisons cite evidence outside the manuscript")
        if comparison_ids:
            related_work = next(
                (item for item in self.sections if item.section_id == "related_work"),
                None,
            )
            if related_work is None or not comparison_ids.issubset(
                set(related_work.citation_ids)
            ):
                raise ValueError(
                    "related-work section must cite every structured closest-prior comparison"
                )
        code_paths = {
            binding.file_path
            for formula in self.formulas
            for binding in formula.code_bindings
        }
        for formula in self.formulas:
            if not set(formula.citation_ids).issubset(citation_set):
                raise ValueError("formula cites evidence outside the paper manifest")
        for section in self.sections:
            if not set(section.citation_ids).issubset(citation_set):
                raise ValueError("section cites evidence outside the paper manifest")
            if not set(section.formula_ids).issubset(formula_set):
                raise ValueError("section references an unknown resolved formula")
            if not set(section.result_keys).issubset(result_set):
                raise ValueError("section references an unknown observed result")
        for claim in self.claims:
            if not set(claim.citation_ids).issubset(citation_set):
                raise ValueError("claim cites evidence outside the paper manifest")
            if not set(claim.result_keys).issubset(result_set):
                raise ValueError("claim references an unknown observed result")
            if claim.claim_type == "method" and not set(claim.code_file_paths).issubset(
                code_paths
            ):
                raise ValueError("method claim lacks a formula-resolved code path")
        result_sections = [item for item in self.sections if item.section_id == "results"]
        if not result_sections or set(result_sections[0].result_keys) != result_set:
            raise ValueError("results section must include every observed experiment result")
        experimental_claim_keys = {
            key
            for claim in self.claims
            if claim.claim_type == "experiment"
            for key in claim.result_keys
        }
        if experimental_claim_keys != result_set:
            raise ValueError("experiment claims must cover every observed result exactly")
        return self


class PaperConsistencyAdvisoryFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^advisory-[a-zA-Z0-9_-]{1,80}$")
    category: Literal[
        "original_paper_comparison",
        "formula_semantics",
        "method_description",
        "result_interpretation",
        "claim_scope",
    ]
    severity: Literal["warning", "error"]
    summary: str = Field(min_length=10, max_length=3000)
    claim_ids: list[str] = Field(default_factory=list, max_length=50)
    formula_ids: list[str] = Field(default_factory=list, max_length=30)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    recommendation: str = Field(min_length=10, max_length=3000)


class PaperConsistencyAdvisoryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=10, max_length=6000)
    findings: list[PaperConsistencyAdvisoryFinding] = Field(
        default_factory=list, max_length=100
    )

    @model_validator(mode="after")
    def validate_unique_findings(self) -> "PaperConsistencyAdvisoryDraft":
        identifiers = [item.finding_id for item in self.findings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("paper advisory finding IDs must be unique")
        return self


class PaperConsistencyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^paper-check-[a-z0-9-]{2,100}$")
    category: Literal[
        "manuscript_integrity",
        "citation_pdf",
        "original_paper_comparison",
        "formula_code",
        "method_code",
        "experiment_result",
        "claim_evidence",
        "claim_scope",
    ]
    status: Literal["passed", "failed", "warning"]
    summary: str = Field(min_length=10, max_length=3000)
    evidence: list[str] = Field(default_factory=list, max_length=100)


class PaperClaimAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^claim-[a-zA-Z0-9_-]{1,80}$")
    status: Literal["passed", "failed", "warning"]
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    code_file_paths: list[str] = Field(default_factory=list, max_length=30)
    result_keys: list[str] = Field(default_factory=list, max_length=100)
    summary: str = Field(min_length=10, max_length=3000)


class PaperCodeConsistencyReport(BaseModel):
    """Deterministic and advisory comparison of paper, original sources, code, and results."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    manuscript_manifest_uri: str
    verdict: Literal["passed", "needs_revision"]
    summary: str = Field(min_length=10, max_length=6000)
    checks: list[PaperConsistencyCheck] = Field(min_length=7, max_length=200)
    claim_audits: list[PaperClaimAudit] = Field(min_length=1, max_length=200)
    advisory: PaperConsistencyAdvisoryDraft
    blockers: list[str] = Field(default_factory=list, max_length=100)
    recommendations: list[str] = Field(default_factory=list, max_length=100)
    recommended_route: AgentName | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_verdict(self) -> "PaperCodeConsistencyReport":
        failed_checks = [item for item in self.checks if item.status == "failed"]
        failed_claims = [item for item in self.claim_audits if item.status == "failed"]
        advisory_errors = [
            item for item in self.advisory.findings if item.severity == "error"
        ]
        failed = bool(failed_checks or failed_claims or advisory_errors)
        if self.verdict == "passed":
            if failed or self.blockers or self.recommended_route is not None:
                raise ValueError("passed paper/code review cannot contain failed evidence")
        elif (
            not failed
            or not self.blockers
            or not self.recommendations
            or self.recommended_route is None
        ):
            raise ValueError("paper revision verdict requires failures and a repair route")
        return self


class PaperReviewFinding(BaseModel):
    """One senior-review finding grounded in the final manuscript evidence graph."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^review-[a-zA-Z0-9_-]{1,80}$")
    category: Literal[
        "novelty",
        "methodology",
        "experiments",
        "reproducibility",
        "results",
        "citations",
        "writing",
        "limitations",
        "submission_format",
    ]
    severity: Literal["blocking", "major", "minor", "note"]
    summary: str = Field(min_length=10, max_length=3000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    required_revision: str = Field(min_length=10, max_length=3000)


class PaperReviewDraft(BaseModel):
    """Bounded qualitative verdict returned by the authoritative reviewer model."""

    model_config = ConfigDict(extra="forbid")

    reviewer_role: str = Field(min_length=10, max_length=300)
    target_venue_profile: Literal[
        "top_ml_conference",
        "authoritative_bioinformatics_journal",
        "authoritative_interdisciplinary_journal",
    ]
    recommendation: Literal[
        "accept",
        "minor_revision",
        "major_revision",
        "reject",
    ]
    summary: str = Field(min_length=30, max_length=6000)
    strengths: list[str] = Field(default_factory=list, max_length=30)
    findings: list[PaperReviewFinding] = Field(default_factory=list, max_length=100)
    required_revisions: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_recommendation(self) -> "PaperReviewDraft":
        identifiers = [item.finding_id for item in self.findings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("paper review finding IDs must be unique")
        major = [
            item
            for item in self.findings
            if item.severity in {"blocking", "major"}
        ]
        if self.recommendation in {"accept", "minor_revision"} and major:
            raise ValueError("accept/minor-revision recommendation cannot contain major findings")
        if self.recommendation in {"major_revision", "reject"} and not major:
            raise ValueError("major/reject recommendation requires a major finding")
        if major and not self.required_revisions:
            raise ValueError("major findings require an explicit revision list")
        return self


class PaperReviewReport(BaseModel):
    """Immutable final peer-review report; only no-major-issue reports pass."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    review_round: int = Field(ge=1)
    manuscript_manifest_uri: str
    consistency_report_uri: str
    verdict: Literal["passed", "needs_revision"]
    reviewer_role: str = Field(min_length=10, max_length=300)
    target_venue_profile: Literal[
        "top_ml_conference",
        "authoritative_bioinformatics_journal",
        "authoritative_interdisciplinary_journal",
    ]
    recommendation: Literal[
        "accept",
        "minor_revision",
        "major_revision",
        "reject",
    ]
    summary: str = Field(min_length=30, max_length=6000)
    strengths: list[str] = Field(default_factory=list, max_length=30)
    findings: list[PaperReviewFinding] = Field(default_factory=list, max_length=100)
    required_revisions: list[str] = Field(default_factory=list, max_length=100)
    major_issue_count: int = Field(ge=0)
    minor_issue_count: int = Field(ge=0)
    recommended_route: AgentName | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_final_verdict(self) -> "PaperReviewReport":
        major = sum(
            item.severity in {"blocking", "major"} for item in self.findings
        )
        minor = sum(item.severity == "minor" for item in self.findings)
        if self.major_issue_count != major or self.minor_issue_count != minor:
            raise ValueError("paper review issue counts do not match findings")
        if self.verdict == "passed":
            if major or self.recommended_route is not None:
                raise ValueError("passed final review cannot contain major issues or a route")
        elif major == 0 or self.recommended_route is None or not self.required_revisions:
            raise ValueError("revision verdict requires major issues, revisions and a route")
        return self


class DispatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    observed_state_version: int = Field(ge=1)
    decision: Literal["dispatch", "wait", "request_human", "complete", "abort"]
    target_agent: AgentName | None = None
    task_template: str | None = None
    reason: str = Field(min_length=1)
    required_inputs: list[str] = Field(default_factory=list)
    needs_human: bool = False


class LeadTransitionOption(BaseModel):
    """One policy-compiled state change that Lead may select.

    Options contain no executable callback or mutable state.  The StateMachine keeps
    the corresponding candidate snapshots private and commits only the option selected
    by Lead after validating the run/version/phase contract.
    """

    model_config = ConfigDict(extra="forbid")

    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,95}$")
    action: Literal[
        "advance",
        "retry",
        "continue",
        "wait",
        "request_human",
        "complete",
        "abort",
    ]
    target_state: WorkflowState
    target_agent: AgentName | None = None
    reason_hint: str = Field(min_length=5, max_length=1000)
    recommended: bool = False


class LeadTransitionDecision(BaseModel):
    """Lead's authoritative semantic choice between policy-safe state candidates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    observed_state_version: int = Field(ge=1)
    phase: Literal[
        "specialist_result",
        "feedback_control",
        "user_evaluation",
        "human_resume",
        "human_retry",
    ]
    selected_option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,95}$")
    action: Literal[
        "advance",
        "retry",
        "continue",
        "wait",
        "request_human",
        "complete",
        "abort",
    ]
    target_state: WorkflowState
    target_agent: AgentName | None = None
    reason: str = Field(min_length=5, max_length=3000)
    considered_issue_codes: list[str] = Field(default_factory=list, max_length=50)


class TaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    parent_task_id: str | None = None
    agent: AgentName
    objective: str = Field(min_length=1)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)
    deadline_seconds: int = Field(default=300, ge=1, le=86400)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning", "error", "blocking"]
    code: str
    message: str
    root_cause_key: str | None = None


class FeedbackControl(BaseModel):
    """Typed, advisory control that the Runtime must validate before applying."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    action: Literal["continue", "retry", "request_human", "abort"]
    severity: Literal["info", "warning", "error", "blocking"]
    reason: str = Field(min_length=5, max_length=3000)
    target_agent: AgentName | None = None
    issue_codes: list[str] = Field(default_factory=list, max_length=30)
    observed_log_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_control_target(self) -> "FeedbackControl":
        if self.action == "retry":
            if self.target_agent in {None, AgentName.LEAD, AgentName.FEEDBACK}:
                raise ValueError("feedback retry requires a specialist target")
        elif self.target_agent is not None:
            raise ValueError("only feedback retry may specify a target_agent")
        return self


class FeedbackCoordinationReport(BaseModel):
    """Persisted workflow-wide plan produced by the read-only Feedback coordinator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    source_task_id: str
    observed_log_count: int = Field(ge=0)
    workflow_health: Literal["on_track", "attention", "blocked", "aborted"]
    progress_summary: str = Field(min_length=10, max_length=5000)
    stage_assessment: str = Field(min_length=10, max_length=5000)
    next_actions: list[str] = Field(default_factory=list, max_length=30)
    risks: list[str] = Field(default_factory=list, max_length=30)
    coordination_notes: list[str] = Field(default_factory=list, max_length=30)
    control: FeedbackControl
    created_at: datetime = Field(default_factory=utc_now)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    task_id: str
    agent: AgentName
    status: Literal["succeeded", "in_progress", "failed", "blocked", "needs_human"]
    summary: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[AgentIssue] = Field(default_factory=list)
    route_hint: AgentName | None = None
    needs_human: bool = False
    feedback_control: FeedbackControl | None = None

    @model_validator(mode="after")
    def validate_feedback_control_owner(self) -> "AgentResult":
        if self.feedback_control is not None and self.agent != AgentName.FEEDBACK:
            raise ValueError("only feedback-agent may return feedback_control")
        return self


class UserReviewRequest(BaseModel):
    """A persisted after-step checkpoint that must be resolved by a user."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    review_id: str = Field(pattern=r"^review_[A-Za-z0-9_-]{1,160}$")
    source_state: WorkflowState
    proposed_state: WorkflowState
    source_agent: AgentName
    source_task_id: str
    result_status: Literal["succeeded", "in_progress", "failed", "blocked", "needs_human"]
    result_summary: str = Field(min_length=1, max_length=3000)
    issue_codes: list[str] = Field(default_factory=list, max_length=30)
    artifact_kinds: list[str] = Field(default_factory=list, max_length=100)
    feedback_action: Literal["continue", "retry", "request_human", "abort"]
    feedback_reason: str = Field(default="", max_length=3000)
    allowed_revision_targets: list[AgentName] = Field(default_factory=list)
    default_revision_target: AgentName | None = None
    prior_artifact_count: int = Field(ge=0)
    prior_completed_agent_count: int = Field(ge=0)
    prior_gate_attempts: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_review_request(self) -> "UserReviewRequest":
        if self.source_agent == AgentName.LEAD:
            raise ValueError("Lead decisions are not user-review execution results")
        if self.proposed_state == WorkflowState.AWAITING_USER_REVIEW:
            raise ValueError("proposed state cannot already await user review")
        if len(self.allowed_revision_targets) != len(set(self.allowed_revision_targets)):
            raise ValueError("allowed revision targets must be unique")
        if any(
            agent in {AgentName.LEAD, AgentName.FEEDBACK}
            for agent in self.allowed_revision_targets
        ):
            raise ValueError("revision targets must be specialist agents")
        if (
            self.default_revision_target is not None
            and self.default_revision_target not in self.allowed_revision_targets
        ):
            raise ValueError("default revision target must be allowlisted")
        return self


class UserEvaluation(BaseModel):
    """A user-authored verdict consumed only by the Runtime control plane."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    review_id: str
    observed_state_version: int = Field(ge=1)
    verdict: Literal["satisfied", "needs_revision"]
    rating: int | None = Field(default=None, ge=1, le=5)
    comments: str = Field(default="", max_length=5000)
    target_agent: AgentName | None = None
    submitted_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evaluation(self) -> "UserEvaluation":
        comments = self.comments.strip()
        if self.verdict == "satisfied":
            if self.target_agent is not None:
                raise ValueError("satisfied evaluation cannot select a revision target")
        else:
            if len(comments) < 3:
                raise ValueError("needs_revision evaluation requires actionable comments")
            if self.target_agent in {AgentName.LEAD, AgentName.FEEDBACK}:
                raise ValueError("revision target must be a specialist agent")
        self.comments = comments
        return self


class UserRevisionNote(BaseModel):
    """Accepted user feedback forwarded to the next allowlisted repair task."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    rejected_agent: AgentName
    target_agent: AgentName
    source_state: WorkflowState
    comments: str = Field(min_length=3, max_length=5000)
    rating: int | None = Field(default=None, ge=1, le=5)
    created_at: datetime = Field(default_factory=utc_now)


class RunSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    project_slug: str = Field(
        default="legacy",
        pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$",
    )
    objective: str
    execution_mode: Literal["mock", "live"] = "mock"
    state: WorkflowState
    version: int = Field(ge=1)
    completed_agents: list[AgentName] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    blocked_from: WorkflowState | None = None
    gate_attempts: dict[str, int] = Field(default_factory=dict)
    pending_user_review: UserReviewRequest | None = None
    queued_user_reviews: list[UserReviewRequest] = Field(default_factory=list, max_length=1)
    revision_history: list[UserRevisionNote] = Field(default_factory=list, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def terminal(self) -> bool:
        return self.state in {
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.ABORTED,
            WorkflowState.FAILED,
        }

    @property
    def awaiting_user_review(self) -> bool:
        return self.state == WorkflowState.AWAITING_USER_REVIEW

    @model_validator(mode="after")
    def validate_pending_user_review(self) -> "RunSnapshot":
        if self.state == WorkflowState.AWAITING_USER_REVIEW:
            if self.pending_user_review is None:
                raise ValueError("awaiting-user-review state requires a pending review")
        elif self.pending_user_review is not None:
            raise ValueError("pending review is only valid in awaiting-user-review state")
        if self.queued_user_reviews and self.pending_user_review is None:
            raise ValueError("queued reviews require an active pending review")
        return self


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)
    run_id: str
    event_type: str
    state: WorkflowState
    state_version: int = Field(ge=1)
    agent: AgentName | None = None
    task_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProcessLogRecord(BaseModel):
    """Redacted observability event written to an Agent-specific JSONL file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    timestamp: datetime = Field(default_factory=utc_now)
    run_id: str
    agent: AgentName
    event: str = Field(min_length=1, max_length=200)
    status: Literal["started", "succeeded", "failed", "blocked", "info"]
    workflow_state: WorkflowState
    state_version: int = Field(ge=1)
    task_id: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    duration_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
