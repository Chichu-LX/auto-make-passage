from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
from typing import Iterator, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .schemas import RunSnapshot, WorkflowState, utc_now
from .storage import ArtifactStore, StoragePolicyError


MONITORED_STATES = {
    WorkflowState.RESULT_CHECK,
    WorkflowState.EXPERIMENT_CODE_RUN,
}
INDEX_RELATIVE_PATH = "project/var/runs/monitoring/index.json"
LOCK_RELATIVE_PATH = "project/var/runs/monitoring/index.lock"
MAX_INDEX_BYTES = 4 * 1024 * 1024


class MonitoringJob(BaseModel):
    """One durable wake-up request; credentials are deliberately never persisted."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    job_id: str = Field(pattern=r"^wake_[A-Za-z0-9_-]{1,96}$")
    run_id: str = Field(pattern=r"^run_[A-Za-z0-9_-]{1,128}$")
    project_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    workflow_state: Literal["RESULT_CHECK", "EXPERIMENT_CODE_RUN"]
    due_at: datetime
    source_artifact_uri: str
    execution_mode: Literal["mock", "live"]
    review_provider: Literal["deepseek", "claude-sdk"] = "deepseek"
    server_backend: Literal["disabled", "local", "ssh"] = "disabled"
    status: Literal["scheduled", "claimed", "fired", "failed", "cancelled"] = (
        "scheduled"
    )
    attempts: int = Field(default=0, ge=0)
    last_error_type: str | None = Field(default=None, max_length=200)
    registered_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    fired_at: datetime | None = None


class MonitoringIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    jobs: list[MonitoringJob] = Field(default_factory=list, max_length=10000)


class MonitoringScheduler:
    """Atomic, append-preserving scheduler for 12-hour result checks.

    The daemon and CLI may be separate processes. A small advisory file lock protects
    the durable index; completed records remain as audit history and are never deleted.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve()
        self.artifacts = ArtifactStore(self.root)
        self.index_path = self.root / INDEX_RELATIVE_PATH
        self.lock_path = self.root / LOCK_RELATIVE_PATH

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise StoragePolicyError("monitoring lock cannot be a symbolic link")
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> MonitoringIndex:
        if not self.index_path.exists():
            return MonitoringIndex()
        if self.index_path.is_symlink() or not self.index_path.is_file():
            raise StoragePolicyError("monitoring index must be a regular file")
        if self.index_path.stat().st_size > MAX_INDEX_BYTES:
            raise StoragePolicyError("monitoring index exceeds its read limit")
        try:
            return MonitoringIndex.model_validate_json(
                self.index_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise StoragePolicyError("monitoring index is invalid") from exc

    def _save_unlocked(self, index: MonitoringIndex) -> None:
        self.artifacts.write_runtime_json(
            INDEX_RELATIVE_PATH,
            index.model_dump(mode="json"),
        )

    @staticmethod
    def _candidate_kinds(state: WorkflowState) -> tuple[str, ...]:
        if state == WorkflowState.RESULT_CHECK:
            return ("result_check_report", "server_schedule_manifest")
        if state == WorkflowState.EXPERIMENT_CODE_RUN:
            return ("experiment_code_run_report", "experiment_code_run_manifest")
        return ()

    def _next_report_evidence(
        self, snapshot: RunSnapshot
    ) -> tuple[datetime, str] | None:
        if snapshot.state not in MONITORED_STATES:
            return None
        allowed_kinds = self._candidate_kinds(snapshot.state)
        for artifact in reversed(snapshot.artifacts):
            if artifact.kind not in allowed_kinds:
                continue
            try:
                payload = self.artifacts.read_artifact_json(
                    artifact, max_bytes=2 * 1024 * 1024
                )
            except StoragePolicyError:
                continue
            if not isinstance(payload, dict) or payload.get("run_id") != snapshot.run_id:
                continue
            if artifact.kind.endswith("report") and payload.get("verdict") != "in_progress":
                continue
            if artifact.kind.endswith("manifest") and payload.get("status") != "launched":
                continue
            raw_due = payload.get("next_report_at")
            if not isinstance(raw_due, str):
                continue
            try:
                due_at = datetime.fromisoformat(raw_due.replace("Z", "+00:00"))
            except ValueError:
                continue
            if due_at.tzinfo is None:
                continue
            return due_at, artifact.uri
        return None

    def schedule_from_snapshot(
        self,
        snapshot: RunSnapshot,
        *,
        review_provider: Literal["deepseek", "claude-sdk"] = "deepseek",
        server_backend: Literal["disabled", "local", "ssh"] = "disabled",
    ) -> MonitoringJob | None:
        """Register a wake-up only after both user reviews have been resolved."""

        if snapshot.awaiting_user_review or snapshot.terminal:
            return None
        evidence = self._next_report_evidence(snapshot)
        if evidence is None:
            return None
        due_at, source_uri = evidence
        now = utc_now()
        if due_at <= now:
            return None
        with self._locked():
            index = self._load_unlocked()
            for position, existing in enumerate(index.jobs):
                if existing.status != "scheduled" or existing.run_id != snapshot.run_id:
                    continue
                if (
                    existing.workflow_state == snapshot.state.value
                    and existing.due_at == due_at
                    and existing.source_artifact_uri == source_uri
                ):
                    return existing
                index.jobs[position] = existing.model_copy(
                    update={"status": "cancelled", "updated_at": now}
                )
            job = MonitoringJob(
                job_id=f"wake_{uuid4().hex[:20]}",
                run_id=snapshot.run_id,
                project_slug=snapshot.project_slug,
                workflow_state=snapshot.state.value,
                due_at=due_at,
                source_artifact_uri=source_uri,
                execution_mode=snapshot.execution_mode,
                review_provider=review_provider,
                server_backend=server_backend,
            )
            index.jobs.append(job)
            self._save_unlocked(index)
            return job

    def claim_due(self, *, now: datetime | None = None) -> list[MonitoringJob]:
        current = now or utc_now()
        with self._locked():
            index = self._load_unlocked()
            claimed: list[MonitoringJob] = []
            for position, job in enumerate(index.jobs):
                if job.status == "scheduled" and job.due_at <= current:
                    value = job.model_copy(
                        update={
                            "status": "claimed",
                            "attempts": job.attempts + 1,
                            "updated_at": current,
                        }
                    )
                    index.jobs[position] = value
                    claimed.append(value)
            if claimed:
                self._save_unlocked(index)
            return claimed

    def mark_fired(self, job_id: str, *, when: datetime | None = None) -> MonitoringJob:
        return self._finish(job_id, status="fired", when=when or utc_now())

    def mark_failed(self, job_id: str, *, error_type: str) -> MonitoringJob:
        return self._finish(
            job_id,
            status="failed",
            when=utc_now(),
            error_type=re.sub(r"[^A-Za-z0-9_.-]", "_", error_type)[:200],
        )

    def _finish(
        self,
        job_id: str,
        *,
        status: Literal["fired", "failed"],
        when: datetime,
        error_type: str | None = None,
    ) -> MonitoringJob:
        with self._locked():
            index = self._load_unlocked()
            for position, job in enumerate(index.jobs):
                if job.job_id != job_id:
                    continue
                if job.status != "claimed":
                    raise StoragePolicyError("only a claimed monitoring job may finish")
                updated = job.model_copy(
                    update={
                        "status": status,
                        "updated_at": when,
                        "fired_at": when if status == "fired" else None,
                        "last_error_type": error_type,
                    }
                )
                index.jobs[position] = updated
                self._save_unlocked(index)
                return updated
        raise StoragePolicyError("monitoring job does not exist")

    def inventory(self) -> MonitoringIndex:
        with self._locked():
            return self._load_unlocked()
