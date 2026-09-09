from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .schemas import AgentName, ArtifactRef, EventRecord, RunSnapshot
from .state_machine import WRITE_SCOPES


class StoragePolicyError(RuntimeError):
    pass


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ArtifactStore:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()

    def _safe_target(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts:
            raise StoragePolicyError("absolute paths and path traversal are forbidden")
        target = self.root / raw
        parent = target.parent.resolve()
        if not _is_within(parent, self.root):
            raise StoragePolicyError("target escapes project root")
        current = self.root
        for part in raw.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise StoragePolicyError("symbolic links are forbidden in write paths")
        if target.is_symlink():
            raise StoragePolicyError("symbolic link targets are forbidden")
        return target

    def _safe_existing_artifact(self, uri: str) -> Path:
        prefix = "artifact://"
        if not uri.startswith(prefix):
            raise StoragePolicyError("only artifact:// references can be read")
        relative_path = uri[len(prefix) :]
        if not relative_path or any(character in relative_path for character in ("?", "#", "%")):
            raise StoragePolicyError("artifact URI is malformed")
        target = self._safe_target(relative_path)
        current = self.root
        for part in Path(relative_path).parts:
            current = current / part
            if current.is_symlink():
                raise StoragePolicyError("symbolic links are forbidden in artifact reads")
        if not target.is_file():
            raise StoragePolicyError("artifact does not exist or is not a file")
        return target

    @staticmethod
    def _verify_digest(payload: bytes, expected_sha256: str) -> None:
        if expected_sha256:
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected_sha256:
                raise StoragePolicyError("artifact SHA-256 does not match its reference")

    def _atomic_write(self, relative_path: str, payload: bytes) -> ArtifactRef:
        target = self._safe_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temp_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        return ArtifactRef(
            uri=f"artifact://{relative_path}", sha256=digest, kind=target.suffix.lstrip(".") or "file"
        )

    @staticmethod
    def _check_agent_scope(agent: AgentName, relative_path: str) -> None:
        allowed = WRITE_SCOPES[agent]
        if not any(relative_path.startswith(prefix) for prefix in allowed):
            raise StoragePolicyError(
                f"{agent.value} cannot write outside its declared scope"
            )

    def write_agent_json(
        self, agent: AgentName, relative_path: str, value: Any
    ) -> ArtifactRef:
        self._check_agent_scope(agent, relative_path)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        return self._atomic_write(relative_path, payload)

    def write_agent_bytes(
        self,
        agent: AgentName,
        relative_path: str,
        payload: bytes,
        *,
        max_bytes: int,
        kind: str,
    ) -> ArtifactRef:
        self._check_agent_scope(agent, relative_path)
        if not payload:
            raise StoragePolicyError("empty artifacts are forbidden")
        if len(payload) > max_bytes:
            raise StoragePolicyError("artifact exceeds its write size limit")
        artifact = self._atomic_write(relative_path, payload)
        return artifact.model_copy(update={"kind": kind})

    def agent_directory_target(
        self, agent: AgentName, relative_directory: str
    ) -> Path:
        """Resolve a new Agent-owned directory without creating the final directory."""

        normalized = relative_directory.strip().rstrip("/")
        if not normalized:
            raise StoragePolicyError("agent directory must not be empty")
        self._check_agent_scope(agent, f"{normalized}/")
        marker = self._safe_target(f"{normalized}/.materialization-marker")
        target = marker.parent
        if target.exists():
            raise StoragePolicyError("agent materialization directory already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def register_agent_file(
        self,
        agent: AgentName,
        relative_path: str,
        *,
        max_bytes: int,
        kind: str,
    ) -> ArtifactRef:
        """Register a safely materialized file without loading it fully into memory."""

        self._check_agent_scope(agent, relative_path)
        target = self._safe_existing_artifact(f"artifact://{relative_path}")
        size = target.stat().st_size
        if size <= 0:
            raise StoragePolicyError("empty artifacts are forbidden")
        if size > max_bytes:
            raise StoragePolicyError("artifact exceeds its registration size limit")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return ArtifactRef(
            uri=f"artifact://{relative_path}",
            sha256=digest.hexdigest(),
            kind=kind,
        )

    def read_artifact_bytes(
        self, artifact: ArtifactRef, *, max_bytes: int
    ) -> bytes:
        target = self._safe_existing_artifact(artifact.uri)
        size = target.stat().st_size
        if size > max_bytes:
            raise StoragePolicyError("artifact exceeds its read size limit")
        payload = target.read_bytes()
        if len(payload) != size:
            raise StoragePolicyError("artifact changed while it was being read")
        self._verify_digest(payload, artifact.sha256)
        return payload

    def verify_artifact_file(
        self, artifact: ArtifactRef, *, max_bytes: int
    ) -> int:
        """Stream-verify an artifact digest and return its size."""

        target = self._safe_existing_artifact(artifact.uri)
        size = target.stat().st_size
        if size <= 0:
            raise StoragePolicyError("empty artifacts are forbidden")
        if size > max_bytes:
            raise StoragePolicyError("artifact exceeds its verification size limit")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise StoragePolicyError("artifact SHA-256 does not match its reference")
        return size

    def read_artifact_json(
        self, artifact: ArtifactRef, *, max_bytes: int = 1024 * 1024
    ) -> Any:
        payload = self.read_artifact_bytes(artifact, max_bytes=max_bytes)
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoragePolicyError("artifact is not valid UTF-8 JSON") from exc

    def write_runtime_json(self, relative_path: str, value: Any) -> ArtifactRef:
        if not relative_path.startswith(("project/var/runs/", "project/var/state/", "project/var/logs/")):
            raise StoragePolicyError("runtime writes are restricted to project/var")
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")
        return self._atomic_write(relative_path, payload)


class EventStore:
    """Single-process event and snapshot store for the V0.1 runtime."""

    def __init__(self, project_root: str | Path, artifacts: ArtifactStore) -> None:
        self.root = Path(project_root).resolve()
        self.artifacts = artifacts
        self._sequences: dict[str, int] = {}

    def append(
        self,
        snapshot: RunSnapshot,
        event_type: str,
        *,
        agent: AgentName | None = None,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> EventRecord:
        sequence = self._sequences.get(snapshot.run_id, 0) + 1
        self._sequences[snapshot.run_id] = sequence
        record = EventRecord(
            sequence=sequence,
            run_id=snapshot.run_id,
            event_type=event_type,
            state=snapshot.state,
            state_version=snapshot.version,
            agent=agent,
            task_id=task_id,
            details=details or {},
        )
        path = self.root / "project" / "var" / "state" / snapshot.run_id / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json() + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def save_snapshot(self, snapshot: RunSnapshot) -> ArtifactRef:
        return self.artifacts.write_runtime_json(
            f"project/var/state/{snapshot.run_id}/snapshot.json",
            snapshot.model_dump(mode="json"),
        )

    def load_snapshot(self, run_id: str) -> RunSnapshot:
        if not re.fullmatch(r"run_[A-Za-z0-9_-]{1,128}", run_id):
            raise StoragePolicyError("invalid run_id")
        state_root = self.root / "project" / "var" / "state"
        run_root = state_root / run_id
        current = self.root
        for part in ("project", "var", "state", run_id):
            current = current / part
            if current.is_symlink():
                raise StoragePolicyError("symbolic links are forbidden in state reads")
        if not _is_within(run_root.resolve(), state_root.resolve()):
            raise StoragePolicyError("run snapshot path escapes state root")
        snapshot_path = run_root / "snapshot.json"
        if not snapshot_path.is_file() or snapshot_path.is_symlink():
            raise StoragePolicyError("run snapshot does not exist")
        if snapshot_path.stat().st_size > 4 * 1024 * 1024:
            raise StoragePolicyError("run snapshot exceeds read limit")
        try:
            snapshot = RunSnapshot.model_validate_json(
                snapshot_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise StoragePolicyError("run snapshot is invalid") from exc
        if snapshot.run_id != run_id:
            raise StoragePolicyError("run snapshot identity mismatch")
        event_path = run_root / "events.jsonl"
        if event_path.is_file() and not event_path.is_symlink():
            if event_path.stat().st_size > 16 * 1024 * 1024:
                raise StoragePolicyError("run event log exceeds read limit")
            self._sequences[run_id] = len(
                event_path.read_text(encoding="utf-8").splitlines()
            )
        return snapshot
