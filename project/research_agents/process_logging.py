from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .schemas import AgentName, ProcessLogRecord


MAX_LOG_FILE_BYTES = 16 * 1024 * 1024
MAX_DETAIL_DEPTH = 7
MAX_DETAIL_ITEMS = 50
MAX_DETAIL_STRING = 1200

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|password|passwd|secret|authorization|cookie|credential|access[_-]?token)",
    re.IGNORECASE,
)
_KEY_VALUE_SECRET = re.compile(
    r"\b(?:sk|tvly|ghp|github_pat)-[A-Za-z0-9_-]{12,}\b",
    re.IGNORECASE,
)
_BEARER_SECRET = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}", re.IGNORECASE)
_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_INLINE_SECRET = re.compile(
    r"\b(api[_-]?key|password|passwd|secret|authorization|cookie|credential|access[_-]?token)"
    r"\s*([:=])\s*([^\s,;]+)",
    re.IGNORECASE,
)


class ProcessLogError(RuntimeError):
    pass


def _safe_segment(value: str) -> str:
    candidate = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", candidate):
        return candidate
    return f"id_{hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:16]}"


def redact_text(value: str, *, max_chars: int = MAX_DETAIL_STRING) -> str:
    """Redact common credentials while preserving a caller-selected bounded length."""
    text = value[:max(0, max_chars)]
    text = _INLINE_SECRET.sub(r"\1\2[REDACTED]", text)
    text = _KEY_VALUE_SECRET.sub("[REDACTED_CREDENTIAL]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)


def redact_details(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_DETAIL_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DETAIL_ITEMS:
                sanitized["_truncated_items"] = len(value) - MAX_DETAIL_ITEMS
                break
            name = redact_text(str(key))
            sanitized[name] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(str(key))
                else redact_details(item, depth=depth + 1)
            )
        return sanitized
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [
            redact_details(item, depth=depth + 1)
            for item in items[:MAX_DETAIL_ITEMS]
        ]
        if len(items) > MAX_DETAIL_ITEMS:
            sanitized.append(f"[TRUNCATED_{len(items) - MAX_DETAIL_ITEMS}_ITEMS]")
        return sanitized
    return redact_text(str(value))


class ProcessLogStore:
    """Append-only, per-Agent JSONL logs plus a redacted workflow timeline."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / "project" / "logging"
        self._lock = threading.Lock()

    def _safe_path(self, *segments: str, filename: str) -> Path:
        safe_segments = [_safe_segment(segment) for segment in segments]
        target = self.root.joinpath(*safe_segments, filename)
        try:
            target.parent.resolve().relative_to(self.workspace_root)
        except ValueError as exc:
            raise ProcessLogError("log path escapes the workspace") from exc
        current = self.workspace_root
        for part in target.relative_to(self.workspace_root).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ProcessLogError("symbolic links are forbidden in log paths")
        if target.is_symlink():
            raise ProcessLogError("symbolic link log targets are forbidden")
        return target

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size + len(line.encode("utf-8")) > MAX_LOG_FILE_BYTES:
            raise ProcessLogError(f"process log exceeds {MAX_LOG_FILE_BYTES} bytes")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, record: ProcessLogRecord) -> tuple[str, str]:
        sanitized = record.model_copy(
            update={"details": redact_details(record.details)}
        )
        line = sanitized.model_dump_json() + "\n"
        run_segment = _safe_segment(record.run_id)
        agent_path = self._safe_path(
            record.agent.value,
            run_segment,
            filename="process.jsonl",
        )
        workflow_path = self._safe_path(
            "_workflow",
            run_segment,
            filename="timeline.jsonl",
        )
        with self._lock:
            self._append_line(agent_path, line)
            self._append_line(workflow_path, line)
        return (
            agent_path.relative_to(self.workspace_root).as_posix(),
            workflow_path.relative_to(self.workspace_root).as_posix(),
        )

    def summary(self, run_id: str, *, max_recent: int = 20) -> dict[str, Any]:
        run_segment = _safe_segment(run_id)
        path = self._safe_path("_workflow", run_segment, filename="timeline.jsonl")
        if not path.is_file():
            return {
                "run_id": run_id,
                "record_count": 0,
                "status_counts": {},
                "agent_counts": {},
                "recent": [],
                "log_file": path.relative_to(self.workspace_root).as_posix(),
            }
        if path.stat().st_size > MAX_LOG_FILE_BYTES:
            raise ProcessLogError("workflow log exceeds read policy")
        records: list[dict[str, Any]] = []
        parse_errors = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
            except json.JSONDecodeError:
                parse_errors += 1
        status_counts: dict[str, int] = {}
        agent_counts: dict[str, int] = {}
        duration_by_agent: dict[str, float] = {}
        for record in records:
            status = str(record.get("status", "unknown"))
            agent = str(record.get("agent", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            duration = record.get("duration_ms")
            if isinstance(duration, (int, float)):
                duration_by_agent[agent] = round(
                    duration_by_agent.get(agent, 0.0) + float(duration), 3
                )
        recent = [
            {
                key: record.get(key)
                for key in (
                    "timestamp",
                    "agent",
                    "event",
                    "status",
                    "workflow_state",
                    "task_id",
                    "attempt",
                    "duration_ms",
                    "details",
                )
            }
            for record in records[-max(0, min(max_recent, 50)) :]
        ]
        return {
            "run_id": run_id,
            "record_count": len(records),
            "parse_errors": parse_errors,
            "status_counts": status_counts,
            "agent_counts": agent_counts,
            "duration_ms_by_agent": duration_by_agent,
            "recent": recent,
            "log_file": path.relative_to(self.workspace_root).as_posix(),
        }
