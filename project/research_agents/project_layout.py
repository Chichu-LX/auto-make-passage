from __future__ import annotations

import re
from pathlib import PurePosixPath

from .schemas import TaskEnvelope


PROJECT_SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$"
_PROJECT_SLUG = re.compile(PROJECT_SLUG_PATTERN)


class ProjectLayoutError(ValueError):
    pass


def normalize_project_slug(value: str) -> str:
    slug = value.strip().lower().replace("_", "-")
    if not _PROJECT_SLUG.fullmatch(slug):
        raise ProjectLayoutError(
            "project slug must be 3-64 lowercase ASCII letters, digits, or internal hyphens"
        )
    return slug


def task_project_slug(task: TaskEnvelope) -> str:
    value = task.context.get("project_slug")
    if value is None:
        # Backward compatibility for persisted V0.x tasks and deterministic unit fixtures.
        return "legacy"
    if not isinstance(value, str):
        raise ProjectLayoutError("task context project_slug must be a string")
    return normalize_project_slug(value)


def scoped_path(area: str, project_slug: str, *parts: str) -> str:
    if area not in {"code", "data", "design", "reference"}:
        raise ProjectLayoutError("unsupported project area")
    slug = normalize_project_slug(project_slug)
    suffix = PurePosixPath(*parts)
    if suffix.is_absolute() or ".." in suffix.parts:
        raise ProjectLayoutError("project path is not traversal-safe")
    return PurePosixPath(area, slug, suffix).as_posix()


def artifact_prefix(area: str, project_slug: str, *parts: str) -> str:
    path = scoped_path(area, project_slug, *parts)
    return f"artifact://{path.rstrip('/')}/"
