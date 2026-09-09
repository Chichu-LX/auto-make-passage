#!/usr/bin/env python3
"""Fail when Git's public boundary contains secrets, local outputs, or vendored Skills."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
REQUIRED_FILES = {
    ".github/workflows/ci.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "project/LICENSE",
    "project/pyproject.toml",
}
FORBIDDEN_EXACT = {
    ".env",
    "project/.env",
    "project/configs/servers.local.json",
    "project/configs/servers.local.yaml",
}
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
}


def tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=WORKSPACE_ROOT,
        check=True,
        capture_output=True,
    )
    return [value.decode("utf-8") for value in completed.stdout.split(b"\0") if value]


def main() -> int:
    errors: list[str] = []
    tracked = tracked_files()
    tracked_set = set(tracked)

    missing = sorted(REQUIRED_FILES - tracked_set)
    if missing:
        errors.append(f"required release files are not tracked: {', '.join(missing)}")

    forbidden = sorted(FORBIDDEN_EXACT & tracked_set)
    if forbidden:
        errors.append(f"private files are tracked: {', '.join(forbidden)}")

    for relative in tracked:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            errors.append(f"tracked file exceeds 10 MiB: {relative}")
        normalized = Path(relative)
        parts = normalized.parts
        if (
            len(parts) >= 5
            and parts[0:2] == ("project", "agents")
            and parts[3] == "skills"
            and relative.endswith("registry.json") is False
        ):
            errors.append(f"third-party Skill content is tracked: {relative}")
        if relative.startswith("project/var/") and relative != "project/var/.gitkeep":
            errors.append(f"runtime state is tracked: {relative}")
        if relative.startswith("project/logging/") and relative != "project/logging/README.md":
            errors.append(f"process log is tracked: {relative}")

        if path.stat().st_size <= 2 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label} in tracked file: {relative}")

    registries = sorted((WORKSPACE_ROOT / "project" / "agents").glob("*/skills/registry.json"))
    if len(registries) != 14:
        errors.append(f"expected 14 Skill registries, found {len(registries)}")
    for registry in registries:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        for entry in payload.get("skills", []):
            source = str(entry.get("source", ""))
            if entry.get("status") != "reserved":
                errors.append(f"non-reserved public Skill entry: {source}")
            if entry.get("local_path"):
                errors.append(f"public Skill entry contains local_path: {source}")
            if not entry.get("source_commit"):
                errors.append(f"Skill entry lacks source provenance: {source}")

    if errors:
        print("Open-source release audit failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Open-source release audit passed: {len(tracked)} tracked files, "
        f"{len(registries)} registry-only Skill inventories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
