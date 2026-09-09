#!/usr/bin/env python3
"""Materialize reserved GitHub skills locally without executing or redistributing them.

The committed registry is an immutable source/provenance manifest. Synced Skill
directories and their generated ``skill.json`` files are local inspection
artifacts ignored by Git; this command never rewrites the public registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIRECTORIES = {
    "lead-agent": "lead_agent",
    "download-paper-agent": "download_paper_agent",
    "download-data-agent": "download_data_agent",
    "make-idea-agent": "make_idea_agent",
    "idea-design-agent": "idea_design_agent",
    "code-make-agent": "code_make_agent",
    "code-review-agent": "code_review_agent",
    "server-schedule-agent": "server_schedule_agent",
    "check-result-agent": "check_result_agent",
    "experiment-code-run-agent": "experiment_code_run_agent",
    "write-paper-agent": "write_paper_agent",
    "check-paper-and-code-agent": "check_paper_and_code_agent",
    "paper-review-agent": "paper_review_agent",
    "feedback-agent": "feedback_agent",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_SKILL_BYTES = 25 * 1024 * 1024
RISK_MARKERS = (
    "rm -rf",
    "subprocess.",
    "os.system(",
    "eval(",
    "exec(",
    "curl ",
    "wget ",
    "bypasspermissions",
)


class SkillSyncError(RuntimeError):
    pass


def _local_directory_name(source: str) -> str:
    package, skill_name = source.rsplit("@", 1)
    value = f"{package.replace('/', '--')}--{skill_name}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)


def _tree_digest(root: Path) -> tuple[str, int, int, list[str], list[str]]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    executable_files: list[str] = []
    risk_markers: set[str] = set()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise SkillSyncError(f"symbolic link is forbidden: {path}")
        if not path.is_file() or path.name == "skill.json":
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise SkillSyncError(f"skill file exceeds size policy: {relative}")
        total_bytes += size
        if total_bytes > MAX_SKILL_BYTES:
            raise SkillSyncError("skill package exceeds total size policy")
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        file_count += 1
        if path.stat().st_mode & 0o111:
            executable_files.append(relative)
        if size <= 1024 * 1024:
            text = content.decode("utf-8", errors="ignore").lower()
            risk_markers.update(marker for marker in RISK_MARKERS if marker in text)
    return digest.hexdigest(), file_count, total_bytes, executable_files, sorted(risk_markers)


def _source_commit(repository: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "ls-remote", repository, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.split()[0]
    except (subprocess.SubprocessError, IndexError):
        return "unknown"


def _download_skill(entry: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    source = str(entry["source"])
    skill_name = source.rsplit("@", 1)[1]
    temp_root = Path(tempfile.mkdtemp(prefix="agent-skill-"))
    subprocess.run(
        ["git", "init", "-q", str(temp_root)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        completed = subprocess.run(
            [
                "npx",
                "--yes",
                "skills",
                "add",
                source,
                "--agent",
                "claude-code",
                "--copy",
                "-y",
            ],
            cwd=temp_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise SkillSyncError(f"download timed out: {source}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout)[-1000:]
        shutil.rmtree(temp_root, ignore_errors=True)
        raise SkillSyncError(f"npx skills failed for {source}: {message}")
    package = temp_root / ".claude" / "skills" / skill_name
    if not (package / "SKILL.md").is_file():
        shutil.rmtree(temp_root, ignore_errors=True)
        raise SkillSyncError(f"downloaded package has no SKILL.md: {source}")
    return entry, package


def _install_one(
    agent: str, entry: dict[str, Any], *, refresh: bool
) -> dict[str, Any]:
    source = str(entry["source"])
    skill_name = source.rsplit("@", 1)[1]
    local_directory = _local_directory_name(source)
    destination = (
        WORKSPACE_ROOT
        / "project"
        / "agents"
        / AGENT_DIRECTORIES[agent]
        / "skills"
        / local_directory
    )
    if destination.exists() and not refresh:
        raise SkillSyncError(f"destination already exists: {destination}")
    original, package = _download_skill(entry)
    temp_root = package.parents[2]
    try:
        digest, count, total, executables, markers = _tree_digest(package)
        staged = destination.with_name(f".{destination.name}.staged")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(package, staged)
        metadata = {
            "schema_version": "1.0",
            "name": skill_name,
            "local_directory": local_directory,
            "agent": agent,
            "source": source,
            "repository": original["repository"],
            "catalog": original.get("catalog"),
            "purpose": original.get("purpose"),
            "status": "downloaded_unreviewed",
            "activated": False,
            "source_commit": _source_commit(str(original["repository"])),
            "tree_sha256": digest,
            "file_count": count,
            "total_bytes": total,
            "executable_files": executables,
            "static_risk_markers": markers,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "notice": "Local copy only. No third-party script has been executed by the Agent runtime.",
        }
        (staged / "skill.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
        return metadata
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def sync(agents: list[str], *, workers: int, refresh: bool) -> dict[str, Any]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for agent in agents:
        directory = AGENT_DIRECTORIES.get(agent)
        if directory is None:
            raise SkillSyncError(f"unsupported Agent package: {agent}")
        path = WORKSPACE_ROOT / "project" / "agents" / directory / "skills" / "registry.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        jobs.extend(
            (agent, entry)
            for entry in payload["skills"]
            if entry.get("status") == "reserved"
        )

    installed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_install_one, agent, entry, refresh=refresh): (agent, entry)
            for agent, entry in jobs
        }
        for future in as_completed(futures):
            agent, entry = futures[future]
            source = str(entry["source"])
            try:
                installed[source] = future.result()
                print(f"installed {agent}: {source}")
            except Exception as exc:
                failures[source] = f"{type(exc).__name__}: {exc}"
                print(f"failed {agent}: {source}: {failures[source]}")

    return {"installed": len(installed), "failed": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents", nargs="+", default=list(AGENT_DIRECTORIES), choices=list(AGENT_DIRECTORIES)
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = sync(args.agents, workers=args.workers, refresh=args.refresh)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
