from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import AgentName


AGENT_DIRECTORIES: dict[AgentName, str] = {
    AgentName.LEAD: "lead_agent",
    AgentName.DOWNLOAD_PAPER: "download_paper_agent",
    AgentName.DOWNLOAD_DATA: "download_data_agent",
    AgentName.MAKE_IDEA: "make_idea_agent",
    AgentName.IDEA_DESIGN: "idea_design_agent",
    AgentName.CODE_MAKE: "code_make_agent",
    AgentName.CODE_REVIEW: "code_review_agent",
    AgentName.SERVER_SCHEDULE: "server_schedule_agent",
    AgentName.CHECK_RESULT: "check_result_agent",
    AgentName.EXPERIMENT_CODE_RUN: "experiment_code_run_agent",
    AgentName.CHECK_PAPER_AND_CODE: "check_paper_and_code_agent",
    AgentName.WRITE_PAPER: "write_paper_agent",
    AgentName.PAPER_REVIEW: "paper_review_agent",
    AgentName.FEEDBACK: "feedback_agent",
}


class SkillReservationError(RuntimeError):
    pass


def _local_skill_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise SkillReservationError(f"local Skill contains a symbolic link: {path}")
        if not path.is_file() or path.name == "skill.json":
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def registry_path(workspace_root: str | Path, agent: AgentName) -> Path:
    return (
        Path(workspace_root).resolve()
        / "project"
        / "agents"
        / AGENT_DIRECTORIES[agent]
        / "skills"
        / "registry.json"
    )


def load_skill_reservations(
    workspace_root: str | Path, agent: AgentName
) -> list[dict[str, Any]]:
    path = registry_path(workspace_root, agent)
    if not path.is_file():
        raise SkillReservationError(f"missing skill registry for {agent.value}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("agent") != agent.value:
        raise SkillReservationError(f"skill registry agent mismatch: {path}")
    skills = payload.get("skills")
    if not isinstance(skills, list) or not 5 <= len(skills) <= 10:
        raise SkillReservationError(
            f"{agent.value} must reserve between 5 and 10 skills"
        )
    sources: set[str] = set()
    for skill in skills:
        if not isinstance(skill, dict):
            raise SkillReservationError(f"invalid skill entry in {path}")
        source = skill.get("source", "")
        repository = skill.get("repository", "")
        if not source or "@" not in source or "/" not in source:
            raise SkillReservationError(f"invalid skill source: {source!r}")
        if not repository.startswith("https://github.com/"):
            raise SkillReservationError(f"skill is not backed by GitHub: {source}")
        status = skill.get("status")
        if status not in {"reserved", "downloaded_unreviewed", "reviewed_active"}:
            raise SkillReservationError(f"invalid Skill lifecycle status: {source}")
        if status != "reserved":
            local_path = skill.get("local_path", "")
            if not isinstance(local_path, str) or not local_path.startswith("skills/"):
                raise SkillReservationError(f"downloaded Skill has no local path: {source}")
            local_root = (path.parent.parent / local_path).resolve()
            skills_root = path.parent.resolve()
            try:
                local_root.relative_to(skills_root)
            except ValueError as exc:
                raise SkillReservationError(f"local Skill escapes Agent package: {source}") from exc
            metadata_path = local_root / "skill.json"
            if not (local_root / "SKILL.md").is_file() or not metadata_path.is_file():
                raise SkillReservationError(f"downloaded Skill package is incomplete: {source}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("source") != source or metadata.get("agent") != agent.value:
                raise SkillReservationError(f"local Skill provenance mismatch: {source}")
            if status == "downloaded_unreviewed" and metadata.get("activated") is not False:
                raise SkillReservationError(f"unreviewed Skill must remain inactive: {source}")
            digest = _local_skill_digest(local_root)
            if digest != skill.get("tree_sha256") or digest != metadata.get("tree_sha256"):
                raise SkillReservationError(f"local Skill hash mismatch: {source}")
        if source in sources:
            raise SkillReservationError(f"duplicate skill source: {source}")
        sources.add(source)
    return skills


def validate_all_skill_reservations(
    workspace_root: str | Path,
) -> dict[str, int]:
    return {
        agent.value: len(load_skill_reservations(workspace_root, agent))
        for agent in AgentName
    }


def summarize_skill_inventory(workspace_root: str | Path) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for agent in AgentName:
        entries = load_skill_reservations(workspace_root, agent)
        statuses: dict[str, int] = {}
        for entry in entries:
            status = str(entry["status"])
            statuses[status] = statuses.get(status, 0) + 1
        inventory[agent.value] = {
            "total": len(entries),
            "local": sum(bool(entry.get("local_path")) for entry in entries),
            "active": sum(entry["status"] == "reviewed_active" for entry in entries),
            **statuses,
        }
    return inventory
