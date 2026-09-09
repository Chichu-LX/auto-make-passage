from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import AgentName


class AgentPackageError(RuntimeError):
    pass


class AgentPackageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    name: AgentName
    role: str = Field(min_length=10)
    execution_mode: Literal["orchestrator", "specialist"]
    model_provider: Literal[
        "deepseek",
        "claude-agent-sdk",
        "deepseek-or-claude-agent-sdk",
    ]
    allowed_tools: list[str]
    write_scopes: list[str]
    system_prompt: str
    input_contracts: list[str] = Field(min_length=1)
    output_contracts: list[str] = Field(min_length=1)
    skill_policy: Literal[
        "registry_only_reserved",
        "local_unreviewed_inactive",
        "local_reviewed_active",
    ]


@dataclass(frozen=True)
class AgentPackage:
    root: Path
    config: AgentPackageConfig
    system_prompt: str


def load_agent_package(module_file: str | Path) -> AgentPackage:
    root = Path(module_file).resolve().parent
    config_path = root / "config" / "agent.json"
    if not config_path.is_file():
        raise AgentPackageError(f"missing Agent package config: {config_path}")
    try:
        config = AgentPackageConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise AgentPackageError(f"invalid Agent package config: {config_path}") from exc
    prompt_path = (root / config.system_prompt).resolve()
    try:
        prompt_path.relative_to(root)
    except ValueError as exc:
        raise AgentPackageError("Agent system prompt escapes its package") from exc
    if not prompt_path.is_file() or prompt_path.suffix.lower() != ".md":
        raise AgentPackageError(f"missing Agent system prompt: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if len(prompt) < 50:
        raise AgentPackageError("Agent system prompt is unexpectedly short")
    return AgentPackage(root=root, config=config, system_prompt=prompt)
