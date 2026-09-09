from __future__ import annotations

import os
import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = WORKSPACE_ROOT / ".env"
MAX_ENV_BYTES = 64 * 1024
ALLOWED_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "CLAUDE_AGENT_MODEL",
    "CLAUDE_AGENT_SDK_ENABLED",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "TAVILY_API_KEY",
    "TAVILY_SEARCH_DEPTH",
}
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")


class WorkspaceEnvError(RuntimeError):
    pass


def _decode_value(raw: str, *, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise WorkspaceEnvError(f"unterminated quoted value on line {line_number}")
        value = value[1:-1]
    if "\x00" in value or "\n" in value or "\r" in value:
        raise WorkspaceEnvError(f"invalid control character on line {line_number}")
    return value


def load_workspace_env(
    path: str | Path | None = None,
    *,
    override: bool = False,
) -> set[str]:
    """Load only allowlisted provider variables without interpolation or logging values."""

    target = Path(path).resolve() if path is not None else DEFAULT_ENV_PATH.resolve()
    if not target.exists():
        return set()
    if target.is_symlink() or not target.is_file():
        raise WorkspaceEnvError("workspace .env must be a regular non-symbolic file")
    if target.stat().st_size > MAX_ENV_BYTES:
        raise WorkspaceEnvError("workspace .env exceeds the bounded size limit")
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceEnvError("workspace .env is not readable UTF-8") from exc
    loaded: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise WorkspaceEnvError(f"invalid .env assignment on line {line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise WorkspaceEnvError(f"invalid environment name on line {line_number}")
        if name not in ALLOWED_ENV_NAMES:
            continue
        value = _decode_value(raw_value, line_number=line_number)
        if override or name not in os.environ:
            os.environ[name] = value
        loaded.add(name)
    return loaded
