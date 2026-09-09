from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .providers import JsonModel, ProviderRequestError


GENERIC_JSON_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}


class ClaudeAgentSdkJsonModel(JsonModel):
    """Synchronous JSON adapter around one bounded Claude Agent SDK session.

    Runtime still owns dispatch and state transitions. The SDK session receives one
    prompt, a fixed working directory and an explicit tool allowlist; it cannot launch
    another workflow Agent or mutate Runtime state.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        model: str = "Codex-opus-4-6",
        allowed_tools: Sequence[str] = (),
        max_turns: int = 4,
        max_budget_usd: float = 1.0,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Claude Agent SDK workspace must be a regular directory")
        if max_turns < 1 or max_turns > 20:
            raise ValueError("Claude Agent SDK max_turns must be between 1 and 20")
        if max_budget_usd <= 0 or max_budget_usd > 20:
            raise ValueError("Claude Agent SDK budget must be within (0, 20]")
        forbidden = {"Write", "Edit", "Bash", "WebSearch", "WebFetch", "Agent"}
        tools = tuple(dict.fromkeys(str(value) for value in allowed_tools))
        if forbidden.intersection(tools):
            raise ValueError("review SDK adapter permits read-only local tools only")
        self.workspace_root = root
        self.model = model
        self.allowed_tools = tools
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

    async def _complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        prompt = (
            f"SYSTEM CONTRACT:\n{system_prompt}\n\n"
            "Return exactly one JSON object satisfying the supplied response contract. "
            "Do not wrap it in Markdown.\n\n"
            f"TASK INPUT:\n{user_prompt}"
        )
        options = ClaudeAgentOptions(
            cwd=str(self.workspace_root),
            model=self.model,
            allowed_tools=list(self.allowed_tools),
            tools=list(self.allowed_tools),
            permission_mode="plan",
            setting_sources=[],
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            thinking={"type": "adaptive"},
            output_format={
                "type": "json_schema",
                "schema": GENERIC_JSON_OBJECT_SCHEMA,
            },
        )
        final: Any = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    final = message.structured_output or message.result
        except Exception as exc:
            raise ProviderRequestError(
                f"Claude Agent SDK request failed: {type(exc).__name__}"
            ) from exc
        if isinstance(final, dict):
            return final
        if not isinstance(final, str) or not final.strip():
            raise ProviderRequestError("Claude Agent SDK returned no JSON result")
        try:
            parsed = json.loads(final)
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("Claude Agent SDK returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderRequestError("Claude Agent SDK result must be a JSON object")
        return parsed

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            return anyio.run(self._complete, system_prompt, user_prompt)
        except RuntimeError as exc:
            if "already running" in str(exc).lower():
                raise ProviderRequestError(
                    "Claude Agent SDK synchronous adapter cannot run inside an active event loop"
                ) from exc
            raise
