from __future__ import annotations

import json
import os
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from requests import RequestException
from tavily import TavilyClient


PAPER_AUTHORITY_DOMAINS = (
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "openreview.net",
    "aclanthology.org",
    "aaai.org",
    "nature.com",
    "academic.oup.com",
    "pmc.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "arxiv.org",
)
DATA_AUTHORITY_DOMAINS = (
    "huggingface.co",
    "zenodo.org",
    "figshare.com",
    "archive.ics.uci.edu",
    "datadryad.org",
    "kaggle.com",
    "github.com",
    "bindingdb.org",
    "rcsb.org",
)


class ProviderConfigurationError(RuntimeError):
    pass


class ProviderRequestError(RuntimeError):
    """Redacted provider failure suitable for bounded Agent retry logic."""


class JsonModel(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]: ...


def _environment_value(*names: str) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)


def _required_key(explicit: str | None, *env_names: str) -> str:
    value = explicit or _environment_value(*env_names)
    if not value:
        joined = " or ".join(env_names)
        raise ProviderConfigurationError(f"missing required environment variable: {joined}")
    return value


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


class DeepSeekModel:
    """Minimal DeepSeek client using the OpenAI-compatible Responses API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model or _environment_value(
            "OPENAI_MODEL", "DEEPSEEK_MODEL"
        ) or "deepseek-v4-pro"
        self.reasoning_effort = reasoning_effort or _environment_value(
            "OPENAI_REASONING_EFFORT", "DEEPSEEK_REASONING_EFFORT"
        ) or "high"
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ProviderConfigurationError(
                "OPENAI_REASONING_EFFORT must be 'low', 'medium', or 'high'"
            )
        self._client = OpenAI(
            api_key=_required_key(api_key, "OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
            base_url=base_url
            or _environment_value("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com",
            timeout=timeout_seconds,
            # The OpenAI SDK retries twice by default.  This provider already owns
            # the retry policy below, so leaving SDK retries enabled multiplies one
            # bounded Agent request into as many as nine full read timeouts.
            max_retries=0,
        )

    def _complete_v4_json(self, system_prompt: str, user_prompt: str) -> str:
        request = {
            "model": self.model,
            "instructions": (
                f"{system_prompt}\n\n"
                "Return exactly one valid JSON object and no surrounding prose."
            ),
            "input": user_prompt,
            "temperature": 0,
            # A streaming read timeout is refreshed by every token and therefore
            # does not bound a model that keeps dribbling an excessively long
            # answer.  Keep JSON generations finite at the API boundary as well
            # as in the Agent-level schema and retry budgets.
            "max_output_tokens": 30000,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"format": {"type": "json_object"}},
            "store": False,
        }
        stream_method = getattr(self._client.responses, "stream", None)
        if callable(stream_method):
            chunks: list[str] = []
            completed_text = ""
            with stream_method(**request) as stream:
                for event in stream:
                    event_type = (
                        event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                    )
                    if event_type == "response.output_text.delta":
                        delta = (
                            event.get("delta")
                            if isinstance(event, dict)
                            else getattr(event, "delta", None)
                        )
                        if isinstance(delta, str):
                            chunks.append(delta)
                    elif event_type == "response.output_text.done":
                        text = (
                            event.get("text")
                            if isinstance(event, dict)
                            else getattr(event, "text", None)
                        )
                        if isinstance(text, str):
                            completed_text = text
                if chunks:
                    return "".join(chunks)
                if completed_text:
                    return completed_text
                get_final_response = getattr(stream, "get_final_response", None)
                if callable(get_final_response):
                    final_response = get_final_response()
                    output_text = getattr(final_response, "output_text", None)
                    if isinstance(output_text, str):
                        return output_text
            return ""
        response = self._client.responses.create(**request)
        return response.output_text

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            for attempt in range(1, 4):
                try:
                    if self.model.startswith("deepseek-v4"):
                        content = self._complete_v4_json(system_prompt, user_prompt)
                    else:
                        response = self._client.chat.completions.create(
                            model=self.model,
                            temperature=0,
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                        content = response.choices[0].message.content
                    if not content:
                        if attempt >= 3:
                            raise ProviderRequestError("DeepSeek returned an empty response")
                        continue
                    return _parse_json_object(content)
                except (
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    RateLimitError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    if attempt >= 3:
                        raise ProviderRequestError(
                            f"DeepSeek response stream failed: {type(exc).__name__}"
                        ) from exc
        except OpenAIError as exc:
            raise ProviderRequestError(
                f"DeepSeek request failed: {type(exc).__name__}"
            ) from exc


class TavilySearch:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = TavilyClient(api_key=_required_key(api_key, "TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        lower = query.lower()
        if "site:arxiv.org" in lower:
            include_domains = ["arxiv.org"]
        elif "github" in lower or "source code repository" in lower:
            include_domains = ["github.com"]
        elif any(
            token in lower
            for token in ("dataset", "data set", "benchmark data", "data repository")
        ):
            include_domains = list(DATA_AUTHORITY_DOMAINS)
        else:
            include_domains = list(PAPER_AUTHORITY_DOMAINS)
        search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced").strip().lower()
        if search_depth not in {"advanced", "basic"}:
            raise ProviderConfigurationError(
                "TAVILY_SEARCH_DEPTH must be 'advanced' or 'basic'"
            )
        response: dict[str, Any] | None = None
        last_error: RequestException | None = None
        for _ in range(3):
            try:
                response = self._client.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=False,
                    include_domains=include_domains,
                )
                break
            except RequestException as exc:
                last_error = exc
        if response is None:
            raise ProviderRequestError(
                f"Tavily request failed: {type(last_error).__name__}"
            ) from last_error
        results = response.get("results", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
            for item in results
            if isinstance(item, dict)
        ]
