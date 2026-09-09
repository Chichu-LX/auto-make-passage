from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openai import APIConnectionError

from research_agents.providers import DeepSeekModel


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text='{"status":"ok"}')


class _FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _FakeResponses()


class _FlakyResponses(_FakeResponses):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise APIConnectionError(request=SimpleNamespace())
        return super().create(**kwargs)


class _FlakyOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _FlakyResponses()


class _MalformedThenResponses(_FakeResponses):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.calls < 3:
            return SimpleNamespace(output_text='{"status":"unterminated}')
        return SimpleNamespace(output_text='{"status":"ok"}')


class _MalformedThenOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _MalformedThenResponses()


class _FakeStream:
    def __init__(self) -> None:
        self.events = [
            SimpleNamespace(type="response.output_text.delta", delta='{"status":'),
            SimpleNamespace(type="response.output_text.delta", delta='"ok"}'),
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.events)


class _StreamingResponses(_FakeResponses):
    def __init__(self) -> None:
        super().__init__()
        self.stream_kwargs = None

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return _FakeStream()


class _StreamingOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _StreamingResponses()


class _EmptyThenStreamingResponses(_StreamingResponses):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        self.stream_kwargs = kwargs
        stream = _FakeStream()
        if self.calls < 3:
            stream.events = []
        return stream


class _EmptyThenStreamingOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _EmptyThenStreamingResponses()


class _IncompleteStream(_FakeStream):
    def __init__(self) -> None:
        self.events = []

    def get_final_response(self):
        raise RuntimeError("Didn't receive a response.completed event")


class _IncompleteThenStreamingResponses(_StreamingResponses):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        self.stream_kwargs = kwargs
        if self.calls < 3:
            return _IncompleteStream()
        return _FakeStream()


class _IncompleteThenStreamingOpenAI:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.responses = _IncompleteThenStreamingResponses()


class ProviderTests(unittest.TestCase):
    def test_deepseek_v4_explicitly_requests_json_for_responses_api(self):
        with patch("research_agents.providers.OpenAI", _FakeOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(
                model.complete_json("Lead state transitions.", "Choose option A."),
                {"status": "ok"},
            )
        instructions = model._client.responses.kwargs["instructions"]
        self.assertIn("JSON", instructions)
        self.assertEqual(model._client.responses.kwargs["max_output_tokens"], 30000)
        self.assertEqual(
            model._client.responses.kwargs["reasoning"], {"effort": "high"}
        )
        self.assertEqual(model._client.init_kwargs["max_retries"], 0)

    def test_deepseek_retries_transient_connection_errors(self):
        with patch("research_agents.providers.OpenAI", _FlakyOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(model.complete_json("JSON", "{}"), {"status": "ok"})
        self.assertEqual(model._client.responses.calls, 3)

    def test_deepseek_retries_truncated_json_responses(self):
        with patch("research_agents.providers.OpenAI", _MalformedThenOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(model.complete_json("JSON", "{}"), {"status": "ok"})
        self.assertEqual(model._client.responses.calls, 3)

    def test_deepseek_v4_collects_streamed_json_deltas(self):
        with patch("research_agents.providers.OpenAI", _StreamingOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(model.complete_json("JSON", "{}"), {"status": "ok"})
        self.assertIsNotNone(model._client.responses.stream_kwargs)
        self.assertEqual(
            model._client.responses.stream_kwargs["text"],
            {"format": {"type": "json_object"}},
        )

    def test_deepseek_v4_retries_empty_streams_inside_provider_budget(self):
        with patch("research_agents.providers.OpenAI", _EmptyThenStreamingOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(model.complete_json("JSON", "{}"), {"status": "ok"})
        self.assertEqual(model._client.responses.calls, 3)

    def test_deepseek_v4_retries_streams_missing_completed_event(self):
        with patch("research_agents.providers.OpenAI", _IncompleteThenStreamingOpenAI):
            model = DeepSeekModel("test-key", model="deepseek-v4-pro")
            self.assertEqual(model.complete_json("JSON", "{}"), {"status": "ok"})
        self.assertEqual(model._client.responses.calls, 3)


if __name__ == "__main__":
    unittest.main()
