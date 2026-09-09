from __future__ import annotations

import unittest

from research_agents.cli import (
    build_parser,
    command_continue_live,
    command_resolve_human,
)


class CliTests(unittest.TestCase):
    def test_continue_live_requires_a_run_and_preserves_safe_defaults(self) -> None:
        args = build_parser().parse_args(
            ["continue-live", "--run-id", "run_example"]
        )
        self.assertIs(args.func, command_continue_live)
        self.assertEqual(args.run_id, "run_example")
        self.assertEqual(args.review_provider, "deepseek")
        self.assertEqual(args.server_backend, "disabled")

    def test_resolve_human_requires_explicit_action_and_reason(self) -> None:
        args = build_parser().parse_args(
            [
                "resolve-human",
                "--run-id",
                "run_example",
                "--action",
                "retry",
                "--reason",
                "retry verified evidence merge",
            ]
        )
        self.assertIs(args.func, command_resolve_human)
        self.assertEqual(args.action, "retry")
        self.assertFalse(args.apply_only)


if __name__ == "__main__":
    unittest.main()
