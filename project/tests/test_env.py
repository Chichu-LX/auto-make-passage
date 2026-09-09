from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agents.env import WorkspaceEnvError, load_workspace_env


class WorkspaceEnvTests(unittest.TestCase):
    def test_loads_only_allowlisted_openai_compatible_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "OPENAI_API_KEY=test-key\n"
                "OPENAI_BASE_URL=https://example.com/v1\n"
                "OPENAI_MODEL=test-model\n"
                "UNRELATED_OPTION=must-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_workspace_env(path)
                self.assertEqual(
                    loaded,
                    {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"},
                )
                self.assertEqual(os.environ["OPENAI_MODEL"], "test-model")
                self.assertNotIn("UNRELATED_OPTION", os.environ)

    def test_existing_environment_wins_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_MODEL=file-model\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_MODEL": "shell-model"}, clear=True):
                load_workspace_env(path)
                self.assertEqual(os.environ["OPENAI_MODEL"], "shell-model")

    def test_rejects_malformed_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY\n", encoding="utf-8")
            with self.assertRaises(WorkspaceEnvError):
                load_workspace_env(path)


if __name__ == "__main__":
    unittest.main()
