import json
import tempfile
import unittest
from pathlib import Path

from research_agents.process_logging import (
    ProcessLogError,
    ProcessLogStore,
    redact_details,
    redact_text,
)
from research_agents.schemas import AgentName, ProcessLogRecord, WorkflowState


class ProcessLogStoreTests(unittest.TestCase):
    def test_nested_validation_error_remains_observable(self) -> None:
        details = redact_details(
            {
                "attempt_errors": [
                    {
                        "attempt": 1,
                        "errors": [
                            {
                                "location": ["experiment_plans", "0", "metrics"],
                                "type": "missing",
                                "message": "Field required",
                            }
                        ],
                    }
                ]
            }
        )
        error = details["attempt_errors"][0]["errors"][0]
        self.assertEqual(error["type"], "missing")
        self.assertEqual(error["location"][0], "experiment_plans")

    def test_user_revision_redaction_preserves_feedback_but_removes_credentials(self) -> None:
        value = redact_text(
            "Keep this methodological feedback. password=unsafe-value "
            "and credential sk-abcdefghijklmnop must be removed.",
            max_chars=5000,
        )
        self.assertIn("methodological feedback", value)
        self.assertNotIn("unsafe-value", value)
        self.assertNotIn("sk-abcdefghijklmnop", value)

    def test_logs_are_partitioned_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProcessLogStore(directory)
            store.append(
                ProcessLogRecord(
                    run_id="run_test",
                    agent=AgentName.LEAD,
                    event="decision_completed",
                    status="succeeded",
                    workflow_state=WorkflowState.IDEATING,
                    state_version=2,
                    details={
                        "target_agent": "make-idea-agent",
                        "api_key": "do-not-log",
                        "message": "credential sk-abcdefghijklmnop",
                    },
                )
            )
            agent_path = (
                Path(directory)
                / "project/logging/lead-agent/run_test/process.jsonl"
            )
            timeline = (
                Path(directory)
                / "project/logging/_workflow/run_test/timeline.jsonl"
            )
            self.assertTrue(agent_path.is_file())
            self.assertTrue(timeline.is_file())
            text = agent_path.read_text(encoding="utf-8")
            self.assertNotIn("do-not-log", text)
            self.assertNotIn("sk-abcdefghijklmnop", text)
            payload = json.loads(text)
            self.assertEqual(payload["details"]["api_key"], "[REDACTED]")
            summary = store.summary("run_test")
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["agent_counts"]["lead-agent"], 1)

    def test_log_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "project").mkdir()
            (root / "project" / "logging").symlink_to(
                Path(outside), target_is_directory=True
            )
            store = ProcessLogStore(root)
            with self.assertRaises(ProcessLogError):
                store.append(
                    ProcessLogRecord(
                        run_id="run_test",
                        agent=AgentName.LEAD,
                        event="run_created",
                        status="info",
                        workflow_state=WorkflowState.ACQUIRING_PAPERS,
                        state_version=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
