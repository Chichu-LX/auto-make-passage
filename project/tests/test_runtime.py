import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from agents.download_data_agent import DownloadDataAgent
from agents.download_paper_agent import DownloadPaperAgent
from agents.idea_design_agent import IdeaDesignAgent
from agents.lead_agent import RuleBasedLeadAgent
from agents.make_idea_agent import MakeIdeaAgent
from research_agents.downloads import DownloadPayload
from research_agents.hooks import AutoApproveUserReviewHook
from research_agents.registry import build_mock_registry
from research_agents.runtime import ResearchRuntime
from research_agents.schemas import AgentName, UserEvaluation, WorkflowState
from research_agents.schemas import AgentResult, FeedbackControl
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore

from test_agents import FakeSearch
from research_fixtures import advisor_response, valid_design_response, valid_idea_response


class FirstFourModel:
    def complete_json(self, system_prompt, user_prompt):
        if "read-only auxiliary Agent" in system_prompt:
            role = system_prompt.split(",", 1)[0].removeprefix("You are ")
            return advisor_response(role)
        if system_prompt.startswith("You are download-paper-agent"):
            return {
                "candidates": [
                    {
                        "paper_id": "paper-one",
                        "title": "PaperOne: integrated affinity model",
                        "authors": ["A. Researcher"],
                        "year": 2025,
                        "doi": "10.1000/integrated",
                        "abstract": "A paper supporting an integrated research gap.",
                        "source": "Journal",
                        "landing_page_url": "https://example.com/paper",
                        "direct_pdf_url": "https://example.com/paper.pdf",
                        "relevance_score": 0.95,
                    },
                    {
                        "paper_id": "paper-two",
                        "title": "PaperTwo: robust affinity evaluation",
                        "authors": ["B. Researcher"],
                        "year": 2025,
                        "doi": "10.1000/integrated-two",
                        "abstract": "A second paper supporting strict affinity evaluation.",
                        "source": "Bioinformatics",
                        "landing_page_url": "https://example.com/paper-two",
                        "direct_pdf_url": "https://example.com/paper-two.pdf",
                        "relevance_score": 0.94,
                    },
                ]
            }
        if system_prompt.startswith("You are download-data-agent"):
            return {
                "candidates": [
                    {
                        "dataset_id": "dataset-one",
                        "requirement_key": "dataset-davis",
                        "name": "Integrated dataset",
                        "description": "A labeled dataset for the experiment.",
                        "source": "Repository",
                        "landing_page_url": "https://example.com/data",
                        "direct_download_url": "https://example.com/data.csv",
                        "license_id": "cc-by-4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "license_evidence": "Licensed under CC BY 4.0.",
                        "format": "csv",
                        "estimated_size_bytes": 100,
                        "relevance_score": 0.95,
                    }
                ]
            }
        if system_prompt.startswith("You are idea-design-agent"):
            design = valid_design_response()
            module = design["reusable_code_modules"][0]
            module.update(
                {
                    "repository_url": "https://github.com/example/paperone",
                    "repository_status": "search_matched",
                    "license_id": None,
                    "reuse_mode": "reference_only",
                }
            )
            required_fields = json.loads(user_prompt).get("required_fields")
            if isinstance(required_fields, list):
                return {field: design[field] for field in required_fields}
            return design
        if system_prompt.startswith("You are make-idea-agent"):
            return valid_idea_response()
        raise AssertionError("unexpected model role")


class FirstFourDownloader:
    def __init__(self):
        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(buffer)
        self.pdf = buffer.getvalue()
        self.csv = b"label,value\na,1\nb,2\n"

    def fetch(self, url, policy):
        if url.endswith(".pdf"):
            if url.endswith("paper-two.pdf"):
                buffer = io.BytesIO()
                writer = PdfWriter()
                writer.add_blank_page(width=73, height=72)
                writer.write(buffer)
                content = buffer.getvalue()
            else:
                content = self.pdf
            content_type = "application/pdf"
        else:
            content = self.csv
            content_type = "text/csv"
        return DownloadPayload(
            content=content,
            final_url=url,
            content_type=content_type,
            sha256=hashlib.sha256(content).hexdigest(),
            redirect_count=0,
        )


class RuntimeTests(unittest.TestCase):
    class InvalidReviewHook:
        def evaluate(self, snapshot, review):
            return UserEvaluation(
                run_id=snapshot.run_id,
                review_id="review_wrong",
                observed_state_version=snapshot.version,
                verdict="satisfied",
            )

    class RetryFeedback:
        name = AgentName.FEEDBACK

        def run(self, task):
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=AgentName.FEEDBACK,
                status="succeeded",
                summary="code review logs require implementation repair",
                feedback_control=FeedbackControl(
                    action="retry",
                    severity="error",
                    reason="Code review evidence requires a Code Make retry.",
                    target_agent=AgentName.CODE_MAKE,
                    issue_codes=["CODE_REVIEW_MISMATCH"],
                ),
            )

    class IntegratedSearch(FakeSearch):
        def search(self, query, max_results=5):
            if "GitHub source code repository" in query and "PaperOne" in query:
                return [
                    {
                        "title": "PaperOne source code",
                        "url": "https://github.com/example/paperone",
                        "content": "PaperOne integrated affinity model",
                        "score": 0.99,
                    }
                ]
            return super().search(query, max_results=max_results)

    def test_first_four_real_agent_implementations_reach_implementing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            model = FirstFourModel()
            search = self.IntegratedSearch()
            downloader = FirstFourDownloader()
            registry = build_mock_registry()
            registry[AgentName.DOWNLOAD_PAPER] = DownloadPaperAgent(
                search, model, store, downloader
            )
            registry[AgentName.MAKE_IDEA] = MakeIdeaAgent(model, store)
            registry[AgentName.IDEA_DESIGN] = IdeaDesignAgent(model, store)
            registry[AgentName.DOWNLOAD_DATA] = DownloadDataAgent(
                search, model, store, downloader
            )
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
                feedback_after_each_step=False,
                after_step_hook=AutoApproveUserReviewHook(),
            )
            snapshot = runtime.run_steps(runtime.create_run("integrated topic"), steps=4)
            self.assertEqual(snapshot.state, WorkflowState.IMPLEMENTING)
            self.assertEqual(
                [artifact.kind for artifact in snapshot.artifacts],
                [
                    "paper_manifest",
                    "idea_proposal",
                    "research_design",
                    "research_design_markdown",
                    "data_manifest",
                    "data_profile_report",
                ],
            )

    def test_bounded_first_four_run_stops_after_idea_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                build_mock_registry(),
                feedback_after_each_step=False,
                after_step_hook=AutoApproveUserReviewHook(),
            )
            snapshot = runtime.run_steps(runtime.create_run("bounded experiment"), steps=4)
            self.assertEqual(snapshot.state, WorkflowState.IMPLEMENTING)
            self.assertEqual(
                snapshot.completed_agents,
                [
                    AgentName.DOWNLOAD_PAPER,
                    AgentName.MAKE_IDEA,
                    AgentName.IDEA_DESIGN,
                    AgentName.DOWNLOAD_DATA,
                ],
            )

    def test_full_mock_workflow_reaches_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = build_mock_registry()
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
                feedback_after_each_step=True,
                after_step_hook=AutoApproveUserReviewHook(),
            )
            snapshot = runtime.run_to_terminal(
                runtime.create_run("multi-agent reproducibility")
            )

            self.assertEqual(snapshot.state, WorkflowState.COMPLETED)
            self.assertEqual(
                snapshot.completed_agents,
                [
                    AgentName.DOWNLOAD_PAPER,
                    AgentName.MAKE_IDEA,
                    AgentName.IDEA_DESIGN,
                    AgentName.DOWNLOAD_DATA,
                    AgentName.CODE_MAKE,
                    AgentName.CODE_REVIEW,
                    AgentName.SERVER_SCHEDULE,
                    AgentName.CHECK_RESULT,
                    AgentName.EXPERIMENT_CODE_RUN,
                    AgentName.WRITE_PAPER,
                    AgentName.CHECK_PAPER_AND_CODE,
                    AgentName.PAPER_REVIEW,
                ],
            )
            feedback = registry[AgentName.FEEDBACK]
            self.assertEqual(len(feedback.calls), 12)

            for agent in AgentName:
                log_path = (
                    Path(directory)
                    / "project"
                    / "logging"
                    / agent.value
                    / snapshot.run_id
                    / "process.jsonl"
                )
                self.assertTrue(log_path.is_file(), agent.value)
                self.assertTrue(log_path.read_text(encoding="utf-8").strip())

            timeline = (
                Path(directory)
                / "project"
                / "logging"
                / "_workflow"
                / snapshot.run_id
                / "timeline.jsonl"
            )
            self.assertTrue(timeline.is_file())
            timeline_records = [
                json.loads(line)
                for line in timeline.read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(timeline_records), 70)
            controls = [
                record
                for record in timeline_records
                if record["event"] == "control_applied"
            ]
            self.assertEqual(len(controls), 12)
            self.assertTrue(
                all(record["details"]["action"] == "continue" for record in controls)
            )
            self.assertTrue(
                all(record["details"]["state_changed"] is False for record in controls)
            )
            self.assertEqual(
                sum(record["event"] == "user_review_requested" for record in timeline_records),
                24,
            )
            self.assertEqual(
                sum(
                    record["event"] == "user_evaluation_received"
                    for record in timeline_records
                ),
                24,
            )
            self.assertEqual(
                sum(
                    record["event"] == "transition_decision_completed"
                    for record in timeline_records
                ),
                48,
            )

            snapshot_path = Path(directory) / "project" / "var" / "state" / snapshot.run_id / "snapshot.json"
            event_path = Path(directory) / "project" / "var" / "state" / snapshot.run_id / "events.jsonl"
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(event_path.exists())
            saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["state"], "COMPLETED")
            self.assertGreaterEqual(len(event_path.read_text(encoding="utf-8").splitlines()), 19)

    def test_feedback_can_route_only_through_runtime_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = build_mock_registry()
            registry[AgentName.FEEDBACK] = self.RetryFeedback()
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
                feedback_after_each_step=True,
                after_step_hook=AutoApproveUserReviewHook(),
            )
            snapshot = runtime.create_run("feedback retry test").model_copy(
                update={"state": WorkflowState.CODE_REVIEW, "version": 7}
            )
            updated = runtime.run_step(snapshot)
            self.assertEqual(updated.state, WorkflowState.IMPLEMENTING)
            self.assertEqual(updated.gate_attempts["feedback:CODE_REVIEW"], 1)
            feedback_log = (
                Path(directory)
                / "project"
                / "logging"
                / AgentName.FEEDBACK.value
                / updated.run_id
                / "process.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("control_applied", feedback_log)

    def test_default_hook_pauses_and_user_revision_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                build_mock_registry(),
                feedback_after_each_step=True,
            )
            snapshot = runtime.run_step(runtime.create_run("human gated research"))
            self.assertEqual(snapshot.state, WorkflowState.AWAITING_USER_REVIEW)
            review = snapshot.pending_user_review
            self.assertIsNotNone(review)
            self.assertEqual(review.source_agent, AgentName.DOWNLOAD_PAPER)
            self.assertEqual(review.proposed_state, WorkflowState.IDEATING)

            snapshot = runtime.submit_user_evaluation(
                snapshot,
                UserEvaluation(
                    run_id=snapshot.run_id,
                    review_id=review.review_id,
                    observed_state_version=snapshot.version,
                    verdict="satisfied",
                    rating=5,
                ),
            )
            self.assertEqual(snapshot.state, WorkflowState.AWAITING_USER_REVIEW)
            feedback_review = snapshot.pending_user_review
            self.assertIsNotNone(feedback_review)
            self.assertEqual(feedback_review.source_agent, AgentName.FEEDBACK)
            snapshot = runtime.submit_user_evaluation(
                snapshot,
                UserEvaluation(
                    run_id=snapshot.run_id,
                    review_id=feedback_review.review_id,
                    observed_state_version=snapshot.version,
                    verdict="satisfied",
                    rating=5,
                ),
            )
            self.assertEqual(snapshot.state, WorkflowState.IDEATING)

            snapshot = runtime.run_step(snapshot)
            review = snapshot.pending_user_review
            self.assertIsNotNone(review)
            self.assertEqual(review.source_agent, AgentName.MAKE_IDEA)
            snapshot = runtime.submit_user_evaluation(
                snapshot,
                UserEvaluation(
                    run_id=snapshot.run_id,
                    review_id=review.review_id,
                    observed_state_version=snapshot.version,
                    verdict="needs_revision",
                    rating=2,
                    comments="创新点证据不足，请增加与权威论文的直接比较。",
                ),
            )
            self.assertEqual(snapshot.state, WorkflowState.IDEATING)
            self.assertEqual(snapshot.completed_agents, [AgentName.DOWNLOAD_PAPER])
            self.assertEqual(snapshot.revision_history[-1].target_agent, AgentName.MAKE_IDEA)

            decision = runtime.lead.decide(
                snapshot,
                runtime.state_machine.allowed_next_agents(snapshot),
            )
            task = runtime.state_machine.build_task(snapshot, decision)
            self.assertIn("权威论文", task.context["user_revision"]["comments"])

    def test_invalid_automatic_hook_fails_closed_at_user_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                build_mock_registry(),
                after_step_hook=self.InvalidReviewHook(),
            )
            snapshot = runtime.run_step(runtime.create_run("invalid hook"))
            self.assertEqual(snapshot.state, WorkflowState.AWAITING_USER_REVIEW)
            lead_log = (
                Path(directory)
                / "project"
                / "logging"
                / AgentName.LEAD.value
                / snapshot.run_id
                / "process.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("user_review_hook_failed", lead_log)


if __name__ == "__main__":
    unittest.main()
