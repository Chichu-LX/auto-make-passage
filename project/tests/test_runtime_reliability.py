from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.lead_agent import RuleBasedLeadAgent
from agents.feedback_agent import FeedbackAgent
from research_agents.claude_sdk_adapter import ClaudeAgentSdkJsonModel
from research_agents.hooks import PauseForUserReviewHook
from research_agents.monitoring import MonitoringScheduler
from research_agents.providers import PAPER_AUTHORITY_DOMAINS, ProviderRequestError, TavilySearch
from research_agents.registry import build_mock_registry
from research_agents.runtime import ResearchRuntime
from research_agents.schemas import (
    AgentName,
    AgentResult,
    FeedbackControl,
    RunSnapshot,
    UserEvaluation,
    WorkflowState,
    utc_now,
)
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore


class _TavilyClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": []}


class _FeedbackModel:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "summary": "The latest stage is on track and has no blocking anomaly.",
            "control": {
                "action": "continue",
                "severity": "info",
                "reason": "The typed result and redacted logs are internally consistent.",
                "issue_codes": [],
            },
            "coordination": {
                "workflow_health": "on_track",
                "progress_summary": "The current specialist stage completed with typed evidence.",
                "stage_assessment": "No blocking mismatch was detected in the bounded log summary.",
                "next_actions": ["Request the two independent user evaluations."],
                "risks": [],
                "coordination_notes": ["Runtime remains the only state authority."],
            },
        }


class _CapturingFeedbackModel(_FeedbackModel):
    def __init__(self) -> None:
        self.payload = None

    def complete_json(self, system_prompt, user_prompt):
        import json

        self.payload = json.loads(user_prompt)
        return super().complete_json(system_prompt, user_prompt)


class _NeedsHumanSpecialist:
    name = AgentName.DOWNLOAD_PAPER

    def run(self, task):
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="needs_human",
            summary="Provider evidence could not be completed safely.",
            issues=[
                {
                    "severity": "blocking",
                    "code": "PROVIDER_EVIDENCE_MISSING",
                    "message": "Provider evidence could not be completed safely.",
                    "root_cause_key": "G-TEST:provider:evidence",
                }
            ],
            needs_human=True,
        )


class _RequestHumanFeedback:
    name = AgentName.FEEDBACK

    def run(self, task):
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded",
            summary="Keep the workflow at its existing human checkpoint.",
            feedback_control=FeedbackControl(
                action="request_human",
                severity="blocking",
                reason="The specialist already failed closed.",
                issue_codes=["PROVIDER_EVIDENCE_MISSING"],
            ),
        )


class _FlakyLead(RuleBasedLeadAgent):
    def __init__(self, state_machine):
        super().__init__(state_machine)
        self.dispatch_calls = 0
        self.transition_calls = 0

    def decide(self, snapshot, allowed_next_agents):
        self.dispatch_calls += 1
        if self.dispatch_calls < 3:
            raise ProviderRequestError("temporary dispatch failure")
        return super().decide(snapshot, allowed_next_agents)

    def decide_transition(self, snapshot, *, phase, options, context):
        self.transition_calls += 1
        if self.transition_calls < 3:
            raise ProviderRequestError("temporary transition failure")
        return super().decide_transition(
            snapshot,
            phase=phase,
            options=options,
            context=context,
        )


class RuntimeReliabilityTests(unittest.TestCase):
    def test_persisted_feedback_checkpoint_recovers_without_rerunning_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            machine = ResearchStateMachine()
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(machine),
                build_mock_registry(),
                execution_mode="live",
            )
            prior = runtime.create_run("recover feedback before user gate")
            decision = RuleBasedLeadAgent(machine).decide(
                prior, machine.allowed_next_agents(prior)
            )
            task = machine.build_task(prior, decision)
            source_result = AgentResult(
                run_id=prior.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="succeeded",
                summary="The persisted specialist result passed its typed contract.",
            )
            committed = machine.apply_result(prior, task, source_result)
            feedback_result = AgentResult(
                run_id=prior.run_id,
                task_id=f"feedback_{task.task_id}",
                agent=AgentName.FEEDBACK,
                status="succeeded",
                summary="The persisted Feedback audit recommends continuing.",
                feedback_control=FeedbackControl(
                    action="continue",
                    severity="info",
                    reason="No blocking inconsistency was found in the typed result.",
                ),
            )

            recovered = runtime.resume_feedback_checkpoint(
                committed,
                source_task=task,
                source_result=source_result,
                feedback_result=feedback_result,
                source_state=WorkflowState.ACQUIRING_PAPERS,
            )

            self.assertEqual(recovered.state, WorkflowState.AWAITING_USER_REVIEW)
            self.assertEqual(
                recovered.pending_user_review.source_state,
                WorkflowState.ACQUIRING_PAPERS,
            )
            self.assertEqual(
                recovered.pending_user_review.source_agent,
                AgentName.DOWNLOAD_PAPER,
            )
            self.assertEqual(len(recovered.queued_user_reviews), 1)
            self.assertEqual(
                recovered.queued_user_reviews[0].source_agent,
                AgentName.FEEDBACK,
            )

    def test_feedback_request_human_is_idempotent_after_specialist_human_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_mock_registry()
            registry[AgentName.DOWNLOAD_PAPER] = _NeedsHumanSpecialist()
            registry[AgentName.FEEDBACK] = _RequestHumanFeedback()
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
            )
            snapshot = runtime.run_step(runtime.create_run("fail closed safely"))
            self.assertEqual(snapshot.state, WorkflowState.AWAITING_USER_REVIEW)
            self.assertEqual(
                snapshot.pending_user_review.proposed_state,
                WorkflowState.NEEDS_HUMAN,
            )
            self.assertEqual(len(snapshot.queued_user_reviews), 1)

    def test_feedback_receives_typed_source_artifact_and_validation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _CapturingFeedbackModel()
            registry = build_mock_registry()
            registry[AgentName.FEEDBACK] = FeedbackAgent(
                model, ArtifactStore(directory)
            )
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
            )
            runtime.run_step(runtime.create_run("feedback evidence visibility"))
            context = model.payload["task"]["context"]
            self.assertIn("source_artifacts", context)
            self.assertIn("source_evidence", context)
            self.assertEqual(context["source_artifacts"], [])
            self.assertEqual(
                context["source_evidence"],
                [{"check": "mock-contract", "status": "passed"}],
            )

    def test_lead_retries_bounded_transient_provider_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            machine = ResearchStateMachine()
            lead = _FlakyLead(machine)
            runtime = ResearchRuntime(directory, lead, build_mock_registry())
            snapshot = runtime.run_step(runtime.create_run("retry transient provider"))
            self.assertTrue(snapshot.awaiting_user_review)
            self.assertEqual(lead.dispatch_calls, 3)
            self.assertGreaterEqual(lead.transition_calls, 3)

    def test_specialist_and_feedback_have_separate_user_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                build_mock_registry(),
                after_step_hook=PauseForUserReviewHook(),
            )
            snapshot = runtime.run_step(runtime.create_run("dual review contract"))
            self.assertEqual(snapshot.pending_user_review.source_agent, AgentName.DOWNLOAD_PAPER)
            self.assertEqual(len(snapshot.queued_user_reviews), 1)

            specialist_review = snapshot.pending_user_review
            snapshot = runtime.submit_user_evaluation(
                snapshot,
                UserEvaluation(
                    run_id=snapshot.run_id,
                    review_id=specialist_review.review_id,
                    observed_state_version=snapshot.version,
                    verdict="satisfied",
                    rating=5,
                ),
            )
            self.assertEqual(snapshot.state, WorkflowState.AWAITING_USER_REVIEW)
            self.assertEqual(snapshot.pending_user_review.source_agent, AgentName.FEEDBACK)

            feedback_review = snapshot.pending_user_review
            snapshot = runtime.submit_user_evaluation(
                snapshot,
                UserEvaluation(
                    run_id=snapshot.run_id,
                    review_id=feedback_review.review_id,
                    observed_state_version=snapshot.version,
                    verdict="needs_revision",
                    rating=2,
                    comments="Feedback did not identify the paper-quality risk; rerun discovery.",
                ),
            )
            self.assertEqual(snapshot.state, WorkflowState.ACQUIRING_PAPERS)
            self.assertEqual(snapshot.revision_history[-1].rejected_agent, AgentName.FEEDBACK)
            self.assertIn(AgentName.DOWNLOAD_PAPER, snapshot.completed_agents)
            self.assertEqual(
                snapshot.revision_history[-1].target_agent,
                AgentName.DOWNLOAD_PAPER,
            )

    def test_feedback_artifacts_use_the_run_project_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_mock_registry()
            registry[AgentName.FEEDBACK] = FeedbackAgent(
                _FeedbackModel(), ArtifactStore(directory)
            )
            runtime = ResearchRuntime(
                directory,
                RuleBasedLeadAgent(ResearchStateMachine()),
                registry,
            )
            snapshot = runtime.run_step(
                runtime.create_run(
                    "feedback project isolation",
                    project_slug="feedback-isolation",
                )
            )
            feedback_refs = [
                item for item in snapshot.artifacts if item.kind.startswith("feedback_")
            ]
            self.assertEqual(len(feedback_refs), 2)
            self.assertTrue(
                all(
                    item.uri.startswith("artifact://project/feedback/feedback-isolation/")
                    for item in feedback_refs
                )
            )

    def test_langgraph_rejects_an_illegal_transition(self):
        machine = ResearchStateMachine()
        with self.assertRaises(ValueError):
            machine.topology.assert_transition(
                WorkflowState.ACQUIRING_PAPERS,
                WorkflowState.WRITING,
            )

    def test_tavily_uses_advanced_authority_whitelist(self):
        with patch.dict(os.environ, {"TAVILY_SEARCH_DEPTH": "advanced"}), patch(
            "research_agents.providers.TavilyClient", _TavilyClient
        ):
            search = TavilySearch("test-key")
            search.search("drug target affinity NeurIPS open access PDF")
            call = search._client.calls[-1]
        self.assertEqual(call["search_depth"], "advanced")
        self.assertEqual(set(call["include_domains"]), set(PAPER_AUTHORITY_DOMAINS))

    def test_future_result_check_is_persisted_and_claimed_only_when_due(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            due_at = utc_now() + timedelta(hours=12)
            report = artifacts.write_runtime_json(
                "project/var/runs/run_monitor_test/reports/result.json",
                {
                    "run_id": "run_monitor_test",
                    "verdict": "in_progress",
                    "next_report_at": due_at.isoformat(),
                },
            ).model_copy(update={"kind": "result_check_report"})
            snapshot = RunSnapshot(
                run_id="run_monitor_test",
                project_slug="monitor-test",
                objective="monitor a long-running result",
                execution_mode="live",
                state=WorkflowState.RESULT_CHECK,
                version=5,
                artifacts=[report],
            )
            scheduler = MonitoringScheduler(directory)
            job = scheduler.schedule_from_snapshot(snapshot, server_backend="disabled")
            self.assertIsNotNone(job)
            self.assertEqual(scheduler.claim_due(now=utc_now()), [])
            claimed = scheduler.claim_due(now=due_at + timedelta(seconds=1))
            self.assertEqual([item.job_id for item in claimed], [job.job_id])
            fired = scheduler.mark_fired(job.job_id, when=due_at + timedelta(seconds=2))
            self.assertEqual(fired.status, "fired")
            self.assertEqual(len(scheduler.inventory().jobs), 1)

    def test_claude_sdk_adapter_rejects_mutating_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ClaudeAgentSdkJsonModel(directory, allowed_tools=["Write"])


if __name__ == "__main__":
    unittest.main()
