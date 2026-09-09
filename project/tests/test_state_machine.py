import unittest

from pydantic import ValidationError

from agents.lead_agent import RuleBasedLeadAgent
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    DispatchDecision,
    FeedbackControl,
    LeadTransitionDecision,
    UserEvaluation,
    WorkflowState,
)
from research_agents.state_machine import (
    CONTROLLED_FAILURE_ROUTES,
    FEEDBACK_RETRY_ROUTES,
    TASK_TEMPLATES,
    ResearchStateMachine,
    StateMachineError,
)


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = ResearchStateMachine()
        self.snapshot = self.machine.create_run("test objective")

    def test_first_allowed_agent_is_download_paper(self) -> None:
        self.assertEqual(
            self.machine.allowed_next_agents(self.snapshot),
            [AgentName.DOWNLOAD_PAPER],
        )

    def test_lead_can_choose_a_policy_safe_alternate_repair_state(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.CODE_REVIEW, "version": 7}
        )
        dispatch = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, dispatch)
        result = AgentResult(
            run_id=snapshot.run_id,
            task_id=task.task_id,
            agent=AgentName.CODE_REVIEW,
            status="succeeded",
            summary="Review passed, but Lead may still select a policy-safe repair.",
        )
        proposed = self.machine.apply_result(snapshot, task, result)
        plan = self.machine.plan_lead_transition(
            snapshot,
            proposed,
            phase="specialist_result",
            context={"result_status": "succeeded", "issue_codes": []},
            source_state=WorkflowState.CODE_REVIEW,
            alternate_routes=CONTROLLED_FAILURE_ROUTES[WorkflowState.CODE_REVIEW],
        )
        option = next(
            item for item in plan.options if item.target_agent == AgentName.CODE_MAKE
        )
        decision = LeadTransitionDecision(
            run_id=snapshot.run_id,
            observed_state_version=snapshot.version,
            phase="specialist_result",
            selected_option_id=option.option_id,
            action=option.action,
            target_state=option.target_state,
            target_agent=option.target_agent,
            reason="Implementation evidence should be repaired before server scheduling.",
        )
        updated = self.machine.commit_lead_transition(plan, decision)
        self.assertEqual(updated.state, WorkflowState.IMPLEMENTING)
        self.assertNotIn(AgentName.CODE_REVIEW, updated.completed_agents)

    def test_lead_cannot_modify_a_selected_transition_option(self) -> None:
        snapshot = self.snapshot
        dispatch = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, dispatch)
        result = AgentResult(
            run_id=snapshot.run_id,
            task_id=task.task_id,
            agent=AgentName.DOWNLOAD_PAPER,
            status="succeeded",
            summary="Paper acquisition completed.",
        )
        proposed = self.machine.apply_result(snapshot, task, result)
        plan = self.machine.plan_lead_transition(
            snapshot,
            proposed,
            phase="specialist_result",
            context={"result_status": "succeeded", "issue_codes": []},
            source_state=WorkflowState.ACQUIRING_PAPERS,
        )
        option = next(item for item in plan.options if item.recommended)
        escaped = LeadTransitionDecision(
            run_id=snapshot.run_id,
            observed_state_version=snapshot.version,
            phase="specialist_result",
            selected_option_id=option.option_id,
            action=option.action,
            target_state=WorkflowState.WRITING,
            target_agent=option.target_agent,
            reason="Attempt to modify the state carried by an allowlisted option.",
        )
        with self.assertRaisesRegex(StateMachineError, "do not match"):
            self.machine.commit_lead_transition(plan, escaped)

    def test_stale_dispatch_is_rejected(self) -> None:
        decision = DispatchDecision(
            run_id=self.snapshot.run_id,
            observed_state_version=self.snapshot.version + 1,
            decision="dispatch",
            target_agent=AgentName.DOWNLOAD_PAPER,
            task_template="discover_research_papers",
            reason="stale",
        )
        with self.assertRaisesRegex(StateMachineError, "stale"):
            self.machine.validate_dispatch(self.snapshot, decision)

    def test_lead_cannot_dispatch_wrong_agent(self) -> None:
        decision = DispatchDecision(
            run_id=self.snapshot.run_id,
            observed_state_version=self.snapshot.version,
            decision="dispatch",
            target_agent=AgentName.CODE_MAKE,
            task_template="implement_research_code",
            reason="wrong agent",
        )
        with self.assertRaisesRegex(StateMachineError, "not allowed"):
            self.machine.validate_dispatch(self.snapshot, decision)

    def test_empty_objective_is_rejected(self) -> None:
        with self.assertRaises(StateMachineError):
            self.machine.create_run("  ")

    def test_same_state_atomic_update_is_not_a_langgraph_edge(self) -> None:
        blocked = self.snapshot.model_copy(
            update={
                "state": WorkflowState.NEEDS_HUMAN,
                "blocked_from": WorkflowState.ACQUIRING_PAPERS,
            }
        )
        control = FeedbackControl(
            action="continue",
            severity="info",
            reason="Attach feedback evidence while preserving the human gate.",
        )
        updated = self.machine.apply_feedback(
            blocked,
            source_state=WorkflowState.ACQUIRING_PAPERS,
            control=control,
            artifacts=[
                ArtifactRef(
                    uri="artifact://project/feedback/test/report.json",
                    sha256="a" * 64,
                    kind="feedback_coordination_report",
                )
            ],
        )
        self.assertEqual(updated.state, WorkflowState.NEEDS_HUMAN)
        self.assertEqual(updated.version, blocked.version + 1)

    def test_dispatch_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            DispatchDecision.model_validate(
                {
                    "run_id": self.snapshot.run_id,
                    "observed_state_version": 1,
                    "decision": "dispatch",
                    "reason": "invalid",
                    "unexpected": True,
                }
            )

    def test_rule_based_lead_is_dispatch_only(self) -> None:
        lead = RuleBasedLeadAgent(self.machine)
        decision = lead.decide(
            self.snapshot, self.machine.allowed_next_agents(self.snapshot)
        )
        self.assertEqual(decision.target_agent, AgentName.DOWNLOAD_PAPER)
        self.assertEqual(self.snapshot.state, WorkflowState.ACQUIRING_PAPERS)

    def test_core_order_places_idea_design_before_data(self) -> None:
        expected = [
            AgentName.DOWNLOAD_PAPER,
            AgentName.MAKE_IDEA,
            AgentName.IDEA_DESIGN,
            AgentName.DOWNLOAD_DATA,
        ]
        snapshot = self.snapshot
        lead = RuleBasedLeadAgent(self.machine)
        observed = []
        for _ in expected:
            allowed = self.machine.allowed_next_agents(snapshot)
            observed.extend(allowed)
            decision = lead.decide(snapshot, allowed)
            task = self.machine.build_task(snapshot, decision)
            snapshot = self.machine.apply_result(
                snapshot,
                task,
                AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=task.agent,
                    status="succeeded",
                    summary="ok",
                ),
            )
        self.assertEqual(observed, expected)
        self.assertEqual(snapshot.state, WorkflowState.IMPLEMENTING)

    def test_result_gate_precedes_supplementary_experiments_and_writing(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.RESULT_CHECK, "version": 12}
        )
        observed = []
        lead = RuleBasedLeadAgent(self.machine)
        for _ in range(2):
            allowed = self.machine.allowed_next_agents(snapshot)
            observed.extend(allowed)
            decision = lead.decide(snapshot, allowed)
            task = self.machine.build_task(snapshot, decision)
            snapshot = self.machine.apply_result(
                snapshot,
                task,
                AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=task.agent,
                    status="succeeded",
                    summary="objective gate passed",
                ),
            )
        self.assertEqual(
            observed, [AgentName.CHECK_RESULT, AgentName.EXPERIMENT_CODE_RUN]
        )
        self.assertEqual(snapshot.state, WorkflowState.WRITING)

    def test_experiment_monitoring_in_progress_is_user_gated_without_retry_cost(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.EXPERIMENT_CODE_RUN,
                "version": 14,
                "gate_attempts": {"EXPERIMENT_CODE_RUN": 99},
            }
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="in_progress",
                summary="supplementary jobs remain active",
            ),
        )
        self.assertEqual(updated.state, WorkflowState.EXPERIMENT_CODE_RUN)
        self.assertEqual(updated.gate_attempts["EXPERIMENT_CODE_RUN"], 99)

    def test_experiment_design_gap_routes_to_idea_design_without_cycle_cap(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.EXPERIMENT_CODE_RUN,
                "version": 14,
                "gate_attempts": {"EXPERIMENT_CODE_RUN": 10000},
            }
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="a necessary ablation is absent",
                route_hint=AgentName.IDEA_DESIGN,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.DESIGNING_IDEA)
        self.assertEqual(updated.gate_attempts["EXPERIMENT_CODE_RUN"], 10001)

    def test_code_review_failure_can_only_route_to_allowlisted_repair(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.CODE_REVIEW, "version": 8}
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="implementation mismatch",
                route_hint=AgentName.CODE_MAKE,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.IMPLEMENTING)
        self.assertEqual(updated.gate_attempts["CODE_REVIEW"], 1)

    def test_code_review_allows_five_iterations_then_requires_human(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.CODE_REVIEW,
                "version": 8,
                "gate_attempts": {"CODE_REVIEW": 4},
            }
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        self.assertEqual(task.attempt, 5)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="fifth review still failed",
                route_hint=AgentName.CODE_MAKE,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.NEEDS_HUMAN)
        self.assertEqual(updated.gate_attempts["CODE_REVIEW"], 5)

    def test_result_monitoring_in_progress_does_not_consume_retry_budget(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.RESULT_CHECK,
                "version": 12,
                "gate_attempts": {"RESULT_CHECK": 99},
            }
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="in_progress",
                summary="jobs are still running; next report is due in twelve hours",
            ),
        )
        self.assertEqual(updated.state, WorkflowState.RESULT_CHECK)
        self.assertEqual(updated.gate_attempts["RESULT_CHECK"], 99)

    def test_result_repair_cycles_have_no_attempt_count_cap(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.RESULT_CHECK,
                "version": 12,
                "gate_attempts": {"RESULT_CHECK": 10000},
            }
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="metric target was not met",
                route_hint=AgentName.CODE_MAKE,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.IMPLEMENTING)
        self.assertEqual(updated.gate_attempts["RESULT_CHECK"], 10001)

    def test_writing_literature_gap_routes_to_download_paper(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.WRITING, "version": 20}
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="Research Design citations lack verified local PDF evidence.",
                route_hint=AgentName.DOWNLOAD_PAPER,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.ACQUIRING_PAPERS)
        self.assertEqual(
            CONTROLLED_FAILURE_ROUTES[WorkflowState.WRITING][
                AgentName.DOWNLOAD_PAPER
            ],
            WorkflowState.ACQUIRING_PAPERS,
        )
        self.assertEqual(
            FEEDBACK_RETRY_ROUTES[WorkflowState.WRITING][AgentName.DOWNLOAD_PAPER],
            WorkflowState.ACQUIRING_PAPERS,
        )

    def test_non_allowlisted_failure_route_requires_human(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.CODE_REVIEW, "version": 8}
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        updated = self.machine.apply_result(
            snapshot,
            task,
            AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=task.agent,
                status="failed",
                summary="invalid repair",
                route_hint=AgentName.DOWNLOAD_PAPER,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.NEEDS_HUMAN)

    def test_human_resume_advances_when_blocked_agent_already_succeeded(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.NEEDS_HUMAN,
                "blocked_from": WorkflowState.ACQUIRING_PAPERS,
                "completed_agents": [AgentName.DOWNLOAD_PAPER],
                "version": 4,
            }
        )
        resumed = self.machine.resume_after_human(snapshot)
        self.assertEqual(resumed.state, WorkflowState.IDEATING)
        self.assertIsNone(resumed.blocked_from)

    def test_human_resume_retries_when_blocked_agent_did_not_succeed(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.NEEDS_HUMAN,
                "blocked_from": WorkflowState.ACQUIRING_PAPERS,
                "version": 4,
            }
        )
        resumed = self.machine.resume_after_human(snapshot)
        self.assertEqual(resumed.state, WorkflowState.ACQUIRING_PAPERS)

    def test_human_can_explicitly_retry_an_earlier_completed_cycle(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={
                "state": WorkflowState.NEEDS_HUMAN,
                "blocked_from": WorkflowState.ACQUIRING_DATA,
                "completed_agents": [AgentName.DOWNLOAD_DATA],
                "version": 7,
            }
        )
        retried = self.machine.retry_after_human(snapshot)
        self.assertEqual(retried.state, WorkflowState.ACQUIRING_DATA)
        self.assertIsNone(retried.blocked_from)

    def test_satisfied_review_accepts_typed_degraded_primary_artifact(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.IDEATING, "version": 5}
        )
        decision = DispatchDecision(
            run_id=snapshot.run_id,
            observed_state_version=snapshot.version,
            decision="dispatch",
            target_agent=AgentName.MAKE_IDEA,
            task_template=TASK_TEMPLATES[AgentName.MAKE_IDEA],
            reason="generate paper-grounded ideas",
        )
        task = self.machine.build_task(snapshot, decision)
        result = AgentResult(
            run_id=snapshot.run_id,
            task_id=task.task_id,
            agent=AgentName.MAKE_IDEA,
            status="needs_human",
            summary="typed proposal uses metadata-only evidence",
            artifacts=[
                ArtifactRef(
                    kind="idea_proposal",
                    uri="artifact://design/test/ideas/proposal.json",
                    sha256="0" * 64,
                )
            ],
            needs_human=True,
        )
        proposed = self.machine.apply_result(snapshot, task, result)
        paused = self.machine.stage_user_review(
            snapshot,
            proposed,
            task=task,
            result=result,
            feedback_control=FeedbackControl(
                action="retry",
                severity="warning",
                reason="try to improve local PDF coverage",
                target_agent=AgentName.MAKE_IDEA,
            ),
        )
        review = paused.pending_user_review
        accepted = self.machine.apply_user_evaluation(
            paused,
            UserEvaluation(
                run_id=paused.run_id,
                review_id=review.review_id,
                observed_state_version=paused.version,
                verdict="satisfied",
                rating=5,
            ),
        )
        self.assertEqual(accepted.state, WorkflowState.DESIGNING_IDEA)
        self.assertIn(AgentName.MAKE_IDEA, accepted.completed_agents)

    def test_feedback_retry_is_runtime_allowlisted(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.RUNNING, "version": 8}
        )
        updated = self.machine.apply_feedback(
            snapshot,
            source_state=WorkflowState.CODE_REVIEW,
            control=FeedbackControl(
                action="retry",
                severity="error",
                reason="Code review found a traceable implementation mismatch.",
                target_agent=AgentName.CODE_MAKE,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.IMPLEMENTING)
        self.assertEqual(updated.gate_attempts["feedback:CODE_REVIEW"], 1)

    def test_feedback_invalid_retry_fails_closed(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.RUNNING, "version": 8}
        )
        updated = self.machine.apply_feedback(
            snapshot,
            source_state=WorkflowState.CODE_REVIEW,
            control=FeedbackControl(
                action="retry",
                severity="blocking",
                reason="Attempt to route outside the review repair allowlist.",
                target_agent=AgentName.DOWNLOAD_PAPER,
            ),
        )
        self.assertEqual(updated.state, WorkflowState.NEEDS_HUMAN)

    def test_user_review_rejects_stale_or_non_allowlisted_revision(self) -> None:
        snapshot = self.snapshot.model_copy(
            update={"state": WorkflowState.CODE_REVIEW, "version": 8}
        )
        decision = RuleBasedLeadAgent(self.machine).decide(
            snapshot, self.machine.allowed_next_agents(snapshot)
        )
        task = self.machine.build_task(snapshot, decision)
        result = AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=task.agent,
            status="succeeded",
            summary="review passed",
        )
        proposed = self.machine.apply_result(snapshot, task, result)
        paused = self.machine.stage_user_review(
            snapshot,
            proposed,
            task=task,
            result=result,
            feedback_control=None,
        )
        review = paused.pending_user_review
        self.assertIsNotNone(review)
        with self.assertRaisesRegex(StateMachineError, "stale"):
            self.machine.apply_user_evaluation(
                paused,
                UserEvaluation(
                    run_id=paused.run_id,
                    review_id=review.review_id,
                    observed_state_version=paused.version - 1,
                    verdict="satisfied",
                ),
            )
        with self.assertRaisesRegex(StateMachineError, "not allowed"):
            self.machine.apply_user_evaluation(
                paused,
                UserEvaluation(
                    run_id=paused.run_id,
                    review_id=review.review_id,
                    observed_state_version=paused.version,
                    verdict="needs_revision",
                    comments="Return to paper discovery even though this gate forbids it.",
                    target_agent=AgentName.DOWNLOAD_PAPER,
                ),
            )


if __name__ == "__main__":
    unittest.main()
