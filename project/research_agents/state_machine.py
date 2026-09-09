from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .project_layout import normalize_project_slug
from .process_logging import redact_text
from .schemas import (
    AgentName,
    AgentResult,
    DispatchDecision,
    FeedbackControl,
    LeadTransitionDecision,
    LeadTransitionOption,
    RunSnapshot,
    TaskEnvelope,
    UserEvaluation,
    UserReviewRequest,
    UserRevisionNote,
    WorkflowState,
    utc_now,
)
from .workflow_graph import LangGraphWorkflowTopology


STATE_AGENT: dict[WorkflowState, AgentName] = {
    WorkflowState.ACQUIRING_PAPERS: AgentName.DOWNLOAD_PAPER,
    WorkflowState.IDEATING: AgentName.MAKE_IDEA,
    WorkflowState.DESIGNING_IDEA: AgentName.IDEA_DESIGN,
    WorkflowState.ACQUIRING_DATA: AgentName.DOWNLOAD_DATA,
    WorkflowState.IMPLEMENTING: AgentName.CODE_MAKE,
    WorkflowState.CODE_REVIEW: AgentName.CODE_REVIEW,
    WorkflowState.RUNNING: AgentName.SERVER_SCHEDULE,
    WorkflowState.RESULT_CHECK: AgentName.CHECK_RESULT,
    WorkflowState.EXPERIMENT_CODE_RUN: AgentName.EXPERIMENT_CODE_RUN,
    WorkflowState.WRITING: AgentName.WRITE_PAPER,
    WorkflowState.CONSISTENCY_REVIEW: AgentName.CHECK_PAPER_AND_CODE,
    WorkflowState.PAPER_REVIEW: AgentName.PAPER_REVIEW,
}

SUCCESS_NEXT: dict[WorkflowState, WorkflowState] = {
    WorkflowState.ACQUIRING_PAPERS: WorkflowState.IDEATING,
    WorkflowState.IDEATING: WorkflowState.DESIGNING_IDEA,
    WorkflowState.DESIGNING_IDEA: WorkflowState.ACQUIRING_DATA,
    WorkflowState.ACQUIRING_DATA: WorkflowState.IMPLEMENTING,
    WorkflowState.IMPLEMENTING: WorkflowState.CODE_REVIEW,
    WorkflowState.CODE_REVIEW: WorkflowState.RUNNING,
    WorkflowState.RUNNING: WorkflowState.RESULT_CHECK,
    WorkflowState.RESULT_CHECK: WorkflowState.EXPERIMENT_CODE_RUN,
    WorkflowState.EXPERIMENT_CODE_RUN: WorkflowState.WRITING,
    WorkflowState.WRITING: WorkflowState.CONSISTENCY_REVIEW,
    WorkflowState.CONSISTENCY_REVIEW: WorkflowState.PAPER_REVIEW,
    WorkflowState.PAPER_REVIEW: WorkflowState.COMPLETED,
}

# A human may explicitly accept a degraded-but-typed primary artifact. Evidence-gap
# files are intentionally absent: approving a schema failure must never advance the
# workflow merely because an automatic hook returned "satisfied".
HUMAN_ACCEPTABLE_PRIMARY_ARTIFACT: dict[WorkflowState, str] = {
    WorkflowState.ACQUIRING_PAPERS: "paper_manifest",
    WorkflowState.IDEATING: "idea_proposal",
    WorkflowState.DESIGNING_IDEA: "research_design",
    WorkflowState.IMPLEMENTING: "code_implementation_manifest",
    WorkflowState.CODE_REVIEW: "code_review_report",
}

TASK_TEMPLATES: dict[AgentName, str] = {
    AgentName.DOWNLOAD_PAPER: "discover_research_papers",
    AgentName.MAKE_IDEA: "rank_paper_grounded_innovation_candidates",
    AgentName.IDEA_DESIGN: "design_executable_research_blueprint",
    AgentName.DOWNLOAD_DATA: "download_design_required_datasets",
    AgentName.CODE_MAKE: "implement_research_code",
    AgentName.CODE_REVIEW: "review_code_against_idea",
    AgentName.SERVER_SCHEDULE: "schedule_and_monitor_experiment",
    AgentName.CHECK_RESULT: "check_experiment_results_against_design",
    AgentName.EXPERIMENT_CODE_RUN: "run_design_and_supplementary_experiments",
    AgentName.WRITE_PAPER: "write_evidence_grounded_paper",
    AgentName.CHECK_PAPER_AND_CODE: "check_claim_evidence_consistency",
    AgentName.PAPER_REVIEW: "review_paper_quality",
    AgentName.FEEDBACK: "audit_workflow_progress",
}

WRITE_SCOPES: dict[AgentName, list[str]] = {
    AgentName.DOWNLOAD_PAPER: ["reference/"],
    AgentName.DOWNLOAD_DATA: ["data/"],
    AgentName.MAKE_IDEA: ["design/"],
    AgentName.IDEA_DESIGN: ["design/"],
    AgentName.CODE_MAKE: ["code/"],
    AgentName.CODE_REVIEW: ["design/"],
    AgentName.SERVER_SCHEDULE: ["project/var/runs/", "project/var/logs/jobs/"],
    AgentName.CHECK_RESULT: ["design/", "project/var/logs/jobs/"],
    AgentName.EXPERIMENT_CODE_RUN: [
        "design/",
        "project/var/runs/",
        "project/var/logs/jobs/",
    ],
    AgentName.WRITE_PAPER: ["project/paper/"],
    AgentName.CHECK_PAPER_AND_CODE: ["project/reviews/consistency/"],
    AgentName.PAPER_REVIEW: ["project/reviews/paper/"],
    AgentName.FEEDBACK: ["project/feedback/"],
    AgentName.LEAD: [],
}


CONTROLLED_FAILURE_ROUTES: dict[WorkflowState, dict[AgentName, WorkflowState]] = {
    WorkflowState.DESIGNING_IDEA: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
    },
    WorkflowState.ACQUIRING_DATA: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
    },
    WorkflowState.IMPLEMENTING: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
    },
    WorkflowState.CODE_REVIEW: {
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.CODE_REVIEW: WorkflowState.CODE_REVIEW,
    },
    WorkflowState.RUNNING: {
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
    },
    WorkflowState.RESULT_CHECK: {
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
    },
    WorkflowState.EXPERIMENT_CODE_RUN: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
    },
    WorkflowState.WRITING: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
    },
    WorkflowState.CONSISTENCY_REVIEW: {
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
        AgentName.CHECK_PAPER_AND_CODE: WorkflowState.CONSISTENCY_REVIEW,
    },
    WorkflowState.PAPER_REVIEW: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
        AgentName.CHECK_PAPER_AND_CODE: WorkflowState.CONSISTENCY_REVIEW,
        AgentName.PAPER_REVIEW: WorkflowState.PAPER_REVIEW,
    },
}

MAX_GATE_ATTEMPTS = 3
CODE_REVIEW_MAX_ATTEMPTS = 5


def gate_attempt_limit(state: WorkflowState) -> int | None:
    """Return a retry cap, or ``None`` for user-gated experiment operations.

    Server execution and result repair can legitimately require an unknown number of
    compute cycles. They remain bounded by explicit user checkpoints and the 12-hour
    reporting contract instead of an arbitrary retry count.
    """

    if state == WorkflowState.CODE_REVIEW:
        return CODE_REVIEW_MAX_ATTEMPTS
    if state in {
        WorkflowState.RUNNING,
        WorkflowState.RESULT_CHECK,
        WorkflowState.EXPERIMENT_CODE_RUN,
        WorkflowState.PAPER_REVIEW,
    }:
        return None
    return MAX_GATE_ATTEMPTS


def retry_budget_remaining(state: WorkflowState, attempts: int) -> bool:
    limit = gate_attempt_limit(state)
    return limit is None or attempts < limit


FEEDBACK_RETRY_ROUTES: dict[WorkflowState, dict[AgentName, WorkflowState]] = {
    WorkflowState.ACQUIRING_PAPERS: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
    },
    WorkflowState.IDEATING: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
    },
    WorkflowState.DESIGNING_IDEA: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
    },
    WorkflowState.ACQUIRING_DATA: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
    },
    WorkflowState.IMPLEMENTING: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
    },
    WorkflowState.CODE_REVIEW: {
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.CODE_REVIEW: WorkflowState.CODE_REVIEW,
    },
    WorkflowState.RUNNING: {
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
    },
    WorkflowState.RESULT_CHECK: {
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
    },
    WorkflowState.EXPERIMENT_CODE_RUN: {
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.DOWNLOAD_DATA: WorkflowState.ACQUIRING_DATA,
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.SERVER_SCHEDULE: WorkflowState.RUNNING,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
    },
    WorkflowState.WRITING: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.CHECK_RESULT: WorkflowState.RESULT_CHECK,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
    },
    WorkflowState.CONSISTENCY_REVIEW: {
        AgentName.CODE_MAKE: WorkflowState.IMPLEMENTING,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
        AgentName.CHECK_PAPER_AND_CODE: WorkflowState.CONSISTENCY_REVIEW,
    },
    WorkflowState.PAPER_REVIEW: {
        AgentName.DOWNLOAD_PAPER: WorkflowState.ACQUIRING_PAPERS,
        AgentName.MAKE_IDEA: WorkflowState.IDEATING,
        AgentName.IDEA_DESIGN: WorkflowState.DESIGNING_IDEA,
        AgentName.EXPERIMENT_CODE_RUN: WorkflowState.EXPERIMENT_CODE_RUN,
        AgentName.WRITE_PAPER: WorkflowState.WRITING,
        AgentName.CHECK_PAPER_AND_CODE: WorkflowState.CONSISTENCY_REVIEW,
        AgentName.PAPER_REVIEW: WorkflowState.PAPER_REVIEW,
    },
}


def workflow_transition_edges() -> set[tuple[WorkflowState, WorkflowState]]:
    """Build the single transition allowlist compiled by LangGraph."""

    edges: set[tuple[WorkflowState, WorkflowState]] = set()
    # A specialist result and Feedback control are staged before the user gate.
    # Their proposed state can be successful, blocked, retried, or terminal; none
    # becomes user-accepted until the review transition is persisted.
    edges.update(
        (state, WorkflowState.AWAITING_USER_REVIEW)
        for state in WorkflowState
        if state != WorkflowState.AWAITING_USER_REVIEW
    )
    for source, target in SUCCESS_NEXT.items():
        edges.update(
            {
                (source, target),
                (source, source),
                (source, WorkflowState.AWAITING_USER_REVIEW),
                (source, WorkflowState.NEEDS_HUMAN),
                (source, WorkflowState.ABORTED),
            }
        )
    for source, routes in CONTROLLED_FAILURE_ROUTES.items():
        edges.update((source, target) for target in routes.values())
    for source, routes in FEEDBACK_RETRY_ROUTES.items():
        edges.update((source, target) for target in routes.values())
    review_targets = {
        *STATE_AGENT,
        WorkflowState.NEEDS_HUMAN,
        WorkflowState.COMPLETED,
        WorkflowState.ABORTED,
        WorkflowState.FAILED,
        WorkflowState.AWAITING_USER_REVIEW,
    }
    edges.update(
        (WorkflowState.AWAITING_USER_REVIEW, target) for target in review_targets
    )
    for blocked in STATE_AGENT:
        edges.add((WorkflowState.NEEDS_HUMAN, blocked))
        if blocked in SUCCESS_NEXT:
            edges.add((WorkflowState.NEEDS_HUMAN, SUCCESS_NEXT[blocked]))
    return edges


class StateMachineError(RuntimeError):
    pass


@dataclass(frozen=True)
class LeadTransitionPlan:
    """Policy-safe candidates presented to Lead without granting mutation access."""

    run_id: str
    observed_state_version: int
    phase: str
    source_state: WorkflowState
    options: tuple[LeadTransitionOption, ...]
    context: dict[str, Any]
    candidates: dict[str, RunSnapshot]


class ResearchStateMachine:
    def __init__(self) -> None:
        self.topology = LangGraphWorkflowTopology(workflow_transition_edges())

    def _copy_to(
        self,
        snapshot: RunSnapshot,
        target: WorkflowState,
        update: dict,
    ) -> RunSnapshot:
        # LangGraph guards state changes. Updating artifacts, counters, or timestamps
        # atomically inside the same state is not an edge traversal.
        if snapshot.state != target:
            try:
                self.topology.assert_transition(snapshot.state, target)
            except ValueError as exc:
                raise StateMachineError(str(exc)) from exc
        return snapshot.model_copy(update={"state": target, **update})

    def create_run(
        self,
        objective: str,
        *,
        execution_mode: str = "mock",
        project_slug: str | None = None,
    ) -> RunSnapshot:
        if not objective.strip():
            raise StateMachineError("objective must not be empty")
        run_id = f"run_{uuid4().hex[:12]}"
        slug = normalize_project_slug(project_slug or f"research-{run_id[-8:]}")
        return RunSnapshot(
            run_id=run_id,
            project_slug=slug,
            objective=objective.strip(),
            execution_mode=execution_mode,
            state=WorkflowState.ACQUIRING_PAPERS,
            version=1,
        )

    @staticmethod
    def _transition_action(
        source: WorkflowState,
        target: WorkflowState,
        *,
        phase: str,
        expected_success: WorkflowState | None,
        context: dict[str, Any],
    ) -> str:
        if target == WorkflowState.COMPLETED:
            return "complete"
        if target == WorkflowState.ABORTED:
            return "abort"
        if target == WorkflowState.NEEDS_HUMAN:
            return "request_human"
        if phase == "human_retry":
            return "retry"
        if target == source:
            if (
                phase == "specialist_result"
                and context.get("result_status") == "in_progress"
            ):
                return "wait"
            return "continue"
        if expected_success is not None and target == expected_success:
            return "advance"
        return "retry"

    @staticmethod
    def _transition_option_id(
        action: str,
        target: WorkflowState,
        target_agent: AgentName | None,
    ) -> str:
        suffix = target_agent.value if target_agent is not None else target.value.lower()
        return f"{action}_{suffix.replace('-', '_').lower()}"

    def plan_lead_transition(
        self,
        prior_snapshot: RunSnapshot,
        proposed_snapshot: RunSnapshot,
        *,
        phase: str,
        context: dict[str, Any],
        source_state: WorkflowState | None = None,
        alternate_routes: dict[AgentName, WorkflowState] | None = None,
        allow_alternate_retries: bool = True,
    ) -> LeadTransitionPlan:
        """Compile complete candidate snapshots; Lead chooses, Runtime later commits one.

        The deterministic StateMachine proposes the safe default and enumerates only
        topology-valid repair/escalation choices.  It does not choose between them.
        """

        if prior_snapshot.run_id != proposed_snapshot.run_id:
            raise StateMachineError("lead transition snapshots belong to different runs")
        if phase not in {
            "specialist_result",
            "feedback_control",
            "user_evaluation",
            "human_resume",
            "human_retry",
        }:
            raise StateMachineError(f"unsupported lead transition phase: {phase}")
        source = source_state or prior_snapshot.state
        expected_success = SUCCESS_NEXT.get(source)
        options: list[LeadTransitionOption] = []
        candidates: dict[str, RunSnapshot] = {}

        def add_candidate(
            candidate: RunSnapshot,
            *,
            action: str,
            target_agent: AgentName | None,
            reason_hint: str,
            recommended: bool = False,
        ) -> None:
            if prior_snapshot.state != candidate.state:
                try:
                    self.topology.assert_transition(
                        prior_snapshot.state, candidate.state
                    )
                except ValueError:
                    return
            option_id = self._transition_option_id(
                action, candidate.state, target_agent
            )
            if option_id in candidates:
                return
            option = LeadTransitionOption(
                option_id=option_id,
                action=action,
                target_state=candidate.state,
                target_agent=target_agent,
                reason_hint=reason_hint,
                recommended=recommended,
            )
            options.append(option)
            candidates[option_id] = candidate

        proposed_action = self._transition_action(
            prior_snapshot.state,
            proposed_snapshot.state,
            phase=phase,
            expected_success=expected_success,
            context=context,
        )
        proposed_agent = STATE_AGENT.get(proposed_snapshot.state)
        add_candidate(
            proposed_snapshot,
            action=proposed_action,
            target_agent=proposed_agent,
            reason_hint=(
                "Policy default derived from the typed result, Feedback recommendation, "
                "or explicit user verdict."
            ),
            recommended=True,
        )

        attempts = dict(proposed_snapshot.gate_attempts)
        attempts_used = attempts.get(source.value, 0)
        if (
            allow_alternate_retries
            and proposed_snapshot.state not in {
                WorkflowState.ABORTED,
                WorkflowState.FAILED,
            }
            and retry_budget_remaining(source, attempts_used)
        ):
            for target_agent, target_state in sorted(
                (alternate_routes or {}).items(), key=lambda item: item[0].value
            ):
                if target_state == proposed_snapshot.state:
                    continue
                retry_attempts = dict(attempts)
                lead_key = f"lead:{source.value}"
                retry_attempts[lead_key] = retry_attempts.get(lead_key, 0) + 1
                candidate = proposed_snapshot.model_copy(
                    update={
                        "state": target_state,
                        "completed_agents": list(prior_snapshot.completed_agents),
                        "blocked_from": None,
                        "gate_attempts": retry_attempts,
                        "pending_user_review": None,
                        "queued_user_reviews": [],
                        "updated_at": utc_now(),
                    }
                )
                add_candidate(
                    candidate,
                    action="retry",
                    target_agent=target_agent,
                    reason_hint=(
                        f"Lead-authorized repair route to {target_agent.value}; the "
                        "Runtime will preserve artifacts and audit history."
                    ),
                )

        if proposed_snapshot.state not in {
            WorkflowState.NEEDS_HUMAN,
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.ABORTED,
            WorkflowState.FAILED,
        }:
            human_candidate = proposed_snapshot.model_copy(
                update={
                    "state": WorkflowState.NEEDS_HUMAN,
                    "blocked_from": source,
                    "pending_user_review": None,
                    "queued_user_reviews": [],
                    "updated_at": utc_now(),
                }
            )
            add_candidate(
                human_candidate,
                action="request_human",
                target_agent=None,
                reason_hint="Pause safely because evidence or routing confidence is insufficient.",
            )

        if proposed_snapshot.state not in {
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.ABORTED,
            WorkflowState.FAILED,
        }:
            abort_candidate = proposed_snapshot.model_copy(
                update={
                    "state": WorkflowState.ABORTED,
                    "blocked_from": source,
                    "pending_user_review": None,
                    "queued_user_reviews": [],
                    "updated_at": utc_now(),
                }
            )
            add_candidate(
                abort_candidate,
                action="abort",
                target_agent=None,
                reason_hint="Terminate only for an explicit unrecoverable or unsafe condition.",
            )

        if not options or sum(item.recommended for item in options) != 1:
            raise StateMachineError("lead transition plan requires one recommended option")
        return LeadTransitionPlan(
            run_id=prior_snapshot.run_id,
            observed_state_version=prior_snapshot.version,
            phase=phase,
            source_state=source,
            options=tuple(options),
            context=dict(context),
            candidates=candidates,
        )

    def commit_lead_transition(
        self,
        plan: LeadTransitionPlan,
        decision: LeadTransitionDecision,
    ) -> RunSnapshot:
        """Validate Lead's semantic decision and return its precompiled snapshot."""

        if decision.run_id != plan.run_id:
            raise StateMachineError("lead transition run_id does not match plan")
        if decision.observed_state_version != plan.observed_state_version:
            raise StateMachineError("stale lead transition decision")
        if decision.phase != plan.phase:
            raise StateMachineError("lead transition phase does not match plan")
        option = next(
            (
                item
                for item in plan.options
                if item.option_id == decision.selected_option_id
            ),
            None,
        )
        if option is None:
            raise StateMachineError("Lead selected an option outside the policy plan")
        if (
            decision.action != option.action
            or decision.target_state != option.target_state
            or decision.target_agent != option.target_agent
        ):
            raise StateMachineError("Lead transition fields do not match selected option")
        return plan.candidates[option.option_id]

    def allowed_next_agents(self, snapshot: RunSnapshot) -> list[AgentName]:
        if snapshot.terminal or snapshot.state == WorkflowState.NEEDS_HUMAN:
            return []
        if snapshot.awaiting_user_review:
            return []
        agent = STATE_AGENT.get(snapshot.state)
        return [agent] if agent else []

    def resume_after_human(self, snapshot: RunSnapshot) -> RunSnapshot:
        """Resume a human-blocked gate without discarding accepted artifacts or audit history."""

        if snapshot.state != WorkflowState.NEEDS_HUMAN or snapshot.blocked_from is None:
            raise StateMachineError("run is not resumable from a human-blocked gate")
        blocked_state = snapshot.blocked_from
        expected_agent = STATE_AGENT.get(blocked_state)
        if expected_agent is not None and expected_agent in snapshot.completed_agents:
            target_state = SUCCESS_NEXT[blocked_state]
        else:
            target_state = blocked_state
        return self._copy_to(
            snapshot,
            target_state,
            {
                "blocked_from": None,
                "pending_user_review": None,
                "queued_user_reviews": [],
                "version": snapshot.version + 1,
                "updated_at": utc_now(),
            },
        )

    def retry_after_human(self, snapshot: RunSnapshot) -> RunSnapshot:
        """Explicitly retry the blocked gate even if an earlier cycle completed it."""

        if snapshot.state != WorkflowState.NEEDS_HUMAN or snapshot.blocked_from is None:
            raise StateMachineError("run is not retryable from a human-blocked gate")
        target_state = snapshot.blocked_from
        return self._copy_to(
            snapshot,
            target_state,
            {
                "blocked_from": None,
                "pending_user_review": None,
                "queued_user_reviews": [],
                "version": snapshot.version + 1,
                "updated_at": utc_now(),
            },
        )

    def validate_dispatch(
        self, snapshot: RunSnapshot, decision: DispatchDecision
    ) -> None:
        if decision.run_id != snapshot.run_id:
            raise StateMachineError("dispatch run_id does not match snapshot")
        if decision.observed_state_version != snapshot.version:
            raise StateMachineError("stale dispatch decision")
        if decision.decision != "dispatch":
            raise StateMachineError(f"unsupported decision for active run: {decision.decision}")
        allowed = self.allowed_next_agents(snapshot)
        if decision.target_agent not in allowed:
            raise StateMachineError(
                f"agent {decision.target_agent!s} not allowed in {snapshot.state.value}"
            )
        expected_template = TASK_TEMPLATES[decision.target_agent]
        if decision.task_template != expected_template:
            raise StateMachineError("task template is not allowed for target agent")

    def build_task(
        self, snapshot: RunSnapshot, decision: DispatchDecision
    ) -> TaskEnvelope:
        if decision.target_agent is None:
            raise StateMachineError("dispatch target is required")
        context = {
            "workflow_state": snapshot.state.value,
            "state_version": snapshot.version,
            "project_slug": snapshot.project_slug,
            "task_template": decision.task_template,
        }
        applicable_revision = next(
            (
                note
                for note in reversed(snapshot.revision_history)
                if note.target_agent == decision.target_agent
            ),
            None,
        )
        if applicable_revision is not None:
            context["user_revision"] = {
                "review_id": applicable_revision.review_id,
                "rejected_agent": applicable_revision.rejected_agent.value,
                "target_agent": applicable_revision.target_agent.value,
                "comments": applicable_revision.comments,
                "rating": applicable_revision.rating,
                "instruction": (
                    "Address this user-authorized revision without expanding the assigned "
                    "role, tool allowlist, or write scope."
                ),
            }
        return TaskEnvelope(
            run_id=snapshot.run_id,
            task_id=f"task_{snapshot.version:03d}_{uuid4().hex[:8]}",
            agent=decision.target_agent,
            objective=snapshot.objective,
            inputs=list(snapshot.artifacts),
            acceptance_criteria=[
                "return a schema-valid AgentResult",
                "do not claim evidence that is not present in registered artifacts",
                "stay within the assigned role and write scope",
            ],
            write_scope=WRITE_SCOPES[decision.target_agent],
            attempt=snapshot.gate_attempts.get(snapshot.state.value, 0) + 1,
            deadline_seconds=900,
            context=context,
        )

    def apply_result(
        self, snapshot: RunSnapshot, task: TaskEnvelope, result: AgentResult
    ) -> RunSnapshot:
        if result.run_id != snapshot.run_id or result.task_id != task.task_id:
            raise StateMachineError("result identity does not match task")
        if result.agent != task.agent:
            raise StateMachineError("result agent does not match task agent")

        if result.status == "in_progress":
            allowed_in_progress = {
                WorkflowState.RESULT_CHECK: AgentName.CHECK_RESULT,
                WorkflowState.EXPERIMENT_CODE_RUN: AgentName.EXPERIMENT_CODE_RUN,
            }
            expected_in_progress = allowed_in_progress.get(snapshot.state)
            if task.agent != expected_in_progress:
                raise StateMachineError(
                    "only the state-owned monitoring Agent may return in_progress"
                )
            return self._copy_to(
                snapshot,
                snapshot.state,
                {
                    "blocked_from": None,
                    "version": snapshot.version + 1,
                    "artifacts": [*snapshot.artifacts, *result.artifacts],
                    "updated_at": utc_now(),
                },
            )

        if result.status != "succeeded":
            attempts = dict(snapshot.gate_attempts)
            attempts[snapshot.state.value] = attempts.get(snapshot.state.value, 0) + 1
            route_state = CONTROLLED_FAILURE_ROUTES.get(snapshot.state, {}).get(
                result.route_hint
            )
            if (
                route_state is not None
                and not result.needs_human
                and retry_budget_remaining(
                    snapshot.state, attempts[snapshot.state.value]
                )
            ):
                return self._copy_to(
                    snapshot,
                    route_state,
                    {
                        "blocked_from": None,
                        "gate_attempts": attempts,
                        "version": snapshot.version + 1,
                        "artifacts": [*snapshot.artifacts, *result.artifacts],
                        "updated_at": utc_now(),
                    },
                )
            return self._copy_to(
                snapshot,
                WorkflowState.NEEDS_HUMAN,
                {
                    "blocked_from": snapshot.state,
                    "gate_attempts": attempts,
                    "version": snapshot.version + 1,
                    "artifacts": [*snapshot.artifacts, *result.artifacts],
                    "updated_at": utc_now(),
                },
            )

        expected = STATE_AGENT.get(snapshot.state)
        if task.agent != expected:
            raise StateMachineError("result cannot advance the current workflow state")
        next_state = SUCCESS_NEXT[snapshot.state]
        return self._copy_to(
            snapshot,
            next_state,
            {
                "version": snapshot.version + 1,
                "completed_agents": [*snapshot.completed_agents, task.agent],
                "artifacts": [*snapshot.artifacts, *result.artifacts],
                "updated_at": utc_now(),
            },
        )

    def apply_feedback(
        self,
        snapshot: RunSnapshot,
        *,
        source_state: WorkflowState,
        control: FeedbackControl | None,
        artifacts: list | None = None,
    ) -> RunSnapshot:
        """Apply only Runtime-allowlisted Feedback control; invalid control fails closed."""
        appended_artifacts = [*snapshot.artifacts, *(artifacts or [])]
        if control is None or control.action == "continue":
            if not artifacts:
                return snapshot
            return self._copy_to(
                snapshot,
                snapshot.state,
                {
                    "artifacts": appended_artifacts,
                    "version": snapshot.version + 1,
                    "updated_at": utc_now(),
                },
            )
        if control.action == "abort":
            return self._copy_to(
                snapshot,
                WorkflowState.ABORTED,
                {
                    "blocked_from": source_state,
                    "artifacts": appended_artifacts,
                    "version": snapshot.version + 1,
                    "updated_at": utc_now(),
                },
            )
        if control.action == "request_human":
            # A specialist failure may already have moved the proposed snapshot to
            # NEEDS_HUMAN. Feedback requesting the same safe pause is an idempotent
            # control decision, not a second topology transition. Preserve its
            # artifacts and audit version without asking LangGraph to validate a
            # meaningless NEEDS_HUMAN -> NEEDS_HUMAN edge.
            if snapshot.state == WorkflowState.NEEDS_HUMAN:
                return snapshot.model_copy(
                    update={
                        "blocked_from": source_state,
                        "artifacts": appended_artifacts,
                        "version": snapshot.version + 1,
                        "updated_at": utc_now(),
                    }
                )
            return self._copy_to(
                snapshot,
                WorkflowState.NEEDS_HUMAN,
                {
                    "blocked_from": source_state,
                    "artifacts": appended_artifacts,
                    "version": snapshot.version + 1,
                    "updated_at": utc_now(),
                },
            )

        attempts = dict(snapshot.gate_attempts)
        attempt_key = f"feedback:{source_state.value}"
        attempts[attempt_key] = attempts.get(attempt_key, 0) + 1
        route_state = FEEDBACK_RETRY_ROUTES.get(source_state, {}).get(
            control.target_agent
        )
        if route_state is None or not retry_budget_remaining(
            source_state, attempts[attempt_key]
        ):
            return self._copy_to(
                snapshot,
                WorkflowState.NEEDS_HUMAN,
                {
                    "blocked_from": source_state,
                    "artifacts": appended_artifacts,
                    "gate_attempts": attempts,
                    "version": snapshot.version + 1,
                    "updated_at": utc_now(),
                },
            )
        return self._copy_to(
            snapshot,
            route_state,
            {
                "blocked_from": None,
                "artifacts": appended_artifacts,
                "gate_attempts": attempts,
                "version": snapshot.version + 1,
                "updated_at": utc_now(),
            },
        )

    def stage_user_review(
        self,
        prior_snapshot: RunSnapshot,
        proposed_snapshot: RunSnapshot,
        *,
        task: TaskEnvelope,
        result: AgentResult,
        feedback_control: FeedbackControl | None,
        feedback_task: TaskEnvelope | None = None,
        feedback_result: AgentResult | None = None,
    ) -> RunSnapshot:
        """Pause after one specialist step without granting the hook mutation authority."""
        if prior_snapshot.pending_user_review is not None:
            raise StateMachineError("cannot stage a second user review")
        if prior_snapshot.run_id != proposed_snapshot.run_id:
            raise StateMachineError("review snapshots belong to different runs")
        if task.run_id != prior_snapshot.run_id or result.run_id != prior_snapshot.run_id:
            raise StateMachineError("review task/result identity does not match run")
        if task.agent != result.agent or task.task_id != result.task_id:
            raise StateMachineError("review task/result identity mismatch")
        allowed_targets = sorted(
            FEEDBACK_RETRY_ROUTES.get(prior_snapshot.state, {}),
            key=lambda agent: agent.value,
        )
        if not allowed_targets:
            raise StateMachineError(
                f"no user revision route exists for {prior_snapshot.state.value}"
            )
        control = feedback_control or FeedbackControl(
            action="continue",
            severity="info",
            reason="No Feedback Agent control was supplied; await user evaluation.",
        )
        review = UserReviewRequest(
            review_id=f"review_{task.task_id}",
            source_state=prior_snapshot.state,
            proposed_state=proposed_snapshot.state,
            source_agent=task.agent,
            source_task_id=task.task_id,
            result_status=result.status,
            result_summary=(
                result.summary.strip()
                or f"{task.agent.value} completed without a textual summary."
            ),
            issue_codes=[issue.code for issue in result.issues],
            artifact_kinds=[artifact.kind for artifact in result.artifacts],
            feedback_action=control.action,
            feedback_reason=control.reason,
            allowed_revision_targets=allowed_targets,
            default_revision_target=(
                task.agent if task.agent in allowed_targets else None
            ),
            prior_artifact_count=len(prior_snapshot.artifacts),
            prior_completed_agent_count=len(prior_snapshot.completed_agents),
            prior_gate_attempts=dict(prior_snapshot.gate_attempts),
        )
        queued: list[UserReviewRequest] = []
        if feedback_task is not None and feedback_result is not None:
            if feedback_task.agent != AgentName.FEEDBACK or feedback_result.agent != AgentName.FEEDBACK:
                raise StateMachineError("queued feedback review must belong to feedback-agent")
            if feedback_task.task_id != feedback_result.task_id:
                raise StateMachineError("feedback task/result identity mismatch")
            queued.append(
                UserReviewRequest(
                    review_id=f"review_{feedback_task.task_id}",
                    source_state=prior_snapshot.state,
                    proposed_state=proposed_snapshot.state,
                    source_agent=AgentName.FEEDBACK,
                    source_task_id=feedback_task.task_id,
                    result_status=feedback_result.status,
                    result_summary=feedback_result.summary.strip()
                    or "Feedback coordination completed without a textual summary.",
                    issue_codes=[item.code for item in feedback_result.issues],
                    artifact_kinds=[item.kind for item in feedback_result.artifacts],
                    feedback_action=control.action,
                    feedback_reason=control.reason,
                    allowed_revision_targets=allowed_targets,
                    default_revision_target=(
                        task.agent if task.agent in allowed_targets else None
                    ),
                    prior_artifact_count=len(prior_snapshot.artifacts),
                    prior_completed_agent_count=len(prior_snapshot.completed_agents),
                    prior_gate_attempts=dict(prior_snapshot.gate_attempts),
                )
            )
        return self._copy_to(
            proposed_snapshot,
            WorkflowState.AWAITING_USER_REVIEW,
            {
                "pending_user_review": review,
                "queued_user_reviews": queued,
                "version": proposed_snapshot.version + 1,
                "updated_at": utc_now(),
            },
        )

    def apply_user_evaluation(
        self,
        snapshot: RunSnapshot,
        evaluation: UserEvaluation,
    ) -> RunSnapshot:
        """Approve the staged result or rollback through a user-visible allowlist."""
        review = snapshot.pending_user_review
        if snapshot.state != WorkflowState.AWAITING_USER_REVIEW or review is None:
            raise StateMachineError("run is not awaiting user review")
        if evaluation.run_id != snapshot.run_id:
            raise StateMachineError("evaluation run_id does not match snapshot")
        if evaluation.review_id != review.review_id:
            raise StateMachineError("evaluation review_id does not match pending review")
        if evaluation.observed_state_version != snapshot.version:
            raise StateMachineError("stale user evaluation")

        if evaluation.verdict == "satisfied":
            accepted_state = review.proposed_state
            completed_agents = list(snapshot.completed_agents)
            blocked_from = snapshot.blocked_from
            primary_kind = HUMAN_ACCEPTABLE_PRIMARY_ARTIFACT.get(review.source_state)
            expected_agent = STATE_AGENT.get(review.source_state)
            accepts_degraded_primary = (
                review.result_status == "needs_human"
                and primary_kind is not None
                and primary_kind in review.artifact_kinds
                and review.source_agent == expected_agent
            )
            # Feedback may request a human after an otherwise successful stage. The
            # explicit human verdict is the final authority for this checkpoint.
            accepts_successful_human_gate = (
                review.result_status == "succeeded"
                and review.proposed_state == WorkflowState.NEEDS_HUMAN
                and review.source_agent == expected_agent
            )
            if accepts_degraded_primary or accepts_successful_human_gate:
                accepted_state = SUCCESS_NEXT[review.source_state]
                blocked_from = None
                if review.source_agent not in completed_agents:
                    completed_agents.append(review.source_agent)
            if snapshot.queued_user_reviews:
                next_review = snapshot.queued_user_reviews[0].model_copy(
                    update={
                        "proposed_state": accepted_state,
                        # The specialist has now been explicitly accepted. A later
                        # negative rating of Feedback must roll back Feedback only,
                        # not discard the accepted specialist artifacts/completion.
                        "prior_artifact_count": (
                            review.prior_artifact_count + len(review.artifact_kinds)
                        ),
                        "prior_completed_agent_count": len(completed_agents),
                        "prior_gate_attempts": dict(snapshot.gate_attempts),
                    }
                )
                return self._copy_to(
                    snapshot,
                    WorkflowState.AWAITING_USER_REVIEW,
                    {
                        "completed_agents": completed_agents,
                        "blocked_from": blocked_from,
                        "pending_user_review": next_review,
                        "queued_user_reviews": snapshot.queued_user_reviews[1:],
                        "version": snapshot.version + 1,
                        "updated_at": utc_now(),
                    },
                )
            return self._copy_to(
                snapshot,
                accepted_state,
                {
                    "completed_agents": completed_agents,
                    "blocked_from": blocked_from,
                    "pending_user_review": None,
                    "queued_user_reviews": [],
                    "version": snapshot.version + 1,
                    "updated_at": utc_now(),
                },
            )

        target = evaluation.target_agent
        if target is None:
            target = review.default_revision_target
        if target not in review.allowed_revision_targets:
            allowed = ", ".join(
                agent.value for agent in review.allowed_revision_targets
            )
            raise StateMachineError(
                f"revision target is not allowed; choose one of: {allowed}"
            )
        route_state = FEEDBACK_RETRY_ROUTES.get(review.source_state, {}).get(target)
        if route_state is None:
            raise StateMachineError("revision route is not defined for this review gate")

        attempt_key = f"user_review:{review.source_state.value}"
        attempts = dict(review.prior_gate_attempts)
        attempts[attempt_key] = snapshot.gate_attempts.get(attempt_key, 0) + 1
        safe_comments = redact_text(evaluation.comments, max_chars=5000)
        note = UserRevisionNote(
            review_id=review.review_id,
            rejected_agent=review.source_agent,
            target_agent=target,
            source_state=review.source_state,
            comments=safe_comments,
            rating=evaluation.rating,
        )
        history = [*snapshot.revision_history[-99:], note]
        return self._copy_to(
            snapshot,
            route_state,
            {
                "completed_agents": snapshot.completed_agents[
                    : review.prior_completed_agent_count
                ],
                "artifacts": snapshot.artifacts[: review.prior_artifact_count],
                "blocked_from": None,
                "gate_attempts": attempts,
                "pending_user_review": None,
                "queued_user_reviews": [],
                "revision_history": history,
                "version": snapshot.version + 1,
                "updated_at": utc_now(),
            },
        )
