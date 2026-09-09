from __future__ import annotations

from pydantic import ValidationError

from research_agents.schemas import (
    AgentName,
    DispatchDecision,
    LeadTransitionDecision,
    LeadTransitionOption,
    RunSnapshot,
)
from research_agents.state_machine import TASK_TEMPLATES


class LeadDecisionError(RuntimeError):
    """Raised when the model tries to escape the runtime-provided dispatch contract."""


def validate_allowed_agents(allowed_next_agents: list[AgentName]) -> None:
    if not allowed_next_agents:
        raise LeadDecisionError("runtime provided no dispatchable agent")
    if len(allowed_next_agents) != len(set(allowed_next_agents)):
        raise LeadDecisionError("runtime provided duplicate allowed agents")
    if any(agent not in TASK_TEMPLATES for agent in allowed_next_agents):
        raise LeadDecisionError("runtime provided an agent without a task template")


def validate_model_decision(
    response: object,
    snapshot: RunSnapshot,
    allowed_next_agents: list[AgentName],
) -> DispatchDecision:
    try:
        decision = DispatchDecision.model_validate(response)
    except ValidationError as exc:
        raise LeadDecisionError("model returned an invalid DispatchDecision") from exc
    if decision.run_id != snapshot.run_id:
        raise LeadDecisionError("model changed the runtime run_id")
    if decision.observed_state_version != snapshot.version:
        raise LeadDecisionError("model returned a stale state version")
    if decision.decision != "dispatch":
        raise LeadDecisionError("active workflow requires a dispatch decision")
    if decision.target_agent not in allowed_next_agents:
        raise LeadDecisionError("model selected an agent outside the runtime allowlist")
    expected_template = TASK_TEMPLATES[decision.target_agent]
    if decision.task_template != expected_template:
        raise LeadDecisionError("model selected a task template outside the runtime contract")
    if decision.needs_human:
        raise LeadDecisionError("dispatch cannot simultaneously request human intervention")
    return decision


def validate_transition_options(options: list[LeadTransitionOption]) -> None:
    if not options:
        raise LeadDecisionError("runtime provided no state-transition options")
    if len({item.option_id for item in options}) != len(options):
        raise LeadDecisionError("runtime provided duplicate state-transition options")
    if sum(item.recommended for item in options) != 1:
        raise LeadDecisionError("runtime must provide exactly one safe default transition")


def validate_model_transition_decision(
    response: object,
    snapshot: RunSnapshot,
    *,
    phase: str,
    options: list[LeadTransitionOption],
) -> LeadTransitionDecision:
    validate_transition_options(options)
    try:
        decision = LeadTransitionDecision.model_validate(response)
    except ValidationError as exc:
        raise LeadDecisionError("model returned an invalid LeadTransitionDecision") from exc
    if decision.run_id != snapshot.run_id:
        raise LeadDecisionError("model changed the transition run_id")
    if decision.observed_state_version != snapshot.version:
        raise LeadDecisionError("model returned a stale transition state version")
    if decision.phase != phase:
        raise LeadDecisionError("model changed the transition decision phase")
    option = next(
        (item for item in options if item.option_id == decision.selected_option_id),
        None,
    )
    if option is None:
        raise LeadDecisionError("model selected a transition outside the runtime allowlist")
    if (
        decision.action != option.action
        or decision.target_state != option.target_state
        or decision.target_agent != option.target_agent
    ):
        raise LeadDecisionError("model altered fields of the selected transition option")
    return decision


def normalize_model_transition_decision(
    response: object,
    snapshot: RunSnapshot,
    *,
    phase: str,
    options: list[LeadTransitionOption],
) -> LeadTransitionDecision:
    """Turn Lead's semantic option selection into the immutable Runtime contract.

    The only state-changing choice delegated to the model is ``selected_option_id``.
    Run identity, version, phase, action and routing fields are Runtime-owned values,
    so they are reconstructed from the selected allowlisted option rather than copied
    from a potentially drifting provider response.  An unknown option still fails
    closed and can never be converted into a state transition.
    """

    validate_transition_options(options)
    if not isinstance(response, dict):
        raise LeadDecisionError("model returned a non-object transition selection")
    selected_option_id = response.get("selected_option_id")
    if not isinstance(selected_option_id, str):
        raise LeadDecisionError("model omitted the selected transition option")
    option = next(
        (item for item in options if item.option_id == selected_option_id),
        None,
    )
    if option is None:
        raise LeadDecisionError("model selected a transition outside the runtime allowlist")

    reason = response.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 5:
        reason = option.reason_hint
    raw_codes = response.get("considered_issue_codes", [])
    issue_codes = (
        [item for item in raw_codes if isinstance(item, str)][:50]
        if isinstance(raw_codes, list)
        else []
    )
    decision = LeadTransitionDecision(
        run_id=snapshot.run_id,
        observed_state_version=snapshot.version,
        phase=phase,
        selected_option_id=option.option_id,
        action=option.action,
        target_state=option.target_state,
        target_agent=option.target_agent,
        reason=reason.strip(),
        considered_issue_codes=issue_codes,
    )
    return validate_model_transition_decision(
        decision,
        snapshot,
        phase=phase,
        options=options,
    )
