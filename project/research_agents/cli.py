from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from agents.lead_agent import DeepSeekLeadAgent, RuleBasedLeadAgent
from .claude_sdk_adapter import ClaudeAgentSdkJsonModel
from .env import load_workspace_env
from .hooks import AutoApproveUserReviewHook
from .monitoring import MonitoringJob, MonitoringScheduler
from .providers import DeepSeekModel, TavilySearch
from .registry import build_live_registry, build_mock_registry
from .remote_execution import DisabledRemoteBackend, SafeLocalBackend, SafeSSHBackend
from .runtime import ResearchRuntime
from .schemas import AgentName, AgentResult, RunSnapshot, TaskEnvelope, UserEvaluation, utc_now
from .state_machine import ResearchStateMachine
from .storage import ArtifactStore, EventStore
from .skills import summarize_skill_inventory

DEFAULT_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[2])


def _keys(prompt: bool) -> tuple[str | None, str | None]:
    if prompt:
        return (
            getpass.getpass("DeepSeek API key: ").strip(),
            getpass.getpass("Tavily API key: ").strip(),
        )
    return (
        os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("TAVILY_API_KEY"),
    )


def _build_live(root: Path, prompt_keys: bool) -> tuple[DeepSeekModel, TavilySearch]:
    deepseek_key, tavily_key = _keys(prompt_keys)
    return DeepSeekModel(deepseek_key), TavilySearch(tavily_key)


def _build_live_runtime(
    root: Path,
    *,
    prompt_keys: bool,
    review_provider: str,
    server_backend: str,
) -> ResearchRuntime:
    model, search = _build_live(root, prompt_keys)
    if review_provider == "claude-sdk":
        enabled = os.getenv("CLAUDE_AGENT_SDK_ENABLED", "false").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            raise ValueError(
                "Claude Agent SDK review provider requires CLAUDE_AGENT_SDK_ENABLED=true"
            )
        review_model = ClaudeAgentSdkJsonModel(
            root,
            model=os.getenv("CLAUDE_AGENT_MODEL", "Codex-opus-4-6"),
        )
    else:
        review_model = model
    if server_backend == "ssh":
        backend = SafeSSHBackend()
    elif server_backend == "local":
        backend = SafeLocalBackend(root)
    else:
        backend = DisabledRemoteBackend()
    return ResearchRuntime(
        root,
        DeepSeekLeadAgent(model),
        build_live_registry(
            model,
            search,
            ArtifactStore(root),
            remote_backend=backend,
            review_model=review_model,
        ),
        feedback_after_each_step=True,
        execution_mode="live",
    )


def _load_snapshot(root: Path, run_id: str) -> RunSnapshot:
    artifacts = ArtifactStore(root)
    return EventStore(root, artifacts).load_snapshot(run_id)


def _snapshot_payload(
    snapshot: RunSnapshot,
    *,
    monitoring_job: MonitoringJob | None = None,
) -> dict:
    review = snapshot.pending_user_review
    accepted_count = (
        review.prior_completed_agent_count
        if review is not None
        else len(snapshot.completed_agents)
    )
    return {
        "run_id": snapshot.run_id,
        "project_slug": snapshot.project_slug,
        "execution_mode": snapshot.execution_mode,
        "state": snapshot.state.value,
        "version": snapshot.version,
        "accepted_agents": [
            agent.value for agent in snapshot.completed_agents[:accepted_count]
        ],
        "staged_agents": [
            agent.value for agent in snapshot.completed_agents[accepted_count:]
        ],
        "queued_user_review_count": len(snapshot.queued_user_reviews),
        "monitoring_wakeup": (
            {
                "job_id": monitoring_job.job_id,
                "state": monitoring_job.workflow_state,
                "due_at": monitoring_job.due_at.isoformat(),
                "status": monitoring_job.status,
                "server_backend": monitoring_job.server_backend,
            }
            if monitoring_job is not None
            else None
        ),
        "pending_user_review": (
            {
                "review_id": review.review_id,
                "source_agent": review.source_agent.value,
                "source_state": review.source_state.value,
                "proposed_state": review.proposed_state.value,
                "result_status": review.result_status,
                "result_summary": review.result_summary,
                "issue_codes": review.issue_codes,
                "artifact_kinds": review.artifact_kinds,
                "feedback_action": review.feedback_action,
                "feedback_reason": review.feedback_reason,
                "allowed_revision_targets": [
                    agent.value for agent in review.allowed_revision_targets
                ],
                "review_command": (
                    "research-agents review --run-id "
                    f"{snapshot.run_id} --verdict satisfied --rating 5"
                ),
            }
            if review
            else None
        ),
    }


def _print_snapshot(
    snapshot: RunSnapshot,
    *,
    monitoring_job: MonitoringJob | None = None,
) -> None:
    print(
        json.dumps(
            _snapshot_payload(snapshot, monitoring_job=monitoring_job),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_demo(args: argparse.Namespace) -> int:
    machine = ResearchStateMachine()
    runtime = ResearchRuntime(
        args.root,
        RuleBasedLeadAgent(machine),
        build_mock_registry(),
        feedback_after_each_step=True,
        execution_mode="mock",
        after_step_hook=(
            AutoApproveUserReviewHook() if args.auto_approve else None
        ),
    )
    snapshot = runtime.create_run(args.topic, project_slug=args.project)
    snapshot = (
        runtime.run_to_terminal(snapshot)
        if args.auto_approve
        else runtime.run_step(snapshot)
    )
    _print_snapshot(snapshot)
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    model, search = _build_live(args.root, args.prompt_keys)
    results = search.search(args.query, max_results=3)
    titles = [item["title"] for item in results]
    response = model.complete_json(
        "You are a connectivity checker. Return JSON with status='ok' and a short Chinese summary.",
        json.dumps({"query": args.query, "titles": titles}, ensure_ascii=False),
    )
    print(
        json.dumps(
            {
                "deepseek_status": response.get("status", "unknown"),
                "summary": response.get("summary", ""),
                "tavily_result_count": len(results),
                "sample_urls": [item["url"] for item in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_run_live(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    runtime = _build_live_runtime(
        root,
        prompt_keys=args.prompt_keys,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    runtime.feedback_after_each_step = args.feedback_each_step
    snapshot = runtime.run_step(
        runtime.create_run(args.topic, project_slug=args.project)
    )
    _print_snapshot(snapshot)
    return 0


def command_continue_live(args: argparse.Namespace) -> int:
    """Resume exactly one live business step from a persisted safe checkpoint."""

    root = Path(args.root).resolve()
    snapshot = _load_snapshot(root, args.run_id)
    if snapshot.execution_mode != "live":
        raise ValueError("continue-live only accepts a live run")
    if snapshot.awaiting_user_review:
        raise ValueError("run is awaiting user review; submit review instead")
    if snapshot.terminal or snapshot.state.value == "NEEDS_HUMAN":
        raise ValueError("run is already terminal or requires an explicit human resume")
    runtime = _build_live_runtime(
        root,
        prompt_keys=args.prompt_keys,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    runtime.events.load_snapshot(snapshot.run_id)
    updated = runtime.run_step(snapshot)
    _print_snapshot(updated)
    return 0


def command_recover_feedback(args: argparse.Namespace) -> int:
    """Resume a persisted Feedback result after a pre-commit Lead failure."""

    root = Path(args.root).resolve()
    snapshot = _load_snapshot(root, args.run_id)
    if snapshot.execution_mode != "live":
        raise ValueError("recover-feedback only accepts a live run")
    if snapshot.awaiting_user_review or snapshot.terminal:
        raise ValueError("run is not recoverable from its current checkpoint")
    result_root = root / "project/var/runs" / snapshot.run_id / "agent_results"
    source_path = result_root / f"{args.source_task_id}.json"
    feedback_path = result_root / f"feedback_{args.source_task_id}.json"
    if any(path.is_symlink() or not path.is_file() for path in (source_path, feedback_path)):
        raise ValueError("persisted specialist and Feedback results are required")
    source_result = AgentResult.model_validate_json(source_path.read_text(encoding="utf-8"))
    feedback_result = AgentResult.model_validate_json(feedback_path.read_text(encoding="utf-8"))
    source_state = args.source_state
    from .schemas import WorkflowState

    typed_source_state = WorkflowState(source_state)
    source_task = TaskEnvelope(
        run_id=snapshot.run_id,
        task_id=args.source_task_id,
        agent=source_result.agent,
        objective=snapshot.objective,
        inputs=list(snapshot.artifacts),
        acceptance_criteria=["recover persisted specialist result"],
        write_scope=[],
        context={"recovery": "persisted_feedback_result"},
    )
    runtime = _build_live_runtime(
        root,
        prompt_keys=args.prompt_keys,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    runtime.events.load_snapshot(snapshot.run_id)
    updated = runtime.resume_feedback_checkpoint(
        snapshot,
        source_task=source_task,
        source_result=source_result,
        feedback_result=feedback_result,
        source_state=typed_source_state,
    )
    _print_snapshot(updated)
    return 0


def command_resolve_human(args: argparse.Namespace) -> int:
    """Apply one explicit resolution to a durable NEEDS_HUMAN checkpoint."""

    root = Path(args.root).resolve()
    snapshot = _load_snapshot(root, args.run_id)
    if snapshot.execution_mode != "live":
        raise ValueError("resolve-human only accepts a live run")
    runtime = _build_live_runtime(
        root,
        prompt_keys=args.prompt_keys,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    runtime.events.load_snapshot(snapshot.run_id)
    if args.action == "retry":
        updated = runtime.retry_after_human(snapshot, reason=args.reason)
    else:
        updated = runtime.resume_after_human(snapshot, reason=args.reason)
    if not args.apply_only and not updated.terminal and not updated.awaiting_user_review:
        updated = runtime.run_step(updated)
    _print_snapshot(updated)
    return 0


def command_run_first_four(args: argparse.Namespace) -> int:
    """Start the first-four chain and stop at its first user-review gate."""
    root = Path(args.root).resolve()
    runtime = _build_live_runtime(
        root,
        prompt_keys=args.prompt_keys,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    snapshot = runtime.run_step(
        runtime.create_run(args.topic, project_slug=args.project)
    )
    _print_snapshot(snapshot)
    return 0


def command_status(args: argparse.Namespace) -> int:
    snapshot = _load_snapshot(Path(args.root).resolve(), args.run_id)
    _print_snapshot(snapshot)
    return 0


def command_review(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    snapshot = _load_snapshot(root, args.run_id)
    review = snapshot.pending_user_review
    if review is None:
        raise ValueError("run is not awaiting user review")
    verdict = args.verdict.replace("-", "_")
    target = AgentName(args.target_agent) if args.target_agent else None
    evaluation = UserEvaluation(
        run_id=snapshot.run_id,
        review_id=review.review_id,
        observed_state_version=snapshot.version,
        verdict=verdict,
        rating=args.rating,
        comments=args.comment,
        target_agent=target,
    )

    if snapshot.execution_mode == "live":
        control_runtime = _build_live_runtime(
            root,
            prompt_keys=args.prompt_keys,
            review_provider=args.review_provider,
            server_backend=args.server_backend,
        )
    else:
        machine = ResearchStateMachine()
        control_runtime = ResearchRuntime(
            root,
            RuleBasedLeadAgent(machine),
            build_mock_registry(),
            feedback_after_each_step=True,
            execution_mode=snapshot.execution_mode,
        )
    control_runtime.events.load_snapshot(snapshot.run_id)
    updated = control_runtime.submit_user_evaluation(snapshot, evaluation)
    if (
        args.apply_only
        or updated.terminal
        or updated.state.value == "NEEDS_HUMAN"
        or updated.awaiting_user_review
    ):
        _print_snapshot(updated)
        return 0

    wakeup = MonitoringScheduler(root).schedule_from_snapshot(
        updated,
        review_provider=args.review_provider,
        server_backend=args.server_backend,
    )
    if wakeup is not None:
        _ensure_monitor_daemon(root)
        _print_snapshot(updated, monitoring_job=wakeup)
        return 0

    runtime = control_runtime
    updated = runtime.run_step(updated)
    _print_snapshot(updated)
    return 0


def command_skills(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            summarize_skill_inventory(args.root),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_graph(args: argparse.Namespace) -> int:
    machine = ResearchStateMachine()
    payload = machine.topology.snapshot()
    payload["transitions"] = [
        {"from": source.value, "to": target.value}
        for source, target in sorted(
            machine.topology.edges,
            key=lambda item: (item[0].value, item[1].value),
        )
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _ensure_monitor_daemon(root: Path) -> int:
    """Start one detached local timer process when no healthy daemon is recorded."""

    artifacts = ArtifactStore(root)
    pid_path = root / "project/var/runs/monitoring/daemon.json"
    if pid_path.is_file() and not pid_path.is_symlink() and pid_path.stat().st_size < 4096:
        try:
            pid = int(json.loads(pid_path.read_text(encoding="utf-8"))["pid"])
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    log_path = root / "project/var/logs/monitoring/daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_symlink():
        raise ValueError("monitor daemon log cannot be a symbolic link")
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "research_agents.cli",
                "monitor-daemon",
                "--root",
                str(root),
                "--forever",
            ],
            cwd=root / "project",
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            close_fds=True,
        )
    artifacts.write_runtime_json(
        "project/var/runs/monitoring/daemon.json",
        {"pid": process.pid, "started_at": utc_now().isoformat()},
    )
    return process.pid


def _fire_monitoring_job(root: Path, job: MonitoringJob) -> RunSnapshot:
    snapshot = _load_snapshot(root, job.run_id)
    if snapshot.awaiting_user_review:
        raise ValueError("monitoring wake-up is blocked by an unresolved user review")
    if snapshot.state.value != job.workflow_state:
        raise ValueError("monitoring wake-up state no longer matches the run")
    if job.execution_mode == "live":
        runtime = _build_live_runtime(
            root,
            prompt_keys=False,
            review_provider=job.review_provider,
            server_backend=job.server_backend,
        )
    else:
        machine = ResearchStateMachine()
        runtime = ResearchRuntime(
            root,
            RuleBasedLeadAgent(machine),
            build_mock_registry(),
            feedback_after_each_step=True,
            execution_mode="mock",
        )
    runtime.events.load_snapshot(snapshot.run_id)
    return runtime.run_step(snapshot)


def command_monitor_daemon(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    scheduler = MonitoringScheduler(root)
    fired: list[dict[str, str]] = []
    while True:
        for job in scheduler.claim_due():
            try:
                snapshot = _fire_monitoring_job(root, job)
                scheduler.mark_fired(job.job_id)
                fired.append(
                    {
                        "job_id": job.job_id,
                        "run_id": job.run_id,
                        "state": snapshot.state.value,
                    }
                )
            except Exception as exc:
                scheduler.mark_failed(job.job_id, error_type=type(exc).__name__)
                fired.append(
                    {
                        "job_id": job.job_id,
                        "run_id": job.run_id,
                        "error_type": type(exc).__name__,
                    }
                )
        if not args.forever:
            print(json.dumps({"processed": fired}, ensure_ascii=False, indent=2))
            return 0
        time.sleep(args.poll_seconds)


def command_monitor_status(args: argparse.Namespace) -> int:
    index = MonitoringScheduler(Path(args.root).resolve()).inventory()
    print(index.model_dump_json(indent=2))
    return 0


def _add_live_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-provider",
        choices=["deepseek", "claude-sdk"],
        default="deepseek",
        help="model provider for Paper Review and Feedback only",
    )
    parser.add_argument(
        "--server-backend",
        choices=["disabled", "local", "ssh"],
        default="disabled",
        help="disabled is the safe default; local CPU or ssh must be selected explicitly",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek + Tavily research agent runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the full workflow with deterministic mock agents")
    demo.add_argument("--topic", default="可复现科研多智能体工作流")
    demo.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    demo.add_argument(
        "--project",
        help="3-64 character project slug; omitted values receive a unique research-<id> namespace",
    )
    demo.add_argument(
        "--auto-approve",
        action="store_true",
        help="explicit CI/test mode: approve every review checkpoint automatically",
    )
    demo.set_defaults(func=command_demo)

    smoke = subparsers.add_parser("smoke", help="make one real Tavily search and one DeepSeek call")
    smoke.add_argument("--query", default="multi-agent scientific research workflow reproducibility")
    smoke.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    smoke.add_argument("--prompt-keys", action="store_true")
    smoke.set_defaults(func=command_smoke)

    live = subparsers.add_parser("run-live", help="experimental full workflow using live providers")
    live.add_argument("--topic", required=True)
    live.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    live.add_argument("--project")
    live.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(live)
    live.add_argument(
        "--feedback-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    live.set_defaults(func=command_run_live)

    continue_live = subparsers.add_parser(
        "continue-live",
        help="resume one live Agent step after an interrupted CLI process",
    )
    continue_live.add_argument("--run-id", required=True)
    continue_live.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    continue_live.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(continue_live)
    continue_live.set_defaults(func=command_continue_live)

    recover_feedback = subparsers.add_parser(
        "recover-feedback",
        help="resume a persisted Feedback result after a pre-commit Lead failure",
    )
    recover_feedback.add_argument("--run-id", required=True)
    recover_feedback.add_argument("--source-task-id", required=True)
    recover_feedback.add_argument("--source-state", required=True)
    recover_feedback.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    recover_feedback.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(recover_feedback)
    recover_feedback.set_defaults(func=command_recover_feedback)

    resolve_human = subparsers.add_parser(
        "resolve-human",
        help="retry or resume one durable NEEDS_HUMAN checkpoint",
    )
    resolve_human.add_argument("--run-id", required=True)
    resolve_human.add_argument("--action", choices=["retry", "resume"], required=True)
    resolve_human.add_argument("--reason", required=True)
    resolve_human.add_argument("--apply-only", action="store_true")
    resolve_human.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    resolve_human.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(resolve_human)
    resolve_human.set_defaults(func=command_resolve_human)

    first_four = subparsers.add_parser(
        "run-first-four",
        help="start the first-four chain and pause after its first user-review checkpoint",
    )
    first_four.add_argument("--topic", required=True)
    first_four.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    first_four.add_argument("--project")
    first_four.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(first_four)
    first_four.set_defaults(func=command_run_first_four)

    status = subparsers.add_parser(
        "status", help="show a persisted run and its pending user-review checkpoint"
    )
    status.add_argument("--run-id", required=True)
    status.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    status.set_defaults(func=command_status)

    review = subparsers.add_parser(
        "review", help="submit one user verdict and execute at most one next/retry agent"
    )
    review.add_argument("--run-id", required=True)
    review.add_argument(
        "--verdict",
        choices=["satisfied", "needs-revision"],
        required=True,
    )
    review.add_argument("--rating", type=int, choices=range(1, 6))
    review.add_argument("--comment", default="")
    review.add_argument(
        "--target-agent",
        choices=[
            agent.value
            for agent in AgentName
            if agent not in {AgentName.LEAD, AgentName.FEEDBACK}
        ],
    )
    review.add_argument(
        "--apply-only",
        action="store_true",
        help="apply the verdict without executing the next/retry agent",
    )
    review.add_argument("--prompt-keys", action="store_true")
    _add_live_safety_options(review)
    review.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    review.set_defaults(func=command_review)

    skills = subparsers.add_parser("skills", help="validate reserved skills for all 14 agents")
    skills.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    skills.set_defaults(func=command_skills)

    graph = subparsers.add_parser("graph", help="show the compiled LangGraph topology")
    graph.set_defaults(func=command_graph)

    monitor = subparsers.add_parser(
        "monitor-daemon",
        help="fire due 12-hour Check Result wake-ups from the durable queue",
    )
    monitor.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    monitor.add_argument("--forever", action="store_true")
    monitor.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=30)
    monitor.set_defaults(func=command_monitor_daemon)

    monitor_status = subparsers.add_parser(
        "monitor-status", help="show durable monitoring wake-up history"
    )
    monitor_status.add_argument("--root", default=DEFAULT_WORKSPACE_ROOT)
    monitor_status.set_defaults(func=command_monitor_status)
    return parser


def main() -> int:
    load_workspace_env()
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        # Provider libraries already redact auth headers. Never print key material here.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
