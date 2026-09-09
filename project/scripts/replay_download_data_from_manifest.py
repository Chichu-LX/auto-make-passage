#!/usr/bin/env python3
"""Rerun Download Data from already verified local raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agents.download_data_agent import DownloadDataAgent
from agents.lead_agent import RuleBasedLeadAgent
from research_agents.downloads import DownloadPayload
from research_agents.registry import build_mock_registry
from research_agents.runtime import ResearchRuntime
from research_agents.schemas import (
    AgentName,
    DatasetCandidate,
    WorkflowState,
)
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore, EventStore, StoragePolicyError


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class ReplaySearch:
    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return []


class ReplayModel:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {"candidates": self.candidates}


class ReplayDownloader:
    def __init__(self, payloads: dict[str, DownloadPayload]) -> None:
        self.payloads = payloads

    def fetch(self, url: str, policy: Any) -> DownloadPayload:
        try:
            payload = self.payloads[url]
        except KeyError as exc:
            raise StoragePolicyError("source manifest does not contain the requested raw file") from exc
        if payload.size_bytes > policy.max_bytes:
            raise StoragePolicyError("replayed raw file exceeds the current download policy")
        return payload


def _load_replay_source(
    root: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, DownloadPayload]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_fields = set(DatasetCandidate.model_fields)
    candidates: list[dict[str, Any]] = []
    payloads: dict[str, DownloadPayload] = {}
    for record in manifest.get("records", []):
        if record.get("status") != "downloaded":
            continue
        candidate = DatasetCandidate.model_validate(
            {key: record.get(key) for key in candidate_fields}
        ).model_dump(mode="json")
        raw = record.get("local_artifact") or {}
        uri = str(raw.get("uri") or "")
        if not uri.startswith("artifact://data/"):
            raise StoragePolicyError("source raw artifact is outside data/")
        path = root / uri.removeprefix("artifact://")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record.get("sha256") or digest != raw.get("sha256"):
            raise StoragePolicyError("source raw artifact checksum mismatch")
        url = str(record.get("direct_download_url") or "")
        payloads[url] = DownloadPayload(
            content=content,
            final_url=str(record.get("final_url") or url),
            content_type=str(record.get("content_type") or "application/octet-stream"),
            sha256=digest,
            redirect_count=0,
        )
        candidates.append(candidate)
    if not candidates:
        raise StoragePolicyError("source manifest contains no downloaded datasets")
    return candidates, payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=WORKSPACE_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    source_manifest = args.source_manifest.resolve()
    try:
        source_manifest.relative_to(root / "data")
    except ValueError as exc:
        raise StoragePolicyError("source manifest must be below the workspace data directory") from exc

    store = ArtifactStore(root)
    events = EventStore(root, store)
    snapshot = events.load_snapshot(args.run_id)
    if snapshot.state != WorkflowState.ACQUIRING_DATA or snapshot.awaiting_user_review:
        raise StoragePolicyError(
            "run must be rolled back to ACQUIRING_DATA before offline replay"
        )
    candidates, payloads = _load_replay_source(root, source_manifest)
    registry = build_mock_registry()
    registry[AgentName.DOWNLOAD_DATA] = DownloadDataAgent(
        ReplaySearch(), ReplayModel(candidates), store, ReplayDownloader(payloads)
    )
    machine = ResearchStateMachine()
    runtime = ResearchRuntime(
        root,
        RuleBasedLeadAgent(machine),
        registry,
        feedback_after_each_step=True,
        execution_mode=snapshot.execution_mode,
    )
    runtime.events.load_snapshot(snapshot.run_id)
    updated = runtime.run_step(snapshot)
    print(
        json.dumps(
            {
                "run_id": updated.run_id,
                "project_slug": updated.project_slug,
                "state": updated.state.value,
                "version": updated.version,
                "review_id": (
                    updated.pending_user_review.review_id
                    if updated.pending_user_review is not None
                    else None
                ),
                "completed_agents": [item.value for item in updated.completed_agents],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
