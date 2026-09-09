#!/usr/bin/env python3
"""Approve the current Idea Design and run scoped AssayCal-DTA data acquisition once."""

from __future__ import annotations

import json
from pathlib import Path

from agents.download_data_agent import DownloadDataAgent
from agents.lead_agent import RuleBasedLeadAgent
from research_agents.registry import build_mock_registry
from research_agents.runtime import ResearchRuntime
from research_agents.schemas import AgentName, UserEvaluation, WorkflowState
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore, EventStore


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "run_dta_100_idea_design_v3"
PROJECT_SLUG = "assaycal-dta"
NHGNN_COMMIT = "57a5fd2e4920f28e5d4cfe854b7a4b81715a47c1"


class OfflineDatasetSearch:
    """Pinned source evidence used when Tavily credentials are intentionally unavailable."""

    def search(self, query: str, max_results: int = 5):
        del query
        return [
            {
                "title": "NHGNN-DTA Apache-2.0 repository data archives",
                "url": "https://github.com/hehh77/NHGNN-DTA/tree/main/Data",
                "content": (
                    "GitHub identifies the repository license as Apache-2.0. The pinned Data/ "
                    "directory contains davis.7z and kiba.7z."
                ),
                "score": 1.0,
            },
            {
                "title": "BindingDB downloads and license statement",
                "url": "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp",
                "content": (
                    "BindingDB reports the 202608 BindingDB-curated articles TSV archive as "
                    "17.43 MB. BindingDB staff-curated data are provided under Creative Commons "
                    "Attribution 3.0."
                ),
                "score": 1.0,
            },
        ][:max_results]


class OfflineDatasetSelectionModel:
    def complete_json(self, system_prompt: str, user_prompt: str):
        if not system_prompt.startswith("You are download-data-agent"):
            raise RuntimeError("offline data selector received an unsupported Agent prompt")
        request = json.loads(user_prompt)
        requirement_keys = {
            item.get("dataset_key")
            for item in request.get("research_design_dataset_requirements", [])
        }
        required = {
            "dataset-davis-corrected",
            "dataset-kiba",
            "dataset-bindingdb-assay",
        }
        if not required.issubset(requirement_keys):
            raise RuntimeError("research design does not contain the expected AssayCal datasets")
        apache_license = "https://www.apache.org/licenses/license-2.0"
        repository = "https://github.com/hehh77/NHGNN-DTA"
        raw_root = (
            "https://raw.githubusercontent.com/hehh77/NHGNN-DTA/"
            f"{NHGNN_COMMIT}/Data"
        )
        return {
            "candidates": [
                {
                    "dataset_id": "dataset-davis-nhgnn",
                    "requirement_key": "dataset-davis-corrected",
                    "name": "Davis affinity CSV snapshot from NHGNN-DTA",
                    "description": (
                        "Baseline-compatible Davis table. It does not by itself prove the "
                        "DTA-093 correction provenance, so that remains an explicit audit gate."
                    ),
                    "source": "hehh77/NHGNN-DTA pinned repository",
                    "landing_page_url": f"{repository}/tree/{NHGNN_COMMIT}/Data",
                    "direct_download_url": f"{raw_root}/davis.7z",
                    "license_id": "apache-2.0",
                    "license_url": apache_license,
                    "license_evidence": "Repository license: Apache License 2.0.",
                    "format": "7z",
                    "estimated_size_bytes": 465512,
                    "expected_record_count": 30056,
                    "expected_entity_scale": "68 compounds, 442 targets, dense 30,056 pairs expected; verify locally.",
                    "version": NHGNN_COMMIT,
                    "relevance_score": 0.96,
                },
                {
                    "dataset_id": "dataset-kiba-nhgnn",
                    "requirement_key": "dataset-kiba",
                    "name": "KIBA affinity CSV snapshot from NHGNN-DTA",
                    "description": "Paper-repository KIBA table for reproducible DTA baselines.",
                    "source": "hehh77/NHGNN-DTA pinned repository",
                    "landing_page_url": f"{repository}/tree/{NHGNN_COMMIT}/Data",
                    "direct_download_url": f"{raw_root}/kiba.7z",
                    "license_id": "apache-2.0",
                    "license_url": apache_license,
                    "license_evidence": "Repository license: Apache License 2.0.",
                    "format": "7z",
                    "estimated_size_bytes": 1377887,
                    "expected_record_count": 118254,
                    "expected_entity_scale": "About 2,111 drugs, 229 targets, and 118,254 interactions reported historically; verify locally.",
                    "version": NHGNN_COMMIT,
                    "relevance_score": 0.97,
                },
                {
                    "dataset_id": "dataset-bindingdb-articles-202608",
                    "requirement_key": "dataset-bindingdb-assay",
                    "name": "BindingDB staff-curated article affinity data",
                    "description": (
                        "The 202608 staff-curated article subset retains assay-oriented TSV fields "
                        "without automatically downloading the 565.52 MB mixed full dump."
                    ),
                    "source": "BindingDB official download service",
                    "landing_page_url": "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp",
                    "direct_download_url": "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202608_tsv.zip",
                    "license_id": "cc-by-3.0",
                    "license_url": "https://creativecommons.org/licenses/by/3.0/us/",
                    "license_evidence": "All data curated by BindingDB staff are provided under the Creative Commons Attribution 3.0 License.",
                    "format": "zip",
                    "estimated_size_bytes": 18271972,
                    "expected_record_count": None,
                    "expected_entity_scale": "Official page reports 17.43 MB compressed; row, compound, target, pair, and missingness counts require local profiling.",
                    "version": "202608",
                    "relevance_score": 0.99,
                },
            ]
        }


def main() -> int:
    store = ArtifactStore(WORKSPACE_ROOT)
    events = EventStore(WORKSPACE_ROOT, store)
    snapshot = events.load_snapshot(RUN_ID)
    if snapshot.project_slug != PROJECT_SLUG:
        raise RuntimeError("run is not assigned to the expected project namespace")

    machine = ResearchStateMachine()
    control_runtime = ResearchRuntime(
        WORKSPACE_ROOT,
        RuleBasedLeadAgent(machine),
        build_mock_registry(),
        feedback_after_each_step=False,
        execution_mode=snapshot.execution_mode,
    )
    control_runtime.events.load_snapshot(snapshot.run_id)
    if snapshot.state == WorkflowState.AWAITING_USER_REVIEW:
        review = snapshot.pending_user_review
        if review is None or review.source_agent != AgentName.IDEA_DESIGN:
            raise RuntimeError("pending review is not the expected Idea Design checkpoint")
        snapshot = control_runtime.submit_user_evaluation(
            snapshot,
            UserEvaluation(
                run_id=snapshot.run_id,
                review_id=review.review_id,
                observed_state_version=snapshot.version,
                verdict="satisfied",
                rating=5,
                comments="User approved Idea Design and explicitly requested Download Data.",
            ),
            submission_source="user_explicit_next_step",
        )
    if snapshot.state != WorkflowState.ACQUIRING_DATA:
        raise RuntimeError("run is not ready for Download Data")

    registry = build_mock_registry()
    registry[AgentName.DOWNLOAD_DATA] = DownloadDataAgent(
        OfflineDatasetSearch(),
        OfflineDatasetSelectionModel(),
        store,
    )
    runtime = ResearchRuntime(
        WORKSPACE_ROOT,
        RuleBasedLeadAgent(machine),
        registry,
        feedback_after_each_step=False,
        execution_mode=snapshot.execution_mode,
    )
    runtime.events.load_snapshot(snapshot.run_id)
    paused = runtime.run_step(snapshot)
    review = paused.pending_user_review
    if paused.state != WorkflowState.AWAITING_USER_REVIEW or review is None:
        raise RuntimeError("Download Data did not stop at the required user-review hook")
    result_path = (
        WORKSPACE_ROOT
        / "project"
        / "var"
        / "runs"
        / paused.run_id
        / "agent_results"
        / f"{review.source_task_id}.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "run_id": paused.run_id,
                "project_slug": paused.project_slug,
                "state": paused.state.value,
                "review_id": review.review_id,
                "proposed_state": review.proposed_state.value,
                "allowed_revision_targets": [
                    item.value for item in review.allowed_revision_targets
                ],
                "download_data": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
