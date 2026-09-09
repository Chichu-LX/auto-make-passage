#!/usr/bin/env python3
"""Recompute measured AssayCal-DTA profiles without redownloading source bytes."""

from __future__ import annotations

import json
from pathlib import Path

from agents.download_data_agent.agent import DownloadDataAgent
from research_agents.data_profiling import profile_dataset_payload
from research_agents.schemas import AgentName, AgentResult, ArtifactRef, DatasetManifest
from research_agents.storage import ArtifactStore, EventStore


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "run_dta_100_idea_design_v3"
MANIFEST_PATH = "data/assaycal-dta/manifests/1f5d0510be23.json"
REPORT_PATH = "data/assaycal-dta/reports/1f5d0510be23.md"


def main() -> int:
    store = ArtifactStore(WORKSPACE_ROOT)
    raw = json.loads((WORKSPACE_ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    for record in raw["records"]:
        artifact = record.get("local_artifact")
        dataset_format = record.get("format")
        if record.get("status") != "downloaded" or not artifact or not dataset_format:
            continue
        payload = store.read_artifact_bytes(
            ArtifactRef.model_validate(artifact),
            max_bytes=int(raw["policy"]["max_bytes"]),
        )
        record["profile"] = profile_dataset_payload(payload, dataset_format).model_dump(
            mode="json"
        )
    downloaded = [record for record in raw["records"] if record["status"] == "downloaded"]
    raw["total_downloaded_bytes"] = sum(int(record["size_bytes"]) for record in downloaded)
    raw["total_uncompressed_bytes"] = sum(
        int(record["profile"]["uncompressed_size_bytes"]) for record in downloaded
    )
    raw["total_profiled_rows"] = sum(
        int(record["profile"]["row_count"]) for record in downloaded
    )
    manifest = DatasetManifest.model_validate(raw)
    manifest_ref = store.write_agent_json(
        AgentName.DOWNLOAD_DATA,
        MANIFEST_PATH,
        manifest.model_dump(mode="json"),
    ).model_copy(update={"kind": "data_manifest"})
    report_ref = store.write_agent_bytes(
        AgentName.DOWNLOAD_DATA,
        REPORT_PATH,
        DownloadDataAgent._markdown(manifest).encode("utf-8"),
        max_bytes=512 * 1024,
        kind="data_profile_report",
    )

    events = EventStore(WORKSPACE_ROOT, store)
    snapshot = events.load_snapshot(RUN_ID)
    replacements = {
        "data_manifest": manifest_ref,
        "data_profile_report": report_ref,
    }
    snapshot = snapshot.model_copy(
        update={
            "artifacts": [
                replacements.get(artifact.kind, artifact)
                for artifact in snapshot.artifacts
            ]
        }
    )
    events.save_snapshot(snapshot)
    review = snapshot.pending_user_review
    if review is not None:
        result_path = (
            f"project/var/runs/{RUN_ID}/agent_results/{review.source_task_id}.json"
        )
        result_payload = json.loads((WORKSPACE_ROOT / result_path).read_text(encoding="utf-8"))
        result = AgentResult.model_validate(result_payload).model_copy(
            update={"artifacts": [manifest_ref, report_ref]}
        )
        store.write_runtime_json(result_path, result.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "manifest": manifest_ref.model_dump(mode="json"),
                "report": report_ref.model_dump(mode="json"),
                "rows": manifest.total_profiled_rows,
                "downloaded_bytes": manifest.total_downloaded_bytes,
                "uncompressed_bytes": manifest.total_uncompressed_bytes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
