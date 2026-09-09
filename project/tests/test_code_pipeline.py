from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agents.code_make_agent import CodeMakeAgent
from agents.code_make_agent.assaycal_scaffold import build_assaycal_scaffold
from agents.code_make_agent.local_validation import (
    LocalValidationError,
    validate_generated_python_for_execution,
)
from agents.code_review_agent import CodeReviewAgent
from agents.code_review_agent.review_engine import _unsafe_cross_dataset_aggregation
from research_agents.schemas import (
    AgentName,
    CodeImplementationManifest,
    DatasetManifest,
    DatasetProfile,
    DatasetRecord,
    DownloadPolicyRecord,
    ResearchDesign,
    TaskEnvelope,
    VendoredSourceRecord,
)
from research_agents.storage import ArtifactStore
from research_fixtures import advisor_response, valid_design_response


class StaticModel:
    def __init__(self, response):
        self.response = response

    def complete_json(self, system_prompt, user_prompt):
        return dict(self.response)


class SequenceModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete_json(self, system_prompt, user_prompt):
        self.prompts.append(json.loads(user_prompt))
        return dict(self.responses.pop(0))


class FakeSourceVendor:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def vendor(self, module, *, project_slug: str) -> VendoredSourceRecord:
        commit = "1" * 40
        root = f"code/{project_slug}/third_party/{module.module_id}/{commit}"
        payload = b"local source snapshot\n"
        source = self.artifacts.write_agent_bytes(
            AgentName.CODE_MAKE,
            f"{root}/source.txt",
            payload,
            max_bytes=1024,
            kind="vendored_source_file",
        )
        relative = "source.txt"
        tree = hashlib.sha256()
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(len(payload)).encode("ascii"))
        tree.update(b"\0")
        tree.update(source.sha256.encode("ascii"))
        source_manifest = self.artifacts.write_agent_json(
            AgentName.CODE_MAKE,
            f"{root}/SOURCE_MANIFEST.json",
            {
                "schema_version": "1.0",
                "module_id": module.module_id,
                "repository_url": module.repository_url,
                "commit_sha": commit,
                "license_id": module.license_id,
                "reuse_mode": "integrate",
                "execution_performed": False,
                "file_count": 1,
                "total_bytes": len(payload),
                "tree_sha256": tree.hexdigest(),
                "files": [
                    {
                        "path": relative,
                        "size_bytes": len(payload),
                        "sha256": source.sha256,
                    }
                ],
            },
        ).model_copy(update={"kind": "vendored_source_manifest"})
        return VendoredSourceRecord(
            module_id=module.module_id,
            repository_url=module.repository_url,
            commit_sha=commit,
            license_id=module.license_id,
            local_root=root,
            file_count=1,
            total_bytes=len(payload),
            tree_sha256=tree.hexdigest(),
            source_manifest=source_manifest,
        )


def _design(run_id: str) -> ResearchDesign:
    payload = valid_design_response()
    payload["advisor_consensus"] = [
        advisor_response(role)
        for role in (
            "method-architecture-advisor",
            "code-reuse-architecture-advisor",
            "data-decision-advisor",
            "experiment-decision-advisor",
            "critical-risk-advisor",
        )
    ]
    return ResearchDesign(
        **payload,
        run_id=run_id,
        task_id="task_design",
        objective="test affinity implementation",
        idea_proposal_uri="artifact://design/assaycal-test/ideas/idea.json",
        paper_evidence_uris=["artifact://reference/assaycal-test/papers/paper.pdf"],
        evidence_quality="downloaded",
    )


def _write_inputs(artifacts: ArtifactStore, run_id: str):
    slug = "assaycal-test"
    design = _design(run_id)
    design_ref = artifacts.write_agent_json(
        AgentName.IDEA_DESIGN,
        f"design/{slug}/research-designs/design.json",
        design.model_dump(mode="json"),
    ).model_copy(update={"kind": "research_design"})
    raw_payload = b"compound_id,target_id,affinity,assay_type\nc1,t1,7.2,Kd\n"
    raw = artifacts.write_agent_bytes(
        AgentName.DOWNLOAD_DATA,
        f"data/{slug}/raw/dataset-local/payload.csv",
        raw_payload,
        max_bytes=1024,
        kind="dataset_raw_file",
    )
    materialized = artifacts.write_agent_bytes(
        AgentName.DOWNLOAD_DATA,
        f"data/{slug}/datasets/dataset-local/v1/data.csv",
        raw_payload,
        max_bytes=1024,
        kind="dataset_materialized_file",
    )
    record = DatasetRecord(
        dataset_id="dataset-local",
        requirement_key="dataset-davis",
        name="Local Davis fixture",
        description="One-row local contract fixture.",
        source="unit-test",
        landing_page_url="https://example.com/davis",
        direct_download_url="https://example.com/davis.csv",
        license_id="cc-by-4.0",
        license_url="https://example.com/license",
        license_evidence="Fixture license evidence.",
        format="csv",
        version="v1",
        relevance_score=1.0,
        status="downloaded",
        final_url="https://example.com/davis.csv",
        content_type="text/csv",
        size_bytes=len(raw_payload),
        sha256=raw.sha256,
        local_artifact=raw,
        profile=DatasetProfile(
            profile_status="complete",
            source_table="data.csv",
            compressed_size_bytes=len(raw_payload),
            uncompressed_size_bytes=len(raw_payload),
            row_count=1,
            column_count=4,
            columns=["compound_id", "target_id", "affinity", "assay_type"],
            unique_compounds=1,
            unique_targets=1,
            unique_pairs=1,
            key_field_coverage={"affinity": 1.0},
        ),
        materialized_root=f"data/{slug}/datasets/dataset-local/v1",
        materialized_files=[
            {
                "relative_path": "data.csv",
                "size_bytes": len(raw_payload),
                "sha256": materialized.sha256,
                "artifact": materialized,
            }
        ],
        materialized_size_bytes=len(raw_payload),
    )
    manifest = DatasetManifest(
        run_id=run_id,
        project_slug=slug,
        task_id="task_data",
        objective="test affinity implementation",
        query="local fixture",
        search_result_count=1,
        records=[record],
        downloaded_count=1,
        requires_review_count=0,
        materialized_count=1,
        server_script_count=0,
        total_downloaded_bytes=len(raw_payload),
        total_uncompressed_bytes=len(raw_payload),
        total_materialized_bytes=len(raw_payload),
        total_profiled_rows=1,
        policy=DownloadPolicyRecord(
            max_bytes=10 * 1024 * 1024 * 1024,
            max_redirects=3,
            allowed_content_types=["text/csv"],
        ),
        allowed_license_ids=["cc-by-4.0"],
    )
    data_ref = artifacts.write_agent_json(
        AgentName.DOWNLOAD_DATA,
        f"data/{slug}/manifests/data.json",
        manifest.model_dump(mode="json"),
    ).model_copy(update={"kind": "data_manifest"})
    return design, design_ref, data_ref


class CodePipelineTests(unittest.TestCase):
    def test_static_review_rejects_direct_cross_dataset_label_pooling(self) -> None:
        source = '''
POOL_MEASUREMENT_TYPES = False
dataset_keys = ["dataset-davis", "dataset-kiba"]
pairs = []
for dataset_key in dataset_keys:
    pairs.extend(_load_dataset_pairs(dataset_key))
'''
        self.assertTrue(_unsafe_cross_dataset_aggregation(source))

    def test_code_make_uses_bounded_patch_for_reviewed_failed_candidate(self) -> None:
        run_id = "run_code_patch_revision"
        source_task_id = "task_failed_candidate"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design, design_ref, data_ref = _write_inputs(artifacts, run_id)
            baseline = build_assaycal_scaffold(design)
            token = hashlib.sha256(source_task_id.encode("utf-8")).hexdigest()[:12]
            for item in baseline.files:
                artifacts.write_agent_bytes(
                    AgentName.CODE_MAKE,
                    f"code/assaycal-test/implementations/{token}/{item.path}",
                    item.content.encode("utf-8"),
                    max_bytes=120 * 1024,
                    kind="generated_code_file",
                )
            gap = artifacts.write_agent_json(
                AgentName.CODE_MAKE,
                f"code/assaycal-test/manifests/{token}.gap.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "task_id": source_task_id,
                    "project_slug": "assaycal-test",
                    "status": "blocked",
                    "reason": "Generated code failed local execution preflight: LocalValidationError.",
                    "details": {"preflight_error": "bounded fixture failure"},
                },
            ).model_copy(update={"kind": "code_generation_gap"})
            interrupted_task_id = "task_patch_interrupted"
            interrupted_token = hashlib.sha256(
                interrupted_task_id.encode("utf-8")
            ).hexdigest()[:12]
            interrupted = artifacts.write_agent_json(
                AgentName.CODE_MAKE,
                f"code/assaycal-test/manifests/{interrupted_token}.gap.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "task_id": interrupted_task_id,
                    "project_slug": "assaycal-test",
                    "status": "blocked",
                    "reason": "Code generation failed its bounded contract: RuntimeError.",
                    "details": {
                        "baseline_task_id": source_task_id,
                        "contract_errors": [
                            {
                                "attempt": 1,
                                "errors": [
                                    {
                                        "location": [],
                                        "type": "RuntimeError",
                                        "message": "Didn't receive a response.completed event.",
                                    }
                                ],
                            }
                        ]
                    },
                },
            ).model_copy(update={"kind": "code_generation_gap"})
            model = SequenceModel(
                [
                    {
                        "baseline_task_id": source_task_id,
                        "architecture_summary": (
                            "The verified candidate receives one README-only bounded revision "
                            "before the complete immutable tree is validated again."
                        ),
                        "files": [
                            {
                                "path": "README.md",
                                "purpose": "Document the bounded candidate revision and validation gate.",
                                "language": "markdown",
                                "content": "# Revised AssayCal candidate\n\nNo measured-result claim is made.\n",
                            }
                        ],
                        "known_gaps": ["README-only unit-test revision fixture."],
                    }
                ]
            )
            result = CodeMakeAgent(
                model,
                artifacts,
                source_vendor=FakeSourceVendor(artifacts),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_123_abcd1234",
                    agent=AgentName.CODE_MAKE,
                    objective="test affinity implementation",
                    inputs=[design_ref, data_ref, gap, interrupted],
                    context={
                        "project_slug": "assaycal-test",
                        "user_revision": {
                            "review_id": f"review_{interrupted_task_id}",
                            "comments": "repair only README.md in the failed candidate",
                        },
                    },
                )
            )

            self.assertEqual(result.status, "succeeded", result.summary)
            self.assertEqual(len(model.prompts), 1)
            self.assertEqual(
                model.prompts[0]["revision_mode"],
                "replace_existing_files_only",
            )
            self.assertEqual(model.prompts[0]["baseline_task_id"], source_task_id)
            self.assertEqual(model.prompts[0]["revision_context_file_count"], 1)
            self.assertIn("README.md", model.prompts[0]["baseline_file_inventory"])
            self.assertIn(
                "README.md",
                {item["path"] for item in model.prompts[0]["baseline_files"]},
            )
            manifest = CodeImplementationManifest.model_validate(
                artifacts.read_artifact_json(result.artifacts[0])
            )
            self.assertNotEqual(
                manifest.implementation_root,
                f"code/assaycal-test/implementations/{token}",
            )
            reviewed_model = SequenceModel(
                [
                    {
                        "baseline_task_id": "task_123_abcd1234",
                        "architecture_summary": (
                            "A manifest-verified implementation receives a second bounded "
                            "README revision before all local gates are executed again."
                        ),
                        "files": [
                            {
                                "path": "README.md",
                                "purpose": "Record the independently reviewed bounded revision.",
                                "language": "markdown",
                                "content": "# Independently reviewed revision\n\nNo result claim.\n",
                            }
                        ],
                        "known_gaps": ["Second bounded revision fixture."],
                    }
                ]
            )
            reviewed_result = CodeMakeAgent(
                reviewed_model,
                artifacts,
                source_vendor=FakeSourceVendor(artifacts),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_124_bcde2345",
                    agent=AgentName.CODE_MAKE,
                    objective="test affinity implementation",
                    inputs=[design_ref, data_ref, result.artifacts[0]],
                    context={
                        "project_slug": "assaycal-test",
                        "user_revision": {
                            "review_id": "review_task_999_deadbeef",
                            "comments": (
                                "Revise reviewed implementation task_123_abcd1234 only."
                            ),
                        },
                    },
                )
            )
            self.assertEqual(
                reviewed_result.status,
                "succeeded",
                reviewed_result.summary,
            )
            self.assertEqual(
                reviewed_model.prompts[0]["baseline_task_id"],
                "task_123_abcd1234",
            )

    def test_code_make_repairs_one_schema_invalid_provider_response(self) -> None:
        run_id = "run_code_contract_repair"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design, design_ref, data_ref = _write_inputs(artifacts, run_id)
            draft = build_assaycal_scaffold(design)
            model = SequenceModel([{}, draft.model_dump(mode="json")])
            make = CodeMakeAgent(
                model,
                artifacts,
                source_vendor=FakeSourceVendor(artifacts),
            )
            result = make.run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_code_contract_repair",
                    agent=AgentName.CODE_MAKE,
                    objective="test affinity implementation",
                    inputs=[design_ref, data_ref],
                    context={
                        "project_slug": "assaycal-test",
                        "user_revision": {"comments": "repair the generated contract"},
                    },
                )
            )

            self.assertEqual(result.status, "succeeded", result.summary)
            self.assertEqual(len(model.prompts), 2)
            self.assertNotIn("contract_repair", model.prompts[0])
            repair = model.prompts[1]["contract_repair"]
            self.assertEqual(repair["attempt"], 2)
            self.assertTrue(repair["validation_errors"])
            self.assertTrue(
                any(
                    item["location"] == ["package_name"]
                    for item in repair["validation_errors"]
                )
            )

    def test_code_make_persists_bounded_contract_diagnostics(self) -> None:
        run_id = "run_code_contract_gap"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            _, design_ref, data_ref = _write_inputs(artifacts, run_id)
            model = SequenceModel([{}, {}])
            result = CodeMakeAgent(
                model,
                artifacts,
                source_vendor=FakeSourceVendor(artifacts),
            ).run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_code_contract_gap",
                    agent=AgentName.CODE_MAKE,
                    objective="test affinity implementation",
                    inputs=[design_ref, data_ref],
                    context={
                        "project_slug": "assaycal-test",
                        "user_revision": {"comments": "repair the generated contract"},
                    },
                )
            )

            self.assertEqual(result.status, "failed")
            self.assertIn("ValidationError", result.summary)
            gap = artifacts.read_artifact_json(result.artifacts[0])
            self.assertEqual(gap["details"]["generation_attempt_count"], 2)
            self.assertEqual(
                [item["attempt"] for item in gap["details"]["contract_errors"]],
                [1, 2],
            )

    def test_code_make_and_review_minimum_pipeline(self) -> None:
        run_id = "run_code_pipeline"
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            design, design_ref, data_ref = _write_inputs(artifacts, run_id)
            draft = build_assaycal_scaffold(design)
            make = CodeMakeAgent(
                StaticModel(draft.model_dump(mode="json")),
                artifacts,
                source_vendor=FakeSourceVendor(artifacts),
            )
            make_task = TaskEnvelope(
                run_id=run_id,
                task_id="task_code_make",
                agent=AgentName.CODE_MAKE,
                objective="test affinity implementation",
                inputs=[design_ref, data_ref],
                context={"project_slug": "assaycal-test"},
            )
            make_result = make.run(make_task)
            self.assertEqual(make_result.status, "succeeded", make_result.summary)
            self.assertEqual(make_result.artifacts[0].kind, "code_implementation_manifest")
            manifest = CodeImplementationManifest.model_validate(
                artifacts.read_artifact_json(make_result.artifacts[0])
            )
            self.assertEqual(
                {item.validation_id: item.status for item in manifest.local_validations},
                {"minimal_demo": "passed", "unit_tests": "passed"},
            )
            self.assertTrue(
                any(item.path.startswith(f"src/{manifest.package_name}/models/assaycal/") for item in manifest.files)
            )
            self.assertTrue(
                any(item.path.startswith(f"src/{manifest.package_name}/experiments/smoke/") for item in manifest.files)
            )

            review = CodeReviewAgent(
                StaticModel({"summary": "Deterministic checks confirm the minimum revision.", "recommendations": []}),
                artifacts,
            )
            review_result = review.run(
                TaskEnvelope(
                    run_id=run_id,
                    task_id="task_code_review",
                    agent=AgentName.CODE_REVIEW,
                    objective="test affinity implementation",
                    inputs=[design_ref, make_result.artifacts[0]],
                    context={"project_slug": "assaycal-test"},
                    attempt=1,
                )
            )
            self.assertEqual(review_result.status, "failed", review_result.summary)
            report = artifacts.read_artifact_json(review_result.artifacts[0])
            self.assertEqual(report["verdict"], "needs_revision")
            protocol = next(
                item
                for item in report["checks"]
                if item["check_id"] == "review-scientific-protocol"
            )
            self.assertEqual(protocol["status"], "failed")
            self.assertIn("hashed identity features", protocol["summary"])
            self.assertEqual(len(report["executions"]), 3)

    def test_static_review_rejects_unbounded_loop_without_execution(self) -> None:
        design = _design("run_static")
        draft = build_assaycal_scaffold(design)
        demo_path = f"src/{draft.package_name}/demo.py"
        modified = []
        for item in draft.files:
            if item.path == demo_path:
                modified.append(item.model_copy(update={"content": "while True:\n    pass\n"}))
            else:
                modified.append(item)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for item in modified:
                target = root / item.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(item.content, encoding="utf-8")
            with self.assertRaises(LocalValidationError):
                validate_generated_python_for_execution(root, draft.package_name)


if __name__ == "__main__":
    unittest.main()
