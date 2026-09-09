import tempfile
import unittest
from pathlib import Path

from research_agents.schemas import AgentName, ArtifactRef
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore, EventStore, StoragePolicyError


class ArtifactStoreTests(unittest.TestCase):
    def test_agent_can_write_inside_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifact = store.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                "reference/manifests/test.json",
                {"ok": True},
            )
            self.assertTrue((Path(directory) / "reference/manifests/test.json").exists())
            self.assertTrue(artifact.sha256)

    def test_agent_cannot_write_outside_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaises(StoragePolicyError):
                store.write_agent_json(
                    AgentName.PAPER_REVIEW,
                    "code/unauthorized.py",
                    {"bad": True},
                )

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaises(StoragePolicyError):
                store.write_runtime_json("project/var/runs/../../escape.json", {"bad": True})

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "project").mkdir()
            (root / "project" / "var").mkdir()
            (root / "project" / "var" / "runs").symlink_to(Path(outside), target_is_directory=True)
            store = ArtifactStore(root)
            with self.assertRaises(StoragePolicyError):
                store.write_runtime_json("project/var/runs/escape.json", {"bad": True})

    def test_binary_write_obeys_agent_scope_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifact = store.write_agent_bytes(
                AgentName.DOWNLOAD_DATA,
                "data/raw/test.csv",
                b"a,b\n1,2\n",
                max_bytes=100,
                kind="dataset_raw",
            )
            self.assertEqual(artifact.kind, "dataset_raw")
            with self.assertRaises(StoragePolicyError):
                store.write_agent_bytes(
                    AgentName.DOWNLOAD_DATA,
                    "reference/test.csv",
                    b"bad",
                    max_bytes=100,
                    kind="dataset_raw",
                )

    def test_artifact_read_verifies_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifact = store.write_agent_json(
                AgentName.DOWNLOAD_PAPER,
                "reference/manifests/test.json",
                {"ok": True},
            )
            self.assertEqual(store.read_artifact_json(artifact), {"ok": True})
            tampered = ArtifactRef(
                uri=artifact.uri,
                sha256="0" * 64,
                kind=artifact.kind,
            )
            with self.assertRaisesRegex(StoragePolicyError, "SHA-256"):
                store.read_artifact_json(tampered)

    def test_large_artifact_can_be_stream_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifact = store.write_agent_bytes(
                AgentName.DOWNLOAD_DATA,
                "data/test-project/datasets/data.csv",
                b"a,b\n1,2\n",
                max_bytes=100,
                kind="dataset_materialized_file",
            )
            self.assertEqual(store.verify_artifact_file(artifact, max_bytes=100), 8)
            (Path(directory) / "data/test-project/datasets/data.csv").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(StoragePolicyError, "SHA-256"):
                store.verify_artifact_file(artifact, max_bytes=100)

    def test_event_store_loads_persisted_reviewable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            events = EventStore(directory, artifacts)
            snapshot = ResearchStateMachine().create_run("resume test")
            events.append(snapshot, "run_created")
            events.save_snapshot(snapshot)
            loaded = EventStore(directory, artifacts).load_snapshot(snapshot.run_id)
            self.assertEqual(loaded, snapshot)

    def test_event_store_rejects_unsafe_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(directory)
            with self.assertRaisesRegex(StoragePolicyError, "invalid run_id"):
                EventStore(directory, artifacts).load_snapshot("../../escape")


if __name__ == "__main__":
    unittest.main()
