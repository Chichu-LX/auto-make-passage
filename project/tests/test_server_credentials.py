from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research_agents.remote_execution import (
    RemoteExecutionError,
    SafeSSHBackend,
    load_server_execution_config,
)
from research_agents.schemas import ServerProfileConfig
from research_agents.server_credentials import (
    ServerCredentialError,
    ServerCredentialStore,
)


def _write_env(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def _profile() -> ServerProfileConfig:
    return ServerProfileConfig(
        server_id="gpu-test",
        credential_prefix="GPU_TEST",
        workspace_root="/srv/research-agent-tests",
        execution_enabled=True,
    )


class ServerCredentialTests(unittest.TestCase):
    def test_password_credentials_load_only_from_agent_env(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            _write_env(
                path,
                "\n".join(
                    [
                        "SERVER_GPU_TEST_HOST=gpu.example.org",
                        "SERVER_GPU_TEST_PORT=2222",
                        "SERVER_GPU_TEST_USER=researcher",
                        "SERVER_GPU_TEST_AUTH_MODE=password",
                        "SERVER_GPU_TEST_PASSWORD=correct-horse-battery-staple",
                    ]
                )
                + "\n",
            )
            credentials = ServerCredentialStore(path).resolve("GPU_TEST")
            self.assertEqual(credentials.host, "gpu.example.org")
            self.assertEqual(credentials.port, 2222)
            self.assertEqual(credentials.auth_mode, "password")
            self.assertNotIn("correct-horse", repr(credentials))

    def test_agent_env_rejects_unsafe_permissions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = root / "unsafe.env"
            _write_env(unsafe, "SERVER_GPU_TEST_HOST=gpu.example.org\n", mode=0o644)
            with self.assertRaises(ServerCredentialError):
                ServerCredentialStore(unsafe).resolve("GPU_TEST")

            target = root / "target.env"
            _write_env(target, "SERVER_GPU_TEST_HOST=gpu.example.org\n")
            link = root / "linked.env"
            link.symlink_to(target)
            with self.assertRaises(ServerCredentialError):
                ServerCredentialStore(link).resolve("GPU_TEST")

    def test_key_credentials_require_absolute_safe_regular_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "id_ed25519"
            key.write_text("test-private-key-material\n", encoding="utf-8")
            key.chmod(0o600)
            path = root / ".env"
            _write_env(
                path,
                "\n".join(
                    [
                        "SERVER_GPU_TEST_HOST=192.0.2.10",
                        "SERVER_GPU_TEST_USER=researcher",
                        "SERVER_GPU_TEST_AUTH_MODE=key",
                        f"SERVER_GPU_TEST_PRIVATE_KEY_PATH={key}",
                        "SERVER_GPU_TEST_PRIVATE_KEY_PASSPHRASE=encrypted-key-secret",
                    ]
                )
                + "\n",
            )
            credentials = ServerCredentialStore(path).resolve("GPU_TEST")
            self.assertEqual(credentials.private_key_path, key.resolve())
            key.chmod(0o644)
            with self.assertRaises(ServerCredentialError):
                ServerCredentialStore(path).resolve("GPU_TEST")

    def test_password_is_only_in_child_environment_not_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            secret = "never-place-this-secret-in-argv"
            _write_env(
                path,
                "\n".join(
                    [
                        "SERVER_GPU_TEST_HOST=gpu.example.org",
                        "SERVER_GPU_TEST_USER=researcher",
                        "SERVER_GPU_TEST_AUTH_MODE=password",
                        f"SERVER_GPU_TEST_PASSWORD={secret}",
                    ]
                )
                + "\n",
            )
            captured: dict[str, object] = {}

            def fake_run(argv, **kwargs):
                captured["argv"] = list(argv)
                captured["env"] = dict(kwargs["env"])
                captured["shell"] = kwargs["shell"]
                captured["askpass_exists"] = Path(kwargs["env"]["SSH_ASKPASS"]).is_file()
                return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

            backend = SafeSSHBackend(credentials_path=path)
            with patch("research_agents.remote_execution.subprocess.run", side_effect=fake_run):
                output = backend._run_ssh(_profile(), "printf ok", ["safe argument"])
            self.assertEqual(output, "ok\n")
            argv_text = "\n".join(captured["argv"])
            self.assertNotIn(secret, argv_text)
            self.assertIn("-oStrictHostKeyChecking=yes", captured["argv"])
            self.assertFalse(captured["shell"])
            self.assertTrue(captured["askpass_exists"])
            self.assertEqual(captured["env"]["SERVER_AGENT_SSH_SECRET"], secret)
            self.assertNotIn("SERVER_AGENT_SSH_SECRET", os.environ)

    def test_missing_agent_env_does_not_fall_back_to_process_or_root_env(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ".env"
            with patch.dict(
                os.environ,
                {"SERVER_GPU_TEST_PASSWORD": "must-not-be-used"},
                clear=False,
            ):
                with self.assertRaises(RemoteExecutionError):
                    with SafeSSHBackend(credentials_path=missing)._connection(_profile()):
                        pass

    def test_safe_backend_parses_foreground_demo_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            _write_env(
                path,
                "\n".join(
                    [
                        "SERVER_GPU_TEST_HOST=gpu.example.org",
                        "SERVER_GPU_TEST_USER=researcher",
                        "SERVER_GPU_TEST_AUTH_MODE=password",
                        "SERVER_GPU_TEST_PASSWORD=demo-secret",
                    ]
                )
                + "\n",
            )
            completed = subprocess.CompletedProcess(
                ["ssh"],
                0,
                stdout="DEMO_RC|0\nDEMO_LOG_BEGIN\nminimal demo ok\nDEMO_LOG_END\n",
                stderr="",
            )
            backend = SafeSSHBackend(credentials_path=path)
            with patch(
                "research_agents.remote_execution.subprocess.run",
                return_value=completed,
            ):
                record = backend.run_minimum_demo(
                    _profile(),
                    environment_name="assaycal_demo",
                    code_root="/srv/research-agent-tests/code",
                    argv=["python", "-m", "assaycal.demo", "--json"],
                    gpu_ids=[0, 1],
                    log_path="/srv/research-agent-tests/logs/minimum-demo.log",
                    timeout_seconds=120,
                )
            self.assertEqual(record.status, "passed")
            self.assertEqual(record.return_code, 0)
            self.assertIn("minimal demo ok", record.output_tail)

    def test_server_json_allows_prefix_but_rejects_secret_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "project" / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "servers.local.json"
            base = {
                "schema_version": "1.0",
                "servers": [
                    {
                        "server_id": "gpu-test",
                        "credential_prefix": "GPU_TEST",
                        "workspace_root": "/srv/research-agent-tests",
                    }
                ],
                "projects": [
                    {
                        "project_slug": "assaycal-test",
                        "train_argv": ["python", "-m", "training.cli"],
                        "metric_name": "validation_mse",
                        "metric_direction": "lower",
                    }
                ],
            }
            config_path.write_text(json.dumps(base), encoding="utf-8")
            loaded = load_server_execution_config(root)
            self.assertEqual(loaded.servers[0].credential_prefix, "GPU_TEST")
            base["servers"][0]["password"] = "secret"
            config_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(RemoteExecutionError):
                load_server_execution_config(root)


if __name__ == "__main__":
    unittest.main()
