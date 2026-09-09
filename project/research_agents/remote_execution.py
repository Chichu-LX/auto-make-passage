from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from pydantic import ValidationError

from .server_credentials import (
    SSHCredentials,
    ServerCredentialError,
    ServerCredentialStore,
)
from .schemas import (
    GPUResourceSnapshot,
    JobObservation,
    MetricObservation,
    ScheduledExperimentJob,
    ServerDemoRecord,
    ServerExecutionConfig,
    ServerProfileConfig,
    ServerResourceSnapshot,
)


DEFAULT_SERVER_CONFIG = Path("project/configs/servers.local.json")
MAX_CONFIG_BYTES = 1024 * 1024
MAX_REMOTE_OUTPUT_BYTES = 4 * 1024 * 1024
_SENSITIVE_CONFIG_KEY = re.compile(
    r"password|passphrase|private[_-]?key|secret|token|api[_-]?key|credential",
    re.IGNORECASE,
)


class RemoteExecutionError(RuntimeError):
    def __init__(self, message: str, *, return_code: int | None = None) -> None:
        super().__init__(message)
        self.return_code = return_code


class RemoteCommandTimedOut(RemoteExecutionError):
    pass


def _reject_sensitive_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key != "credential_prefix" and _SENSITIVE_CONFIG_KEY.search(
                normalized_key
            ):
                raise RemoteExecutionError(
                    "server config cannot contain credentials; use the Server Schedule Agent .env"
                )
            _reject_sensitive_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_keys(item)


def load_server_execution_config(
    workspace_root: str | Path,
    path: str | Path | None = None,
) -> ServerExecutionConfig:
    root = Path(workspace_root).resolve()
    unresolved = Path(path) if path is not None else root / DEFAULT_SERVER_CONFIG
    if not unresolved.is_absolute():
        unresolved = root / unresolved
    if unresolved.is_symlink():
        raise RemoteExecutionError("server config cannot be a symbolic link")
    target = unresolved.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RemoteExecutionError("server config must stay inside the workspace") from exc
    if not target.is_file() or target.is_symlink():
        raise RemoteExecutionError(
            "missing regular project/configs/servers.local.json"
        )
    if target.stat().st_size > MAX_CONFIG_BYTES:
        raise RemoteExecutionError("server config exceeds the bounded size limit")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        _reject_sensitive_keys(payload)
        return ServerExecutionConfig.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise RemoteExecutionError("server config is invalid") from exc


def project_execution(config: ServerExecutionConfig, project_slug: str):
    for project in config.projects:
        if project.project_slug == project_slug:
            return project
    raise RemoteExecutionError(
        f"servers.local.json has no execution contract for project {project_slug}"
    )


def _safe_remote_path(value: str, *, root: str | None = None) -> str:
    candidate = value.strip().rstrip("/")
    path = PurePosixPath(candidate)
    if not path.is_absolute() or ".." in path.parts or any(
        character.isspace() for character in candidate
    ):
        raise RemoteExecutionError("remote path must be absolute, traversal-free, and whitespace-free")
    if root is not None:
        root_path = PurePosixPath(root)
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise RemoteExecutionError("remote path escaped the configured server workspace") from exc
    return candidate


def available_gpu_keys(
    profile: ServerProfileConfig,
    probe: ServerResourceSnapshot,
    *,
    transfer_bytes: int,
) -> list[str]:
    if (
        not profile.enabled
        or not profile.execution_enabled
        or probe.status != "available"
        or not probe.conda_available
        or probe.memory_available_bytes < profile.min_available_memory_bytes
        or probe.disk_available_bytes
        < max(profile.min_available_disk_bytes, transfer_bytes * 2)
    ):
        return []
    if profile.accelerator == "cpu":
        return [f"{profile.server_id}:0"]
    return [
        f"{profile.server_id}:{gpu.index}"
        for gpu in probe.gpus
        if gpu.utilization_percent <= profile.max_gpu_utilization_percent
        and gpu.memory_free_mib >= profile.gpu_memory_reserve_mib
    ]


class RemoteExecutionBackend(Protocol):
    def probe(self, profile: ServerProfileConfig) -> ServerResourceSnapshot: ...

    def prepare_environment(
        self,
        profile: ServerProfileConfig,
        *,
        workspace: str,
        environment_name: str,
        conda_dependencies: list[str],
    ) -> None: ...

    def transfer_file(
        self,
        profile: ServerProfileConfig,
        *,
        local_path: Path,
        remote_path: str,
    ) -> None: ...

    def install_project(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        pip_install_argv: list[str],
    ) -> None: ...

    def validate_gpu_environment(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        gpu_ids: list[int],
    ) -> None: ...

    def run_minimum_demo(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        argv: list[str],
        gpu_ids: list[int],
        log_path: str,
        timeout_seconds: int,
    ) -> ServerDemoRecord: ...

    def launch_job(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        runner_path: str,
        spec_path: str,
        bootstrap_log_path: str,
    ) -> str: ...

    def observe_job(
        self,
        profile: ServerProfileConfig,
        job: ScheduledExperimentJob,
    ) -> JobObservation: ...


class DisabledRemoteBackend:
    """Hard no-network backend for local design runs and acceptance pipelines."""

    @staticmethod
    def _disabled() -> None:
        raise RemoteExecutionError(
            "remote execution is disabled for this Runtime; no SSH/GPU connection was attempted"
        )

    def probe(self, profile: ServerProfileConfig) -> ServerResourceSnapshot:
        return ServerResourceSnapshot(
            server_id=profile.server_id,
            status="unreachable",
            error_code="REMOTE_EXECUTION_DISABLED",
        )

    def prepare_environment(self, *args, **kwargs) -> None:
        self._disabled()

    def transfer_file(self, *args, **kwargs) -> None:
        self._disabled()

    def install_project(self, *args, **kwargs) -> None:
        self._disabled()

    def validate_gpu_environment(self, *args, **kwargs) -> None:
        self._disabled()

    def run_minimum_demo(self, *args, **kwargs) -> ServerDemoRecord:
        self._disabled()

    def launch_job(self, *args, **kwargs) -> str:
        self._disabled()

    def observe_job(self, *args, **kwargs) -> JobObservation:
        self._disabled()


class SafeLocalBackend:
    """Run bounded small-scale jobs in a workspace-local Conda environment.

    It implements the remote backend contract without SSH, shell evaluation, network
    access, CUDA assumptions, or destructive directory operations.  The configured
    workspace must stay under ``project/var/local-compute``.
    """

    def __init__(self, workspace_root: str | Path, *, conda_executable: str = "conda") -> None:
        self.root = Path(workspace_root).resolve()
        self.allowed_root = (self.root / "project/var/local-compute").resolve()
        self.allowed_root.mkdir(parents=True, exist_ok=True)
        resolved_conda = shutil.which(conda_executable)
        if resolved_conda is None:
            raise RemoteExecutionError("local backend requires a conda executable")
        self.conda_executable = resolved_conda

    def _path(self, value: str) -> Path:
        path = Path(value).resolve()
        try:
            path.relative_to(self.allowed_root)
        except ValueError as exc:
            raise RemoteExecutionError(
                "local compute path escaped project/var/local-compute"
            ) from exc
        return path

    @staticmethod
    def _bounded_run(
        argv: list[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        stdout=None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not argv or any("\x00" in value for value in argv):
            raise RemoteExecutionError("local command argv is invalid")
        try:
            return subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout if stdout is not None else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteCommandTimedOut("bounded local command timed out") from exc
        except OSError as exc:
            raise RemoteExecutionError("bounded local command failed to start") from exc

    def probe(self, profile: ServerProfileConfig) -> ServerResourceSnapshot:
        workspace = self._path(profile.workspace_root)
        workspace.mkdir(parents=True, exist_ok=True)
        memory_total = 0
        memory_available = 0
        try:
            memory_total = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.strip()
            )
            memory_available = memory_total
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        disk = shutil.disk_usage(workspace)
        return ServerResourceSnapshot(
            server_id=profile.server_id,
            status="available",
            cpu_count=os.cpu_count() or 1,
            load_1m=os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
            memory_total_bytes=memory_total,
            memory_available_bytes=memory_available,
            disk_total_bytes=disk.total,
            disk_available_bytes=disk.free,
            conda_available=True,
            gpus=[],
        )

    def prepare_environment(
        self,
        profile: ServerProfileConfig,
        *,
        workspace: str,
        environment_name: str,
        conda_dependencies: list[str],
    ) -> None:
        target = self._path(str(Path(workspace) / "conda-env"))
        if target.is_dir() and (target / "conda-meta").is_dir():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        dependencies = [
            "python=3.12" if item.startswith("python=") else item
            for item in conda_dependencies
        ]
        result = self._bounded_run(
            [
                self.conda_executable,
                "create",
                "--offline",
                "--yes",
                "--prefix",
                str(target),
                *dependencies,
            ],
            cwd=target.parent,
            timeout_seconds=900,
        )
        if result.returncode != 0:
            raise RemoteExecutionError("local Conda environment creation failed")

    def transfer_file(
        self,
        profile: ServerProfileConfig,
        *,
        local_path: Path,
        remote_path: str,
    ) -> None:
        source = local_path.resolve()
        source.relative_to(self.root)
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink() or not source.is_file():
            raise RemoteExecutionError("local transfer source must be a regular file")
        shutil.copy2(source, target)

    def install_project(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        pip_install_argv: list[str],
    ) -> None:
        root = self._path(code_root)
        env_prefix = self._path(str(root.parent / "conda-env"))
        pyprojects = list(root.glob("**/pyproject.toml"))
        install_root = pyprojects[0].parent if len(pyprojects) == 1 else root
        argv = list(pip_install_argv)
        argv[0] = str(env_prefix / "bin/python")
        result = self._bounded_run(
            argv,
            cwd=install_root,
            timeout_seconds=600,
        )
        if result.returncode != 0:
            raise RemoteExecutionError("local project installation failed")

    def validate_gpu_environment(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        gpu_ids: list[int],
    ) -> None:
        if profile.accelerator != "cpu" or gpu_ids != [0]:
            raise RemoteExecutionError("local backend supports one CPU slot only")

    def run_minimum_demo(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        argv: list[str],
        gpu_ids: list[int],
        log_path: str,
        timeout_seconds: int,
    ) -> ServerDemoRecord:
        started = perf_counter()
        root = self._path(code_root)
        env_prefix = self._path(str(root.parent / "conda-env"))
        pyprojects = list(root.glob("**/pyproject.toml"))
        cwd = pyprojects[0].parent if len(pyprojects) == 1 else root
        command = list(argv)
        command[0] = str(env_prefix / "bin/python")
        log = self._path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = self._bounded_run(command, cwd=cwd, timeout_seconds=timeout_seconds)
            output = result.stdout or ""
            log.write_text(output[-16000:], encoding="utf-8")
            return ServerDemoRecord(
                server_id=profile.server_id,
                argv=argv,
                gpu_ids=gpu_ids,
                status="passed" if result.returncode == 0 else "failed",
                return_code=result.returncode,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                timeout_seconds=timeout_seconds,
                log_path=str(log),
                output_tail=output[-16000:],
            )
        except RemoteCommandTimedOut:
            return ServerDemoRecord(
                server_id=profile.server_id,
                argv=argv,
                gpu_ids=gpu_ids,
                status="timed_out",
                return_code=None,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                timeout_seconds=timeout_seconds,
                log_path=str(log),
                output_tail="bounded local demo timed out",
            )

    def launch_job(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        runner_path: str,
        spec_path: str,
        bootstrap_log_path: str,
    ) -> str:
        runner = self._path(runner_path)
        spec = self._path(spec_path)
        bootstrap = self._path(bootstrap_log_path)
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        environment_roots = list(self.allowed_root.glob("**/conda-env"))
        matches = [
            item for item in environment_roots
            if item.is_dir() and environment_name in str(item.parent)
        ]
        env_prefix = matches[0] if len(matches) == 1 else None
        if env_prefix is None:
            # The run-scoped environment is always the nearest conda-env ancestor
            # sibling of the workspace's code directory.
            candidates = [
                parent / "conda-env" for parent in runner.parents
                if parent != self.allowed_root.parent
            ]
            env_prefix = next(
                (item for item in candidates if item.is_dir()), None
            )
        if env_prefix is None:
            raise RemoteExecutionError("local launch cannot resolve its Conda environment")
        child_environment = dict(os.environ)
        child_environment["PATH"] = os.pathsep.join(
            [
                str(env_prefix / "bin"),
                child_environment.get("PATH", ""),
            ]
        ).rstrip(os.pathsep)
        child_environment["CONDA_PREFIX"] = str(env_prefix)
        child_environment["CONDA_DEFAULT_ENV"] = environment_name
        child_environment["PYTHONNOUSERSITE"] = "1"
        with bootstrap.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                [str(env_prefix / "bin/python"), str(runner), "--spec", str(spec)],
                cwd=runner.parent,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=handle,
                start_new_session=True,
                close_fds=True,
            )
        return f"local-{process.pid}-{uuid4().hex[:8]}"

    def observe_job(
        self,
        profile: ServerProfileConfig,
        job: ScheduledExperimentJob,
    ) -> JobObservation:
        workdir = self._path(job.remote_workdir)
        status_path = workdir / "status.json"
        metrics_path = self._path(job.metrics_path)
        log_path = self._path(job.log_path)
        payload: dict[str, object] = {}
        if status_path.is_file() and not status_path.is_symlink():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
        raw_status = str(payload.get("status") or "unknown")
        status = raw_status if raw_status in {
            "queued", "running", "completed", "failed"
        } else "unknown"
        metrics: list[MetricObservation] = []
        if metrics_path.is_file() and not metrics_path.is_symlink():
            for line in metrics_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    value = json.loads(line)
                    metrics.append(
                        MetricObservation(
                            metric_name=str(value["metric_name"]),
                            value=float(value["value"]),
                            step=value.get("step"),
                            checkpoint_path=value.get("checkpoint_path"),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        matching = [item for item in metrics if item.metric_name == job.metric_name]
        selector = min if job.metric_direction == "lower" else max
        best = selector(matching, key=lambda item: item.value) if matching else None
        exit_code = payload.get("exit_code")
        return JobObservation(
            job_id=job.job_id,
            status=status,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            metrics=metrics,
            best_metric=best,
            best_checkpoint_path=(
                str(payload.get("best_checkpoint_path"))
                if payload.get("best_checkpoint_path")
                else (best.checkpoint_path if best else None)
            ),
            log_tail=(
                log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                if log_path.is_file() and not log_path.is_symlink()
                else ""
            ),
        )


class SafeSSHBackend:
    """Fixed-script SSH backend with an Agent-owned credential store.

    Passwords and key passphrases are exposed only to the child SSH/rsync process
    through a temporary ``SSH_ASKPASS`` environment. All local processes use
    ``shell=False``. Remote scripts receive quoted positional arguments, create only
    run-scoped directories, and never contain recursive deletion.
    """

    _PROBE_SCRIPT = r"""
set -eu
root="$1"
mkdir -p -- "$root"
cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '0')
load_1m=$(awk '{print $1}' /proc/loadavg 2>/dev/null || printf '0')
if command -v free >/dev/null 2>&1; then
  memory_total=$(free -b | awk '/^Mem:/ {print $2}')
  memory_available=$(free -b | awk '/^Mem:/ {print $7}')
else
  memory_total=0
  memory_available=0
fi
disk_line=$(df -PB1 -- "$root" | awk 'NR==2 {print $2 "|" $4}')
disk_total=${disk_line%%|*}
disk_available=${disk_line##*|}
if command -v conda >/dev/null 2>&1; then conda_available=1; else conda_available=0; fi
printf 'HOST|%s|%s|%s|%s|%s|%s\n' "$cpu_count" "$load_1m" "$memory_total" "$memory_available" "$disk_total" "$disk_available"
printf 'CONDA|%s\n' "$conda_available"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | sed 's/, /|/g; s/^/GPU|/'
fi
""".strip()

    _PREPARE_SCRIPT = r"""
set -eu
workspace="$1"
environment_name="$2"
conda_executable="$3"
shift 3
mkdir -p -- "$workspace/data" "$workspace/code" "$workspace/models" "$workspace/runtime" "$workspace/jobs" "$workspace/logs" "$workspace/best"
if "$conda_executable" env list --json | grep -Fq "\"$environment_name\""; then
  printf 'environment-exists\n'
else
  "$conda_executable" create -y -n "$environment_name" "$@"
  printf 'environment-created\n'
fi
""".strip()

    _MKPARENT_SCRIPT = r"""
set -eu
target="$1"
parent=${target%/*}
mkdir -p -- "$parent"
""".strip()

    _LAUNCH_SCRIPT = r"""
set -eu
environment_name="$1"
conda_executable="$2"
runner_path="$3"
spec_path="$4"
bootstrap_log_path="$5"
mkdir -p -- "${bootstrap_log_path%/*}"
nohup "$conda_executable" run --no-capture-output -n "$environment_name" python "$runner_path" --spec "$spec_path" >> "$bootstrap_log_path" 2>&1 < /dev/null &
printf '%s\n' "$!"
""".strip()

    _INSTALL_SCRIPT = r"""
set -eu
environment_name="$1"
conda_executable="$2"
code_root="$3"
shift 3
cd -- "$code_root"
"$conda_executable" run --no-capture-output -n "$environment_name" "$@"
""".strip()

    _GPU_PREFLIGHT_SCRIPT = r"""
set -eu
environment_name="$1"
conda_executable="$2"
gpu_ids="$3"
expected_count="$4"
CUDA_VISIBLE_DEVICES="$gpu_ids" "$conda_executable" run --no-capture-output -n "$environment_name" python -c 'import sys; import torch; expected=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable"; actual=torch.cuda.device_count(); assert actual == expected, f"visible GPU count {actual} != {expected}"; print(f"cuda-ok:{actual}")' "$expected_count"
""".strip()

    _DEMO_SCRIPT = r"""
set -eu
environment_name="$1"
conda_executable="$2"
code_root="$3"
gpu_ids="$4"
log_path="$5"
timeout_seconds="$6"
shift 6
case "$timeout_seconds" in *[!0-9]*|'') printf 'invalid demo timeout\n' >&2; exit 2;; esac
mkdir -p -- "${log_path%/*}"
cd -- "$code_root"
set +e
CUDA_VISIBLE_DEVICES="$gpu_ids" timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" "$conda_executable" run --no-capture-output -n "$environment_name" "$@" > "$log_path" 2>&1
return_code=$?
set -e
printf 'DEMO_RC|%s\nDEMO_LOG_BEGIN\n' "$return_code"
if test -f "$log_path"; then tail -c 16000 -- "$log_path"; fi
printf '\nDEMO_LOG_END\n'
""".strip()

    _OBSERVE_SCRIPT = r"""
set -eu
status_path="$1"
metrics_path="$2"
log_path="$3"
pid="$4"
printf 'STATUS_BEGIN\n'
if test -f "$status_path"; then cat -- "$status_path"; else printf '{}\n'; fi
printf '\nSTATUS_END\nMETRICS_BEGIN\n'
if test -f "$metrics_path"; then tail -n 10000 -- "$metrics_path"; fi
printf '\nMETRICS_END\nLOG_BEGIN\n'
if test -f "$log_path"; then tail -n 200 -- "$log_path"; fi
printf '\nLOG_END\nPID_STATE|'
if kill -0 "$pid" 2>/dev/null; then printf 'running\n'; else printf 'stopped\n'; fi
""".strip()

    def __init__(
        self,
        *,
        ssh_timeout_seconds: int = 30,
        credential_store: ServerCredentialStore | None = None,
        credentials_path: str | Path | None = None,
    ) -> None:
        self.ssh_timeout_seconds = max(5, min(ssh_timeout_seconds, 300))
        if credential_store is not None and credentials_path is not None:
            raise ValueError("provide credential_store or credentials_path, not both")
        self.credential_store = credential_store or ServerCredentialStore(credentials_path)

    def _run(
        self,
        argv: list[str],
        *,
        input_text: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        try:
            completed = subprocess.run(
                argv,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.ssh_timeout_seconds,
                check=False,
                shell=False,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteCommandTimedOut("remote command exceeded its timeout") from exc
        except OSError as exc:
            raise RemoteExecutionError(f"remote command failed: {type(exc).__name__}") from exc
        output_bytes = len(completed.stdout.encode("utf-8", errors="replace")) + len(
            completed.stderr.encode("utf-8", errors="replace")
        )
        if output_bytes > MAX_REMOTE_OUTPUT_BYTES:
            raise RemoteExecutionError("remote command output exceeds safety limit")
        if completed.returncode != 0:
            raise RemoteExecutionError(
                f"remote command returned {completed.returncode}; inspect redacted process logs",
                return_code=completed.returncode,
            )
        return completed.stdout

    @staticmethod
    def _ssh_options(credentials: SSHCredentials) -> list[str]:
        options = [
            "-F",
            "/dev/null",
            "-p",
            str(credentials.port),
            "-l",
            credentials.user,
            "-oConnectTimeout=15",
            "-oStrictHostKeyChecking=yes",
            "-oLogLevel=ERROR",
            "-oCanonicalizeHostname=no",
            "-oClearAllForwardings=yes",
            "-oForwardAgent=no",
            "-oPermitLocalCommand=no",
            "-oProxyCommand=none",
            "-oProxyJump=none",
            "-oRequestTTY=no",
            "-oSendEnv=-*",
        ]
        if credentials.auth_mode == "password":
            options.extend(
                [
                    "-oBatchMode=no",
                    "-oNumberOfPasswordPrompts=1",
                    "-oPreferredAuthentications=password,keyboard-interactive",
                    "-oPubkeyAuthentication=no",
                    "-oPasswordAuthentication=yes",
                    "-oKbdInteractiveAuthentication=yes",
                    "-oIdentityAgent=none",
                ]
            )
        else:
            assert credentials.private_key_path is not None
            options.extend(
                [
                    "-oBatchMode="
                    + ("no" if credentials.private_key_passphrase else "yes"),
                    "-oNumberOfPasswordPrompts=1",
                    "-oPreferredAuthentications=publickey",
                    "-oPasswordAuthentication=no",
                    "-oKbdInteractiveAuthentication=no",
                    "-oIdentitiesOnly=yes",
                    "-oIdentityAgent=none",
                    "-i",
                    str(credentials.private_key_path),
                ]
            )
        return options

    @contextmanager
    def _connection(self, profile: ServerProfileConfig):
        try:
            credentials = self.credential_store.resolve(profile.credential_prefix)
        except ServerCredentialError as exc:
            raise RemoteExecutionError("server SSH credentials are unavailable or invalid") from exc
        inherited_names = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LOGNAME",
            "PATH",
            "SHELL",
            "TMPDIR",
            "USER",
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in inherited_names or key.startswith("LC_")
        }
        secret = credentials.askpass_secret
        if not secret:
            yield credentials, environment
            return
        with tempfile.TemporaryDirectory(prefix="server-agent-askpass-") as directory:
            helper = Path(directory) / "askpass.sh"
            helper.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$SERVER_AGENT_SSH_SECRET\"\n",
                encoding="utf-8",
            )
            helper.chmod(0o700)
            environment.update(
                {
                    "SSH_ASKPASS": str(helper),
                    "SSH_ASKPASS_REQUIRE": "force",
                    "DISPLAY": "server-agent-askpass:0",
                    "SERVER_AGENT_SSH_SECRET": secret,
                }
            )
            yield credentials, environment

    @staticmethod
    def _ssh_argv(credentials: SSHCredentials, args: list[str]) -> list[str]:
        remote_command = "bash -s --"
        if args:
            remote_command += " " + " ".join(shlex.quote(value) for value in args)
        return [
            "ssh",
            *SafeSSHBackend._ssh_options(credentials),
            credentials.host,
            remote_command,
        ]

    def _run_ssh(
        self,
        profile: ServerProfileConfig,
        script: str,
        args: list[str],
        *,
        timeout: int | None = None,
    ) -> str:
        with self._connection(profile) as (credentials, environment):
            return self._run(
                self._ssh_argv(credentials, args),
                input_text=script,
                timeout=timeout,
                env=environment,
            )

    def probe(self, profile: ServerProfileConfig) -> ServerResourceSnapshot:
        try:
            output = self._run_ssh(
                profile,
                self._PROBE_SCRIPT,
                [profile.workspace_root],
            )
            host: list[str] | None = None
            conda_available = False
            gpus: list[GPUResourceSnapshot] = []
            for line in output.splitlines():
                fields = line.split("|")
                if fields[0] == "HOST" and len(fields) == 7:
                    host = fields
                elif fields[0] == "CONDA" and len(fields) == 2:
                    conda_available = fields[1] == "1"
                elif fields[0] == "GPU" and len(fields) == 6:
                    gpus.append(
                        GPUResourceSnapshot(
                            index=int(fields[1]),
                            name=fields[2].strip(),
                            memory_total_mib=int(fields[3]),
                            memory_used_mib=int(fields[4]),
                            utilization_percent=float(fields[5]),
                        )
                    )
            if host is None:
                raise RemoteExecutionError("remote probe omitted host resources")
            return ServerResourceSnapshot(
                server_id=profile.server_id,
                status="available",
                cpu_count=int(host[1]),
                load_1m=float(host[2]),
                memory_total_bytes=int(host[3]),
                memory_available_bytes=int(host[4]),
                disk_total_bytes=int(host[5]),
                disk_available_bytes=int(host[6]),
                conda_available=conda_available,
                gpus=gpus,
            )
        except (RemoteExecutionError, ValueError) as exc:
            return ServerResourceSnapshot(
                server_id=profile.server_id,
                status="unreachable",
                error_code=type(exc).__name__,
            )

    def prepare_environment(
        self,
        profile: ServerProfileConfig,
        *,
        workspace: str,
        environment_name: str,
        conda_dependencies: list[str],
    ) -> None:
        workspace = _safe_remote_path(workspace, root=profile.workspace_root)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", environment_name):
            raise RemoteExecutionError("invalid conda environment name")
        self._run_ssh(
            profile,
            self._PREPARE_SCRIPT,
            [workspace, environment_name, profile.conda_executable, *conda_dependencies],
            timeout=900,
        )

    def transfer_file(
        self,
        profile: ServerProfileConfig,
        *,
        local_path: Path,
        remote_path: str,
    ) -> None:
        if not local_path.is_file() or local_path.is_symlink():
            raise RemoteExecutionError("transfer source must be a regular local file")
        remote_path = _safe_remote_path(remote_path, root=profile.workspace_root)
        self._run_ssh(profile, self._MKPARENT_SCRIPT, [remote_path])
        with self._connection(profile) as (credentials, environment):
            remote_shell = " ".join(
                shlex.quote(value)
                for value in ["ssh", *self._ssh_options(credentials)]
            )
            host = (
                f"[{credentials.host}]"
                if ":" in credentials.host
                else credentials.host
            )
            self._run(
                [
                    "rsync",
                    "-rlpt",
                    "--checksum",
                    "--protect-args",
                    "-e",
                    remote_shell,
                    "--",
                    str(local_path),
                    f"{host}:{remote_path}",
                ],
                timeout=3600,
                env=environment,
            )

    def launch_job(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        runner_path: str,
        spec_path: str,
        bootstrap_log_path: str,
    ) -> str:
        values = [
            _safe_remote_path(value, root=profile.workspace_root)
            for value in (runner_path, spec_path, bootstrap_log_path)
        ]
        output = self._run_ssh(
            profile,
            self._LAUNCH_SCRIPT,
            [
                environment_name,
                profile.conda_executable,
                *values,
            ],
        ).strip()
        if not re.fullmatch(r"[1-9][0-9]{0,19}", output):
            raise RemoteExecutionError("remote launcher did not return a PID")
        return output

    def install_project(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        pip_install_argv: list[str],
    ) -> None:
        code_root = _safe_remote_path(code_root, root=profile.workspace_root)
        self._run_ssh(
            profile,
            self._INSTALL_SCRIPT,
            [
                environment_name,
                profile.conda_executable,
                code_root,
                *pip_install_argv,
            ],
            timeout=1800,
        )

    def validate_gpu_environment(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        gpu_ids: list[int],
    ) -> None:
        if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
            raise RemoteExecutionError("GPU preflight requires unique selected GPU IDs")
        self._run_ssh(
            profile,
            self._GPU_PREFLIGHT_SCRIPT,
            [
                environment_name,
                profile.conda_executable,
                ",".join(str(value) for value in gpu_ids),
                str(len(gpu_ids)),
            ],
            timeout=300,
        )

    def run_minimum_demo(
        self,
        profile: ServerProfileConfig,
        *,
        environment_name: str,
        code_root: str,
        argv: list[str],
        gpu_ids: list[int],
        log_path: str,
        timeout_seconds: int,
    ) -> ServerDemoRecord:
        code_root = _safe_remote_path(code_root, root=profile.workspace_root)
        log_path = _safe_remote_path(log_path, root=profile.workspace_root)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", environment_name):
            raise RemoteExecutionError("invalid conda environment name")
        if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(
            value < 0 for value in gpu_ids
        ):
            raise RemoteExecutionError("minimum demo requires unique GPU IDs")
        if not 10 <= timeout_seconds <= 1800:
            raise RemoteExecutionError("minimum demo timeout is outside its safety range")
        if not 2 <= len(argv) <= 30 or PurePosixPath(argv[0]).name not in {
            "python",
            "python3",
        }:
            raise RemoteExecutionError("minimum demo argv must invoke Python")
        if any(
            not value
            or len(value) > 1000
            or any(character in value for character in ("\x00", "\n", "\r"))
            for value in argv
        ):
            raise RemoteExecutionError("minimum demo argv contains an invalid token")
        started = perf_counter()
        try:
            output = self._run_ssh(
                profile,
                self._DEMO_SCRIPT,
                [
                    environment_name,
                    profile.conda_executable,
                    code_root,
                    ",".join(str(value) for value in gpu_ids),
                    log_path,
                    str(timeout_seconds),
                    *argv,
                ],
                timeout=timeout_seconds + 30,
            )
        except RemoteCommandTimedOut:
            return ServerDemoRecord(
                server_id=profile.server_id,
                argv=argv,
                gpu_ids=gpu_ids,
                status="timed_out",
                return_code=None,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                output_tail="Remote foreground demo exceeded its bounded timeout.",
            )
        except RemoteExecutionError as exc:
            return ServerDemoRecord(
                server_id=profile.server_id,
                argv=argv,
                gpu_ids=gpu_ids,
                status="failed",
                return_code=exc.return_code,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                timeout_seconds=timeout_seconds,
                log_path=log_path,
                output_tail="Remote foreground demo connection or execution failed.",
            )
        match = re.search(r"(?:^|\n)DEMO_RC\|(-?[0-9]+)(?:\n|$)", output)
        if match is None:
            return_code = None
            status = "failed"
        else:
            return_code = int(match.group(1))
            if return_code == 0:
                status = "passed"
            elif return_code in {124, 137}:
                status = "timed_out"
                return_code = None
            else:
                status = "failed"
        return ServerDemoRecord(
            server_id=profile.server_id,
            argv=argv,
            gpu_ids=gpu_ids,
            status=status,
            return_code=return_code,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            timeout_seconds=timeout_seconds,
            log_path=log_path,
            output_tail=self._section(output, "DEMO_LOG")[-16000:],
        )

    @staticmethod
    def _section(output: str, name: str) -> str:
        start = f"{name}_BEGIN\n"
        end = f"\n{name}_END"
        if start not in output or end not in output:
            return ""
        return output.split(start, 1)[1].split(end, 1)[0]

    def observe_job(
        self,
        profile: ServerProfileConfig,
        job: ScheduledExperimentJob,
    ) -> JobObservation:
        if not job.remote_job_id:
            return JobObservation(job_id=job.job_id, status="unknown")
        status_path = str(PurePosixPath(job.remote_workdir) / "status.json")
        for value in (status_path, job.metrics_path, job.log_path):
            _safe_remote_path(value, root=profile.workspace_root)
        try:
            output = self._run_ssh(
                profile,
                self._OBSERVE_SCRIPT,
                [status_path, job.metrics_path, job.log_path, job.remote_job_id],
            )
        except RemoteExecutionError:
            return JobObservation(job_id=job.job_id, status="unknown")
        status_payload: dict[str, object] = {}
        try:
            parsed = json.loads(self._section(output, "STATUS") or "{}")
            if isinstance(parsed, dict):
                status_payload = parsed
        except json.JSONDecodeError:
            status_payload = {}
        metrics: list[MetricObservation] = []
        for line in self._section(output, "METRICS").splitlines():
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    continue
                metrics.append(
                    MetricObservation(
                        metric_name=str(value.get("metric_name", "")),
                        value=float(value["value"]),
                        step=value.get("step"),
                        checkpoint_path=value.get("checkpoint_path"),
                    )
                )
            except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
                continue
        pid_state = "running" if "PID_STATE|running" in output else "stopped"
        raw_status = str(status_payload.get("status") or "")
        if raw_status == "completed":
            observed_status = "completed"
        elif raw_status == "failed":
            observed_status = "failed"
        elif raw_status == "queued":
            observed_status = "queued"
        elif pid_state == "running":
            observed_status = "running"
        else:
            observed_status = "unknown"
        exit_code = status_payload.get("exit_code")
        if not isinstance(exit_code, int):
            exit_code = None
        matching = [item for item in metrics if item.metric_name == job.metric_name]
        best = None
        if matching:
            selector = min if job.metric_direction == "lower" else max
            best = selector(matching, key=lambda item: item.value)
        return JobObservation(
            job_id=job.job_id,
            status=observed_status,
            exit_code=exit_code,
            metrics=metrics,
            best_metric=best,
            best_checkpoint_path=(
                str(status_payload.get("best_checkpoint_path"))
                if status_payload.get("best_checkpoint_path")
                else (best.checkpoint_path if best else None)
            ),
            log_tail=self._section(output, "LOG")[-16000:],
        )


MANAGED_RUNNER_SOURCE = r'''from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    workdir = Path(spec["remote_workdir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    status_path = workdir / "status.json"
    log_path = Path(spec["log_path"]).resolve()
    metrics_path = Path(spec["metrics_path"]).resolve()
    best_dir = Path(spec["best_checkpoint_dir"]).resolve()
    for path in (log_path, metrics_path, best_dir):
        if not inside(path, workdir.parent.parent):
            raise RuntimeError("managed output escaped the run workspace")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in spec["gpu_ids"])
    environment["EXPERIMENT_METRICS_PATH"] = str(metrics_path)
    lock_path = Path(spec["gpu_lock_path"]).resolve()
    if not inside(lock_path, workdir.parent.parent):
        raise RuntimeError("GPU queue lock escaped the run workspace")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    runner_pid = os.getpid()
    atomic_json(
        status_path,
        {
            "status": "queued",
            "runner_pid": runner_pid,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        started_at = datetime.now(timezone.utc).isoformat()
        atomic_json(
            status_path,
            {"status": "running", "runner_pid": runner_pid, "started_at": started_at},
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                spec["command_argv"],
                cwd=workdir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "runner_pid": runner_pid,
                    "child_pid": process.pid,
                    "started_at": started_at,
                },
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
            return_code = process.wait()
    observations = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
                if value.get("metric_name") == spec["metric_name"] and math.isfinite(float(value["value"])):
                    observations.append(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    best = None
    if observations:
        best = (min if spec["metric_direction"] == "lower" else max)(
            observations, key=lambda value: float(value["value"])
        )
    best_checkpoint_path = None
    if best and best.get("checkpoint_path"):
        source = Path(best["checkpoint_path"]).resolve()
        if source.is_file() and inside(source, workdir):
            destination = best_dir / source.name
            shutil.copy2(source, destination)
            best_checkpoint_path = str(destination)
    atomic_json(
        status_path,
        {
            "status": "completed" if return_code == 0 else "failed",
            "runner_pid": runner_pid,
            "child_pid": process.pid,
            "exit_code": return_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "best_metric": best,
            "best_checkpoint_path": best_checkpoint_path,
        },
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def script_has_forbidden_directory_deletion(source: str) -> bool:
    """Conservative check used before any generated helper reaches a server."""

    lowered = source.lower()
    forbidden = (
        "rm -r",
        "rm --recursive",
        "rmdir",
        "shutil.rmtree",
        "os.removedirs",
        "find -delete",
    )
    return any(value in lowered for value in forbidden)
