from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from research_agents.schemas import LocalValidationRecord


EXECUTION_TIMEOUT_SECONDS = 30
OUTPUT_TAIL_CHARS = 8000
BANNED_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "http",
    "httpx",
    "multiprocessing",
    "requests",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "urllib",
}
BANNED_CALL_NAMES = {"compile", "eval", "exec", "__import__"}
BANNED_ATTRIBUTES = {
    ("os", "popen"),
    ("os", "remove"),
    ("os", "removedirs"),
    ("os", "rmdir"),
    ("os", "system"),
    ("os", "unlink"),
}
FILESYSTEM_CALLS = {
    "open",
    "read_bytes",
    "read_text",
    "rename",
    "replace",
    "rmdir",
    "unlink",
    "write_bytes",
    "write_text",
}


class LocalValidationError(RuntimeError):
    pass


def _call_name(node: ast.Call) -> tuple[str | None, str | None]:
    if isinstance(node.func, ast.Name):
        return None, node.func.id
    if isinstance(node.func, ast.Attribute):
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        return owner, node.func.attr
    return None, None


def validate_generated_python_for_execution(code_root: Path, package_name: str) -> None:
    """Fail closed before executing model-produced demo or test files."""

    python_files = sorted([*(code_root / "src").rglob("*.py"), *(code_root / "tests").rglob("*.py")])
    if not python_files:
        raise LocalValidationError("generated project contains no Python files")
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise LocalValidationError(f"Python syntax/read failure: {path.name}") from exc
        relative = path.relative_to(code_root).as_posix()
        executable_surface = relative == f"src/{package_name}/demo.py" or relative.startswith("tests/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                if roots & BANNED_IMPORT_ROOTS:
                    raise LocalValidationError(f"forbidden import in {relative}: {sorted(roots & BANNED_IMPORT_ROOTS)}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    raise LocalValidationError(f"forbidden import in {relative}: {root}")
            elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
                raise LocalValidationError(f"unbounded while True loop in {relative}")
            elif isinstance(node, ast.Call):
                owner, name = _call_name(node)
                if name in BANNED_CALL_NAMES or (owner, name) in BANNED_ATTRIBUTES:
                    raise LocalValidationError(f"forbidden call in {relative}: {owner or ''}.{name}")
                if executable_surface and name in FILESYSTEM_CALLS:
                    raise LocalValidationError(f"filesystem call is forbidden in demo/tests: {relative}")


def _sanitized_environment(code_root: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str((code_root / "src").resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _metrics(stdout: str) -> dict[str, float | int | str | bool]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise LocalValidationError("minimal demo produced no JSON")
    try:
        payload = json.loads(lines[-1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise LocalValidationError("minimal demo output is not a JSON object") from exc
    if not isinstance(payload, dict):
        raise LocalValidationError("minimal demo output must be an object")
    required = ("initial_loss", "final_loss", "parameter_delta", "iterations")
    if any(key not in payload for key in required):
        raise LocalValidationError("minimal demo omitted training-update evidence")
    initial = float(payload["initial_loss"])
    final = float(payload["final_loss"])
    delta = float(payload["parameter_delta"])
    iterations = int(payload["iterations"])
    if not all(math.isfinite(value) for value in (initial, final, delta)):
        raise LocalValidationError("minimal demo returned a non-finite number")
    if not final < initial or not delta > 0 or iterations < 2:
        raise LocalValidationError("minimal demo did not prove an iterative parameter update")
    clean: dict[str, float | int | str | bool] = {}
    for key, value in payload.items():
        if isinstance(value, (str, bool, int, float)):
            clean[str(key)] = value
    return clean


def _run(
    code_root: Path,
    validation_id: str,
    argv: list[str],
    *,
    parse_metrics: bool,
) -> LocalValidationRecord:
    started = perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=code_root,
            env=_sanitized_environment(code_root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return LocalValidationRecord(
            validation_id=validation_id,
            argv=argv,
            status="timed_out",
            return_code=None,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
            stdout_tail=(exc.stdout or "")[-OUTPUT_TAIL_CHARS:] if isinstance(exc.stdout, str) else "",
            stderr_tail=(exc.stderr or "")[-OUTPUT_TAIL_CHARS:] if isinstance(exc.stderr, str) else "",
        )
    stdout = completed.stdout[-OUTPUT_TAIL_CHARS:]
    stderr = completed.stderr[-OUTPUT_TAIL_CHARS:]
    metrics: dict[str, float | int | str | bool] = {}
    status = "passed" if completed.returncode == 0 else "failed"
    if status == "passed" and parse_metrics:
        try:
            metrics = _metrics(completed.stdout)
        except LocalValidationError as exc:
            status = "failed"
            stderr = f"{stderr}\n{exc}".strip()[-OUTPUT_TAIL_CHARS:]
    return LocalValidationRecord(
        validation_id=validation_id,
        argv=argv,
        status=status,
        return_code=completed.returncode,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        stdout_tail=stdout,
        stderr_tail=stderr,
        metrics=metrics,
    )


def run_local_validations(code_root: Path, package_name: str) -> list[LocalValidationRecord]:
    validate_generated_python_for_execution(code_root, package_name)
    demo = _run(
        code_root,
        "minimal_demo",
        [sys.executable, "-m", f"{package_name}.demo", "--json"],
        parse_metrics=True,
    )
    tests = _run(
        code_root,
        "unit_tests",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        parse_metrics=False,
    )
    return [demo, tests]
