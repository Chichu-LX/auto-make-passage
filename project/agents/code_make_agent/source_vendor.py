from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from research_agents.schemas import AgentName, ArtifactRef, ReusableCodeModule, VendoredSourceRecord
from research_agents.storage import ArtifactStore, StoragePolicyError


MAX_VENDOR_FILES = 20_000
MAX_VENDOR_FILE_BYTES = 50 * 1024 * 1024
MAX_VENDOR_TOTAL_BYTES = 512 * 1024 * 1024
VENDOR_TIMEOUT_SECONDS = 180
LICENSE_FILENAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt")


class SourceVendorError(RuntimeError):
    pass


def _repository_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SourceVendorError("only credential-free HTTPS GitHub repositories are allowed")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise SourceVendorError("GitHub repository URL must identify exactly owner/repository")
    return f"https://github.com/{parts[0]}/{parts[1].removesuffix('.git')}"


def _run_git(argv: list[str], *, cwd: Path | None = None, timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            ["git", *argv],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise SourceVendorError(f"bounded git operation failed: {argv[0]}") from exc
    return completed.stdout.strip()


def _head_commit(repository: str) -> str:
    output = _run_git(["ls-remote", repository, "HEAD"], timeout=30)
    try:
        commit = output.splitlines()[0].split()[0]
    except IndexError as exc:
        raise SourceVendorError("repository HEAD could not be resolved") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SourceVendorError("repository HEAD is not a full commit SHA")
    return commit


def _license_file(root: Path, license_id: str) -> str:
    candidates = [root / name for name in LICENSE_FILENAMES]
    candidates.extend(sorted(root.glob("LICENSE*")))
    for path in candidates:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if license_id == "MIT" and "permission is hereby granted, free of charge" in text:
            return path.relative_to(root).as_posix()
        if license_id == "Apache-2.0" and "apache license" in text and "version 2.0" in text:
            return path.relative_to(root).as_posix()
    raise SourceVendorError(f"declared {license_id} license text was not found in the repository")


def _tree_inventory(root: Path) -> tuple[list[dict[str, str | int]], str, int]:
    records: list[dict[str, str | int]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SourceVendorError(f"vendored source contains a symbolic link: {path.name}")
        if not path.is_file() or path.name == "SOURCE_MANIFEST.json":
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size <= 0:
            continue
        if size > MAX_VENDOR_FILE_BYTES:
            raise SourceVendorError(f"vendored source file exceeds limit: {relative}")
        total_bytes += size
        if total_bytes > MAX_VENDOR_TOTAL_BYTES:
            raise SourceVendorError("vendored repository exceeds total size limit")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_sha = digest.hexdigest()
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(file_sha.encode("ascii"))
        records.append({"path": relative, "size_bytes": size, "sha256": file_sha})
        if len(records) > MAX_VENDOR_FILES:
            raise SourceVendorError("vendored repository exceeds file-count limit")
    if not records:
        raise SourceVendorError("vendored repository contains no regular source files")
    return records, tree_digest.hexdigest(), total_bytes


class GitHubSourceVendor:
    """Download licensed integration sources locally without executing any repository file."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def _existing_record(
        self,
        *,
        module: ReusableCodeModule,
        repository: str,
        commit: str,
        relative_root: str,
    ) -> VendoredSourceRecord | None:
        manifest_path = self.artifacts.root / relative_root / "SOURCE_MANIFEST.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return None
        manifest_ref = self.artifacts.register_agent_file(
            AgentName.CODE_MAKE,
            f"{relative_root}/SOURCE_MANIFEST.json",
            max_bytes=16 * 1024 * 1024,
            kind="vendored_source_manifest",
        )
        try:
            payload = self.artifacts.read_artifact_json(manifest_ref, max_bytes=16 * 1024 * 1024)
        except StoragePolicyError as exc:
            raise SourceVendorError("existing source manifest is unreadable") from exc
        if (
            payload.get("module_id") != module.module_id
            or payload.get("repository_url") != repository
            or payload.get("commit_sha") != commit
            or payload.get("license_id") != module.license_id
        ):
            raise SourceVendorError("existing source snapshot provenance differs from the design")
        records, tree_sha, total = _tree_inventory(manifest_path.parent)
        if (
            tree_sha != payload.get("tree_sha256")
            or total != payload.get("total_bytes")
            or len(records) != payload.get("file_count")
        ):
            raise SourceVendorError("existing source snapshot no longer matches its manifest")
        return VendoredSourceRecord(
            module_id=module.module_id,
            repository_url=repository,
            commit_sha=commit,
            license_id=str(module.license_id),
            local_root=relative_root,
            file_count=len(records),
            total_bytes=total,
            tree_sha256=tree_sha,
            source_manifest=manifest_ref,
        )

    def vendor(self, module: ReusableCodeModule, *, project_slug: str) -> VendoredSourceRecord:
        if module.reuse_mode != "integrate":
            raise SourceVendorError("only integrate-mode modules may be vendored")
        if module.repository_status != "verified_open_source" or not module.license_id:
            raise SourceVendorError("integration requires verified source and an SPDX license")
        repository = _repository_url(module.repository_url)
        commit = _head_commit(repository)
        relative_root = f"code/{project_slug}/third_party/{module.module_id}/{commit}"
        existing = self._existing_record(
            module=module,
            repository=repository,
            commit=commit,
            relative_root=relative_root,
        )
        if existing is not None:
            return existing

        target = self.artifacts.agent_directory_target(AgentName.CODE_MAKE, relative_root)
        temp_root = Path(tempfile.mkdtemp(prefix=".vendor-stage-", dir=target.parent))
        checkout = temp_root / "checkout"
        try:
            _run_git(
                ["clone", "--depth", "1", "--no-tags", repository, str(checkout)],
                timeout=VENDOR_TIMEOUT_SECONDS,
            )
            checked_out = _run_git(["rev-parse", "HEAD"], cwd=checkout, timeout=30)
            if checked_out != commit:
                raise SourceVendorError("repository HEAD changed during bounded download")
            git_metadata = checkout / ".git"
            if not git_metadata.is_dir() or git_metadata.is_symlink():
                raise SourceVendorError("clone did not produce ordinary Git metadata")
            shutil.rmtree(git_metadata)
            license_path = _license_file(checkout, str(module.license_id))
            records, tree_sha, total = _tree_inventory(checkout)
            os.replace(checkout, target)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        manifest_payload = {
            "schema_version": "1.0",
            "module_id": module.module_id,
            "repository_url": repository,
            "commit_sha": commit,
            "license_id": module.license_id,
            "license_path": license_path,
            "reuse_mode": "integrate",
            "execution_performed": False,
            "file_count": len(records),
            "total_bytes": total,
            "tree_sha256": tree_sha,
            "files": records,
        }
        manifest_ref = self.artifacts.write_agent_json(
            AgentName.CODE_MAKE,
            f"{relative_root}/SOURCE_MANIFEST.json",
            manifest_payload,
        ).model_copy(update={"kind": "vendored_source_manifest"})
        return VendoredSourceRecord(
            module_id=module.module_id,
            repository_url=repository,
            commit_sha=commit,
            license_id=str(module.license_id),
            local_root=relative_root,
            file_count=len(records),
            total_bytes=total,
            tree_sha256=tree_sha,
            source_manifest=manifest_ref,
        )

