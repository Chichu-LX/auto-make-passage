from __future__ import annotations

import io
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import py7zr

from .downloads import DownloadPolicyError, SAFE_ARCHIVE_MEMBER_EXTENSIONS
from .schemas import AgentName, DatasetMaterializedFile
from .storage import ArtifactStore, StoragePolicyError


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise DownloadPolicyError("archive contains a path traversal entry")
    if normalized.suffix.lower() not in SAFE_ARCHIVE_MEMBER_EXTENSIONS:
        raise DownloadPolicyError("archive contains a non-data file type")
    return normalized


def _write_plain(payload: bytes, dataset_format: str, staging: Path) -> None:
    target = staging / f"dataset.{dataset_format}"
    target.write_bytes(payload)


def _write_zip(payload: bytes, staging: Path, max_bytes: int) -> None:
    total = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            unix_mode = member.external_attr >> 16
            if member.is_dir():
                continue
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise DownloadPolicyError("ZIP symbolic links are forbidden")
            relative = _safe_member_path(member.filename)
            total += member.file_size
            if total > max_bytes:
                raise DownloadPolicyError("materialized ZIP size exceeds the 10 GiB policy")
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def _write_7z(payload: bytes, staging: Path, max_bytes: int) -> None:
    with py7zr.SevenZipFile(io.BytesIO(payload), mode="r") as archive:
        total = 0
        for member in archive.list():
            if member.is_directory:
                continue
            if member.is_symlink:
                raise DownloadPolicyError("7z symbolic links are forbidden")
            _safe_member_path(member.filename)
            total += int(member.uncompressed or 0)
            if total > max_bytes:
                raise DownloadPolicyError("materialized 7z size exceeds the 10 GiB policy")
        archive.extractall(path=staging)


def materialize_dataset_payload(
    payload: bytes,
    dataset_format: str,
    *,
    artifacts: ArtifactStore,
    agent: AgentName,
    relative_root: str,
    max_bytes: int,
) -> list[DatasetMaterializedFile]:
    """Persist usable dataset files atomically below an Agent-owned dataset root."""

    target = artifacts.agent_directory_target(agent, relative_root)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-materializing-", dir=target.parent)
    )
    committed = False
    try:
        if dataset_format == "zip":
            _write_zip(payload, staging, max_bytes)
        elif dataset_format == "7z":
            _write_7z(payload, staging, max_bytes)
        else:
            _write_plain(payload, dataset_format, staging)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        if not files:
            raise DownloadPolicyError("dataset materialization produced no files")
        total = 0
        for path in files:
            if path.is_symlink():
                raise DownloadPolicyError("materialized symbolic links are forbidden")
            relative = path.relative_to(staging)
            _safe_member_path(relative.as_posix())
            total += path.stat().st_size
            if total > max_bytes:
                raise DownloadPolicyError("materialized dataset exceeds the 10 GiB policy")

        os.replace(staging, target)
        committed = True
        records: list[DatasetMaterializedFile] = []
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            workspace_relative = path.relative_to(artifacts.root).as_posix()
            artifact = artifacts.register_agent_file(
                agent,
                workspace_relative,
                max_bytes=max_bytes,
                kind="dataset_materialized_file",
            )
            records.append(
                DatasetMaterializedFile(
                    relative_path=path.relative_to(target).as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=artifact.sha256,
                    artifact=artifact,
                )
            )
        return records
    except (OSError, zipfile.BadZipFile, py7zr.Bad7zFile, StoragePolicyError) as exc:
        if isinstance(exc, (DownloadPolicyError, StoragePolicyError)):
            raise
        raise DownloadPolicyError(
            f"dataset materialization failed: {type(exc).__name__}"
        ) from exc
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)
