from __future__ import annotations

import hashlib


REPORT_INTERVAL_SECONDS = 12 * 60 * 60
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_TRANSFER_FILE_BYTES = 10 * 1024**3
MAX_RUNTIME_HELPER_BYTES = 256 * 1024
MAX_SCHEDULE_REPORT_BYTES = 512 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def environment_name(project_slug: str, task_id: str) -> str:
    prefix = project_slug.replace("-", "_")[:30].strip("_") or "research"
    return f"run_{prefix}_{file_token(task_id)[:8]}"[:63]
