from __future__ import annotations

import hashlib


REPORT_INTERVAL_SECONDS = 12 * 60 * 60
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_LOG_SNAPSHOT_BYTES = 64 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

