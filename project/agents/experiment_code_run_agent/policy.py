from __future__ import annotations

import hashlib


REPORT_INTERVAL_SECONDS = 12 * 60 * 60
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 512 * 1024
MAX_JOB_SPEC_BYTES = 128 * 1024
MAX_LOG_SNAPSHOT_BYTES = 64 * 1024
MAX_SUPPLEMENTARY_JOBS = 1500


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
