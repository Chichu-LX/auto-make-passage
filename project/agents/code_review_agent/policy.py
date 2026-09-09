from __future__ import annotations

import hashlib


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_CODE_FILE_BYTES = 120 * 1024
MAX_SOURCE_FILE_BYTES = 50 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 512 * 1024 * 1024
EXECUTION_TIMEOUT_SECONDS = 30
OUTPUT_TAIL_CHARS = 8000
MAX_REVIEW_ITERATIONS = 5


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

