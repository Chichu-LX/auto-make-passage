from __future__ import annotations

import hashlib


MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_DATA_FILE_BYTES = 10 * 1024 * 1024 * 1024
MAX_CODE_FILE_BYTES = 120 * 1024
MAX_CODE_TOTAL_BYTES = 1024 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
