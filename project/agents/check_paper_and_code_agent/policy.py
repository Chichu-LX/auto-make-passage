from __future__ import annotations

import hashlib


MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_MANUSCRIPT_BYTES = 4 * 1024 * 1024
MAX_PDF_EXCERPT_CHARS = 5_000
MAX_ADVISORY_CONTEXT_CHARS = 900_000
MAX_REPORT_BYTES = 2 * 1024 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
