from __future__ import annotations

import hashlib


MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 5
MAX_PAPER_TEXT_CHARS = 3_000
MAX_MODEL_CONTEXT_BYTES = 64 * 1024


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
