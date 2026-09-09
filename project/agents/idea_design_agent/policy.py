from __future__ import annotations

import hashlib


MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_MODEL_CONTEXT_BYTES = 220 * 1024
MAX_PAPERS_IN_CONTEXT = 50


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
