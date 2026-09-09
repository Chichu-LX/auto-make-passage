from __future__ import annotations

import hashlib


MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_MANUSCRIPT_BYTES = 4 * 1024 * 1024
MAX_MODEL_CONTEXT_CHARS = 900_000
MAX_CODE_EXCERPT_CHARS = 1_500
MAX_CODE_SYMBOLS_IN_PROMPT = 160
MAX_PAPER_ABSTRACT_CHARS = 2_400


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
