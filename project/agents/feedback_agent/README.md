# Feedback Agent

Feedback is the read-only workflow coordinator. After every specialist step, Runtime supplies a
redacted summary of `project/logging/_workflow/<run_id>/timeline.jsonl`, the source result status, and
an explicit retry-target allowlist.

Feedback returns a typed `FeedbackControl` plus a persisted progress/risk/next-action report:
`continue`, `retry`, `request_human`, or `abort`. It has no dispatch, state mutation, research-file
edit, shell, network, or server authority. Runtime validates the
target against the source workflow gate and the three-attempt budget before changing state. Invalid
Feedback output fails closed to `NEEDS_HUMAN`.

Every Feedback result receives its own Runtime user-review checkpoint after the specialist review.
Feedback cannot approve its own `continue` recommendation. A normal `user_review_requested` event uses the
log status `blocked` only to represent an intentional pause and must not be treated as an Agent
failure.
