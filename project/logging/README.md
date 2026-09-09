# Agent Process Logging

Runtime writes redacted, append-only JSONL process logs here:

```text
project/logging/
├── _workflow/<run_id>/timeline.jsonl
├── lead-agent/<run_id>/process.jsonl
├── download-paper-agent/<run_id>/process.jsonl
├── make-idea-agent/<run_id>/process.jsonl
├── idea-design-agent/<run_id>/process.jsonl
├── download-data-agent/<run_id>/process.jsonl
├── ...
└── feedback-agent/<run_id>/process.jsonl
```

Each record contains timestamp, run/task identity, Agent, event, status, workflow state/version,
attempt, duration, and bounded metadata such as artifact kinds and issue codes. Prompt bodies, paper
text, dataset rows, result payloads, authorization headers, and API keys are not logged. Sensitive
keys and common credential patterns are redacted before disk writes.

Feedback Agent receives a bounded redacted summary of the workflow timeline after every specialist
step. It may recommend `continue`, an allowlisted `retry`, `request_human`, or `abort`. Only Runtime
can apply the control. Ordinary research gates share a three-attempt budget and Code Review allows
five iterations. Server Schedule, Check Result, and Experiment Code Run are intentionally
count-unbounded: their resource use is constrained by the user-review hook and persistent 12-hour
reporting windows. An `in_progress` result is logged as informational, remains in `RESULT_CHECK`
or `EXPERIMENT_CODE_RUN`, and consumes no failure attempt. Experiment Code Run additionally logs
its main-result gate, coverage audit, launch, and monitoring phases; per-job snapshots remain under
`project/var/logs/jobs/<run_id>/experiment_code_run/` rather than in the process log.

Write Paper additionally logs evidence resolution, model synthesis, formula binding, deterministic
rendering, and the three output hashes. Check Paper and Code logs source resolution, independent
deterministic reconstruction, bounded qualitative advisory, and report persistence. The process log
contains counts, issue codes, routes, URIs, and hashes only; manuscript text, PDF excerpts, source
excerpts, and experimental values remain in their content-addressed artifacts and are not copied into
the logging tree.

After Feedback, Runtime invokes the after-step user-review hook and persists
`AWAITING_USER_REVIEW`. The Lead log and workflow timeline record `user_review_requested` and
`user_evaluation_received`. The full user comment is stored only in the ignored run snapshot so it
can be forwarded to the allowlisted repair task; process logs contain only verdict, rating, comment
length, and selected target. A `satisfied` verdict accepts the staged state. A `needs_revision`
verdict rolls back staged artifact references/completion markers and executes at most one repair
Agent before pausing again.

Generated `.jsonl` files are intentionally ignored by Git but remain on disk for local monitoring.
If a log cannot be written, Runtime stops instead of continuing without observability.
