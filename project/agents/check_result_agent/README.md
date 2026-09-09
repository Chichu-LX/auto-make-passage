# Check Result Agent

This Agent monitors the immutable jobs launched by Server Schedule. A running job is represented by `AgentResult.status = in_progress`; that status remains in `RESULT_CHECK`, appends a report artifact, and does not consume a failure retry. The normal production hook then pauses for user evaluation.

Checks are due at expected completion or after at most 12 hours. If a prior report's `next_report_at` is still in the future, the Agent reuses it without polling the server. Long experiments can therefore continue for any number of 12-hour cycles without a busy loop or an arbitrary retry cap.

A passed report advances to Experiment Code Run, not directly to paper writing. That next gate audits and runs every approved non-main computational experiment only after the main metric objective has been met.

Each actual poll records per-job status, exit code, bounded log tail, all finite declared metric observations, the per-job best observation, and the remote best-checkpoint path. Local log snapshots live under `project/var/logs/jobs/<run_id>/`; result JSON and Markdown reports live under `design/<project_slug>/reviews/results/`.

Successful completion requires every job, seed coverage, and the project metric threshold. Failed or unsatisfactory results route to the smallest allowlisted repair stage. User dissatisfaction at the review hook may explicitly choose Server Schedule, Check Result, Code Make, Download Data, Idea Design, or Make Idea as permitted by the Runtime.
