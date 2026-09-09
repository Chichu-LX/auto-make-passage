# Changelog

All notable changes are documented in this file. The project follows semantic versioning while it
remains in active development.

## [0.1.0] - 2026-09-09

### Added

- Fourteen auditable Agent roles covering the research lifecycle.
- Pydantic/JSON Schema contracts for dispatch, transitions, tasks, results, feedback, reviews, and
  durable snapshots.
- LangGraph-backed transition validation across 18 workflow states.
- Independent Specialist and Feedback human-review checkpoints.
- Content-addressed artifact storage, redacted process logs, and checkpoint recovery.
- Secure paper/data ingestion and guarded local/remote experiment execution.
- Evidence-grounded paper writing, deterministic paper/code/result consistency checking, and final
  venue-style review.
- Public documentation, contribution/security policies, CI, and release audit tooling.

### Changed

- Optional third-party Skills are now represented by source registries only. Their contents are no
  longer redistributed and remain inactive after an explicit local sync.

[0.1.0]: https://github.com/Chichu-LX/auto-make-passage/releases/tag/v0.1.0
