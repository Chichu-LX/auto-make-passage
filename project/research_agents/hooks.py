from __future__ import annotations

from typing import Protocol

from .schemas import RunSnapshot, UserEvaluation, UserReviewRequest


class AfterAgentStepHook(Protocol):
    """Observe an after-step checkpoint and optionally return a user verdict."""

    def evaluate(
        self,
        snapshot: RunSnapshot,
        review: UserReviewRequest,
    ) -> UserEvaluation | None: ...


class PauseForUserReviewHook:
    """Production default: persist the checkpoint and wait for a real user verdict."""

    def evaluate(
        self,
        snapshot: RunSnapshot,
        review: UserReviewRequest,
    ) -> UserEvaluation | None:
        return None


class AutoApproveUserReviewHook:
    """Explicit test/demo hook; never used by live execution unless requested."""

    def evaluate(
        self,
        snapshot: RunSnapshot,
        review: UserReviewRequest,
    ) -> UserEvaluation | None:
        return UserEvaluation(
            run_id=snapshot.run_id,
            review_id=review.review_id,
            observed_state_version=snapshot.version,
            verdict="satisfied",
            rating=5,
            comments="Explicit automatic approval for deterministic test or demo execution.",
        )
