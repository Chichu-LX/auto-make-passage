#!/usr/bin/env python3
"""Export Pydantic contracts into each independently inspectable Agent package."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from research_agents.schemas import (
    AgentResult,
    AdvisorReport,
    CodeGenerationDraft,
    CodeRevisionPatchDraft,
    CodeImplementationManifest,
    CodeReviewReport,
    DatasetManifest,
    DatasetSelection,
    DispatchDecision,
    FeedbackControl,
    FeedbackCoordinationReport,
    LeadTransitionDecision,
    LeadTransitionOption,
    ExperimentCodeRunManifest,
    ExperimentCodeRunReport,
    ExperimentCoverageAudit,
    ExperimentCoverageDraft,
    IdeaDraft,
    IdeaProposal,
    PaperManifest,
    PaperManuscriptManifest,
    PaperReviewDraft,
    PaperReviewReport,
    PaperWritingDraft,
    PaperConsistencyAdvisoryDraft,
    PaperCodeConsistencyReport,
    PaperSelection,
    ResearchDesign,
    ResearchDesignDraft,
    ResultCheckReport,
    ServerExecutionConfig,
    ServerScheduleManifest,
    ProcessLogRecord,
    RunSnapshot,
    TaskEnvelope,
    UserEvaluation,
    UserReviewRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS: dict[str, list[type[BaseModel]]] = {
    "lead_agent": [
        RunSnapshot,
        DispatchDecision,
        LeadTransitionOption,
        LeadTransitionDecision,
        UserReviewRequest,
        UserEvaluation,
    ],
    "download_paper_agent": [TaskEnvelope, PaperSelection, PaperManifest, AgentResult],
    "download_data_agent": [TaskEnvelope, ResearchDesign, DatasetSelection, DatasetManifest, AgentResult],
    "make_idea_agent": [TaskEnvelope, PaperManifest, IdeaDraft, IdeaProposal, AgentResult],
    "idea_design_agent": [TaskEnvelope, IdeaProposal, PaperManifest, ExperimentCoverageAudit, AdvisorReport, ResearchDesignDraft, ResearchDesign, AgentResult],
    "code_make_agent": [TaskEnvelope, ResearchDesign, DatasetManifest, CodeGenerationDraft, CodeRevisionPatchDraft, CodeImplementationManifest, AgentResult],
    "code_review_agent": [TaskEnvelope, ResearchDesign, CodeImplementationManifest, CodeReviewReport, AgentResult],
    "server_schedule_agent": [
        TaskEnvelope,
        ResearchDesign,
        DatasetManifest,
        CodeImplementationManifest,
        CodeReviewReport,
        ServerExecutionConfig,
        ServerScheduleManifest,
        AgentResult,
    ],
    "check_result_agent": [
        TaskEnvelope,
        ResearchDesign,
        ServerExecutionConfig,
        ServerScheduleManifest,
        ResultCheckReport,
        AgentResult,
    ],
    "experiment_code_run_agent": [
        TaskEnvelope,
        ResearchDesign,
        CodeImplementationManifest,
        CodeReviewReport,
        ServerExecutionConfig,
        ServerScheduleManifest,
        ResultCheckReport,
        ExperimentCoverageDraft,
        ExperimentCoverageAudit,
        ExperimentCodeRunManifest,
        ExperimentCodeRunReport,
        AgentResult,
    ],
    "write_paper_agent": [
        TaskEnvelope,
        PaperManifest,
        ResearchDesign,
        DatasetManifest,
        CodeImplementationManifest,
        CodeReviewReport,
        ServerScheduleManifest,
        ResultCheckReport,
        ExperimentCodeRunManifest,
        ExperimentCodeRunReport,
        PaperWritingDraft,
        PaperManuscriptManifest,
        AgentResult,
    ],
    "check_paper_and_code_agent": [
        TaskEnvelope,
        PaperManuscriptManifest,
        PaperConsistencyAdvisoryDraft,
        PaperCodeConsistencyReport,
        AgentResult,
    ],
    "paper_review_agent": [
        TaskEnvelope,
        PaperManuscriptManifest,
        PaperCodeConsistencyReport,
        PaperReviewDraft,
        PaperReviewReport,
        AgentResult,
    ],
    "feedback_agent": [
        TaskEnvelope,
        ProcessLogRecord,
        FeedbackControl,
        FeedbackCoordinationReport,
        AgentResult,
    ],
}


def main() -> int:
    for agent_directory, models in EXPORTS.items():
        target = PROJECT_ROOT / "agents" / agent_directory / "schemas"
        target.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        for model in models:
            filename = f"{model.__name__}.schema.json"
            expected.add(filename)
            (target / filename).write_text(
                json.dumps(
                    model.model_json_schema(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        for existing in target.glob("*.schema.json"):
            if existing.name not in expected:
                existing.unlink()
        print(f"{agent_directory}: {len(models)} schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
