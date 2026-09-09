from __future__ import annotations

from agents.base import MockFeedbackAgent, MockSpecialistAgent, SpecialistAgent
from agents.check_paper_and_code_agent import CheckPaperAndCodeAgent
from agents.check_result_agent import CheckResultAgent
from agents.code_make_agent import CodeMakeAgent
from agents.code_review_agent import CodeReviewAgent
from agents.download_data_agent import DownloadDataAgent
from agents.download_paper_agent import DownloadPaperAgent
from agents.feedback_agent import FeedbackAgent
from agents.experiment_code_run_agent import ExperimentCodeRunAgent
from agents.make_idea_agent import MakeIdeaAgent
from agents.idea_design_agent import IdeaDesignAgent
from agents.paper_review_agent import PaperReviewAgent
from agents.server_schedule_agent import ServerScheduleAgent
from agents.write_paper_agent import WritePaperAgent
from .providers import JsonModel, SearchProvider
from .remote_execution import RemoteExecutionBackend, SafeSSHBackend
from .schemas import AgentName
from .storage import ArtifactStore


SPECIALIST_NAMES = [
    AgentName.DOWNLOAD_PAPER,
    AgentName.DOWNLOAD_DATA,
    AgentName.MAKE_IDEA,
    AgentName.IDEA_DESIGN,
    AgentName.CODE_MAKE,
    AgentName.CODE_REVIEW,
    AgentName.SERVER_SCHEDULE,
    AgentName.CHECK_RESULT,
    AgentName.EXPERIMENT_CODE_RUN,
    AgentName.WRITE_PAPER,
    AgentName.CHECK_PAPER_AND_CODE,
    AgentName.PAPER_REVIEW,
    AgentName.FEEDBACK,
]


def build_mock_registry() -> dict[AgentName, SpecialistAgent]:
    registry: dict[AgentName, SpecialistAgent] = {
        name: MockSpecialistAgent(name) for name in SPECIALIST_NAMES
    }
    registry[AgentName.FEEDBACK] = MockFeedbackAgent()
    return registry


def build_live_registry(
    model: JsonModel,
    search: SearchProvider,
    artifacts: ArtifactStore,
    *,
    remote_backend: RemoteExecutionBackend | None = None,
    review_model: JsonModel | None = None,
) -> dict[AgentName, SpecialistAgent]:
    backend = remote_backend or SafeSSHBackend()
    reviewer = review_model or model
    agents: list[SpecialistAgent] = [
        DownloadPaperAgent(search, model, artifacts),
        DownloadDataAgent(search, model, artifacts),
        MakeIdeaAgent(model, artifacts),
        IdeaDesignAgent(model, artifacts),
        CodeMakeAgent(model, artifacts),
        CodeReviewAgent(model, artifacts),
        ServerScheduleAgent(model, artifacts, backend=backend),
        CheckResultAgent(model, artifacts, backend=backend),
        ExperimentCodeRunAgent(model, artifacts, backend=backend),
        WritePaperAgent(model, artifacts),
        CheckPaperAndCodeAgent(model, artifacts),
        PaperReviewAgent(reviewer, artifacts),
        FeedbackAgent(reviewer, artifacts),
    ]
    return {agent.name: agent for agent in agents}
