import unittest
from pathlib import Path

from agents.download_data_agent.agent import DownloadDataAgent
from agents.download_paper_agent.agent import DownloadPaperAgent
from agents.code_make_agent.agent import CodeMakeAgent
from agents.code_review_agent.agent import CodeReviewAgent
from agents.lead_agent.agent import DeepSeekLeadAgent
from agents.idea_design_agent.agent import IdeaDesignAgent
from agents.feedback_agent.agent import FeedbackAgent
from agents.make_idea_agent.agent import MakeIdeaAgent
from agents.server_schedule_agent.agent import ServerScheduleAgent
from agents.check_result_agent.agent import CheckResultAgent
from agents.experiment_code_run_agent.agent import ExperimentCodeRunAgent
from agents.write_paper_agent.agent import WritePaperAgent
from agents.check_paper_and_code_agent.agent import CheckPaperAndCodeAgent
from agents.paper_review_agent.agent import PaperReviewAgent
from research_agents.agent_package import load_agent_package
from research_agents.schemas import AgentName


class AgentPackageTests(unittest.TestCase):
    def test_config_and_prompt_are_runtime_loaded(self):
        project = Path(__file__).resolve().parents[1]
        cases = (
            ("lead_agent", AgentName.LEAD, DeepSeekLeadAgent),
            ("download_paper_agent", AgentName.DOWNLOAD_PAPER, DownloadPaperAgent),
            ("download_data_agent", AgentName.DOWNLOAD_DATA, DownloadDataAgent),
            ("make_idea_agent", AgentName.MAKE_IDEA, MakeIdeaAgent),
            ("idea_design_agent", AgentName.IDEA_DESIGN, IdeaDesignAgent),
            ("code_make_agent", AgentName.CODE_MAKE, CodeMakeAgent),
            ("code_review_agent", AgentName.CODE_REVIEW, CodeReviewAgent),
            ("server_schedule_agent", AgentName.SERVER_SCHEDULE, ServerScheduleAgent),
            ("check_result_agent", AgentName.CHECK_RESULT, CheckResultAgent),
            ("experiment_code_run_agent", AgentName.EXPERIMENT_CODE_RUN, ExperimentCodeRunAgent),
            ("write_paper_agent", AgentName.WRITE_PAPER, WritePaperAgent),
            ("check_paper_and_code_agent", AgentName.CHECK_PAPER_AND_CODE, CheckPaperAndCodeAgent),
            ("paper_review_agent", AgentName.PAPER_REVIEW, PaperReviewAgent),
            ("feedback_agent", AgentName.FEEDBACK, FeedbackAgent),
        )
        for directory, name, implementation in cases:
            with self.subTest(agent=name.value):
                package = load_agent_package(project / "agents" / directory / "agent.py")
                self.assertEqual(package.config.name, name)
                self.assertEqual(
                    tuple(package.config.allowed_tools), implementation.allowed_tools
                )
                self.assertGreater(len(package.system_prompt), 100)

    def test_lead_package_has_no_tools_or_write_scope(self):
        project = Path(__file__).resolve().parents[1]
        package = load_agent_package(project / "agents" / "lead_agent" / "agent.py")
        self.assertEqual(package.config.allowed_tools, [])
        self.assertEqual(package.config.write_scopes, [])


if __name__ == "__main__":
    unittest.main()
