import unittest
from pathlib import Path

from research_agents.skills import AGENT_DIRECTORIES


class LayoutTests(unittest.TestCase):
    # Only project/ is published. The other four are local runtime output
    # directories that exist once a run has written to them, so a fresh clone
    # legitimately has none of them.
    BUSINESS_DIRECTORIES = {"code", "data", "design", "docs", "project", "reference"}
    LOCAL_BUILD_DIRECTORIES = {"output", "tmp"}
    ROOT_FILES = {
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
    }

    def test_workspace_has_only_expected_visible_entries(self):
        workspace = Path(__file__).resolve().parents[2]
        visible_directories = {
            item.name
            for item in workspace.iterdir()
            if item.is_dir()
            and not item.name.startswith(".")
            and item.name != "make paper-all skill"
        }
        self.assertIn("project", visible_directories)
        self.assertEqual(
            visible_directories
            - self.BUSINESS_DIRECTORIES
            - self.LOCAL_BUILD_DIRECTORIES,
            set(),
        )
        visible_files = {
            item.name
            for item in workspace.iterdir()
            if item.is_file() and not item.name.startswith(".")
        }
        self.assertEqual(visible_files - self.ROOT_FILES, set())

    def test_each_agent_has_implementation_and_skill_registry(self):
        workspace = Path(__file__).resolve().parents[2]
        agents_root = workspace / "project" / "agents"
        for directory in AGENT_DIRECTORIES.values():
            with self.subTest(agent=directory):
                agent_root = agents_root / directory
                self.assertTrue((agent_root / "agent.py").is_file())
                self.assertTrue((agent_root / "__init__.py").is_file())
                self.assertTrue((agent_root / "skills" / "registry.json").is_file())

    def test_core_pipeline_agents_are_complete_inspectable_packages(self):
        workspace = Path(__file__).resolve().parents[2]
        for directory in (
            "lead_agent",
            "download_paper_agent",
            "download_data_agent",
            "make_idea_agent",
            "idea_design_agent",
            "code_make_agent",
            "code_review_agent",
            "server_schedule_agent",
            "check_result_agent",
            "experiment_code_run_agent",
            "write_paper_agent",
            "check_paper_and_code_agent",
        ):
            with self.subTest(agent=directory):
                root = workspace / "project" / "agents" / directory
                self.assertTrue((root / "README.md").is_file())
                self.assertTrue((root / "policy.py").is_file())
                self.assertTrue((root / "config" / "agent.json").is_file())
                self.assertTrue((root / "prompts" / "system.md").is_file())
                self.assertTrue(any((root / "schemas").glob("*.schema.json")))


if __name__ == "__main__":
    unittest.main()
