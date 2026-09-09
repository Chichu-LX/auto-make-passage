import unittest
from pathlib import Path

from research_agents.schemas import AgentName
from research_agents.skills import load_skill_reservations, validate_all_skill_reservations


class SkillReservationTests(unittest.TestCase):
    def test_every_agent_has_five_to_ten_github_skills(self):
        workspace_root = Path(__file__).resolve().parents[2]
        counts = validate_all_skill_reservations(workspace_root)
        self.assertEqual(len(counts), len(AgentName))
        self.assertTrue(all(5 <= count <= 10 for count in counts.values()))

    def test_public_registries_reserve_skills_without_redistributing_them(self):
        workspace_root = Path(__file__).resolve().parents[2]
        for agent in AgentName:
            with self.subTest(agent=agent.value):
                entries = load_skill_reservations(workspace_root, agent)
                self.assertTrue(all(entry["status"] == "reserved" for entry in entries))
                self.assertTrue(all(not entry.get("local_path") for entry in entries))
                self.assertTrue(all(entry.get("source_commit") for entry in entries))


if __name__ == "__main__":
    unittest.main()
