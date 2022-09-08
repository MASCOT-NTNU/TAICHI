""" Unit test for Agent zigzag.
This module tests the agent object.
"""

from unittest import TestCase
from Agents.Agent_zigzag import Agent


class TestAgent(TestCase):
    """
    Test class for agent module.
    """

    def setUp(self) -> None:
        self.agent = Agent()

    def test_run(self):
        self.agent.run()


