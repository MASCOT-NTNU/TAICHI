"""
TEST module for EDA analysis

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-06-21

"""

from unittest import TestCase
from EDA.EDA import Agent


class TestEDA(Agent):

    def setUp(self) -> None:
        self.agent = Agent()
        self.agent.run()

    def test_is_equal(self) -> None:
        a = 1
        b = 1
        self.assertEqual(a, b)
        print("hello")



