""" Unit test for planner

This module tests the planner object.

"""

import unittest
from src.Planner.Planner import Planner


class TestPlanner(unittest.TestCase):
    """ Test class for planner.

    """

    def test_planner_update_planner(self):
        """ Tests update planner

        """
        planner = Planner()
        planner.update_planner()
        actual = Planner._ind_next
        expected = 0
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main(exit=False)


