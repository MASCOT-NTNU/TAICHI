""" Unit test for planner

This module tests the planner object.

"""

from unittest import TestCase
from Planner.Planner import Planner


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        self.planner = Planner()

    def test_initial_indices(self):
        """ Test initial indices to be 0. """
        self.assertEqual(self.planner.get_next_index(), 0)
        self.assertEqual(self.planner.get_current_index(), 0)
        self.assertEqual(self.planner.get_pioneer_index(), 0)

    def test_set_indices(self):
        """ Test individual index setting function. """
        id_next = 10
        id_now = 4
        id_pion = 12
        self.planner.set_next_index(id_next)
        self.planner.set_current_index(id_now)
        self.planner.set_pioneer_index(id_pion)

        self.assertEqual(self.planner.get_next_index(), id_next)
        self.assertEqual(self.planner.get_current_index(), id_now)
        self.assertEqual(self.planner.get_pioneer_index(), id_pion)

    def test_update_planner(self):
        """ Test update planner method. """
        id_pion = self.planner.get_pioneer_index()
        id_next = self.planner.get_next_index()
        id_pion_new = 22
        self.planner.update_planner()
        self.planner.set_pioneer_index(id_pion_new)
        self.assertEqual(id_pion_new, self.planner.get_pioneer_index())
        self.assertEqual(id_pion, self.planner.get_next_index())
        self.assertEqual(id_next, self.planner.get_current_index())



