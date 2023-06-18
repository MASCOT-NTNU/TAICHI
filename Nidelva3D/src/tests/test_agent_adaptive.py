""" Unit test for Agent.
This module tests the agent object.
"""

from unittest import TestCase
from Agents.Agent_adaptive import Agent
from numpy.testing import assert_array_equal
import numpy as np
import matplotlib.pyplot as plt


class TestAgent(TestCase):
    """
    Test class for agent module.
    """

    def setUp(self) -> None:
        # kernel = "GRF"
        # random_seed = 0
        # num_steps = 10
        # self.agent_grf = Agent(kernel=kernel, num_steps=num_steps, random_seed=random_seed, debug=True)

        kernel = "GMRF"
        random_seed = 0
        num_steps = 10
        self.agent_gmrf = Agent(kernel=kernel, num_steps=num_steps, random_seed=random_seed, debug=True)

    # def test_compare_gmrf_grf(self) -> None:
    #     wp_grf = self.agent_grf.myopic.waypoint_graph.get_waypoints()
    #     wp_gmrf = self.agent_gmrf.myopic.waypoint_graph.get_waypoints()
    #     self.assertTrue(np.all(wp_grf == wp_gmrf))
    #
    #     rotated_angle_grf = self.agent_grf.myopic.kernel.get_rotated_angle()
    #     rotated_angle_gmrf = self.agent_gmrf.myopic.kernel.get_rotated_angle()
    #     self.assertEqual(rotated_angle_grf, rotated_angle_gmrf)


    def test_run(self):
        kernel = "GMRF"
        self.agent_gmrf.run()
        ibv, vr, rmse = self.agent_gmrf.get_metrics()
