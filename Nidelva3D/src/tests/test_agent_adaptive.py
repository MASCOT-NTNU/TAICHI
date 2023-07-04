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
        kernel = "GRF"
        random_seed = 9
        num_steps = 10
        debug = True
        self.agent_grf = Agent(kernel=kernel, num_steps=num_steps, random_seed=random_seed, debug=debug)

        kernel = "GMRF"
        self.agent_gmrf = Agent(kernel=kernel, num_steps=num_steps, random_seed=random_seed, debug=debug)

    # def test_compare_gmrf_grf(self) -> None:
    #     wp_grf = self.agent_grf.myopic.waypoint_graph.get_waypoints()
    #     wp_gmrf = self.agent_gmrf.myopic.waypoint_graph.get_waypoints()
    #     self.assertTrue(np.all(wp_grf == wp_gmrf))
    #
    #     rotated_angle_grf = self.agent_grf.myopic.kernel.get_rotated_angle()
    #     rotated_angle_gmrf = self.agent_gmrf.myopic.kernel.get_rotated_angle()
    #     self.assertEqual(rotated_angle_grf, rotated_angle_gmrf)

    def test_run(self):
        self.agent_gmrf.run()
        ibv_gmrf, vr_gmrf, rmse_gmrf = self.agent_gmrf.get_metrics()
        self.agent_grf.run()
        ibv_grf, vr_grf, rmse_grf = self.agent_grf.get_metrics()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(25, 8))
        plt.subplot(131)
        plt.plot(ibv_grf, label="GRF")
        plt.plot(ibv_gmrf, label="GMRF")
        plt.legend()
        plt.title("IBV")
        plt.subplot(132)
        plt.plot(vr_grf, label="GRF")
        plt.plot(vr_gmrf, label="GMRF")
        plt.legend()
        plt.title("VR")
        plt.subplot(133)
        plt.plot(rmse_grf, label="GRF")
        plt.plot(rmse_gmrf, label="GMRF")
        plt.legend()
        plt.title("RMSE")
        plt.show()

        ibv