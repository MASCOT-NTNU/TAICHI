""" Unit test for Myopic3D

This module tests the myopic3d object.

"""

from unittest import TestCase
from Planner.Myopic3D import Myopic3D
from WaypointGraph import WaypointGraph
import numpy as np


class TestMyopic(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        self.polygon_border = np.array([[0, 0],
                                        [1000, 0],
                                        [1000, 1000],
                                        [0, 1000]])
        self.polygon_obstacle = [[[]]]
        # self.polygon_obstacle = [np.array([[200, 200],
        #                           [400, 200],
        #                           [400, 400],
        #                           [200, 400]])]
        self.depths = [.5, 1., 1.5, 2., 2.5]
        self.neighbour_distance = 120
        self.wp = WaypointGraph()
        self.wp.set_neighbour_distance(self.neighbour_distance)
        self.wp.set_depth_layers(self.depths)
        self.wp.set_polygon_border(self.polygon_border)
        self.wp.set_polygon_obstacles(self.polygon_obstacle)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()
        self.waypoints = self.wp.get_waypoints()
        self.hash_neighbours = self.wp.get_hash_neighbour()

        self.myopic = Myopic3D(self.wp)

    def test_get_candidates(self):
        wp = self.myopic.get_next_waypoint()

        pass




