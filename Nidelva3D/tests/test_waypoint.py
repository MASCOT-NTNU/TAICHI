""" Unit test for WaypointGraph
This module tests the planner object.

"""

from unittest import TestCase
from Nidelva3D.src.WaypointGraph.WaypointGraph import WaypointGraph
from Nidelva3D.src.usr_func.is_list_empty import is_list_empty
import numpy as np
from numpy import testing
import random
import math
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class TestWaypoint(TestCase):
    """ Test class waypoint graph.

    """

    def setUp(self) -> None:
        self.polygon_border = np.array([[0, 0],
                                        [1000, 0],
                                        [1000, 1000],
                                        [0, 1000]])

        self.polygon_obstacle = [[[]]]
        self.distance_neighbour = 120
        self.depths = [0., 1., 1.5]
        self.w = WaypointGraph()
        self.w.construct_waypoints_and_neighbours(self.polygon_border, self.polygon_obstacle, self.depths,
                                                  self.distance_neighbour)
        self.waypoints = self.w.get_waypoints()
        self.hash_neighbours = self.w.get_hash_neighbour()

    def test_empty_waypoints(self):
        actual_len = len(self.waypoints)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def test_illegal_waypoints(self):
        pb = Polygon(self.polygon_border)
        pos = []
        if not is_list_empty(self.polygon_obstacle):
            for po in self.polygon_obstacle:
                pos.append(Polygon(po))
        s = True
        for i in range(len(self.waypoints)):
            p = Point(self.waypoints[i, :2])
            if not pb.contains(p) or [l.contains(p) for l in pos]:
                s = False
                break
        self.assertTrue(s)

    def test_if_neighbours_are_legal(self):
        for i in range(len(self.hash_neighbours)):
            ind_n = self.hash_neighbours[i]
            w = self.waypoints[i]
            wn = self.waypoints[ind_n]
            # print(w, wn)
            # pass
            d = cdist(w.reshape(1, -1), wn)
            print(d)
            # print(self.waypoints[i])
        # print(len(self.hash_neighbours))
        # print("end")
        # dm = cdist(self.waypoints, self.waypoints)
        #
        # plt.imshow(dm)
        # plt.colorbar()
        # plt.show()
        #
        # print(dm)

    def test_depths(self):
        ud = np.unique(self.waypoints[:, 2])
        self.assertIsNone(testing.assert_array_almost_equal(ud, np.array(self.depths)))

    def test_waypoint_construction(self):
        """ Tests the construction of the waypoint

        """
        # pb = np.array([[0, 0],
        #                [100, 0],
        #                [100, 100],
        #                [0, 100]])

        # plt.plot(self.waypoints[:, 1], self.waypoints[:, 0], 'k.')
        # pb = np.vstack((self.polygon_border, self.polygon_border[0, :].reshape(1, -1)))
        # plt.plot(pb[:, 1], pb[:, 0], 'r-.')
        # plt.show()

        # pb = np.array([[0, 0],
        #                [100, 0],
        #                [110, 50],
        #                [200, 100],
        #                [100, 200],
        #                [20, 100],
        #                [-30, 200],
        #                [-100, 100],
        #                [-50, 50],
        #                [-10, 50]])




# if __name__ == "__main__":
#     unittest.main(exit=False)
    # t = TestWaypoint()
    # t.test_waypoint_construction()

# #%%
# from scipy.spatial.distance import cdist
# x1 = np.array([[0, 0]])
# x2 = np.array([[100, 100]])
# print("Distance", cdist(x1, x2, 'euclidean'))
# print("test", 100 * np.sqrt(2))

