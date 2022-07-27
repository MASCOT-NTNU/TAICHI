""" Unit test for WaypointGraph
This module tests the planner object.

"""

import unittest
from Nidelva3D.src.WaypointGraph.WaypointGraph import WaypointGraph
from Nidelva3D.src.usr_func.is_list_empty import is_list_empty
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from shapely.geometry import Polygon, Point


class TestWaypoint(unittest.TestCase):
    """ Test class waypoint graph.

    """

    def setUp(self) -> None:
        self.pb = np.array([[0, 0],
                            [1000, 0],
                            [1000, 1000],
                            [0, 1000]])
        self.po = [[[]]]
        self.dn = 120
        self.depths = [0., 1., 1.5]
        self.w = WaypointGraph()
        self.w.construct_waypoints(self.pb, self.po, self.depths, self.dn)
        self.locs = self.w.get_waypoints()

    def test_empty_waypoints(self):
        actual_len = len(self.locs)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def test_illegal_waypoints(self):
        pb = Polygon(self.pb)
        pos = []
        if not is_list_empty(self.po):
            for po in self.po:
                pos.append(Polygon(po))
        s = True
        for i in range(len(self.locs)):
            p = Point(self.locs[i, :2])
            if not pb.contains(p) or [l.contains(p) for l in pos]:
                s = False
                break
        self.assertTrue(s)

    def test_equal_distance(self):
        pass

    def test_waypoint_construction(self):
        """ Tests the construction of the waypoint

        """
        # pb = np.array([[0, 0],
        #                [100, 0],
        #                [100, 100],
        #                [0, 100]])

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


if __name__ == "__main__":
    unittest.main(exit=False)
    # t = TestWaypoint()
    # t.test_waypoint_construction()


#%%
# a = [[],[]]

a = [[[]]]

# if not any(a):
if not bool(a):
    print('List is empty')
else:
    print('List is not empty')

