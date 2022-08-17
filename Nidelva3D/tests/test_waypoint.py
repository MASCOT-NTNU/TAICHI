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
        self.neighbour_distance = 120
        self.depths = [0., 1., 1.5]
        self.wg = WaypointGraph()
        self.wg.set_neighbour_distance(self.neighbour_distance)
        self.wg.set_depth_layers(self.depths)
        self.wg.set_polygon_border(self.polygon_border)
        self.wg.set_polygon_obstacles(self.polygon_obstacle)
        self.wg.construct_waypoints()
        self.wg.construct_hash_neighbours()
        self.waypoints = self.wg.get_waypoints()
        self.hash_neighbours = self.wg.get_hash_neighbour()

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
        case = True
        for i in range(len(self.hash_neighbours)):
            ind_n = self.hash_neighbours[i]
            w = self.waypoints[i]
            wn = self.waypoints[ind_n]
            d = cdist(w.reshape(1, -1), wn)
            if (np.amax(d) > self.neighbour_distance + 1) or (np.amin(d) < self.neighbour_distance - 1):
                case = False
        self.assertTrue(case)

    def test_neighbours_plotting(self):
        import plotly
        import plotly.graph_objects as go
        ind_r = np.random.randint(0, self.waypoints.shape[0])

        fig = go.Figure(data=[go.Scatter3d(
            x=self.waypoints[:, 0],
            y=self.waypoints[:, 1],
            z=self.waypoints[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='black',
                opacity=0.8
            )
        )])

        fig.add_trace(go.Scatter3d(
            x=[self.waypoints[ind_r, 0]],
            y=[self.waypoints[ind_r, 1]],
            z=[self.waypoints[ind_r, 2]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                opacity=0.8
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=self.waypoints[self.hash_neighbours[ind_r], 0],
            y=self.waypoints[self.hash_neighbours[ind_r], 1],
            z=self.waypoints[self.hash_neighbours[ind_r], 2],
            mode='markers',
            marker=dict(
                size=20,
                color='blue',
                opacity=0.8
            )
        ))

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename='/Users/yaolin/Downloads/test.html', auto_open=True)
        pass

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


