""" Unit test for WaypointGraph
This module tests the planner object.

"""

from unittest import TestCase
from WaypointGraph import WaypointGraph
from usr_func.is_list_empty import is_list_empty
from numpy import testing
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from numpy import testing


class TestWaypoint(TestCase):
    """ Common test class for the waypoint graph module
    """

    def run_test_empty_waypoints(self):
        """ Test if it generates empty waypoint. """
        actual_len = len(self.waypoints)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def run_test_illegal_waypoints(self):
        """ Test if any waypoints are not within the border polygon or colliding with obstacles. """
        pb = Polygon(self.polygon_border)
        pos = []
        if not is_list_empty(self.polygon_obstacle):
            for po in self.polygon_obstacle:
                pos.append(Polygon(po))
        s = True

        for i in range(len(self.waypoints)):
            p = Point(self.waypoints[i, :2])
            in_border = pb.contains(p)
            in_obs = False
            for po in pos:
                if po.contains(p):
                    in_obs = True
                    break
            if in_obs or not in_border:
                s = False
                break
        self.assertTrue(s)

    def run_test_illegal_lateral_neighbours(self):
        """ Test if lateral neighbours are legal. """
        case = True
        ERROR_BUFFER = 1
        for i in range(len(self.hash_neighbours)):
            ind_n = self.hash_neighbours[i]
            w = self.waypoints[i, :2]
            wn = self.waypoints[ind_n, :2]
            d = cdist(w.reshape(1, -1), wn)
            if ((np.amax(d) > self.neighbour_distance + ERROR_BUFFER) or
                (np.amin(d) < self.neighbour_distance - ERROR_BUFFER)):
                case = False
        self.assertTrue(case, msg="Neighbour lateral distance is illegal!")

    def run_test_depths(self):
        """ Test if depths are properly generated. """
        ud = np.unique(self.waypoints[:, 2])
        self.assertIsNone(testing.assert_array_almost_equal(ud, np.array(self.depths)))

    def run_test_get_vector_between_two_waypoints(self):
        wp1 = [1, 2, 3]
        wp2 = [3, 4, 5]
        vec = self.wg.get_vector_between_two_waypoints(wp1, wp2)
        self.assertIsNone(testing.assert_array_equal(np.array([[wp2[0]-wp1[0]],
                                   [wp2[1]-wp1[1]],
                                   [wp2[2]-wp1[2]]]), vec))

    def run_test_get_waypoints_from_ind(self):
        # c1: empty ind
        wp = self.wg.get_waypoint_from_ind([])
        self.assertEqual(wp.shape[0], 0)
        # c2: one ind
        ids = 10
        wp = self.wg.get_waypoint_from_ind(ids).reshape(-1, 3)
        self.assertEqual(wp.shape[0], 1)
        # c3: multiple inds
        ids = [11, 13, 15]
        wp = self.wg.get_waypoint_from_ind(ids)
        self.assertEqual(wp.shape[0], len(ids))

    def run_test_get_ind_from_waypoints(self):
        """ Test waypoint interpolation works. Given random location, it should return indices for the nearest locations. """
        # c1: empty wp
        ind = self.wg.get_ind_from_waypoint([])
        self.assertIsNone(ind)

        xmin, ymin, zmin = map(np.amin, [self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]])
        xmax, ymax, zmax = map(np.amax, [self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]])

        # c2: one wp
        # s1: generate random waypoint
        xr = np.random.uniform(xmin, xmax)
        yr = np.random.uniform(ymin, ymax)
        zr = np.random.uniform(zmin, zmax)
        wp = np.array([xr, yr, zr])

        # s2: get indice from function
        ind = self.wg.get_ind_from_waypoint(wp)
        self.assertIsNotNone(ind)

        # s3: get waypoint from indice
        wr = self.waypoints[ind]

        # s4: compute distance between nearest waypoint and rest
        d = cdist(wr.reshape(1, -1), wp.reshape(1, -1))
        da = cdist(self.waypoints, wp.reshape(1, -1))

        # s5: see if it is nearest waypoint
        self.assertTrue(d, da.min())

        # c3: more than one wp
        # t = np.random.randint(0, len(self.waypoints))
        t = 10
        xr = np.random.uniform(xmin, xmax, t)
        yr = np.random.uniform(ymin, ymax, t)
        zr = np.random.uniform(zmin, zmax, t)
        wp = np.stack((xr, yr, zr), axis=1)

        ind = self.wg.get_ind_from_waypoint(wp)
        self.assertIsNotNone(ind)
        wr = self.waypoints[ind]
        d = np.diag(cdist(wr, wp))
        da = cdist(self.waypoints, wp)
        self.assertTrue(np.all(d == np.amin(da, axis=0)))

        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'k.', alpha=.1)
        for i in range(len(wp)):
            plt.plot([wp[i, 0], wr[i, 0]], [wp[i, 1], wr[i, 1]], 'r.-')
            # plt.plot(wr[i, 0], wr[i, 1], '.', alpha=.3)
        plt.show()

    def run_test_neighbours_plotting(self):
        import matplotlib.pyplot as plt

        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'k.')
        plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
        if not is_list_empty(self.polygon_obstacle):
            for p in self.polygon_obstacle:
               plt.plot(p[:, 0], p[:, 1], 'b-.')
            plt.show()

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
                size=10,
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
                size=10,
                color='blue',
                opacity=0.8
            )
        ))
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename='/Users/yaolin/Downloads/test.html', auto_open=True)


class TC1(TestWaypoint):

    def setUp(self) -> None:
        """
        """
        self.polygon_border = np.array([[0, 0],
                                       [100, 0],
                                       [110, 50],
                                       [200, 100],
                                       [100, 200],
                                       [20, 100],
                                       [-30, 200],
                                       [-100, 100],
                                       [-50, 50],
                                       [-10, 50]])
        self.polygon_obstacle = [np.array([[50, 50],
                                          [80, 80],
                                          [65, 90],
                                          [25, 70]]),
                                 np.array([[-25, 70],
                                          [0, 90],
                                          [-10, 100],
                                          [-50, 100]]),
                                 np.array([[100, 80],
                                           [150, 100],
                                           [90, 150],
                                           [70, 125]])]

        self.neighbour_distance = 12
        self.depths = [0., 1., 1.5, 3.5, 10]
        self.wg = WaypointGraph()
        self.wg.set_neighbour_distance(self.neighbour_distance)
        self.wg.set_depth_layers(self.depths)
        self.wg.set_polygon_border(self.polygon_border)
        self.wg.set_polygon_obstacles(self.polygon_obstacle)
        self.wg.construct_waypoints()
        self.wg.construct_hash_neighbours()
        self.waypoints = self.wg.get_waypoints()
        self.hash_neighbours = self.wg.get_hash_neighbour()

    def test_all(self):
        self.run_test_get_vector_between_two_waypoints()
        self.run_test_illegal_waypoints()
        self.run_test_get_ind_from_waypoints()
        self.run_test_illegal_lateral_neighbours()
        self.run_test_empty_waypoints()
        self.run_test_depths()
        self.run_test_neighbours_plotting()
        self.run_test_get_waypoints_from_ind()


class TC2(TestWaypoint):

    def setUp(self) -> None:
        """ setup parameters """
        self.polygon_border = np.array([[0, 0],
                                        [1000, 0],
                                        [1000, 1000],
                                        [0, 1000]])
        self.polygon_obstacle = [np.array([[200, 200],
                                  [400, 200],
                                  [400, 400],
                                  [200, 400]])]
        self.depths = [-1, 2, 10]
        self.neighbour_distance = 120

        self.wg = WaypointGraph()
        self.wg.set_neighbour_distance(self.neighbour_distance)
        self.wg.set_depth_layers(self.depths)
        self.wg.set_polygon_border(self.polygon_border)
        self.wg.set_polygon_obstacles(self.polygon_obstacle)
        self.wg.construct_waypoints()
        self.wg.construct_hash_neighbours()
        self.waypoints = self.wg.get_waypoints()
        self.hash_neighbours = self.wg.get_hash_neighbour()

    def test_all(self):
        self.run_test_get_vector_between_two_waypoints()
        self.run_test_illegal_waypoints()
        self.run_test_get_ind_from_waypoints()
        self.run_test_illegal_lateral_neighbours()
        self.run_test_empty_waypoints()
        self.run_test_depths()
        self.run_test_neighbours_plotting()
        self.run_test_get_waypoints_from_ind()


if __name__ == "__main__":
    # T1: test setup 1
    tc1 = TC1()
    tc1.test_all()

    # T2: test setup 2
    tc2 = TC2()
    tc2.test_all()




