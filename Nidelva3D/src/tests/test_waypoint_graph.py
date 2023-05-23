"""
Unit tests for WaypointGraph class.
"""
from unittest import TestCase
from WaypointGraph import WaypointGraph
from usr_func.is_list_empty import is_list_empty
import numpy as np
import matplotlib.pyplot as plt
from numpy import testing
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point


class TestWaypointGraph(TestCase):

    def setUp(self) -> None:
        self.neighbour_distance = 32
        self.depths = np.array([0.0, .5, 1.5])
        self.polygon_border = np.array([[0, 0], [0, 1000], [1000, 1000], [1000, 0]])
        self.polygon_obstacles = [np.array([[100, 100], [100, 200], [200, 200], [200, 100]]),
                                  np.array([[800, 800], [800, 900], [900, 900], [900, 800]])]
        self.wg = WaypointGraph(neighbour_distance=self.neighbour_distance, depths=self.depths,
                                polygon_border=self.polygon_border, polygon_obstacles=self.polygon_obstacles )
        self.waypoints = self.wg.get_waypoints()
        self.neighbour_hash_table = self.wg.get_neighbour_hash_table()

    def test_construct_waypoint(self) -> None:
        wp = self.wg.get_waypoints()
        plt.scatter(wp[:, 1], wp[:, 0])
        plt.show()

    def test_empty_waypoints(self):
        """ Test if it generates empty waypoint. """
        actual_len = len(self.waypoints)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def test_illegal_waypoints(self):
        """ Test if any waypoints are not within the border polygon or colliding with obstacles. """
        polygon_border_shapely = Polygon(self.polygon_border)
        polygon_obstacles_shapely = []
        if not is_list_empty(self.polygon_obstacles):
            for polygon_obstacle in self.polygon_obstacles:
                polygon_obstacles_shapely.append(Polygon(polygon_obstacle))
        legal = True
        for i in range(len(self.waypoints)):
            p = Point(self.waypoints[i, :2])
            in_border = polygon_border_shapely.contains(p)
            in_obstacle = False
            for polygon_obstacle in polygon_obstacles_shapely:
                if polygon_obstacle.contains(p):
                    in_obstacle = True
                    break
            if in_obstacle or not in_border:
                legal = False
                break
        self.assertTrue(legal)

    def test_illegal_lateral_neighbours(self):
        """ Test if lateral neighbours are legal. """
        legal = True
        ERROR_BUFFER = 1
        for i in range(len(self.neighbour_hash_table)):
            ind_n = self.neighbour_hash_table[i]
            w = self.waypoints[i, :2]
            wn = self.waypoints[ind_n, :2]
            d = cdist(w.reshape(1, -1), wn)
            if ((np.amax(d) > self.neighbour_distance + ERROR_BUFFER) or
                (np.amin(d) < self.neighbour_distance - ERROR_BUFFER)):
                legal = False
        self.assertTrue(legal, msg="Neighbour lateral distance is illegal!")

    def test_depths(self):
        """ Test if depths are properly generated. """
        ud = np.unique(self.waypoints[:, 2])
        self.assertIsNone(testing.assert_array_almost_equal(ud, np.array(self.depths)))

    def test_get_vector_between_two_waypoints(self):
        wp1 = [1, 2, 3]
        wp2 = [3, 4, 5]
        vec = self.wg.get_vector_between_two_waypoints(wp1, wp2)
        self.assertIsNone(testing.assert_array_equal(np.array([[wp2[0]-wp1[0]],
                                                               [wp2[1]-wp1[1]],
                                                               [wp2[2]-wp1[2]]]), vec))

    def test_get_waypoints_from_ind(self):
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

    def test_get_ind_from_waypoints(self):
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

    def test_neighbours_plotting(self):
        import matplotlib.pyplot as plt

        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'k.')
        plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
        if not is_list_empty(self.polygon_obstacles):
            for p in self.polygon_obstacles:
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
            x=self.waypoints[self.neighbour_hash_table[ind_r], 0],
            y=self.waypoints[self.neighbour_hash_table[ind_r], 1],
            z=self.waypoints[self.neighbour_hash_table[ind_r], 2],
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