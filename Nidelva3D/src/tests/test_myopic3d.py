""" Unit test for Myopic3D

This module tests the myopic3d object.

"""

from unittest import TestCase
from WGS import WGS
from numpy import testing
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from Planner.Myopic3D import Myopic3D
from WaypointGraph import WaypointGraph
import numpy as np
import os


class TestMyopic(TestCase):
    """ Common test class for myopic3d planner module.
    """

    def setUp(self) -> None:
        box = np.load(os.getcwd() + "/GMRF/models/grid.npy")
        polygon = box[:, 2:]
        polygon_xy = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
        polygon_b = sort_polygon_vertices(polygon_xy)

        self.polygon_border = polygon_b
        self.polygon_obstacle = [[[]]]
        self.depths = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
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
        """
        Test if candidate locations are legal or not.
        """
        cand = True
        # c1: one waypoint.
        self.myopic.set_current_index(10)
        id_curr = self.myopic.get_current_index()
        id_neigh = self.wp.get_ind_neighbours(id_curr)
        self.myopic.set_previous_index(id_neigh[0])
        id_prev = self.myopic.get_previous_index()

        wp_prev = self.wp.get_waypoint_from_ind(id_prev)
        wp_curr = self.wp.get_waypoint_from_ind(id_curr)
        vec1 = WaypointGraph.get_vector_between_two_waypoints(wp_prev, wp_curr)

        id_c, id_n = self.myopic.get_candidates()
        for i in range(len(id_c)):
            wp_i = self.wp.get_waypoint_from_ind(id_c[i])
            vec2 = WaypointGraph.get_vector_between_two_waypoints(wp_curr, wp_i)
            dot_prod = vec1.T @ vec2
            if dot_prod < 0:
                cand = False
                break
        self.assertTrue(cand)

        # c2: some random combinations, to make test fast.
        id_currs = np.random.randint(0, len(self.waypoints), 100)
        for i in range(len(id_currs)):
            self.myopic.set_current_index(id_currs[i])
            id_curr = self.myopic.get_current_index()
            wp_curr = self.waypoints[id_curr]
            id_neigh = self.wp.get_ind_neighbours(id_curr)
            for j in range(len(id_neigh)):
                self.myopic.set_previous_index(id_neigh[j])
                id_c, id_n = self.myopic.get_candidates()
                wp_prev = self.wp.get_waypoint_from_ind(self.myopic.get_previous_index())
                vec1 = WaypointGraph.get_vector_between_two_waypoints(wp_prev, wp_curr)
                for k in range(len(id_c)):
                    wp_i = self.wp.get_waypoint_from_ind(id_c[k])
                    vec2 = WaypointGraph.get_vector_between_two_waypoints(wp_curr, wp_i)
                    dot_prod = vec1.T @ vec2
                    if dot_prod < 0:
                        cand = False
                        break
        self.assertTrue(cand, msg="Candidates have illegal combinations, please check!")

        # c3: all possible combinations.
        # for i in range(len(self.waypoints)):
        #     self.myopic.set_current_index(i)
        #     id_curr = self.myopic.get_current_index()
        #     wp_curr = self.waypoints[id_curr]
        #     id_neigh = self.wp.get_ind_neighbours(id_curr)
        #     for j in range(len(id_neigh)):
        #         self.myopic.set_previous_index(id_neigh[j])
        #         id_c, id_n = self.myopic.get_candidates()
        #         wp_prev = self.wp.get_waypoint_from_ind(self.myopic.get_previous_index())
        #         vec1 = WaypointGraph.get_vector_between_two_waypoints(wp_prev, wp_curr)
        #         for k in range(len(id_c)):
        #             wp_i = self.wp.get_waypoint_from_ind(id_c[k])
        #             vec2 = WaypointGraph.get_vector_between_two_waypoints(wp_curr, wp_i)
        #             dot_prod = vec1.T @ vec2
        #             if dot_prod < 0:
        #                 cand = False
        #                 break
        # self.assertTrue(cand)

    def test_get_next_waypoint(self):
        """
        Test if next waypoint is properly selected.
        """
        # c1: one waypoint
        self.myopic.set_current_index(100)
        id_curr = self.myopic.get_current_index()
        id_neigh = self.wp.get_ind_neighbours(id_curr)
        self.myopic.set_previous_index(id_neigh[0])
        id_c, id_n = self.myopic.get_candidates()
        wp_cand = self.wp.get_waypoint_from_ind(id_c)
        wp_next = self.myopic.get_next_waypoint()
        id_next = np.where((wp_cand[:, 0] == wp_next[0]) *
                           (wp_cand[:, 1] == wp_next[1]) *
                           (wp_cand[:, 2] == wp_next[2]))[0]
        eibv_cand = self.myopic.gmrf.get_eibv_at_locations(wp_cand)
        id_min = np.argmin(eibv_cand)
        self.assertEqual(id_min, id_next)


