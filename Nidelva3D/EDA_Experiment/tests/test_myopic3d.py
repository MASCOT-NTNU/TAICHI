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
import time


class TestMyopic(TestCase):
    """ Common test class for myopic3d planner module.
    """

    def setUp(self) -> None:
        self.myopic = Myopic3D()
        self.wp = self.myopic.wp
        self.waypoints = self.wp.get_waypoints()

    def test_get_candidates(self):
        """
        Test if candidate locations are legal or not.
        """
        cand = True
        # c1: one waypoint.
        self.myopic.set_next_index(10)
        id_next = self.myopic.get_next_index()
        id_neigh = self.wp.get_ind_neighbours(id_next)
        id_curr = id_neigh[0]
        # id_curr = id_neigh[np.random.randint(0, len(id_neigh))]
        wp_curr = self.wp.get_waypoint_from_ind(id_curr)
        wp_next = self.wp.get_waypoint_from_ind(id_next)
        vec1 = WaypointGraph.get_vector_between_two_waypoints(wp_curr, wp_next)
        id_c, id_n = self.myopic.get_candidates_indices()
        for i in range(len(id_c)):
            wp_i = self.wp.get_waypoint_from_ind(id_c[i])
            vec2 = WaypointGraph.get_vector_between_two_waypoints(wp_next, wp_i)
            dot_prod = vec1.T @ vec2
            if dot_prod < 0:
                cand = False
                break
        self.assertTrue(cand)

        # c2: some random combinations, to make test fast.
        id_nexts = np.random.randint(0, len(self.waypoints), 100)
        for i in range(len(id_nexts)):
            self.myopic.set_next_index(id_nexts[i])
            id_next = self.myopic.get_next_index()
            wp_next = self.waypoints[id_next]
            id_neigh = self.wp.get_ind_neighbours(id_next)
            for j in range(len(id_neigh)):
                self.myopic.set_current_index(id_neigh[j])
                id_c, id_n = self.myopic.get_candidates_indices()
                wp_curr = self.wp.get_waypoint_from_ind(self.myopic.get_current_index())
                vec1 = WaypointGraph.get_vector_between_two_waypoints(wp_curr, wp_next)
                for k in range(len(id_c)):
                    wp_i = self.wp.get_waypoint_from_ind(id_c[k])
                    vec2 = WaypointGraph.get_vector_between_two_waypoints(wp_next, wp_i)
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

    # def test_get_pioneer_waypoint(self):
    #     """
    #     Test if next waypoint is properly selected.
    #     """
    #     # c1: one waypoint
    #     self.myopic.set_next_index(100) # set random index
    #     id_next = self.myopic.get_next_index()
    #     id_neigh = self.wp.get_ind_neighbours(id_next)
    #     self.myopic.set_current_index(id_neigh[0])
    #     id_c, id_n = self.myopic.get_candidates_indices()
    #     wp_cand = self.wp.get_waypoint_from_ind(id_c)
    #     id_pion = self.myopic.get_pioneer_waypoint_index()
    #     wp_pioneer = self.wp.get_waypoint_from_ind(id_pion)
    #     id_next = np.where((wp_cand[:, 0] == wp_pioneer[0]) *
    #                        (wp_cand[:, 1] == wp_pioneer[1]) *
    #                        (wp_cand[:, 2] == wp_pioneer[2]))[0]
    #     eibv_cand = self.myopic.gmrf.get_eibv_at_locations(wp_cand)
    #     id_min = np.argmin(eibv_cand) # not working because the uncertainity in SPDE.
    #     # self.assertEqual(id_min, id_next)

    def test_planning_two_step_ahead(self):
        """
        Test the planning mechanisms for the workflow.
        """
        # c1: start the operation from scratch.
        id_start = np.random.randint(0, len(self.waypoints))
        id_curr = id_start

        # s1: setup the planner -> only once
        self.myopic.set_current_index(id_curr)
        self.myopic.set_next_index(id_curr)

        # s2: get next waypoint using get_pioneer_waypoint
        id_next = self.myopic.get_pioneer_waypoint_index()

        # s3: update planner -> so curr and next waypoint is updated, and trajectory is expanded.
        self.myopic.update_planner()

        # s4: get pioneer waypoint
        id_pion = self.myopic.get_pioneer_waypoint_index()

        for i in range(50):
            print(i)
            t1 = time.time()
            # ss1: update planner
            self.myopic.update_planner()
            # ss2: plan ahead.
            id_pion = self.myopic.get_pioneer_waypoint_index()
            t2 = time.time()
            print("Each step takes: ", t2 - t1)

        trajectory = self.waypoints[self.myopic.get_trajectory_indices()]
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        dz = np.diff(trajectory[:, 2])
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.assertLess(np.amax(dist), 120.1)

        # c2: check if it has sharp turns
        sharp_turn = False
        for i in range(0, len(dx)-1):
            v1 = np.stack((dx[i], dy[i], dz[i]), axis=0)
            v2 = np.stack((dx[i+1], dy[i+1], dz[i+1]), axis=0)
            if np.dot(v1, v2) < 0:
                sharp_turn = True
                break
        self.assertFalse(sharp_turn)







#%% Separate part used for visualisation
#             # here comes the plotting
#             # p0: setup the plotting env
#             import matplotlib.pyplot as plt
#             import plotly
#             import plotly.graph_objects as go
#
#             # p1: all waypoints
#             fig = go.Figure(data=[go.Scatter3d(
#                 x=self.waypoints[:, 1],
#                 y=self.waypoints[:, 0],
#                 z=-self.waypoints[:, 2],
#                 mode='markers',
#                 marker=dict(
#                     size=1,
#                     color="black",
#                     opacity=.3,
#                 ),
#             )])
#
#             # p2: current waypoint in red
#             id = self.myopic.get_current_index()
#             fig.add_trace(go.Scatter3d(
#                 x=[self.waypoints[id, 1]],
#                 y=[self.waypoints[id, 0]],
#                 z=[-self.waypoints[id, 2]],
#                 mode='markers',
#                 marker=dict(
#                     size=10,
#                     color="red",
#                 ),
#             ))
#
#             # p3: next waypoint in orange
#             id = self.myopic.get_next_index()
#             fig.add_trace(go.Scatter3d(
#                 x=[self.waypoints[id, 1]],
#                 y=[self.waypoints[id, 0]],
#                 z=[-self.waypoints[id, 2]],
#                 mode='markers',
#                 marker=dict(
#                     size=10,
#                     color="orange",
#                 ),
#             ))
#
#             # p4: pioneer waypoint in green
#             id = self.myopic.get_pioneer_index()
#             fig.add_trace(go.Scatter3d(
#                 x=[self.waypoints[id, 1]],
#                 y=[self.waypoints[id, 0]],
#                 z=[-self.waypoints[id, 2]],
#                 mode='markers',
#                 marker=dict(
#                     size=10,
#                     color="green",
#                 ),
#             ))
#
#             # p5: trajectory, needs to convert indices to locations.
#             id = self.myopic.get_trajectory_indices()
#             fig.add_trace(go.Scatter3d(
#                 x=self.waypoints[id, 1],
#                 y=self.waypoints[id, 0],
#                 z=-self.waypoints[id, 2],
#                 mode='markers + lines',
#                 marker=dict(
#                     size=4,
#                     color="black",
#                 ),
#                 line=dict(
#                     width=2,
#                     color="blue"
#                 )
#             ))
#
#             fig.update_layout(
#                 title={
#                     'text': "Myopic3D planning illustration",
#                     'y': 0.9,
#                     'x': 0.5,
#                     'xanchor': 'center',
#                     'yanchor': 'top',
#                     'font': dict(size=30, family="Times New Roman"),
#                 },
#                 scene=dict(
#                     # xaxis=dict(range=[np.amin(points_int[:, 0]), np.amax(points_int[:, 0])]),
#                     # yaxis=dict(range=[np.amin(points_int[:, 1]), np.amax(points_int[:, 1])]),
#                     # zaxis=dict(nticks=4, range=[-6, .5], ),
#                     xaxis_tickfont=dict(size=14, family="Times New Roman"),
#                     yaxis_tickfont=dict(size=14, family="Times New Roman"),
#                     zaxis_tickfont=dict(size=14, family="Times New Roman"),
#                     xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
#                     yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
#                     zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
#                 ),
#                 scene_aspectmode='manual',
#                 scene_aspectratio=dict(x=1, y=1, z=.4),
#                 # scene_camera=camera,
#             )
#
#             # p6: save it to image or html
#             figpath = os.getcwd() + "/../fig/Myopic3D/P_{:03d}.html".format(i)
#             plotly.offline.plot(fig, filename=figpath, auto_open=False)

