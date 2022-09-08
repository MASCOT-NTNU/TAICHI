""" Unit test for Agent.
This module tests the agent object.
"""

from unittest import TestCase
from Planner.ZigZag import ZigZag
from WGS import WGS
from GMRF.GMRF import GMRF
from usr_func.sort_polygon_vertices import sort_polygon_vertices
import numpy as np
import os


class TestZigZag(TestCase):

    def setUp(self) -> None:
        self.zz = ZigZag()
        # self.zz.set_depth_limit([0.5, 5.5])
        # self.zz.set_maximum_pitch(10)
        # self.zz.set_marginal_distance(100)
        # __BOX = np.load(os.getcwd() + "/GMRF/models/grid.npy")
        # __POLYGON = __BOX[:, 2:]
        # __POLYGON_XY = np.stack((WGS.latlon2xy(__POLYGON[:, 0], __POLYGON[:, 1])), axis=1)
        # __POLYGON_BORDER = sort_polygon_vertices(__POLYGON_XY)
        self.g = GMRF()
        # rot_angle = self.g.get_rotated_angle()
        # self.zz.set_polygon_border(__POLYGON_BORDER)
        # self.zz.set_rotated_angle(rot_angle)

    def test_construct_path(self):
        self.zz.construct_path()
        path = self.zz.get_zigzag_path()
        path[:, 2] = -path[:, 2]
        grid = self.g.get_gmrf_grid()
        grid[:, 2] = -grid[:, 2]

        import plotly
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter3d(
            x=path[:, 1],
            y=path[:, 0],
            z=path[:, 2],
            mode="markers + lines",
            marker=dict(
                size=5,
                color='black',
            ),
            line=dict(
                width=2,
                color="yellow",
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=grid[:, 1],
            y=grid[:, 0],
            z=grid[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color='blue',
                opacity=.1,
            ),
        ))

        id = 0
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='green',
            ),
        ))

        id = 10
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='red',
            ),
        ))

        id = 50
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='pink',
            ),
        ))

        id = 100
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='blue',
            ),
        ))

        id = 150
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='purple',
            ),
        ))

        id = 200
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='cyan',
            ),
        ))

        id = -1
        fig.add_trace(go.Scatter3d(
            x=[path[id, 1]],
            y=[path[id, 0]],
            z=[path[id, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color='orange',
            ),
        ))
        plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/test_yoyo.html", auto_open=True)

    def test_get_yoyo_between_two_vertices(self):
        v1 = np.array([1600, 2000])
        v2 = np.array([2100, 2800])
        self.zz.construct_path()
        path = self.zz.get_yoyo_path_between_vertices(v1, v2)

        import plotly
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter3d(
            x=path[:, 1],
            y=path[:, 0],
            z=path[:, 2],
            mode="markers + lines",
            marker=dict(
                size=5,
                color='black',
            ),
            line=dict(
                width=2,
                color="yellow",
            )
        ))

        plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/test_yoyo.html", auto_open=True)


