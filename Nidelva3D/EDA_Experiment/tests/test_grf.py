"""
Test module for the GRF class.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

"""

from unittest import TestCase
from GRF.GRF import GRF
from WGS import WGS
from usr_func.interpolate_3d import interpolate_3d
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import get_cmap
from numpy import testing
import os
import plotly.graph_objects as go
import plotly


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.grf = GRF()
        self.grid = self.grf.get_grid()
        self.mu = self.grf.get_mu()
        self.cov = self.grf.get_covariance_matrix()
        self.sigma_diag = np.diag(self.cov)

    def test_prior_mu(self) -> None:
        self.plot_mu("/Users/yaolin/Downloads/mu")
        self.plot_var("/Users/yaolin/Downloads/var")

    def test_get_ind_from_location(self) -> None:
        """
        Test if given a location, it will return the correct index in GMRF grid.
        """
        # c1: one location
        ide = 10
        loc = self.grf.get_location_from_ind(ide)
        id = self.grf.get_ind_from_location(loc)
        self.assertEqual(ide, id)

        # c2: more locations
        ide = [10, 12]
        loc = self.grf.get_location_from_ind(ide)
        id = self.grf.get_ind_from_location(loc)
        self.assertIsNone(testing.assert_array_equal(ide, id))

    def test_get_eibv_at_locations(self) -> None:
        """
        Test if it can return eibv for the given locations.
        """
        # id = [1, 2, 3, 4, 5]
        ind = np.random.randint(0, 1000, 5)
        loc = self.grf.get_location_from_ind(ind)
        eibv = self.grf.get_eibv_at_locations(loc)
        print("EIBV: ", eibv)
        self.assertIsNotNone(eibv)

    def test_assimilate_data(self) -> None:
        """
        Test if it can assimilate data with given dataset.
        - 100 grid points within the grid.
        - 10 replicates with 10 grid points not within the grid.
        - no location.
        """
        # c1: grid points on grid
        ind = np.random.randint(0, self.grid.shape[0], 100)
        x = self.grid[ind, 0]
        y = self.grid[ind, 1]
        z = self.grid[ind, 2]
        v = np.zeros_like(z)
        dataset = np.stack((x, y, z, v), axis=1)
        ida, sal_a, ind_min = self.grf.assimilate_data(dataset)
        dx = self.grid[ind_min, 0] - x
        dy = self.grid[ind_min, 1] - y
        dz = self.grid[ind_min, 2] - z
        gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.assertLess(np.amax(gap), .01)

        # c2: random locations
        for i in range(10):
            ind = np.random.randint(0, self.grid.shape[0], 10)
            x = self.grid[ind, 0] + np.random.randn(len(ind))
            y = self.grid[ind, 1] + np.random.randn(len(ind))
            z = self.grid[ind, 2] + np.random.randn(len(ind))
            v = np.zeros_like(z)
            dataset = np.stack((x, y, z, v), axis=1)
            ida, sal_a, ind_min = self.grf.assimilate_data(dataset)
            id = np.where(np.abs(z) >= .25)[0]
            dx = self.grid[ind_min, 0] - x[id]
            dy = self.grid[ind_min, 1] - y[id]
            dz = self.grid[ind_min, 2] - z[id]
            gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            self.assertLess(np.amax(gap), 5.2)

        # c3: no location
        dataset = np.empty([0, 4])
        ida, sal_a, ind_min = self.grf.assimilate_data(dataset)
        self.assertTrue([True if len(ida) == 0 else False])

    def test_get_updated_mu_mvar(self):
        """
        Test if the assimilation works as desired.
        - c1: 1 value.
        - c2: many values.
        """
        rotated_angle = self.grf.get_rotated_angle()

        # c1: random location
        lat, lon = 63.450604, 10.418646
        depth = .5
        value = 21
        x, y = WGS.latlon2xy(lat, lon)
        dataset = np.array([[x, y, depth, value]])
        self.grf.assimilate_data(dataset)
        file = "/Users/yaolin/Downloads/"
        self.plot_mu(file + "mu_cond")
        self.plot_var(file + "var_cond")

        # c2: desired location
        lat, lon = 63.453648,10.420787
        depth = .5
        value = 10
        x, y = WGS.latlon2xy(lat, lon)
        dataset = np.array([[x, y, depth, value]])
        self.grf.assimilate_data(dataset)
        # file = "/Users/yaoling/Downloads/"
        self.plot_mu(file + "mu_cond2")
        self.plot_var(file + "var_cond2")

        # c3: mutiple desired location
        loc = np.array([[63.449821,10.397046],
                        [63.447082,10.408625],
                        [63.455714,10.409890]])
        lat = loc[:, 0]
        lon = loc[:, 1]
        depth = np.array([.5, 1.5, 2.5])
        value = 10
        x, y = WGS.latlon2xy(lat, lon)
        value = np.array([30, 28, 25])
        dataset = np.vstack((x, y, depth, value)).T
        self.grf.assimilate_data(dataset)
        # file = "/Users/yaoling/Downloads/"
        self.plot_mu(file + "mu_cond3")
        self.plot_var(file + "var_cond3")

    def plot_mu(self, filename=None):
        vmin = 0
        vmax = 28
        ind_remove_top_layer = np.where(self.grid[:, 2] > 0)[0]
        xgrid = self.grid[ind_remove_top_layer, 0]
        ygrid = self.grid[ind_remove_top_layer, 1]
        rotated_angle = self.grf.get_rotated_angle()
        xrotated = xgrid * np.cos(rotated_angle) - ygrid * np.sin(rotated_angle)
        yrotated = xgrid * np.sin(rotated_angle) + ygrid * np.cos(rotated_angle)
        xplot = yrotated
        yplot = xrotated
        mu = self.grf.get_mu()
        mu[mu < 0] = 0
        ind_selected_to_plot = np.where(mu[ind_remove_top_layer] >= 0)[0]
        xplot = xplot[ind_selected_to_plot]
        yplot = yplot[ind_selected_to_plot]
        zplot = -self.grid[ind_remove_top_layer, 2][ind_selected_to_plot]
        points_mean, values_mean = interpolate_3d(xplot, yplot, zplot, mu[ind_remove_top_layer][ind_selected_to_plot])
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(go.Volume(
            x=points_mean[:, 0],
            y=points_mean[:, 1],
            z=points_mean[:, 2],
            value=values_mean,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=10,
            colorscale="BrBG",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Mean field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename + ".html", auto_open=True)

    def plot_var(self, filename=None):
        ind_remove_top_layer = np.where(self.grid[:, 2] > 0)[0]
        xgrid = self.grid[ind_remove_top_layer, 0]
        ygrid = self.grid[ind_remove_top_layer, 1]
        rotated_angle = self.grf.get_rotated_angle()
        xrotated = xgrid * np.cos(rotated_angle) - ygrid * np.sin(rotated_angle)
        yrotated = xgrid * np.sin(rotated_angle) + ygrid * np.cos(rotated_angle)
        xplot = yrotated
        yplot = xrotated
        mvar = np.diag(self.grf.get_covariance_matrix())
        vmin = np.amin(mvar)
        vmax = np.amax(mvar)

        zplot = -self.grid[ind_remove_top_layer, 2]
        points_mean, values_mean = interpolate_3d(xplot, yplot, zplot, mvar[ind_remove_top_layer])

        # fig = go.Figure(data=go.Scatter3d(
        #     x=points_mean[:, 0],
        #     y=points_mean[:, 1],
        #     z=points_mean[:, 2],
        #     mode="markers",
        #     marker=dict(
        #         size=4,
        #         color=values_mean,
        #         colorscale="RdBu",
        #         showscale=True,
        #     )
        # ))

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        fig.add_trace(go.Volume(
            x=points_mean[:, 0],
            y=points_mean[:, 1],
            z=points_mean[:, 2],
            value=values_mean,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=10,
            colorscale="RdBu",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Marginal variance field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename + ".html", auto_open=True)
