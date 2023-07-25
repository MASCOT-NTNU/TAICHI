""" Unit test for CTD simulator
"""

from unittest import TestCase
from WGS import WGS
from AUVSimulator.CTDSimulator import CTDSimulator
from SINMOD import SINMOD
from Planner.Myopic3D import Myopic3D
import numpy as np
from numpy import testing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from datetime import datetime
import os


class TestCTDSimulator(TestCase):

    def setUp(self) -> None:
        self.ctd = CTDSimulator(random_seed=12)
        # path = os.getcwd() + "/AUVSimulator/simulated_truth_wgs.csv"
        # self.truth = pd.read_csv(path).to_numpy()
        filepath_sinmod = os.getcwd() + "/../sinmod/truth_samples_2022.09.09.nc"
        self.sinmod = SINMOD(filepath_sinmod)

    def test_get_salinity_at_time_loc(self) -> None:
        # c1, any time stamp
        dt = 1200
        x, y, depth = self.sinmod.get_coordinates()
        z = np.ones_like(x) * .5
        loc = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        salinity = self.ctd.get_salinity_at_dt_loc(dt, loc)
        plt.figure()
        plt.scatter(loc[:, 1], loc[:, 0], c=salinity, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
        plt.colorbar()
        plt.title("Time: " + datetime.fromtimestamp(self.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        plt.show()

        # c2, any time stamp
        dt = 120000
        salinity = self.ctd.get_salinity_at_dt_loc(dt, loc)
        plt.figure()
        plt.scatter(loc[:, 1], loc[:, 0], c=salinity, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
        plt.colorbar()
        plt.title("Time: " + datetime.fromtimestamp(self.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        plt.show()

        # c3, any locations
        dt = 14400
        xx = np.linspace(2000, 4000, 100)
        yy = np.linspace(0, 2000, 100)
        xv, yv = np.meshgrid(xx, yy)
        x = xv.flatten()
        y = yv.flatten()
        z = np.ones_like(x) * .5
        loc = np.vstack((x, y, z)).T
        salinity = self.ctd.get_salinity_at_dt_loc(dt, loc)
        plt.figure()
        plt.scatter(loc[:, 1], loc[:, 0], c=salinity, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
        plt.colorbar()
        plt.title("Time: " + datetime.fromtimestamp(self.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        plt.show()

    # def test_get_salinity_at_loc(self):
    #     """
    #     Test get salinity from location
    #     """
    #     # c1: value at the corners
    #     path = os.getcwd() + "/GMRF/models/grid.npy"
    #     box = np.load(path)
    #     lat = box[:, 2]
    #     lon = box[:, -1]
    #     x, y = WGS.latlon2xy(lat, lon)
    #     z = .5 * np.ones_like(x)
    #     loc = np.vstack((x, y, z)).T
    #     v = self.ctd.get_salinity_at_loc(loc)
    #     ve = self.sinmod.get_data_at_locations(loc)[:, -1]
    #     self.assertIsNone(testing.assert_array_almost_equal(v, ve))

        # # c2: value within the field
        # lat, lon = 63.456175, 10.402070
        # x, y = WGS.latlon2xy(lat, lon)
        # z = 0.5
        # loc = np.array([x, y, z])
        # v = self.ctd.get_salinity_at_loc(loc)
        # loc_wgs = np.array([lat, lon, z]).reshape(1, -1)
        # ve = self.sinmod.get_data_at_coordinates(loc_wgs)[:, -1]
        # self.assertIsNone(testing.assert_array_almost_equal(v, ve))

        # # c3: value outside
        # lat, lon = 63.446239, 10.380011
        # x, y = WGS.latlon2xy(lat, lon)
        # z = 0.5
        # loc = np.array([x, y, z])
        # v = self.ctd.get_salinity_at_loc(loc)
        # loc_wgs = np.array([lat, lon, z]).reshape(1, -1)
        # ve = self.sinmod.get_data_at_coordinates(loc_wgs)[:, -1]
        # self.assertIsNone(testing.assert_array_almost_equal(v, ve))

        # # c4: all values inside
        # m = Myopic3D()
        # g = m.gmrf.get_gmrf_grid()
        # ind = np.random.randint(0, len(g), 200)
        # loc = g[ind, :]
        # v = self.ctd.get_salinity_at_loc(loc)
        # lat, lon = WGS.xy2latlon(loc[:, 0], loc[:, 1])
        # loc_wgs = np.vstack((lat, lon, loc[:, 2])).T
        # ve = self.sinmod.get_data_at_coordinates(loc_wgs)[:, -1]
        # self.assertIsNone(testing.assert_array_almost_equal(v, ve))
