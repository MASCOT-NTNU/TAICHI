"""
Test module for the CTD.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

"""

from unittest import TestCase
from CTD import CTD
import numpy as np
from matplotlib.pyplot import get_cmap
import matplotlib.pyplot as plt


class TestCTD(TestCase):

    def setUp(self) -> None:
        self.ctd = CTD(loc_start=np.array([0, 0, 0]), random_seed=0)
        self.mu = self.ctd.get_ground_truth()
        self.grid = self.ctd.grid

    def test_get_salinity_at_loc(self) -> None:
        """
        Test get salinity from location
        """
        # c1: value at the corners
        loc = np.array([2750, 1000, .5])
        self.ctd.get_salinity_at_loc(loc)
        truth = self.ctd.get_ground_truth()
        ind_depth = np.where(self.grid[:, -1] == loc[:, -1])[0]
        value = truth[ind_depth]

        plt.scatter(self.grid[ind_depth, 1], self.grid[ind_depth, 0], c=value, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

    def test_get_data_along_path(self) -> None:
        # c1: move to one step
        loc = np.array([3000, 1000, .5])
        data = self.ctd.get_ctd_data(loc)
        grid = self.grid
        truth = self.ctd.get_ground_truth()
        ind_depth = np.where(grid[:, -1] == loc[-1])[0]
        plt.figure()
        plt.scatter(grid[ind_depth, 1], grid[ind_depth, 0], c=truth[ind_depth], cmap=get_cmap("BrBG", 10),
                    vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

        # c2: move to another direction
        loc = np.array([2250, 500, 2.5])
        data = self.ctd.get_ctd_data(loc)
        ind_depth = np.where(grid[:, -1] == loc[-1])[0]
        plt.figure()
        plt.scatter(grid[ind_depth, 1], grid[ind_depth, 0], c=truth[ind_depth], cmap=get_cmap("BrBG", 10),
                    vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

        # c2: move to another direction
        loc = np.array([3500, 750, 4.5])
        data = self.ctd.get_ctd_data(loc)
        ind_depth = np.where(grid[:, -1] == loc[-1])[0]
        plt.figure()
        plt.scatter(grid[ind_depth, 1], grid[ind_depth, 0], c=truth[ind_depth], cmap=get_cmap("BrBG", 10),
                    vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()
