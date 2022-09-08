""" Unit test for SINMOD data handler
"""

from unittest import TestCase
from AUVSimulator.SINMOD import SINMOD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from shapely.geometry import Polygon, Point


class TestSINMOD(TestCase):
    """

    """
    def setUp(self) -> None:
        self.sinmod = SINMOD()

    def test_get_data_from_sinmod(self) -> None:
        # c1: one depth layer
        N = 40
        lat = np.linspace(63.447973, 63.463720, N)
        lon = np.linspace(10.395031, 10.425818, N)
        # depth = np.random.uniform(0.5, 5.5, 5)
        depth = [0.5]
        coord = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                for k in range(len(depth)):
                    coord.append([lat[i], lon[j], depth[k]])
        coord = np.array(coord)
        df = self.sinmod.get_data_at_coordinates(coord)
        plt.scatter(df[:, 1], df[:, 0], c=df[:, -1],
                    cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
        plt.show()

        # c2: many depth layers
        N = 30
        lat = np.linspace(63.447973, 63.463720, N)
        lon = np.linspace(10.395031, 10.425818, N)
        depth = np.linspace(0.5, 5.5, 3)
        coord = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                for k in range(len(depth)):
                    coord.append([lat[i], lon[j], depth[k]])
        coord = np.array(coord)
        df = self.sinmod.get_data_at_coordinates(coord)
        fig = plt.figure(figsize=(len(depth) * 8, 8))
        gs = GridSpec(ncols=len(depth), nrows=1, figure=fig)
        for i in range(len(depth)):
            ind = np.where(coord[:, 2] == depth[i])[0]
            fig.add_subplot(gs[i])
            plt.scatter(df[ind, 1], df[ind, 0], c=df[ind, -1],
                        cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
            plt.colorbar()
        plt.show()


