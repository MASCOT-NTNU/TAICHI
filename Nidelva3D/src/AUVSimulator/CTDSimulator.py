"""
CTDSimulator module simulates CTD sensor.
"""
from GMRF.GMRF import GMRF
from AUVSimulator.SINMOD import SINMOD
from WGS import WGS
import os
import numpy as np
import pandas as pd
from typing import Union
from scipy.spatial.distance import cdist
from pykdtree.kdtree import KDTree


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    __field = None
    __field_grid = None
    __field_salinity = None

    def __init__(self):
        """
        Set up the CTD simulated truth field.
        """
        gmrf = GMRF()
        sinmod = SINMOD()
        grid_xy = gmrf.get_gmrf_grid()
        lat, lon = WGS.xy2latlon(grid_xy[:, 0], grid_xy[:, 1])
        grid_wgs = np.stack((lat, lon, grid_xy[:, 2]), axis=1)
        grid_salinity = sinmod.get_data_at_coordinates(grid_wgs)[:, -1]
        self.__field = np.hstack((grid_xy, grid_salinity.reshape(-1, 1)))
        self.__field_grid = self.__field[:, :3]
        self.__field_grid_tree = KDTree(self.__field_grid)
        self.__field_salinity = self.__field[:, -1]

    def __get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                return self.__field_grid_tree.query(loc.reshape(1, -1))[1]
            elif dm == 2:
                return self.__field_grid_tree.query(loc)[1]
            else:
                return None
        else:
            return None

    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y, z]])

        Returns:
            salinity value at loc
        """
        ind = self.__get_ind_from_location(loc)
        if ind is not None:
            return self.__field_salinity[ind]
        else:
            return None


if __name__ == "__main__":
    c = CTDSimulator()


