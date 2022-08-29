"""
CTDSimulator module simulates CTD sensor.
"""

import numpy as np
from typing import Union
from scipy.spatial.distance import cdist


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    __field = np.empty([0, 4])
    __field_grid = np.empty([0, 3])
    __field_salinity = np.empty([0, 1])

    def __int__(self):
        pass

    def setup_ctd(self, field_truth: np.ndarray):
        """
        Set the simulated field.
        Args:
            field_truth: data matrix containing np.array([[x, y, z, s]]).
            - x: distance along east direction.
            - y: distance along north direction.
            - z: distance along depth direction.
        """
        self.__field = field_truth
        self.__field_grid = self.__field[:, :3]
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
                d = cdist(self.__field_grid, loc.reshape(1, -1))
                return np.argmin(d, axis=0)
            elif dm == 2:
                d = cdist(self.__field_grid, loc)
                return np.argmin(d, axis=0)
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


