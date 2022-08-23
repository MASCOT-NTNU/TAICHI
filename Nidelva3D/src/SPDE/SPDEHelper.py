"""
This helper solves the auxiliary problems associated with SPDE class.
"""

from SPDE.spde import spde
from typing import Union
from WGS import WGS
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import os


class SPDEHelper:
    __gmrf_grid = None

    def __init__(self):
        self.__spde = spde()
        self.construct_gmrf_grid()

    def construct_gmrf_grid(self) -> None:
        """
        Construct GMRF grid by converting lats, lons to xy.
        """
        filepath = os.getcwd() + "/SPDE/models/"
        lat = np.load(filepath + "lats.npy")
        lon = np.load(filepath + "lons.npy")
        depth = np.load(filepath + "depth.npy")
        x, y = WGS.latlon2xy(lat, lon)
        z = depth
        self.__gmrf_grid = np.stack((x, y, z), axis=1)

    def get_eibv_at_locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Get EIBV at candidate locations.
        Args:
            locations: np.array([[x1, y1, z1],
                                 [x2, y2, z2],
                                 ...
                                 [xn, yn, zn]])
        Returns:
            EIBV associated with each location.
        """
        # s1: get indices
        id = self.get_ind_from_location(locations)
        # s2: get post variance from spde
        post_var = self.__spde.candidate(ks=id)
        # s3: get eibv using post variance
        eibv = []
        for i in range(len(id)):
            p = norm.cdf(self.__spde.threshold, self.__spde.mu, post_var[:, i])
            bv = p * (1 - p)
            ibv = np.sum(bv)
            eibv.append(ibv)
        return np.array(eibv)

    def get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            location: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                d = cdist(self.__gmrf_grid, loc.reshape(1, -1))
                return np.argmin(d, axis=0)[0]
            elif dm == 2:
                d = cdist(self.__gmrf_grid, loc)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_location_from_ind(self, ind: Union[int, list]) -> np.ndarray:
        return self.__gmrf_grid[ind]

    def get_gmrf_grid(self):
        return self.__gmrf_grid


if __name__ == "__main__":
    s = SPDEHelper()


