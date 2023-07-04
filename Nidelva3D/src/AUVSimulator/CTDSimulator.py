"""
CTDSimulator simulates the CTD sensor sampling in the ground truth field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28

"""
from SINMOD import SINMOD
from GMRF.spde import spde
import os
from datetime import datetime
import numpy as np
from typing import Union
from pykdtree.kdtree import KDTree
from time import time


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    def __init__(self, random_seed: int = 0,
                 filepath: str = os.getcwd() + "/../sinmod/truth_samples_2022.09.08.nc") -> None:
        """
        Set up the CTD simulated truth field.
        """
        self.sigma = spde().sigma
        np.random.seed(random_seed)
        filepath_sinmod = filepath
        # date_string = filepath_sinmod.split("/")[-1].split("_")[-1][:-3]
        self.timestamp = datetime.strptime("2022-09-08 10:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
        self.sinmod = SINMOD(filepath_sinmod)

        self.timestamp_sinmod = self.sinmod.get_timestamp()
        self.timestamp_sinmod_tree = KDTree(self.timestamp_sinmod.reshape(-1, 1))
        self.salinity_sinmod = self.sinmod.get_salinity()
        self.grid_sinmod = self.sinmod.get_data()[:, :3]
        self.grid_sinmod_tree = KDTree(self.grid_sinmod)

    def get_salinity_at_dt_loc(self, dt: float, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a given time and location.
        """
        self.timestamp += dt
        print("Current datetime: ", datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        ts = np.array([self.timestamp])
        dist, ind_time = self.timestamp_sinmod_tree.query(ts)
        t1 = time()
        sorted_salinity = []
        for i in range(self.salinity_sinmod.shape[-2]):
            for j in range(self.salinity_sinmod.shape[-1]):
                for k in range(self.salinity_sinmod.shape[1]):
                    sorted_salinity.append(self.salinity_sinmod[ind_time, k, i, j])
        sorted_salinity = np.array(sorted_salinity)
        dist, ind_loc = self.grid_sinmod_tree.query(loc)
        print("Query salinity at timestamp and location takes: ", time() - t1)
        return sorted_salinity[ind_loc, 0] + np.random.normal() * self.sigma

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


