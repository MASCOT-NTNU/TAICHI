"""
SINMOD module handles the data interpolation for a given set of coordinates.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

Methodology:
    1. Read SINMOD data from netCDF file.
    2. Construct KDTree for the SINMOD grid.
    3. For a given set of coordinates, find the nearest SINMOD grid point.
    4. Interpolate the data using the nearest SINMOD grid point.
"""
from WGS import WGS
from pykdtree.kdtree import KDTree
import os
import re
import numpy as np
import netCDF4
from datetime import datetime
import time


class SINMOD:
    """
    SINMOD class handles the data interpolation for a given set of coordinates.
    """
    __SINMOD_MAX_DEPTH_LAYER = 10
    __sinmod_filepath = os.getcwd() + "/../sinmod/"
    __sinmod_files = os.listdir(__sinmod_filepath)
    __sinmod_files.sort()
    __sinmod_file = __sinmod_files[-1]  # always get the latest file
    __sinmod_data = netCDF4.Dataset(__sinmod_filepath + __sinmod_file)

    __ind_before = re.search("samples_", __sinmod_file)
    __ind_after = re.search(".nc", __sinmod_file)
    __date_string = __sinmod_file[__ind_before.end():__ind_after.start()]
    __ref_timestamp = datetime.strptime(__date_string, "%Y.%m.%d").timestamp()
    __timestamp = np.array(__sinmod_data["time"]) * 24 * 3600 + __ref_timestamp  # change ref timestamp

    __lat_sinmod = np.array(__sinmod_data['gridLats'])
    __lon_sinmod = np.array(__sinmod_data['gridLons'])
    __depth_sinmod = np.array(__sinmod_data['zc'])[:__SINMOD_MAX_DEPTH_LAYER]

    __salinity_time_ave = np.mean(np.array(__sinmod_data['salinity'])[:, :__SINMOD_MAX_DEPTH_LAYER, :, :], axis=0)
    __df_sinmod = []
    for i in range(__lat_sinmod.shape[0]):
        for j in range(__lat_sinmod.shape[1]):
            for k in range(len(__depth_sinmod)):
                __df_sinmod.append([__lat_sinmod[i, j], __lon_sinmod[i, j],
                                    __depth_sinmod[k], __salinity_time_ave[k, i, j]])
    __df_sinmod = np.array(__df_sinmod)
    x, y = WGS.latlon2xy(__df_sinmod[:, 0], __df_sinmod[:, 1])
    __data_sinmod = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), __df_sinmod[:, 2:]))
    t1 = time.time()
    __sinmod_grid_tree = KDTree(__data_sinmod[:, :3])
    t2 = time.time()
    print("KDTree construction time: ", t2 - t1)

    def __init__(self) -> None:
        pass

    @staticmethod
    def __vectorize(value: np.ndarray) -> np.ndarray:
        """
        Reshape value to a column vector.
        Args:
            value: flattened array with shape (n,)
        Returns:
            vectorized array with shape (n, 1)
        """
        return value.reshape(-1, 1)

    @staticmethod
    def __get_distance_matrix(s1, s2):
        """
        Return distance matrix between two column vectors.
        Args:
            s1: site one
            s2: site two
        Returns:
            Distance matrix from site one to site two.
        """
        s1 = np.array(s1).reshape(-1, 1)
        s2 = np.array(s2).reshape(-1, 1)
        dx = np.dot(s1, np.ones([1, len(s2)]))
        dy = np.dot(np.ones([len(s1), 1]), s2.T)
        return dx - dy

    def get_data_at_locations(self, locations: np.array) -> np.ndarray:
        """
        Get SINMOD data values at given locations.

        Args:
            location: x, y, depth coordinates
            Example: np.array([[x1, y1, depth1],
                               [x2, y2, depth2],
                               ...
                               [xn, yn, depthn]])
        Returns:
            SINMOD data values at given locations.
        """
        ts = time.time()
        dist, ind = self.__sinmod_grid_tree.query(locations.astype(np.float32))
        sal_interpolated = self.__df_sinmod[ind, -1].reshape(-1, 1)
        df_interpolated = np.hstack((locations, sal_interpolated))
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated


if __name__ == "__main__":
    s = SINMOD()