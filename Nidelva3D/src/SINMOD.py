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
    def __init__(self, filepath: str = None) -> None:
        if filepath is None:
            raise ValueError("Please provide the filepath to SINMOD data.")
        else:
            self.sinmod_filepath = filepath
            self.sinmod_data = netCDF4.Dataset(self.sinmod_filepath)
            ind_before = re.search("samples_", self.sinmod_filepath)
            ind_after = re.search(".nc", self.sinmod_filepath)
            date_string = self.sinmod_filepath[ind_before.end():ind_after.start()]
            ref_timestamp = datetime.strptime(date_string, "%Y.%m.%d").timestamp()
            timestamp = np.array(self.sinmod_data["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp

            lat_sinmod = np.array(self.sinmod_data['gridLats'])
            lon_sinmod = np.array(self.sinmod_data['gridLons'])
            depth_sinmod = np.array(self.sinmod_data['zc'])
            salinity_time_ave = np.mean(np.array(self.sinmod_data['salinity'])[:, :, :, :], axis=0)
            df_sinmod = []
            for i in range(lat_sinmod.shape[0]):
                for j in range(lat_sinmod.shape[1]):
                    for k in range(len(depth_sinmod)):
                        df_sinmod.append([lat_sinmod[i, j], lon_sinmod[i, j],
                                            depth_sinmod[k], salinity_time_ave[k, i, j]])
            df_sinmod = np.array(df_sinmod)
            x, y = WGS.latlon2xy(df_sinmod[:, 0], df_sinmod[:, 1])

            self.data_sinmod = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), df_sinmod[:, 2:]))
            t1 = time.time()
            self.sinmod_grid_tree = KDTree(self.data_sinmod[:, :3])
            t2 = time.time()
            print("KDTree construction time: ", t2 - t1)

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
        dist, ind = self.sinmod_grid_tree.query(locations.astype(np.float32))
        sal_interpolated = self.data_sinmod[ind, -1].reshape(-1, 1)
        df_interpolated = np.hstack((locations, sal_interpolated))
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated

    def get_data(self) -> np.ndarray:
        """
        Return the dataset of SINMOD data.
        """
        return self.data_sinmod


if __name__ == "__main__":
    s = SINMOD()